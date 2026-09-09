"""Persistent, single-GPU execution of the frozen Stage 11 campaign.

Public API: ``supervise(root, phase='core', resume_supervisor=False)``.
The core phase trains the fixed matrix, evaluates validation controls, then
stops at ``awaiting_primary_review``. ``resume_after_review(root)`` consumes
an authenticated test_unlock.json; it cannot create or infer an unlock.
``phase='diagnostic_render'`` is the primary agent's explicit choice to render
validation diagnostics without opening test data.

Completed runs are reused only after authentication and artifact checks.
Restarting an interrupted supervisor requires ``resume_supervisor=True``.
An incomplete training run additionally needs authenticated
``recovery_authorizations.json`` with ``protocol_sha256`` and an
``authorizations`` list containing the exact ``arm``, ``seed``, nonempty
``reason``, ``no_observed_final: true`` and ``checkpoint_sha256`` for latest.pt.
No retry is inferred from a bad score or a nonzero process exit.

The optional runner hook is for CPU tests. Production commands are built
from fixed operations and validated arm/seed values, never from free-form
protocol commands. Python is dispatched through a pinned PATH and imports
the frozen source snapshot, independent of the live checkout.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import threading
import time
from typing import Callable

from .stage11_common import (
    ROOT, atomic_json, authenticated_json, digest, json_digest, read_authenticated,
    read_json, require_space, verify_artifacts, verify_sources,
)


DESCRIPTOR_ARMS = ("D_old", "D_full")
SCENE_ARMS = ("S_pointwise", "S_temporal", "S_swapped")
ALL_ARMS = DESCRIPTOR_ARMS + SCENE_ARMS
TORCH_ENV = Path("/mnt/lightsem-data/venv-torch")
MIN_TRAINING_SECONDS = 10 * 3600
EVALUATION_RESERVE_SECONDS = 2 * 3600


@dataclass(frozen=True)
class Job:
    operation: str
    arm: str | None = None
    seed: int | None = None
    split: str | None = None

    @property
    def key(self) -> str:
        return ":".join(str(value) for value in
                        (self.operation, self.split, self.arm, self.seed) if value is not None)

    @property
    def uses_gpu(self) -> bool:
        return self.operation in {"train", "evaluate", "render"}


class CampaignDeadlineExceeded(RuntimeError):
    pass


def training_queue(protocol: dict) -> list[Job]:
    """Matched seed rounds, rotating arm order to balance hardware chronology."""
    seeds = protocol.get("training_seeds")
    if seeds != list(range(5)):
        raise ValueError("the frozen core requires training_seeds [0, 1, 2, 3, 4]")
    for arm in ALL_ARMS:
        cfg = protocol.get("arms", {}).get(arm)
        if not cfg or not isinstance(cfg.get("steps"), int) or cfg["steps"] <= 0:
            raise ValueError(f"missing/invalid frozen steps for {arm}")
        expected_family = "motion" if arm in DESCRIPTOR_ARMS else "scene"
        if cfg.get("family") != expected_family:
            raise ValueError(f"{arm} must belong to family {expected_family}")
    jobs = []
    for arms in (DESCRIPTOR_ARMS, SCENE_ARMS):
        for seed in seeds:
            rotation = seed % len(arms)
            for arm in arms[rotation:] + arms[:rotation]:
                jobs.append(Job("train", arm, seed))
    return jobs


def command_for_job(root, job: Job, *, resume_training: bool = False) -> list[str]:
    if job.operation not in {"train", "evaluate", "aggregate", "render"}:
        raise ValueError("unknown campaign operation")
    if job.operation in {"train", "evaluate"}:
        if job.arm not in ALL_ARMS or job.seed not in range(6):
            raise ValueError("arm/seed outside the frozen campaign interface")
    if job.operation != "train" and job.split not in {"val", "test"}:
        raise ValueError("evaluation operations require split val or test")
    command = ["python", "-m", "lightsem.experiments.stage11", "--root", str(Path(root).resolve()), job.operation]
    if job.arm is not None:
        command += ["--arm", job.arm, "--seed", str(job.seed)]
    if job.split is not None:
        command += ["--split", job.split]
    if job.operation == "evaluate":
        command += ["--controls", "--steps", "8"]
    if resume_training:
        if job.operation != "train":
            raise ValueError("only training supports checkpoint resume")
        command.append("--resume")
    return command


def child_environment(root) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PATH"] = str(TORCH_ENV / "bin") + os.pathsep + environment.get("PATH", "")
    environment["PYTHONPATH"] = str(Path(root).resolve() / "source")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = "2"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["WANDB_MODE"] = "offline"
    environment["WANDB_DISABLED"] = "true"
    environment["WEAVE_TRACE_DISABLED"] = "1"
    environment["UV_CACHE_DIR"] = "/mnt/lightsem-data/uv-cache"
    return environment


def _append_event(root: Path, event: dict) -> None:
    with (root / "campaign_events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": time.time(), **event}, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _supervisor_lock(root: Path):
    with (root / "supervisor.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Stage11 supervisor owns this campaign") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def _interrupt_handlers():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    original = {}

    def stop(signum, frame):
        raise InterruptedError(f"supervisor interrupted by signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        original[sig] = signal.signal(sig, stop)
    try:
        yield
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def check_gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    others = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if others:
        raise RuntimeError("unexpected accelerator process before child launch: " + "; ".join(others))


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_child(
    command: list[str], *, cwd: Path, env: dict, log_path: Path, deadline: float,
    on_start: Callable[[int], None], on_heartbeat: Callable[[], None],
) -> int:
    """Stream output while enforcing the global deadline; no implicit retries."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True, bufsize=0,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    last_heartbeat = 0.0
    try:
        on_start(process.pid)
        with log_path.open("ab", buffering=0) as log:
            while True:
                now = time.time()
                if now >= deadline:
                    raise CampaignDeadlineExceeded("the campaign's global elapsed-time deadline was reached")
                if now - last_heartbeat >= 5:
                    on_heartbeat()
                    last_heartbeat = now
                ready = selector.select(timeout=min(1.0, max(deadline - now, 0)))
                for key, _ in ready:
                    chunk = os.read(key.fd, 65536)
                    if chunk:
                        log.write(chunk)
                        print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                    else:
                        selector.unregister(key.fileobj)
                if process.poll() is not None and not selector.get_map():
                    return int(process.returncode)
    except BaseException:
        _stop_child(process)
        raise
    finally:
        selector.close()
        process.stdout.close()


def report_path(root: Path, job: Job) -> Path:
    if job.operation == "train":
        return root / "runs" / job.arm / f"seed_{job.seed}" / "report.json"
    if job.operation == "evaluate":
        return root / "results" / job.split / job.arm / f"seed_{job.seed}" / "report.json"
    if job.operation == "aggregate":
        return root / "results" / f"aggregate_{job.split}.json"
    if job.operation == "render":
        return root / "results" / f"render_{job.split}.json"
    raise ValueError("unknown operation")


def _pin_identity(root: Path) -> dict:
    read_authenticated(root / "protocol.json")
    freeze = verify_sources(root)
    read_authenticated(root / "training_data.json")
    read_authenticated(root / "normalizers.json")
    return {
        "protocol_sha256": digest(root / "protocol.json"),
        "data_sha256": digest(root / "training_data.json"),
        "normalizers_sha256": digest(root / "normalizers.json"),
        "source_sha256": freeze["source_sha256"],
    }


def _expected_groups(root: Path, split: str, family: str) -> set[str]:
    data = read_authenticated(root / "training_data.json")
    return {str(row["canonical_id"]) for row in data[family] if row["partition"] == split}


def verify_completion(root, job: Job, protocol: dict, pinned: dict) -> dict:
    """Authenticate identity, expected counts and actual files after process exit."""
    root = Path(root)
    path = report_path(root, job)
    report = read_authenticated(path)
    identity = report.get("identity", {})
    if identity:
        for name, expected in pinned.items():
            if name in identity and identity[name] != expected:
                raise RuntimeError(f"{job.key}: {name} identity mismatch")
    if report.get("protocol_sha256", pinned["protocol_sha256"]) != pinned["protocol_sha256"]:
        raise RuntimeError(f"{job.key}: aggregate protocol identity mismatch")
    if job.operation in {"train", "evaluate"}:
        for name, expected in pinned.items():
            if identity.get(name) != expected:
                raise RuntimeError(f"{job.key}: missing/different {name}")
        if identity.get("arm") != job.arm or identity.get("seed") != job.seed:
            raise RuntimeError(f"{job.key}: report arm/seed mismatch")
        if identity.get("config") != protocol["arms"][job.arm]:
            raise RuntimeError(f"{job.key}: configuration mismatch")
        if report.get("status") != "completed":
            raise RuntimeError(f"{job.key}: report is not completed")
        verify_artifacts(report.get("artifacts"))
    if job.operation == "train":
        cfg = protocol["arms"][job.arm]
        if report.get("steps") != cfg["steps"] or report.get("nonfinite") is not False or report.get("oom") is not False:
            raise RuntimeError(f"{job.key}: incomplete or invalid training result")
        if report.get("precision") != "fp32":
            raise RuntimeError(f"{job.key}: precision differs from the frozen recipe")
        if not math.isfinite(report.get("training_seconds", float("nan"))) or report["training_seconds"] <= 0:
            raise RuntimeError(f"{job.key}: invalid measured training duration")
        expected_files = {str((path.parent / name).resolve())
                          for name in ("best.pt", "latest.pt", "identity.json", "training.jsonl")}
        if not expected_files.issubset(report["artifacts"]):
            raise RuntimeError(f"{job.key}: incomplete checkpoint/artifact matrix")
        for step in cfg.get("validation_steps", [cfg["steps"]]):
            validation = read_authenticated(path.parent / f"validation_step_{step}.json")
            if validation.get("identity") != identity or validation.get("step") != step or not validation.get("rows"):
                raise RuntimeError(f"{job.key}: invalid validation checkpoint {step}")
            if not math.isfinite(validation.get("selection_score", float("nan"))):
                raise RuntimeError(f"{job.key}: nonfinite validation selection score")
    elif job.operation == "evaluate":
        verify_completion(root, Job("train", job.arm, job.seed), protocol, pinned)
        training_path = report_path(root, Job("train", job.arm, job.seed))
        if identity.get("training_report_sha256") != digest(training_path):
            raise RuntimeError(f"{job.key}: training report changed after evaluation")
        if identity.get("checkpoint_sha256") != digest(training_path.parent / "best.pt"):
            raise RuntimeError(f"{job.key}: selected checkpoint changed")
        family = protocol["arms"][job.arm]["family"]
        if identity.get("split") != job.split or identity.get("family") != family:
            raise RuntimeError(f"{job.key}: split/family mismatch")
        if identity.get("integration_steps") != 8 or identity.get("controls") is not True:
            raise RuntimeError(f"{job.key}: primary evaluation recipe differs")
        expected_seeds = protocol.get("generation_seeds", [101, 102, 103, 104]) if family == "scene" else []
        if identity.get("generation_seeds") != expected_seeds:
            raise RuntimeError(f"{job.key}: incomplete generation-seed matrix")
        expected = _expected_groups(root, job.split, family)
        if not expected or set(report.get("expected_canonical_ids", [])) != expected:
            raise RuntimeError(f"{job.key}: evaluation canonical-song matrix differs")
        rows = report.get("rows", [])
        if report.get("canonical_count") != len(expected) or report.get("row_count") != len(rows) or not rows:
            raise RuntimeError(f"{job.key}: incomplete evaluation rows")
        entries = [row for row in read_authenticated(root / "training_data.json")[family]
                   if row["partition"] == job.split]
        expected_matrix = Counter((entry["key"], generation_seed) for entry in entries
                                  for generation_seed in (expected_seeds if family == "scene" else [None]))
        actual_matrix = Counter((row.get("key"), row.get("generation_seed")) for row in rows)
        if actual_matrix != expected_matrix:
            raise RuntimeError(f"{job.key}: incomplete recording/generation-seed matrix")
        for row in rows:
            capture = row.get("capture_path")
            if capture not in report["artifacts"] or row.get("capture_sha256") != report["artifacts"][capture]["sha256"]:
                raise RuntimeError(f"{job.key}: missing or inconsistent captured trajectory")
        if job.arm == "S_temporal" and identity.get("donor_controls") is not True:
            raise RuntimeError(f"{job.key}: matched-audio evaluation lacks donor controls")
    elif job.operation == "aggregate":
        if report.get("status") != "complete" or report.get("missing_runs"):
            raise RuntimeError(f"{job.key}: aggregate reports an incomplete matrix")
        if "protocol_sha256" not in report and identity.get("protocol_sha256") != pinned["protocol_sha256"]:
            raise RuntimeError(f"{job.key}: aggregate lacks protocol binding")
        if not isinstance(report.get("gates"), dict) or not isinstance(report.get("eligible_families"), list):
            raise RuntimeError(f"{job.key}: aggregate lacks gates/family results")
        if job.split == "val":
            expected_runs = {(arm, seed) for arm in ALL_ARMS for seed in protocol["training_seeds"]}
            decision_path = root / "extension_decision.json"
            if decision_path.exists() and read_authenticated(decision_path).get("execute"):
                expected_runs.update((arm, 5) for arm in SCENE_ARMS)
        else:
            unlock = read_authenticated(root / "test_unlock.json")
            expected_runs = {(arm, seed) for arm, selection in unlock["arms"].items()
                             for seed in selection["training_seeds"]}
        actual_runs = [(row.get("arm"), row.get("seed")) for row in report.get("expected_runs", [])]
        if len(actual_runs) != len(expected_runs) or set(actual_runs) != expected_runs:
            raise RuntimeError(f"{job.key}: aggregate run matrix differs")
        inputs = report.get("input_reports", {})
        expected_paths = {str(report_path(root, Job("evaluate", arm, seed, job.split)))
                          for arm, seed in expected_runs}
        if set(inputs) != expected_paths:
            raise RuntimeError(f"{job.key}: aggregate input-report matrix differs")
        for input_path, sha256 in inputs.items():
            read_authenticated(input_path)
            if digest(input_path) != sha256:
                raise RuntimeError(f"{job.key}: an aggregated evaluation report changed")
        if report.get("artifacts"):
            verify_artifacts(report["artifacts"])
    elif job.operation == "render":
        if report.get("status") not in {"complete", "completed"}:
            raise RuntimeError(f"{job.key}: incomplete render report")
        if report.get("protocol_sha256", identity.get("protocol_sha256")) != pinned["protocol_sha256"]:
            raise RuntimeError(f"{job.key}: render lacks protocol binding")
        verify_artifacts(report.get("artifacts"))
    return report


def _training_recovery_authorized(root: Path, job: Job, pinned: dict, *, resume_supervisor: bool) -> bool:
    run = root / "runs" / job.arm / f"seed_{job.seed}"
    if not run.exists() or not any(run.iterdir()):
        return False
    if not resume_supervisor:
        raise RuntimeError(f"{job.key}: incomplete run requires explicit supervisor recovery")
    authorization_path = root / "recovery_authorizations.json"
    if not authorization_path.exists():
        raise RuntimeError(f"{job.key}: missing exact checkpoint recovery authorization")
    record = read_authenticated(authorization_path)
    if record.get("protocol_sha256") != pinned["protocol_sha256"]:
        raise RuntimeError("recovery authorization belongs to a different protocol")
    matches = [row for row in record.get("authorizations", [])
               if row.get("arm") == job.arm and row.get("seed") == job.seed]
    if len(matches) != 1:
        raise RuntimeError(f"{job.key}: require exactly one matching recovery authorization")
    authorization = matches[0]
    if not str(authorization.get("reason", "")).strip() or authorization.get("no_observed_final") is not True:
        raise RuntimeError(f"{job.key}: recovery needs a reason and no observed final result")
    if authorization.get("checkpoint_sha256") != digest(run / "latest.pt"):
        raise RuntimeError(f"{job.key}: recovery checkpoint digest mismatch")
    if (run / "report.json").exists():
        raise RuntimeError(f"{job.key}: recovery cannot replace a final report")
    return True


def validate_test_unlock(root, protocol: dict, pinned: dict) -> tuple[dict, list[Job]]:
    root = Path(root)
    unlock = read_authenticated(root / "test_unlock.json")
    aggregate_path = report_path(root, Job("aggregate", split="val"))
    aggregate = verify_completion(root, Job("aggregate", split="val"), protocol, pinned)
    if unlock.get("schema_version") != 1 or unlock.get("protocol_sha256") != pinned["protocol_sha256"]:
        raise RuntimeError("test unlock belongs to a different protocol/schema")
    if unlock.get("validation_aggregate_sha256") != digest(aggregate_path):
        raise RuntimeError("test unlock does not bind the reviewed validation aggregate")
    families = unlock.get("eligible_families")
    if not families or not set(families).issubset(set(aggregate["eligible_families"])):
        raise RuntimeError("test unlock names ineligible or zero families")
    allowed = set()
    for family in families:
        if family not in {"motion", "scene"}:
            raise RuntimeError("unknown test-unlock family")
        gates = ("H11_D",) if family == "motion" else ("H11_T", "H11_A")
        if any(aggregate.get("gates", {}).get(gate, {}).get("passed") is not True for gate in gates):
            raise RuntimeError("test unlock eligibility disagrees with the reviewed numerical gates")
        allowed.update(DESCRIPTOR_ARMS if family == "motion" else SCENE_ARMS)
    if set(unlock.get("arms", {})) != allowed:
        raise RuntimeError("test unlock must retain every control arm of each eligible family")
    jobs = []
    for arm in ALL_ARMS:
        if arm not in allowed:
            continue
        selection = unlock["arms"][arm]
        if selection.get("config_sha256") != json_digest(protocol["arms"][arm]):
            raise RuntimeError(f"test unlock configuration mismatch: {arm}")
        validated_seeds = aggregate.get("arm_seeds", {}).get(arm)
        extension_path = root / "extension_decision.json"
        extension_ran = extension_path.exists() and read_authenticated(extension_path).get("execute")
        expected_seeds = protocol["training_seeds"] + ([5] if arm in SCENE_ARMS and extension_ran else [])
        if validated_seeds != expected_seeds or selection.get("training_seeds") != validated_seeds:
            raise RuntimeError(f"test unlock must include every validated predeclared seed: {arm}")
        if set(selection.get("checkpoints", {})) != {str(seed) for seed in validated_seeds}:
            raise RuntimeError(f"test unlock checkpoint matrix differs: {arm}")
        for seed in selection["training_seeds"]:
            job = Job("evaluate", arm, seed, "test")
            verify_completion(root, Job("evaluate", arm, seed, "val"), protocol, pinned)
            checkpoint = root / "runs" / arm / f"seed_{seed}" / "best.pt"
            if selection.get("checkpoints", {}).get(str(seed)) != digest(checkpoint):
                raise RuntimeError(f"test unlock checkpoint mismatch: {arm}/{seed}")
            jobs.append(job)
    return unlock, jobs


def _freeze_queue(root: Path, phase: str, definition: dict) -> dict:
    path = root / f"campaign_queue_{phase}.json"
    if path.exists():
        stored = read_authenticated(path)
        if stored != definition:
            raise RuntimeError("frozen campaign queue differs from the requested phase")
        return stored
    authenticated_json(path, definition)
    return definition


def _extension_decision(root: Path, protocol: dict, pinned: dict, deadline: float) -> dict:
    path = root / "extension_decision.json"
    if path.exists():
        decision = read_authenticated(path)
        if decision.get("protocol_sha256") != pinned["protocol_sha256"]:
            raise RuntimeError("extension decision belongs to a different protocol")
        for job in training_queue(protocol):
            if decision.get("training_reports_sha256", {}).get(job.key) != digest(report_path(root, job)):
                raise RuntimeError("a core report changed after the frozen extension decision")
        return decision
    reports = {job.key: verify_completion(root, job, protocol, pinned) for job in training_queue(protocol)}
    total = sum(report["training_seconds"] for report in reports.values())
    estimate = 1.25 * sum(
        max(max(reports[Job("train", arm, seed).key].get("wall_seconds", 0),
                reports[Job("train", arm, seed).key]["training_seconds"])
            for seed in protocol["training_seeds"])
        for arm in SCENE_ARMS
    )
    remaining = deadline - time.time()
    predeclared = protocol.get("extension_seeds", []) == [5]
    execute = total < MIN_TRAINING_SECONDS and predeclared and remaining >= estimate + EVALUATION_RESERVE_SECONDS
    reason = ("predeclared seed-5 triplet fits the remaining budget" if execute else
              "measured core training already meets ten hours" if total >= MIN_TRAINING_SECONDS else
              "seed-5 extension was not predeclared" if not predeclared else
              "insufficient time for the whole triplet plus two-hour evaluation reserve")
    decision = {
        "schema_version": 1, "protocol_sha256": pinned["protocol_sha256"],
        "execute": execute, "reason": reason, "training_seconds_core": total,
        "estimated_extension_seconds": estimate, "remaining_seconds_at_decision": remaining,
        "evaluation_reserve_seconds": EVALUATION_RESERVE_SECONDS,
        "arms": list(SCENE_ARMS), "seed": 5, "at": time.time(),
        "training_reports_sha256": {job.key: digest(report_path(root, job)) for job in training_queue(protocol)},
    }
    authenticated_json(path, decision)
    return decision


def supervise(
    root=ROOT, *, phase: str = "core", resume_supervisor: bool = False,
    runner: Callable | None = None,
) -> dict:
    """Run one frozen phase and persist truthful status on completion or failure."""
    if phase not in {"core", "test", "diagnostic_render"}:
        raise ValueError("phase must be core, test, or diagnostic_render")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runner = runner or run_child
    with _supervisor_lock(root), _interrupt_handlers():
        try:
            pinned = _pin_identity(root)
        except BaseException as error:
            status_path = root / "campaign_status.json"
            prior = read_json(status_path) if status_path.exists() else {}
            failure = {**prior, "status": "failed", "phase": phase, "supervisor_pid": os.getpid(),
                       "child_pid": None, "updated_at": time.time(),
                       "error": {"type": type(error).__name__, "message": str(error)}}
            atomic_json(status_path, failure)
            _append_event(root, {"event": "preflight_failed", "phase": phase, "error": failure["error"]})
            raise
        protocol = read_authenticated(root / "protocol.json")
        core = training_queue(protocol)
        path = root / "campaign_status.json"
        previous = read_json(path) if path.exists() else None
        now = time.time()
        hours = float(protocol.get("max_wall_hours", 24))
        if not math.isfinite(hours) or not 0 < hours <= 24:
            raise ValueError("max_wall_hours must be finite and no greater than 24")
        if previous:
            if previous.get("identity") != pinned:
                raise RuntimeError("campaign identity changed since the previous supervisor")
            transition = previous.get("status") == "awaiting_primary_review" and phase in {"test", "diagnostic_render"}
            if not transition and not resume_supervisor:
                raise RuntimeError("existing campaign requires explicit --resume-supervisor")
            started_at, deadline = previous["started_at"], previous["deadline"]
        else:
            if phase != "core":
                raise RuntimeError("a reviewed phase requires an existing core campaign")
            started_at = float(protocol.get("campaign_started_at", now))
            if not math.isfinite(started_at) or started_at > now:
                raise ValueError("campaign_started_at must be a finite, nonfuture timestamp")
            deadline = started_at + hours * 3600
        state = {
            "schema_version": 1, "identity": pinned, "phase": phase, "status": "starting",
            "supervisor_pid": os.getpid(), "child_pid": None, "started_at": started_at,
            "deadline": deadline, "updated_at": now, "current_job": None,
            "completed": dict(previous.get("completed", {})) if previous else {},
            "remaining_jobs": [], "resume_supervisor": resume_supervisor,
            "remaining_jobs_scope": "current stage; later stages listed in pending_stages",
            "pending_stages": [],
        }

        def save(**updates):
            state.update(updates)
            state["updated_at"] = time.time()
            atomic_json(path, state)

        def before_child(job):
            if time.time() >= deadline:
                raise CampaignDeadlineExceeded("no campaign time remains before the next operation")
            if _pin_identity(root) != pinned:
                raise RuntimeError("frozen inputs/sources changed before child launch")
            if job.uses_gpu:
                require_space(root, gb=15)
                check_gpu_idle()

        def execute(jobs):
            for index, job in enumerate(jobs):
                save(current_job=asdict(job), remaining_jobs=[asdict(item) for item in jobs[index:]])
                completion = report_path(root, job)
                if completion.exists():
                    if not resume_supervisor:
                        raise RuntimeError(f"{job.key}: completed artifact requires explicit supervisor resume to reuse")
                    prior_completion = state["completed"].get(job.key)
                    if prior_completion and prior_completion.get("sha256") != digest(completion):
                        raise RuntimeError(f"{job.key}: a previously completed report was replaced")
                    report = verify_completion(root, job, protocol, pinned)
                    state["completed"][job.key] = {"report": str(completion), "sha256": digest(completion), "reused": True}
                    _append_event(root, {"event": "verified_reuse", "job": asdict(job)})
                    save()
                    continue
                recovery = _training_recovery_authorized(root, job, pinned, resume_supervisor=resume_supervisor) if job.operation == "train" else False
                if job.operation == "evaluate" and completion.parent.exists() and any(completion.parent.iterdir()):
                    raise RuntimeError(f"{job.key}: incomplete evaluation artifacts need explicit primary recovery")
                before_child(job)
                command = command_for_job(root, job, resume_training=recovery)
                save(status="running", child_pid=None, command=command)
                _append_event(root, {"event": "child_starting", "job": asdict(job), "command": command,
                                     "checkpoint_recovery": recovery})

                def on_start(pid):
                    save(child_pid=pid)
                    _append_event(root, {"event": "child_started", "job": asdict(job), "pid": pid})

                def on_heartbeat():
                    progress_path = root / "runs" / str(job.arm) / f"seed_{job.seed}" / "status.json"
                    progress = read_json(progress_path) if job.operation == "train" and progress_path.exists() else None
                    save(child_progress=progress)

                returncode = runner(
                    command, cwd=root / "source", env=child_environment(root),
                    log_path=root / "logs" / f"{job.key.replace(':', '_')}.log",
                    deadline=deadline, on_start=on_start, on_heartbeat=on_heartbeat,
                )
                save(child_pid=None, last_returncode=returncode)
                _append_event(root, {"event": "child_exited", "job": asdict(job), "returncode": returncode})
                if returncode != 0:
                    raise RuntimeError(f"{job.key}: child exited {returncode}; automatic retry is disabled")
                report = verify_completion(root, job, protocol, pinned)
                state["completed"][job.key] = {"report": str(completion), "sha256": digest(completion), "reused": False}
                _append_event(root, {"event": "verified_completion", "job": asdict(job), "report_sha256": digest(completion)})
                save(remaining_jobs=[asdict(item) for item in jobs[index + 1:]])

        save()
        _append_event(root, {"event": "supervisor_started", "phase": phase, "pid": os.getpid(),
                             "resume_supervisor": resume_supervisor, "deadline": deadline})
        try:
            if phase == "core":
                definition = {"identity": pinned, "phase": phase, "training": [asdict(job) for job in core],
                              "extension": {"arms": list(SCENE_ARMS), "seed": 5, "conditional": True},
                              "validation": [asdict(Job("evaluate", job.arm, job.seed, "val")) for job in core],
                              "final": asdict(Job("aggregate", split="val")), "stop_status": "awaiting_primary_review"}
                _freeze_queue(root, phase, definition)
                save(stage="core_training", pending_stages=["conditional_extension", "validation_evaluation", "validation_aggregation", "primary_review"])
                execute(core)
                decision = _extension_decision(root, protocol, pinned, deadline)
                extension = [Job("train", arm, 5) for arm in SCENE_ARMS] if decision["execute"] else []
                save(extension_decision=decision, stage="conditional_extension",
                     pending_stages=["validation_evaluation", "validation_aggregation", "primary_review"])
                execute(extension)
                all_training = core + extension
                save(stage="validation_evaluation", pending_stages=["validation_aggregation", "primary_review"])
                execute([Job("evaluate", job.arm, job.seed, "val") for job in all_training])
                save(stage="validation_aggregation", pending_stages=["primary_review"])
                execute([Job("aggregate", split="val")])
                seconds = sum(verify_completion(root, job, protocol, pinned)["training_seconds"] for job in all_training)
                save(status="awaiting_primary_review", current_job=None, child_pid=None, remaining_jobs=[],
                     stage="primary_review", pending_stages=[],
                     training_seconds=seconds, minimum_gpu_training_duration_met=seconds >= MIN_TRAINING_SECONDS,
                     note="validation aggregate is ready; locked test remains sealed pending primary review")
            elif phase == "test":
                _, evaluation = validate_test_unlock(root, protocol, pinned)
                jobs = evaluation + [Job("aggregate", split="test"), Job("render", split="test")]
                _freeze_queue(root, phase, {"identity": pinned, "phase": phase,
                                           "test_unlock_sha256": digest(root / "test_unlock.json"),
                                           "jobs": [asdict(job) for job in jobs]})
                save(stage="approved_test_evaluation_and_render", pending_stages=[])
                execute(jobs)
                save(status="completed", current_job=None, child_pid=None, remaining_jobs=[],
                     note="approved locked evaluation and blind materials completed; human preference remains pending")
            else:
                verify_completion(root, Job("aggregate", split="val"), protocol, pinned)
                jobs = [Job("render", split="val")]
                _freeze_queue(root, phase, {"identity": pinned, "phase": phase, "jobs": [asdict(job) for job in jobs]})
                save(stage="diagnostic_render", pending_stages=[])
                execute(jobs)
                save(status="completed_diagnostics", current_job=None, child_pid=None, remaining_jobs=[],
                     note="validation diagnostics rendered by explicit primary choice; test remains sealed")
            _append_event(root, {"event": "supervisor_stopped", "phase": phase, "status": state["status"]})
            return state
        except BaseException as error:
            status = ("deadline_exceeded" if isinstance(error, CampaignDeadlineExceeded) else
                      "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed")
            save(status=status, child_pid=None, error={"type": type(error).__name__, "message": str(error)})
            _append_event(root, {"event": "supervisor_failed", "phase": phase, "status": status,
                                 "error": state["error"], "job": state["current_job"]})
            raise


def resume_after_review(root=ROOT, *, resume_supervisor: bool = True, runner=None) -> dict:
    return supervise(root, phase="test", resume_supervisor=resume_supervisor, runner=runner)
