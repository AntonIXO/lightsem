"""Scheduling-only parallel driver for the frozen observer prediction CLI.

No fit, model, precision, target, window, or scoring implementation changes.
Run through bash run_torch.sh. --self-test uses synthetic CPU subprocesses only.
The output root must be new. Failed/partial outputs are never retried or removed.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
MIB = 1024 ** 2
JOB_RESERVE = 16 * MIB
METADATA_RESERVE = 2 * MIB
MAX_SLOTS = 3
TIMEOUT_SECONDS = 900


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def write_exclusive(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_refs(references):
    for item in references:
        if ref(item["path"]) != item:
            raise RuntimeError("Pinned source/input changed: " + item["path"])


def gpu_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15)
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if any(not row.isdecimal() for row in rows):
        raise RuntimeError("Unexpected NVIDIA compute-process response")
    return {int(row) for row in rows}


def require_only_owned_gpu(owned=()):
    other = gpu_pids() - set(owned)
    if other:
        raise RuntimeError("Another GPU compute process is active: " + str(sorted(other)))


@contextmanager
def gpu_owner(campaign_root):
    # Reuse the trainer's existing campaign lock. Prediction workers themselves
    # have no lock, so the parent holds ownership across all concurrent children.
    with (Path(campaign_root) / "gpu.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Campaign GPU ownership is already locked") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Job:
    key: str
    role: str
    checkpoint: dict
    output: str
    log: str
    argv: tuple[str, ...]


def validate_plan(args):
    """Authenticate JSON, frozen source, receipts, and weights; never use CUDA."""
    queue_root = args.queue_root.resolve()
    completion_path = queue_root / "queue_complete.json"
    if not completion_path.is_file():
        raise RuntimeError("Full original 15-fit queue completion is required")
    from lightsem.experiments.light_layers import training as t, evaluation as e
    import run_light_layers_campaign as q

    stored = read_json(queue_root / "queue_manifest.json")
    original = stored["plan"]
    if Path(original["inputs"]["output_root"]).resolve() != queue_root:
        raise RuntimeError("Queue root differs from authenticated original plan")
    fresh, training_jobs = q.validate_inputs(q.QueueInputs(**original["inputs"]))
    if fresh != original:
        raise RuntimeError("Original queue inputs/source/config/paths changed")
    completion = read_json(completion_path)
    completed = [q.verify_completed_job(job) for job in training_jobs]
    if (len(training_jobs) != 15 or completion.get("status") != "complete"
            or completion.get("plan_sha256") != original["plan_sha256"]
            or completion.get("queue_source_sha256") != original["queue_source_sha256"]
            or completion.get("jobs") != completed):
        raise RuntimeError("Original complete 15-fit queue receipt is invalid")
    for name in ("manifest", "transforms", "torch_env"):
        if Path(original["inputs"][name]).resolve() != getattr(args, name).resolve():
            raise RuntimeError("Prediction changed original " + name)
    if not (args.torch_env / "bin" / "python").is_file():
        raise RuntimeError("Dedicated Torch environment is unavailable")

    registration, manifest, transforms, _ = e._inputs(args, "predict")
    baseline_ref = registration.get("baselines")
    if baseline_ref is None or e.checked_ref(baseline_ref).resolve() != args.baselines.resolve():
        raise RuntimeError("Frozen TRAIN baseline reference mismatch")
    fitted = read_json(args.baselines)
    for name in ("norm_sha256", "vocab_sha256"):
        if fitted.get(name) != transforms[name]:
            raise RuntimeError("Frozen baseline transform mismatch")
    for name in ("bindings", "donor_map", "donor_tempos_bpm"):
        if fitted.get(name) != registration[name]:
            raise RuntimeError("Frozen baseline registration mismatch: " + name)

    expected = {}
    for job in training_jobs:
        directory = Path(job.output_dir)
        run_path = directory / "complete.json"
        run = read_json(run_path)
        if run["config"]["steps"] != 6000 or run["config"]["batch"] != 4:
            raise RuntimeError("Completed fit is not the prescribed 6000/batch4")
        for role, filename in (("selected", "best.pt"), ("final", "final.pt")):
            key = f"{job.index:02d}_seed{job.seed}_{job.recipe}_{role}"
            expected[key] = {**ref(directory / filename), "recipe": job.recipe,
                "seed": job.seed, "role": role, "bindings": run["bindings"],
                "training_run_receipt": ref(run_path)}
    if len(expected) != 30 or registration.get("checkpoints") != expected:
        raise RuntimeError("Need exact 30 bound best/final checkpoints from this completed queue")
    for checkpoint in expected.values():
        e.validate_training_receipt(checkpoint, registration)

    campaign_root = Path(original["campaign_root"]).resolve()
    output = args.output.resolve()
    if (output.exists() or output == campaign_root
            or not output.is_relative_to(campaign_root)
            or output.is_relative_to(queue_root)):
        raise RuntimeError("Output must be a NEW campaign subdirectory outside the training queue")
    sources = {**q.required_test_sources(), **e.implementation_hashes()}
    for path in (Path(__file__), REPO / "run_light_layers_evaluate.py",
                 Path(__file__).with_name("run_observer_prediction_queue_v1.py")):
        sources[str(path.resolve())] = sha256(path)
    policy = Path(__file__).with_name("OBSERVER_GPU_THROUGHPUT_POLICY_V1.json")
    policy_receipt = Path(__file__).with_name("observer_gpu_throughput_vault_receipt.json")
    fixed_refs = [ref(path) for path in (args.manifest, args.transforms,
        args.registration, args.baselines, queue_root / "queue_manifest.json", completion_path,
        policy, policy_receipt)]
    fixed_refs += [{"path": str(Path(path).resolve()), "sha256": digest}
                   for path, digest in sorted(sources.items())]
    jobs = []
    for role in ("selected", "final"):
        for key, checkpoint in sorted(expected.items()):
            if checkpoint["role"] != role:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_]+", key):
                raise RuntimeError("Unsafe checkpoint key")
            destination = output / role / key
            command = ("bash", "run_torch.sh", "-u", "run_light_layers_evaluate.py", "predict",
                "--manifest", str(args.manifest.resolve()),
                "--transforms", str(args.transforms.resolve()),
                "--registration", str(args.registration.resolve()),
                "--baselines", str(args.baselines.resolve()), "--checkpoint", key,
                "--output", str(destination), "--device", "cuda", "--gpu-authorized")
            jobs.append(Job(key, role, checkpoint, str(destination), str(output / (key + ".log")), command))
    plan = {"schema_version": 1, "scope": "resource_only_prediction_scheduling",
        "science_change": False, "fits_launched": 0, "scoring_performed": False,
        "prediction_implementation": "existing frozen CLI; unchanged FP32 and single 64s window",
        "original_queue_plan_sha256": original["plan_sha256"],
        "campaign_root": str(campaign_root), "output": str(output),
        "torch_env": str(args.torch_env.resolve()), "slots": args.slots,
        "timeout_seconds": TIMEOUT_SECONDS, "per_active_job_reservation_bytes": JOB_RESERVE,
        "metadata_reservation_bytes": METADATA_RESERVE,
        "storage_policy": "actual campaign bytes + 16MiB per active job + 16MiB next job + metadata; no 30-job upfront reservation",
        "fixed_references": fixed_refs, "registration": ref(args.registration),
        "resource_only_policy": ref(policy), "resource_only_policy_vault_receipt": ref(policy_receipt),
        "source_files": sources, "jobs": [asdict(job) for job in jobs]}
    plan["plan_sha256"] = t.json_hash(plan)
    # This guard reserves all three possible children globally, unlike separate
    # child-only checks. Repeat with current actual bytes before EVERY launch.
    t.campaign_space(campaign_root, args.slots * JOB_RESERVE + METADATA_RESERVE)
    dev_rows = {row["recording_id"]: row for row in manifest["records"]
                if row["partition"] in ("dev", "val")}
    manifest_rows = {row["recording_id"]: row for row in manifest["records"]}

    def verify_inputs(job):
        check_refs(fixed_refs)
        e.checked_ref(job.checkpoint)
        e.validate_training_receipt(job.checkpoint, registration)

    def verify_result(job):
        receipt_path = Path(job.output) / "complete.json"
        receipt = read_json(receipt_path)
        checkpoint = e.validate_prediction_receipt(receipt, registration, baseline_ref)
        if checkpoint != job.checkpoint:
            raise RuntimeError("Prediction receipt belongs to a different checkpoint")
        rows = receipt.get("records", [])
        if len(rows) != len(dev_rows) or {row["recording_id"] for row in rows} != set(dev_rows):
            raise RuntimeError("Prediction does not contain exact DEV inventory")
        for row in rows:
            rid = row["recording_id"]
            donor = registration["donor_map"][rid]
            if (row.get("source_row") != dev_rows[rid]
                    or row.get("canonical_id") != dev_rows[rid]["canonical_id"]
                    or row.get("donor_source_row") != manifest_rows[donor]
                    or row.get("target_sha256") != registration["target_hashes"][rid]):
                raise RuntimeError("Prediction source/donor/target reference changed")
            artifact = e.checked_ref(row["artifact"])
            if not artifact.resolve().is_relative_to(Path(job.output).resolve()):
                raise RuntimeError("Prediction artifact escaped its distinct job directory")
        return ref(receipt_path)

    def space_check(reserve):
        return t.campaign_space(campaign_root, reserve)

    return plan, jobs, verify_inputs, verify_result, space_check


async def stop_processes(running):
    """Stop whole process groups, including a failed leader's descendants."""
    for item in running:
        try:
            os.killpg(item["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    waits = [asyncio.create_task(item["process"].wait()) for item in running]
    if waits:
        _, pending = await asyncio.wait(waits, timeout=10)
        # Always kill remaining group members, even when their leader exited.
        for item in running:
            try:
                os.killpg(item["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.gather(*pending, return_exceptions=True)


async def execute(plan, jobs, verify_inputs, verify_result, space_check, *,
                  gpu_check=require_only_owned_gpu, timeout=TIMEOUT_SECONDS):
    """Event-driven bounded scheduler. Callbacks allow data-free CPU tests."""
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    for role in {job.role for job in jobs}:
        (output / role).mkdir(exist_ok=False)
    write_exclusive(output / "plan.json", plan)
    started = time.monotonic()
    states = {job.key: "pending" for job in jobs}
    receipts = {}
    active = {}
    launched = []
    pending_jobs = iter(jobs)
    next_job = next(pending_jobs, None)
    env = dict(os.environ, TORCH_ENV=plan.get("torch_env", ""))
    progress = (output / "progress.jsonl").open("x")

    def event(kind, **extra):
        live = [{"key": item["job"].key, "pid": item["process"].pid}
                for item in active.values() if item["process"].returncode is None]
        row = {"event": kind, "time": now(), "elapsed_s": time.monotonic() - started,
               "active": live, "active_count": len(live), **extra}
        line = json.dumps(row, sort_keys=True, allow_nan=False)
        progress.write(line + "\n"); progress.flush()
        print(line, flush=True)

    async def wait_child(item):
        try:
            remaining = max(0., timeout - (time.monotonic() - item["start"]))
            return await asyncio.wait_for(item["process"].wait(), timeout=remaining)
        except asyncio.TimeoutError as error:
            raise RuntimeError("Prediction exceeded timeout: " + item["job"].key) from error

    try:
        event("queue_start", jobs=len(jobs), slots=plan["slots"], plan_sha256=plan.get("plan_sha256"))
        while next_job is not None or active:
            while next_job is not None and len(active) < plan["slots"]:
                job = next_job
                # Never refill a slot while a child has an unhandled exit.
                if any(task.done() for task in active):
                    break
                verify_inputs(job)
                gpu_check([item["process"].pid for item in active.values()])
                storage = space_check((len(active) + 1) * JOB_RESERVE + METADATA_RESERVE)
                if Path(job.output).exists() or Path(job.log).exists():
                    raise RuntimeError("No overwrite or retry: " + job.key)
                stream = Path(job.log).open("xb")
                item = {"job": job, "stream": stream, "start": time.monotonic(), "start_time": now()}
                try:
                    item["process"] = await asyncio.create_subprocess_exec(
                        *job.argv, cwd=REPO, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    states[job.key] = "launch_failed"
                    raise
                launched.append(item)
                states[job.key] = "running"
                task = asyncio.create_task(wait_child(item))
                active[task] = item
                event("predict_start", key=job.key, role=job.role, pid=item["process"].pid,
                      start_time=item["start_time"], checkpoint=job.checkpoint,
                      output=job.output, log=job.log, storage=storage)
                next_job = next(pending_jobs, None)
            if not active:
                continue
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            # Detect errors in every ready child before validating successes or
            # launching a replacement. On error the outer handler stops all.
            for task in done:
                item = active[task]
                try:
                    code = task.result()
                except BaseException:
                    states[item["job"].key] = "failed"
                    raise
                if code != 0:
                    states[item["job"].key] = "failed"
                    raise RuntimeError(f"Prediction failed ({code}): {item['job'].key}")
            for task in done:
                item = active[task]
                job = item["job"]
                item["stream"].close()
                try:
                    receipt = verify_result(job)
                except BaseException:
                    states[job.key] = "failed_receipt_validation"
                    raise
                job_bytes = Path(job.log).stat().st_size + sum(
                    path.stat().st_size for path in Path(job.output).rglob("*") if path.is_file())
                if job_bytes > JOB_RESERVE:
                    raise RuntimeError("Prediction exceeded its 16MiB output/log reservation: " + job.key)
                receipts[job.key] = receipt
                states[job.key] = "complete"
                del active[task]
                space_check(len(active) * JOB_RESERVE + METADATA_RESERVE)
                event("predict_complete", key=job.key, role=job.role, pid=item["process"].pid,
                      start_time=item["start_time"], end_time=now(),
                      job_elapsed_s=time.monotonic() - item["start"], receipt=receipt,
                      log=ref(job.log), output_bytes=job_bytes)
        event("queue_predictions_complete", completed=len(receipts), end_time=now())
        return receipts
    except BaseException as error:
        unfinished = [item for item in launched if states[item["job"].key] != "complete"]
        await stop_processes(unfinished)
        for task in active:
            if not task.done():
                task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        for item in unfinished:
            job = item["job"]
            if states[job.key] == "running":
                states[job.key] = "terminated_after_error"
            item["stream"].close()
            event("predict_stopped", key=job.key, pid=item["process"].pid,
                  status=states[job.key], start_time=item["start_time"], end_time=now(),
                  job_elapsed_s=time.monotonic() - item["start"],
                  returncode=item["process"].returncode)
        active.clear()
        failure = {"status": "partial_execution_blocked", "error": f"{type(error).__name__}: {error}",
                   "time": now(), "elapsed_s": time.monotonic() - started,
                   "plan_sha256": plan.get("plan_sha256"), "states": states,
                   "completed_predictions": receipts, "resume_supported": False,
                   "artifacts_preserved": True, "science_conclusion": "none"}
        write_exclusive(output / "partial.json", failure)
        event("queue_blocked", **failure)
        raise
    finally:
        for item in launched:
            item["stream"].close()
        progress.close()


async def launch_authorized(plan, jobs, verify_inputs, verify_result, space_check):
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, current.cancel)
    try:
        return await execute(plan, jobs, verify_inputs, verify_result, space_check)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def self_test():
    """Synthetic CPU processes only; no project imports, datasets, or CUDA."""
    import tempfile
    child = ("import asyncio,json,os,pathlib,subprocess,sys; "
             "p=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; "
             "p.mkdir(parents=True,exist_ok=False); "
             "(p/'pid').write_text(str(os.getpid())); "
             "g=subprocess.Popen([sys.executable,'-c','import asyncio;asyncio.run(asyncio.sleep(30))']) if mode=='fork' else None; "
             "(p/'grandchild_pid').write_text(str(g.pid)) if g else None; "
             "asyncio.run(asyncio.sleep(0.01 if mode=='fail' else float(sys.argv[3]))); "
             "sys.exit(7) if mode=='fail' else None; "
             "(p/'complete.json').write_text(json.dumps({'status':'complete'}))")
    checks = []
    with tempfile.TemporaryDirectory(prefix="observer-parallel-cpu-") as temporary:
        base = Path(temporary)

        def setup(name, modes, delay="0.1"):
            output = base / name
            jobs = [Job(str(i), "selected", {"synthetic": True},
                        str(output / "selected" / str(i)), str(output / (str(i) + ".log")),
                        (sys.executable, "-c", child, str(output / "selected" / str(i)), mode, delay))
                    for i, mode in enumerate(modes)]
            plan = {"output": str(output), "slots": 3, "scope": "synthetic_cpu_only"}
            reserves = []

            def space(reserve):
                reserves.append(reserve)
                return {"synthetic_reserve": reserve}

            def validate(job):
                assert read_json(Path(job.output) / "complete.json")["status"] == "complete"
                return ref(Path(job.output) / "complete.json")

            return output, jobs, plan, reserves, space, validate

        async def run_cases():
            output, jobs, plan, reserves, space, validate = setup("success", ["ok"] * 7)
            receipts = await execute(plan, jobs, lambda job: None, validate, space, gpu_check=lambda owned: None)
            progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
            assert len(receipts) == 7 and max(row["active_count"] for row in progress) == 3
            assert max(reserves) == 3 * JOB_RESERVE + METADATA_RESERVE
            assert {row["event"] for row in progress} >= {"predict_start", "predict_complete"}
            assert len([row for row in progress if row["event"] == "predict_complete"]) == 7
            checks.append("success_exact_inventory_max3_and_global_sum_reservation")
            try:
                await execute(plan, jobs, lambda job: None, validate, space, gpu_check=lambda owned: None)
                raise AssertionError("Existing output was accepted")
            except FileExistsError:
                checks.append("no_overwrite_or_retry")

            output, jobs, plan, reserves, space, validate = setup("failure", ["fail"] + ["ok"] * 4, "30")
            try:
                await execute(plan, jobs, lambda job: None, validate, space, gpu_check=lambda owned: None)
                raise AssertionError("Failed process accepted")
            except RuntimeError:
                partial = read_json(output / "partial.json")
                assert partial["status"] == "partial_execution_blocked"
                assert partial["states"]["0"] == "failed"
                assert partial["states"]["3"] == partial["states"]["4"] == "pending"
                assert not Path(jobs[3].log).exists()
                checks.append("first_error_no_refill_partials_preserved")

            output, jobs, plan, reserves, space, validate = setup("timeout", ["fork", "ok", "ok", "ok"], "30")
            try:
                await execute(plan, jobs, lambda job: None, validate, space,
                              gpu_check=lambda owned: None, timeout=0.1)
                raise AssertionError("Timeout accepted")
            except RuntimeError:
                partial = read_json(output / "partial.json")
                assert "timeout" in partial["error"] and partial["states"]["3"] == "pending"
                for job in jobs[:3]:
                    pid_path = Path(job.output) / "pid"
                    if pid_path.exists():
                        assert not Path("/proc", pid_path.read_text()).exists()
                descendant = Path(jobs[0].output) / "grandchild_pid"
                assert descendant.exists(), "Synthetic descendant did not start"
                stat = Path("/proc", descendant.read_text(), "stat")
                assert not stat.exists() or stat.read_text().split(")", 1)[1].split()[0] == "Z"
                checks.append("timeout_terminates_child_process_groups_including_descendants")

            output, jobs, plan, reserves, space, validate = setup("storage", ["ok"] * 4, "30")
            def full(reserve):
                if reserve > JOB_RESERVE + METADATA_RESERVE:
                    raise RuntimeError("synthetic global storage gate")
                return {}
            try:
                await execute(plan, jobs, lambda job: None, validate, full, gpu_check=lambda owned: None)
                raise AssertionError("Storage guard bypassed")
            except RuntimeError:
                partial = read_json(output / "partial.json")
                assert partial["states"]["1"] == "pending" and not Path(jobs[1].log).exists()
                checks.append("global_storage_error_stops_active_children")

            output, jobs, plan, reserves, space, validate = setup("source", ["ok"])
            def changed(job):
                raise RuntimeError("synthetic pinned input changed")
            try:
                await execute(plan, jobs, changed, validate, space, gpu_check=lambda owned: None)
                raise AssertionError("Changed source accepted")
            except RuntimeError:
                assert read_json(output / "partial.json")["states"]["0"] == "pending"
                assert not Path(jobs[0].log).exists()
                checks.append("changed_input_refuses_before_child_launch")

        asyncio.run(run_cases())
        try:
            validate_plan(argparse.Namespace(queue_root=base / "missing-training-queue"))
            raise AssertionError("Incomplete training queue accepted")
        except RuntimeError as error:
            assert "15-fit queue completion" in str(error)
            checks.append("incomplete_training_queue_refused_before_project_or_gpu_access")
    assert "torch" not in sys.modules
    print(json.dumps({"status": "passed", "scope": "synthetic_cpu_only", "cuda_used": False,
                      "dataset_access": False, "checks": checks, "source": ref(__file__)}))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--self-test"]:
        return self_test()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue-root", "registration", "output", "manifest", "transforms", "baselines"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--torch-env", type=Path, default=Path("/root/lightsem/.venv/torch-access"))
    parser.add_argument("--slots", type=int, choices=range(1, MAX_SLOTS + 1), default=3)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.gpu_authorized:
        raise RuntimeError("Parent must explicitly authorize GPU predictions (--gpu-authorized)")
    plan, jobs, verify_inputs, verify_result, space_check = validate_plan(args)
    require_only_owned_gpu()
    if args.dry_run:
        print(json.dumps({"status": "validated_not_launched", "plan": plan}, sort_keys=True))
        return 0
    with gpu_owner(plan["campaign_root"]):
        require_only_owned_gpu()
        receipts = asyncio.run(launch_authorized(plan, jobs, verify_inputs, verify_result, space_check))
    output = Path(plan["output"])
    try:
        indices = {role: [receipts[job.key] for job in jobs if job.role == role]
                   for role in ("selected", "final")}
        if any(len(index) != 15 for index in indices.values()) or len(receipts) != 30:
            raise RuntimeError("Completed inventory must be separate 15-cell selected/final indices")
        space_check(METADATA_RESERVE)
        for role, index in indices.items():
            write_exclusive(output / (role + "_index.json"), index)
        progress_rows = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
        write_exclusive(output / "complete.json", {
            "schema_version": 1, "status": "complete", "scope": "resource_only_prediction_scheduling",
            "start_time": progress_rows[0]["time"], "end_time": now(),
            "elapsed_s": progress_rows[-1]["elapsed_s"],
            "checkpoints": 30, "registration": plan["registration"], "plan": ref(output / "plan.json"),
            "progress": ref(output / "progress.jsonl"), "slots": plan["slots"],
            "indices": {role: ref(output / (role + "_index.json")) for role in indices},
            "fits_launched": 0, "science_change": False,
            "scoring": "Not performed; parent must bind each complete index before separate scoring"})
    except BaseException as error:
        write_exclusive(output / "partial.json", {"status": "partial_finalization_blocked",
            "error": f"{type(error).__name__}: {error}", "completed_predictions": receipts,
            "resume_supported": False, "artifacts_preserved": True, "science_conclusion": "none"})
        raise
    print(json.dumps({"event": "queue_complete", "checkpoints": 30, **ref(output / "complete.json")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as error:
        print(f"EXECUTION BLOCKED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
