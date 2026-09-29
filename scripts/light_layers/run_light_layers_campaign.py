#!/usr/bin/env python3
"""Parent-launched, fail-stop queue for the isolated historical observer proxy.

Exactly five recipes x seeds0/1/2, seed outermost, under ONE aggregate vault
experiment already registered by the parent. This program makes no vault calls.
It does not create new independent hypotheses, replace data, tune parameters,
resume optimizers, or delete artifacts. Completed jobs may be reused only with
--resume-exact and exact manifest/config/scope/source/weight verification.

Every training child runs through ``bash run_torch.sh -u scripts/light_layers/run_light_layers_train.py``.
Its stdout/stderr go directly to a file (not a pipe). The parent must launch this
queue nonblocking, then inspect the wrapper completion and queue reports.

Required extra preregistration bindings:
  campaign_queue_source_sha256 = queue_source_hash()
  cpu_test_receipt_sha256 = SHA256 of the supplied CPU-test receipt

CPU-test receipt schema1: status="passed", scope="synthetic_cpu_only",
cuda_used=false, dataset_access=false, tests_failed=0, tests_passed>0,
source_files={absolute_path:sha256} including every required_test_sources() entry.
The parent writes this receipt from the completed native CPU test result, then
binds its hash in the single aggregate preregistration. The queue never invents
or executes a test receipt as proof.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable

from lightsem.model.light_layers.training import (
    ContractError, OBSERVER_RECIPES, OBSERVER_SCOPE_BINDINGS, OBSERVER_STUDY_KIND,
    RECIPES, SourceStore, TrainConfig, campaign_space, implementation_hashes,
    json_hash, provenance, sha256, validate_run_scope, verify_preregistration,
    write_json)

REPO = Path(__file__).resolve().parents[2]
MIB = 1024**2
CHILD_LOG_CAP = 16*MIB
MAX_UPDATES_LOG = 6000*2048
QUEUE_METADATA_RESERVE = 2*MIB
PER_FIT_SECONDS = 3600.


def queue_source_files():
    return {str(path):sha256(path) for path in (
        Path(__file__).resolve(), REPO/"tests"/"test_light_layers_campaign.py")}


def queue_source_hash():
    return json_hash(queue_source_files())


def required_test_sources():
    """Exact tested source graph, without reading any source dataset/result."""
    result = implementation_hashes()
    result.update(queue_source_files())
    for name in ("test_light_layers_training.py", "test_light_layers_model.py",
                 "test_light_layers_targets.py", "test_light_layers_overlay.py"):
        path = REPO/"tests"/name
        result[str(path)] = sha256(path)
    # Queue launches the wrapper, so it is also a pinned implementation input.
    result[str(REPO/"run_torch.sh")] = sha256(REPO/"run_torch.sh")
    return result


def verify_cpu_receipt(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1 or value.get("status") != "passed"
            or value.get("scope") != "synthetic_cpu_only"
            or value.get("cuda_used") is not False or value.get("dataset_access") is not False
            or value.get("tests_failed") != 0 or type(value.get("tests_passed")) is not int
            or value["tests_passed"] < 1):
        raise ContractError("A passed synthetic-CPU-only test receipt is required")
    sources = value.get("source_files", {})
    for source_path, expected in required_test_sources().items():
        if sources.get(source_path) != expected:
            raise ContractError(f"CPU receipt did not test the current source: {source_path}")
    return sha256(path)


@dataclass(frozen=True)
class QueueInputs:
    manifest: Path
    transforms: Path
    preregistration: Path
    protocol: Path
    cpu_test_receipt: Path
    output_root: Path
    torch_env: Path

    def paths(self):
        return {name:str(Path(value).resolve()) for name,value in asdict(self).items()}


@dataclass(frozen=True)
class Job:
    index: int
    recipe: str
    seed: int
    config_sha256: str
    output_dir: str
    log_path: str
    argv: tuple[str, ...]
    bindings: dict
    model_parameters: int
    launch_reserve_bytes: int


def _model_parameters(recipe):
    # Meta-device count only; no allocation/initialization on CUDA or data read.
    import torch
    from lightsem.model.light_layers.model import (
        LightLayersConfig, build_light_layers, count_trainable_parameters)
    architecture, visual, pose = RECIPES[recipe]
    with torch.device("meta"):
        model = build_light_layers(LightLayersConfig(visual_aux_dim=visual,pose_aux_dim=pose),
                                   architecture=architecture)
    return count_trainable_parameters(model)


def validate_inputs(inputs: QueueInputs):
    """Validate frozen JSON/code only; does not load feature/target NPZ arrays."""
    values = inputs.paths()
    for name in ("manifest", "transforms", "preregistration", "protocol", "cpu_test_receipt"):
        if not Path(values[name]).is_file():
            raise ContractError(f"Missing frozen queue input: {name}")
    if not (Path(values["torch_env"])/"bin"/"python").is_file():
        raise ContractError("Dedicated TORCH_ENV is missing; no interpreter fallback")
    manifest = json.loads(Path(values["manifest"]).read_text())
    transforms = json.loads(Path(values["transforms"]).read_text())
    registration = json.loads(Path(values["preregistration"]).read_text())
    scope = validate_run_scope(manifest, TrainConfig(), observer_proxy=True)
    if str(Path(values["protocol"]).resolve()) != scope["observer_protocol_path"]:
        raise ContractError("Queue --protocol differs from the manifest scope protocol")
    if sha256(values["protocol"]) != scope["observer_protocol_sha256"]:
        raise ContractError("Queue protocol hash mismatch")
    if transforms.get("smoke") is not False or transforms.get("study_scope") != scope:
        raise ContractError("Queue requires frozen non-smoke transforms in exact observer scope")
    store = SourceStore(manifest)
    bindings = provenance(values["manifest"], store, TrainConfig(), transforms, scope=scope)
    bindings["transforms_sha256"] = sha256(values["transforms"])
    for key in ("manifest_sha256", "source_sha256", "split_sha256", *OBSERVER_SCOPE_BINDINGS):
        if transforms.get("bindings", {}).get(key) != bindings[key]:
            raise ContractError(f"Frozen transform provenance changed: {key}")
    receipt_sha = verify_cpu_receipt(values["cpu_test_receipt"])
    source_sha = queue_source_hash()
    for key, value in (("campaign_queue_source_sha256", source_sha),
                       ("cpu_test_receipt_sha256", receipt_sha)):
        if registration.get("bindings", {}).get(key) != value:
            raise ContractError(f"Aggregate preregistration queue binding missing/changed: {key}")
    experiment = registration.get("vault_experiment_id")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ContractError("One parent-registered aggregate experiment id is required")
    if registration.get("per_job_experiment_ids"):
        raise ContractError("Queue is ONE aggregate experiment, not15 independent experiments")
    root = Path(values["output_root"])
    campaign_root = Path(manifest["campaign_root"]).resolve()
    if root == campaign_root or not root.is_relative_to(campaign_root):
        raise ContractError("Queue output root must be a new subdirectory of the accounted campaign root")
    protocol = json.loads(Path(values["protocol"]).read_text())
    recipes = protocol["student"]["recipes"]
    if recipes != list(OBSERVER_RECIPES):
        raise ContractError("Protocol recipe order changed")
    jobs = []
    parameters = {recipe:_model_parameters(recipe) for recipe in recipes}
    for seed in (0,1,2):
        for recipe in recipes:
            config = TrainConfig(recipe=recipe,seed=seed)
            job_bindings = {**bindings,"config_sha256":json_hash(asdict(config))}
            job_bindings["preregistration_sha256"] = verify_preregistration(
                values["preregistration"], job_bindings, recipe)
            index = len(jobs)
            name = f"{index+1:02d}_seed{seed}_{recipe}"
            output = root/name
            log = root/(name+".log")
            argv = ("bash","run_torch.sh","-u","scripts/light_layers/run_light_layers_train.py",
                "--phase","train","--manifest",values["manifest"],
                "--transforms",values["transforms"],"--preregistration",values["preregistration"],
                "--output",str(output),"--recipe",recipe,"--seed",str(seed),
                "--steps","6000","--batch","4","--observer-proxy","--gpu-authorized")
            reserve = parameters[recipe]*4*3 + MAX_UPDATES_LOG + CHILD_LOG_CAP + 8*MIB
            jobs.append(Job(index,recipe,seed,job_bindings["config_sha256"],str(output),
                            str(log),argv,job_bindings,parameters[recipe],reserve))
    plan = {"schema_version":1,"study_kind":OBSERVER_STUDY_KIND,"inputs":values,
        "aggregate_experiment_id":experiment,"independent_hypothesis_tests":False,
        "study_scope":scope,"campaign_root":str(campaign_root),
        "queue_source_files":queue_source_files(),"queue_source_sha256":source_sha,
        "cpu_test_receipt_sha256":receipt_sha,"wrapper_sha256":sha256(REPO/"run_torch.sh"),
        "preregistration_sha256":sha256(values["preregistration"]),
        "jobs":[asdict(job) for job in jobs],"order":"seed_outer0_1_2_then_protocol_recipe_order",
        "resume_policy":"explicit_exact_completed_jobs_only; incomplete jobs never retried",
        "child_log_cap_bytes":CHILD_LOG_CAP,"per_fit_wall_seconds_ceiling":PER_FIT_SECONDS}
    plan = json.loads(json.dumps(plan,sort_keys=True,allow_nan=False))
    plan["plan_sha256"] = json_hash(plan)
    return plan, jobs


def verify_completed_job(job: Job):
    """Authenticate completed weights/config/scope before any reuse or success."""
    output = Path(job.output_dir)
    marker = output/"complete.json"
    if not marker.is_file():
        raise ContractError(f"Incomplete/failed job blocks queue; no optimizer resume: {output}")
    result = json.loads(marker.read_text())
    if (result.get("status") != "complete" or result.get("smoke") is not False
            or result.get("study_scope", {}).get("study_kind") != OBSERVER_STUDY_KIND):
        raise ContractError(f"Job is not a completed non-smoke observer fit: {output}")
    if result.get("bindings") != job.bindings:
        raise ContractError(f"Completed job bindings changed: {output}")
    if json_hash(result.get("config")) != job.config_sha256:
        raise ContractError(f"Completed job config changed: {output}")
    if result.get("seed") != job.seed:
        raise ContractError(f"Completed job seed changed: {output}")
    scope = result["study_scope"]
    if any(scope.get(key) != job.bindings[key] for key in OBSERVER_SCOPE_BINDINGS):
        raise ContractError(f"Completed job scope changed: {output}")
    for name in ("final","best"):
        path = output/(name+".pt")
        if not path.is_file() or result.get(name+"_weights_sha256") != sha256(path):
            raise ContractError(f"Completed {name} weights missing/changed: {output}")
    if not result.get("best_step") or not math.isfinite(float(result.get("elapsed_s",math.nan))):
        raise ContractError(f"Incomplete checkpoint/timing receipt: {output}")
    return {"job_index":job.index,"recipe":job.recipe,"seed":job.seed,
            "complete_sha256":sha256(marker),"final_weights_sha256":result["final_weights_sha256"],
            "best_weights_sha256":result["best_weights_sha256"],
            "training_elapsed_s":float(result["elapsed_s"])}


@contextmanager
def queue_lock(campaign_root):
    with (Path(campaign_root)/"campaign_queue.lock").open("a+") as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError("Another campaign queue process is active") from error
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def _stop_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid,signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)


def run_child(job: Job, *, torch_env, campaign_root, timeout_s):
    """Native launch, direct file output, hard wall/storage/log-size limits."""
    log = Path(job.log_path)
    if log.exists() or Path(job.output_dir).exists():
        raise ContractError("Refuse overwrite of a child log or run directory")
    environment = os.environ.copy()
    environment["TORCH_ENV"] = str(torch_env)
    started = time.monotonic()
    process = None
    with log.open("xb") as stream:
        try:
            process = subprocess.Popen(list(job.argv),cwd=REPO,env=environment,
                                       stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                remaining = timeout_s-(time.monotonic()-started)
                if remaining <= 0:
                    raise ContractError("Child exceeded per-fit/parent residual wall budget")
                try:
                    code = process.wait(timeout=min(10.,remaining))
                    break
                except subprocess.TimeoutExpired:
                    if log.stat().st_size > CHILD_LOG_CAP:
                        raise ContractError("Child output exceeded bounded log reservation")
                    campaign_space(campaign_root)
            if log.stat().st_size > CHILD_LOG_CAP:
                raise ContractError("Child output exceeded bounded log reservation")
            campaign_space(campaign_root)
            return {"exit_code":code,"elapsed_s":time.monotonic()-started,"pid":process.pid,
                    "log_sha256":sha256(log),"log_bytes":log.stat().st_size}
        finally:
            if process is not None and process.poll() is None:
                _stop_child(process)


def _progress(root, value):
    encoded = json.dumps(value,sort_keys=True,allow_nan=False)
    if len(encoded.encode()) > 8192:
        raise ContractError("Queue progress row exceeded metadata reservation")
    with (Path(root)/"queue_progress.jsonl").open("a") as stream:
        stream.write(encoded+"\n")
    print(encoded,flush=True)


def execute_plan(plan, jobs, *, residual_gpu_hours, resume_exact=False,
                 launch:Callable=run_child, space_check:Callable=campaign_space,
                 refresh:Callable|None=None):
    """Run sequentially; injected launcher is for synthetic CPU tests only."""
    if not math.isfinite(residual_gpu_hours) or not 0 < residual_gpu_hours <= 36:
        raise ContractError("Parent residual GPU budget must be in (0,36] hours")
    root = Path(plan["inputs"]["output_root"])
    campaign_root = Path(plan["campaign_root"])
    current_budget = residual_gpu_hours*3600
    if root.exists():
        if not resume_exact:
            raise ContractError("Queue output exists; explicit --resume-exact required")
        marker = root/"queue_manifest.json"
        if not marker.is_file():
            raise ContractError("Existing directory has no authenticated queue manifest")
        original = json.loads(marker.read_text())
        if original.get("plan") != plan:
            raise ContractError("Resume bindings/source/config/paths differ from original queue")
        if (root/"queue_complete.json").exists():
            # Verify every completed artifact, then return; do not overwrite a
            # completed queue's markers/logs merely to say it is still complete.
            receipts = [verify_completed_job(job) for job in jobs]
            result = json.loads((root/"queue_complete.json").read_text())
            if result.get("plan_sha256") != plan["plan_sha256"] or result.get("jobs") != receipts:
                raise ContractError("Completed queue receipt changed")
            return result
        spent = 0.
        if (root/"queue_progress.jsonl").exists():
            for line in (root/"queue_progress.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row.get("event") == "job_complete":
                    spent += float(row["child"]["elapsed_s"])
        current_budget = min(current_budget,float(original["initial_residual_gpu_hours"])*3600-spent)
    else:
        if resume_exact:
            raise ContractError("--resume-exact requires an existing matching queue")
        space_check(campaign_root,QUEUE_METADATA_RESERVE)
        root.mkdir(parents=True,exist_ok=False)
        write_json(root/"queue_manifest.json",{"plan":plan,"initial_residual_gpu_hours":residual_gpu_hours})
    started = time.monotonic()
    receipts = []
    progress_path = root/"queue_progress.jsonl"
    event_counter = 0
    if progress_path.exists():
        with progress_path.open() as stream:
            event_counter = sum(1 for _ in stream)
    try:
        with queue_lock(campaign_root):
            for job in jobs:
                output = Path(job.output_dir)
                if output.exists():
                    if not resume_exact:
                        raise ContractError("Unexpected existing child output; no overwrite")
                    receipts.append(verify_completed_job(job))
                    continue
                if Path(job.log_path).exists():
                    raise ContractError("An incomplete prior child log blocks retry; preserve and inspect")
                remaining = current_budget-(time.monotonic()-started)
                if remaining <= 0:
                    raise ContractError("Parent residual GPU budget exhausted")
                if refresh is not None:
                    fresh = refresh()
                    if fresh["plan_sha256"] != plan["plan_sha256"]:
                        raise ContractError("Frozen queue inputs/source changed before child launch")
                space_check(campaign_root,job.launch_reserve_bytes+QUEUE_METADATA_RESERVE)
                _progress(root,{"event":"job_start","job_index":job.index,"recipe":job.recipe,
                    "seed":job.seed,"aggregate_experiment_id":plan["aggregate_experiment_id"],
                    "plan_sha256":plan["plan_sha256"],"remaining_budget_s":remaining})
                event_counter += 1
                child = launch(job,torch_env=plan["inputs"]["torch_env"],campaign_root=campaign_root,
                               timeout_s=min(PER_FIT_SECONDS,remaining))
                if child["exit_code"] != 0:
                    raise ContractError(f"Training child failed with exit{child['exit_code']}; stop, no replacements")
                receipt = verify_completed_job(job)
                receipts.append(receipt)
                _progress(root,{"event":"job_complete","receipt":receipt,"child":child,
                    "plan_sha256":plan["plan_sha256"],"aggregate_experiment_id":plan["aggregate_experiment_id"]})
                event_counter += 1
            result = {"schema_version":1,"status":"complete","plan_sha256":plan["plan_sha256"],
                "aggregate_experiment_id":plan["aggregate_experiment_id"],"independent_hypothesis_tests":False,
                "study_scope":plan["study_scope"],"jobs":receipts,"elapsed_s_this_invocation":time.monotonic()-started,
                "parent_residual_budget_hours_this_invocation":residual_gpu_hours,
                "queue_source_sha256":plan["queue_source_sha256"],
                "cpu_test_receipt_sha256":plan["cpu_test_receipt_sha256"]}
            space_check(campaign_root,QUEUE_METADATA_RESERVE)
            write_json(root/"queue_complete.json",result)
            print(json.dumps({"event":"queue_complete","jobs":len(receipts),
                              "plan_sha256":plan["plan_sha256"]}),flush=True)
            return result
    except BaseException as error:
        # Preserve every failure attempt as a unique small marker. No old logs,
        # old weights, or prior blocked markers are overwritten/deleted.
        failure = root/f"queue_blocked_{event_counter:04d}.json"
        if failure.exists():
            failure = root/f"queue_blocked_{event_counter:04d}_{time.time_ns()}.json"
        write_json(failure,{"status":"execution_blocked","error":f"{type(error).__name__}: {error}",
            "plan_sha256":plan["plan_sha256"],"aggregate_experiment_id":plan["aggregate_experiment_id"],
            "completed_jobs":receipts,"elapsed_s":time.monotonic()-started,
            "interpretation":"execution failure, not hypothesis refutation"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("manifest","transforms","preregistration","protocol","cpu-test-receipt","output-root","torch-env"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--residual-gpu-hours",type=float,required=True)
    parser.add_argument("--gpu-authorized",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--resume-exact",action="store_true")
    args = parser.parse_args(argv)
    inputs = QueueInputs(args.manifest,args.transforms,args.preregistration,args.protocol,
                         args.cpu_test_receipt,args.output_root,args.torch_env)
    plan,jobs = validate_inputs(inputs)
    if not math.isfinite(args.residual_gpu_hours) or not 0 < args.residual_gpu_hours <= 36:
        raise ContractError("Parent residual GPU budget must be in (0,36] hours")
    if args.dry_run:
        print(json.dumps({"status":"validated_not_launched","plan":plan,
                          "residual_gpu_hours":args.residual_gpu_hours},sort_keys=True),flush=True)
        return 0
    if not args.gpu_authorized:
        raise ContractError("Parent must explicitly pass --gpu-authorized to launch the queue")
    execute_plan(plan,jobs,residual_gpu_hours=args.residual_gpu_hours,resume_exact=args.resume_exact,
                 refresh=lambda:validate_inputs(inputs)[0])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(f"EXECUTION BLOCKED: {error}",file=sys.stderr,flush=True)
        raise SystemExit(2)
