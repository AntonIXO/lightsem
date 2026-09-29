"""CPU-only queue controls. All child launches are mocks; no GPU/data fits."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

import run_light_layers_campaign as queue
from lightsem.experiments.light_layers.training import (
    ContractError, OBSERVER_RECIPES, SourceStore, TrainConfig, json_hash,
    provenance, sha256, validate_run_scope)
from test_light_layers_training import observer_scope_fixture


def reference(path):
    return {"path":str(path),"sha256":sha256(path)}


@pytest.fixture
def frozen_inputs(tmp_path):
    manifest,protocol,protocol_path=observer_scope_fixture(tmp_path)
    manifest.update(schema_version=1,campaign_root=str(tmp_path))
    capture=json.loads(Path(manifest["video_input_manifest"]["path"]).read_text())
    split=tmp_path/"split.json"
    split.write_text(json.dumps({"recordings":{r["recording_id"]:
        {"canonical_id":r["canonical_id"],"partition":r["partition"]} for r in capture["inputs"]}}))
    manifest["split_manifest"]=reference(split)
    path=tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    scope=validate_run_scope(manifest,TrainConfig(),observer_proxy=True)
    transforms={"schema_version":1,"smoke":False,"study_scope":scope,
        "audio":{"mean":[0.]*94,"std":[1.]*94,"counts":[100]*94},
        "aux":{},"palette_prototypes":[[1/72]*72 for _ in range(32)]}
    transforms["norm_sha256"]=json_hash({"audio":transforms["audio"],"aux":transforms["aux"]})
    transforms["vocab_sha256"]=json_hash(transforms["palette_prototypes"])
    bindings=provenance(path,SourceStore(manifest),TrainConfig(),transforms,scope=scope)
    transforms["bindings"]=bindings
    transform_path=tmp_path/"transforms.json";transform_path.write_text(json.dumps(transforms))
    receipt={"schema_version":1,"status":"passed","scope":"synthetic_cpu_only",
        "cuda_used":False,"dataset_access":False,"tests_failed":0,"tests_passed":1,
        "source_files":queue.required_test_sources()}
    receipt_path=tmp_path/"cpu_receipt.json";receipt_path.write_text(json.dumps(receipt))
    registration={"registered_before_fit":True,"vault_experiment_id":"synthetic-aggregate-only",
        "study_kind":scope["study_kind"],"unknown_cuts_policy":scope["unknown_cuts_policy"],
        "bindings":{**bindings,"transforms_sha256":sha256(transform_path),
            "campaign_queue_source_sha256":queue.queue_source_hash(),
            "cpu_test_receipt_sha256":sha256(receipt_path)},
        "recipes":{recipe:{"config_sha256":[json_hash(asdict(TrainConfig(recipe=recipe,seed=seed)))
                                           for seed in (0,1,2)]} for recipe in OBSERVER_RECIPES}}
    reg_path=tmp_path/"preregistration.json";reg_path.write_text(json.dumps(registration))
    env=tmp_path/"env";(env/"bin").mkdir(parents=True);(env/"bin"/"python").write_text("synthetic executable not invoked")
    return queue.QueueInputs(path,transform_path,reg_path,protocol_path,receipt_path,tmp_path/"fits",env)


def mock_launcher(scope,calls,*,fail_at=None,interrupt_at=None):
    def launch(job,**kwargs):
        calls.append((job.recipe,job.seed,kwargs))
        if interrupt_at == job.index:
            raise KeyboardInterrupt("synthetic interruption before child creation")
        output=Path(job.output_dir);output.mkdir()
        Path(job.log_path).write_text("synthetic mocked child output\n")
        if fail_at == job.index:
            (output/"blocked.json").write_text(json.dumps({"status":"execution_blocked"}))
            return {"exit_code":2,"elapsed_s":1.,"pid":-1,"log_bytes":32,"log_sha256":sha256(job.log_path)}
        for name in ("best","final"):
            (output/(name+".pt")).write_bytes(("mock model not trained "+name).encode())
        report={"status":"complete","smoke":False,"study_scope":scope,"bindings":job.bindings,
            "config":asdict(TrainConfig(recipe=job.recipe,seed=job.seed)),"seed":job.seed,
            "final_weights_sha256":sha256(output/"final.pt"),"best_weights_sha256":sha256(output/"best.pt"),
            "best_step":300,"elapsed_s":1.}
        (output/"complete.json").write_text(json.dumps(report))
        return {"exit_code":0,"elapsed_s":1.,"pid":-1,"log_bytes":32,"log_sha256":sha256(job.log_path)}
    return launch


def test_plan_exact15_seed_outer_native_cli_and_aggregate_id(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs)
    assert len(jobs) == 15
    assert [(j.recipe,j.seed) for j in jobs] == [(r,s) for s in (0,1,2) for r in OBSERVER_RECIPES]
    assert all(j.argv[:4] == ("bash","run_torch.sh","-u","run_light_layers_train.py") for j in jobs)
    assert all("--observer-proxy" in j.argv and "--gpu-authorized" in j.argv for j in jobs)
    assert all(j.argv[j.argv.index("--steps")+1] == "6000" for j in jobs)
    assert all(j.argv[j.argv.index("--batch")+1] == "4" for j in jobs)
    assert not plan["independent_hypothesis_tests"]
    assert plan["aggregate_experiment_id"] == "synthetic-aggregate-only"
    assert not frozen_inputs.output_root.exists()
    assert len({j.bindings["preregistration_sha256"] for j in jobs}) == 1
    assert all(j.launch_reserve_bytes > queue.CHILD_LOG_CAP for j in jobs)


def test_queue_sequential_mocked_children_and_no_overwrite(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs);calls=[]
    result=queue.execute_plan(plan,jobs,residual_gpu_hours=20,
        launch=mock_launcher(plan["study_scope"],calls),space_check=lambda *args:None)
    assert result["status"] == "complete" and len(result["jobs"]) == 15
    assert [(r,s) for r,s,_ in calls] == [(j.recipe,j.seed) for j in jobs]
    assert all(k["timeout_s"] <= 3600 for _,_,k in calls)
    assert (frozen_inputs.output_root/"queue_complete.json").is_file()
    with pytest.raises(ContractError,match="resume-exact"):
        queue.execute_plan(plan,jobs,residual_gpu_hours=20,
            launch=lambda *a,**k:pytest.fail("must not launch"),space_check=lambda *args:None)


def test_first_actual_child_error_stops_and_preserves_artifacts(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs);calls=[]
    with pytest.raises(ContractError,match="exit2"):
        queue.execute_plan(plan,jobs,residual_gpu_hours=20,
            launch=mock_launcher(plan["study_scope"],calls,fail_at=1),space_check=lambda *args:None)
    assert len(calls) == 2
    assert Path(jobs[0].output_dir,"complete.json").is_file()
    assert Path(jobs[1].output_dir,"blocked.json").is_file()
    assert Path(jobs[1].log_path).is_file()
    assert not Path(jobs[2].output_dir).exists()
    assert list(frozen_inputs.output_root.glob("queue_blocked_*.json"))
    with pytest.raises(ContractError,match="Incomplete/failed"):
        queue.execute_plan(plan,jobs,residual_gpu_hours=20,resume_exact=True,
            launch=lambda *a,**k:pytest.fail("failed jobs must not retry"),space_check=lambda *args:None)


def test_explicit_resume_reuses_only_exact_complete_jobs(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs);calls=[]
    with pytest.raises(KeyboardInterrupt):
        queue.execute_plan(plan,jobs,residual_gpu_hours=20,
            launch=mock_launcher(plan["study_scope"],calls,interrupt_at=1),space_check=lambda *args:None)
    first_hash=sha256(Path(jobs[0].output_dir)/"complete.json")
    calls=[]
    result=queue.execute_plan(plan,jobs,residual_gpu_hours=19,resume_exact=True,
        launch=mock_launcher(plan["study_scope"],calls),space_check=lambda *args:None)
    assert len(calls) == 14
    assert sha256(Path(jobs[0].output_dir)/"complete.json") == first_hash
    complete_hash=sha256(frozen_inputs.output_root/"queue_complete.json")
    assert queue.execute_plan(plan,jobs,residual_gpu_hours=19,resume_exact=True,
        launch=lambda *a,**k:pytest.fail("complete queue must not rerun"),space_check=lambda *args:None) == result
    assert sha256(frozen_inputs.output_root/"queue_complete.json") == complete_hash
    Path(jobs[0].output_dir,"best.pt").write_bytes(b"tampered")
    with pytest.raises(ContractError,match="weights missing/changed"):
        queue.execute_plan(plan,jobs,residual_gpu_hours=19,resume_exact=True,
            launch=lambda *a,**k:pytest.fail("tamper must block"),space_check=lambda *args:None)


def test_cpu_receipt_and_queue_source_bindings_fail_closed(frozen_inputs):
    receipt=json.loads(frozen_inputs.cpu_test_receipt.read_text());receipt["cuda_used"]=True
    frozen_inputs.cpu_test_receipt.write_text(json.dumps(receipt))
    with pytest.raises(ContractError,match="CPU-only"):
        queue.validate_inputs(frozen_inputs)
    receipt["cuda_used"]=False;receipt["source_files"][str(queue.REPO/"run_torch.sh")]="wrong"
    frozen_inputs.cpu_test_receipt.write_text(json.dumps(receipt))
    with pytest.raises(ContractError,match="current source"):
        queue.validate_inputs(frozen_inputs)


def test_full_prereg_and_transform_tampering_block(frozen_inputs):
    registration=json.loads(frozen_inputs.preregistration.read_text())
    registration["recipes"]["S_VR_P0"]["config_sha256"][0]="wrong"
    frozen_inputs.preregistration.write_text(json.dumps(registration))
    with pytest.raises(ContractError,match="exact6000"):
        queue.validate_inputs(frozen_inputs)


def test_refresh_source_drift_stops_before_any_child(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs)
    with pytest.raises(ContractError,match="source changed"):
        queue.execute_plan(plan,jobs,residual_gpu_hours=20,
            launch=lambda *a,**k:pytest.fail("must not launch"),space_check=lambda *args:None,
            refresh=lambda:{"plan_sha256":"changed"})


def test_storage_failure_blocks_before_launch(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs)
    def no_space(*args): raise ContractError("synthetic disk cap")
    with pytest.raises(ContractError,match="disk cap"):
        queue.execute_plan(plan,jobs,residual_gpu_hours=20,
            launch=lambda *a,**k:pytest.fail("must not launch"),space_check=no_space)
    assert not frozen_inputs.output_root.exists()


def test_dry_run_does_not_launch_or_create_output(frozen_inputs,monkeypatch,capsys):
    args=[]
    for name,path in frozen_inputs.paths().items():
        args += ["--"+name.replace("_","-"),path]
    args += ["--residual-gpu-hours","20","--dry-run"]
    monkeypatch.setattr(queue,"execute_plan",lambda *a,**k:pytest.fail("dry-run launched"))
    assert queue.main(args) == 0
    output=json.loads(capsys.readouterr().out)
    assert output["status"] == "validated_not_launched"
    assert len(output["plan"]["jobs"]) == 15
    assert not frozen_inputs.output_root.exists()


def test_residual_budget_range_and_explicit_launch_authorization(frozen_inputs):
    plan,jobs=queue.validate_inputs(frozen_inputs)
    for budget in (0,37,float("nan")):
        with pytest.raises(ContractError,match="residual GPU budget"):
            queue.execute_plan(plan,jobs,residual_gpu_hours=budget,
                launch=lambda *a,**k:pytest.fail("must not launch"),space_check=lambda *args:None)
    args=[]
    for name,path in frozen_inputs.paths().items():args += ["--"+name.replace("_","-"),path]
    with pytest.raises(ContractError,match="gpu-authorized"):
        queue.main(args+["--residual-gpu-hours","20"])
