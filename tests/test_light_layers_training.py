"""Synthetic CPU tests only. No source dataset, TEST or GPU access."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightsem.model.light_layers.model import LightLayersConfig, SharedLightLayers
from lightsem.model.light_layers.training import (
    AUDIO_SCALARS, CENTER10, CENTER40, BatchSampler, ContractError, SourceStore,
    TrainConfig, TrainingData, VisualRecord, compute_loss, deterministic_donors,
    fit_transforms, generated_history, implementation_hashes, intensity_novelty,
    make_targets, positive_weight, predict, sample_linear, sha256,
    validate_smoke_selection, verify_preregistration)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record(rid="r0", partition="train", offset=17., n=640, varying=True):
    times = offset+np.arange(n)/10
    hs = np.zeros((n,72),np.float32)
    hue = ((np.arange(n)//20)%4 if varying else np.zeros(n,int))
    hs[np.arange(n),hue] = 1
    rich = np.zeros((n,96),np.float32); rich[:,:72] = hs
    for first in range(72,96,6):
        rich[:,first] = 1; rich[:,first+2:first+6] = .5
    values = np.where((np.arange(n)//20)%2, .8, .2).astype(np.float32) if varying else np.full(n,.5,np.float32)
    return VisualRecord(rid, "song-"+rid, partition, times, rich, hs,
        np.ones(n,bool), np.full(n,.5), np.full(n,.8), values,
        np.ones(n,bool), np.zeros(n,bool))


def tiny_model(visual=0, pose=0):
    return SharedLightLayers(LightLayersConfig(d_model=8,n_blocks=1,n_heads=2,
        ff_mult=1,conv_kernel=3,attn_window=4,dropout=0,palette_size=4,
        visual_aux_dim=visual,pose_aux_dim=pose))


def batch():
    generator = torch.Generator().manual_seed(5)
    return {"audio40":torch.randn(1,2560,94,generator=generator),
        "accent":torch.zeros(1,2560), "accent_mask":torch.ones(1,2560,dtype=torch.bool),
        "color":torch.zeros(1,640), "color_mask":torch.ones(1,640,dtype=torch.bool),
        "palette":torch.zeros(1,640,dtype=torch.long),
        "palette_mask":torch.ones(1,640,dtype=torch.bool)}


def norm():
    return {"pos_weight":{"accent":1.,"color":1.}}


def test_fit_rejects_development_and_test():
    for split in ("val","dev","test"):
        with pytest.raises(ContractError,match="TRAIN-only"):
            fit_transforms([record(),record("other",split)],palette_size=4)


def test_audio_normalizer_rejects_heldout_tag():
    with pytest.raises(ContractError,match="TRAIN only"):
        fit_transforms([record()],palette_size=4,
                       audio_rows=[("dev",np.ones((10,94),np.float32))])


def test_fit_is_frozen_reproducible_and_zero_events_finite():
    r = record(varying=False)
    one = fit_transforms([r],palette_size=2)
    two = fit_transforms([r],palette_size=2)
    assert one == two
    assert one["event_counts"] == {"color":0,"accent":0}
    assert one["pos_weight"] == {"color":1.,"accent":1.}
    target = make_targets(r,one)
    assert target["palette"].shape == (640,)
    assert target["color"].sum() == 0
    assert np.isfinite(target["rich"]).all()
    assert positive_weight(np.ones(100),np.ones(100,bool)) == 0
    assert positive_weight(np.r_[1,np.zeros(1000)],np.ones(1001,bool)) == 50


def test_fit_rejects_zero_eligibility():
    r = record(); r.color_valid[:] = False; r.hs[:] = 0; r.chromatic[:] = 0
    with pytest.raises(ContractError,match="Zero eligible"):
        fit_transforms([r],palette_size=4)


def test_intensity_gain_invariance_and_actual_irregular_time():
    r = record()
    source = r.times+.012*np.sin(np.arange(640))
    a,_ = intensity_novelty(source,r.values,r.value_valid,r.cuts)
    b,_ = intensity_novelty(source,r.values*.5,r.value_valid,r.cuts)
    np.testing.assert_allclose(a.scores,b.scores,atol=1e-7)
    np.testing.assert_array_equal(a.times_s,source)
    assert a.eligible.sum() > 600


def test_scalar_interpolation_offsets_rates_no_extrapolation():
    times = np.arange(1000)/50
    probability = np.column_stack((times,2*times))
    query = 3.125+np.arange(16)/40
    value,valid = sample_linear(times,probability,query)
    np.testing.assert_allclose(value[:,0],query,atol=1e-6)
    assert valid.all()
    _,valid = sample_linear(times,probability,np.array([-1.,100.]))
    assert not valid.any()


def test_reproducible_sampling_balances_songs_not_coaches():
    rows = [record("a"),record("b"),record("c")]
    rows[1].canonical_id = rows[0].canonical_id
    first,second = BatchSampler(rows,11),BatchSampler(rows,11)
    assert first.sample(100) == second.sample(100)
    draws = BatchSampler(rows,11).sample(2000)
    p = np.mean([rid == "c" for rid,_ in draws])
    assert .45 < p < .55
    assert all(start == 0 for _,start in draws)


def test_donor_never_same_canonical_or_partition():
    rows = [record("a"),record("b"),record("c")]
    donors = deterministic_donors(rows,{"a":100,"b":101,"c":200})
    assert donors["a"] == "b"
    assert all(key != value for key,value in donors.items())
    with pytest.raises(ContractError,match="No different"):
        deterministic_donors([rows[0]])


class OwnStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.cfg = SimpleNamespace(palette_size=4,null_palette_id=4)
    def palette_logits(self,context,ids):
        # Next generated mark follows previous generated state, never any target.
        chosen = torch.where(ids == 4,0,(ids+1)%4)
        return torch.nn.functional.one_hot(chosen,4).float()*20


def test_generated_history_is_own_cues_and_never_gold():
    model = OwnStateModel()
    context = torch.zeros(1,20,2)
    logits = torch.full((1,20),-20.); logits[0,7] = 20.; logits[0,15] = 20.
    state = generated_history(model,context,logits)
    assert state["previous_ids"][0,0] == 4
    assert state["previous_ids"][0,7] == 0
    assert state["generated_palette_ids"][0,7] == 1
    assert state["generated_palette_ids"][0,14] == 1
    assert state["generated_palette_ids"][0,15] == 2
    null = generated_history(model,context,logits,state_ablation=True)
    assert torch.all(null["previous_ids"] == 4)
    assert torch.all(null["generated_palette_ids"] == 0)


def test_correct_center_loss_and_gradients():
    model = tiny_model()
    b = batch()
    losses, outputs = compute_loss(model,b,norm(),state_dropout=0)
    outputs["accent_logit"].retain_grad(); outputs["chromatic_change_logit"].retain_grad()
    outputs["palette_logits"].retain_grad()
    losses["total"].backward()
    grad = outputs["accent_logit"].grad
    assert torch.count_nonzero(grad[:,:1120]) == 0
    assert torch.count_nonzero(grad[:,1440:]) == 0
    assert torch.count_nonzero(grad[:,CENTER40]) > 0
    grad = outputs["chromatic_change_logit"].grad
    assert torch.count_nonzero(grad[:,:280]) == 0
    assert torch.count_nonzero(grad[:,360:]) == 0
    assert torch.count_nonzero(grad[:,CENTER10]) > 0
    assert torch.isfinite(losses["total"])


def test_changing_gold_cannot_change_generated_history():
    model = tiny_model(); b = batch()
    _, first = compute_loss(model,b,norm(),state_dropout=0)
    b["palette"][:] = 3; b["color"][:] = 1
    _,second = compute_loss(model,b,norm(),state_dropout=0)
    assert torch.equal(first["generated_previous_ids"],second["generated_previous_ids"])


def test_whole_sequence_null_dropout_and_empty_masks_are_finite():
    model = tiny_model(); b = batch()
    for key in ("accent_mask","color_mask","palette_mask"):
        b[key][:] = False
    losses, outputs = compute_loss(model,b,norm(),state_dropout=1)
    assert losses["total"] == 0
    losses["total"].backward()
    assert torch.all(outputs["generated_previous_ids"] == model.cfg.null_palette_id)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_aux_loss_coordinate_average_not_dimension_weight():
    from lightsem.model.light_layers.training import _aux_loss
    a = _aux_loss(torch.ones(1,80,6),torch.zeros(1,80,6),torch.ones(1,80,6,dtype=torch.bool))
    b = _aux_loss(torch.ones(1,80,96),torch.zeros(1,80,96),torch.ones(1,80,96,dtype=torch.bool))
    assert a == b == 1


def test_logits_and_predict_output_schema_clip_true_edges():
    model = tiny_model(visual=6,pose=4)
    output = model(torch.zeros(1,40,94))
    assert output["accent_logit"].shape == (1,40)
    assert output["chromatic_change_logit"].shape == (1,10)
    assert output["visual_aux"].shape == (1,10,6)
    assert output["pose_aux"].shape == (1,10,4)
    observed = []
    hook = model.register_forward_pre_hook(lambda module,args: observed.append(args[0].shape[1]))
    prediction = predict(model,np.zeros((3600,94),np.float32))
    hook.remove()
    assert prediction["accent_probability"].shape == (3600,)
    assert prediction["color_probability"].shape == (900,)
    assert prediction["palette_probability"].shape == (900,4)
    assert prediction["generated_palette_ids"].min() >= 0
    assert prediction["generated_palette_ids"].max() < 4
    assert not prediction["chunk_full_equivalence_verified"]
    assert observed[0] == 1440  # true edge clipped, not padded to2560
    assert all(length <= 2560 and length%4 == 0 for length in observed)
    assert model.training


def _reference(path):
    return {"path":str(path),"sha256":sha256(path)}


def source_fixture(tmp_path):
    rid="r0"
    split=tmp_path/"split.json"
    split.write_text(json.dumps({"recordings":{rid:{"canonical_id":"song-r0","partition":"train"}}}))
    audio=tmp_path/"audio.npz"; rhythm=tmp_path/"rhythm.npz"; visual=tmp_path/"visual.npz"
    n=4000; t=np.arange(n)/40
    np.savez(audio,times=t.astype(np.float32),mel=np.broadcast_to(t,(64,n)).astype(np.float16),
             meta=json.dumps({"fps":40}),**{"ch__"+key:t.astype(np.float32) for key in AUDIO_SCALARS})
    rt=np.arange(5000)/50
    np.savez(rhythm,times_s=rt,beat_logit=np.zeros(len(rt)),downbeat_logit=np.full(len(rt),np.log(3)))
    r=record(offset=17)
    np.savez(visual,times_s=r.times,sam2_rich=r.rich,sam2_hs_histogram=r.hs,
             sam2_color_valid=r.color_valid,sam2_retained_fraction=r.retained,
             sam2_chromatic_fraction=r.chromatic,sam2_value_mean=r.values,
             sam2_value_valid=r.value_valid,cuts=r.cuts)
    row={"recording_id":rid,"canonical_id":"song-r0","partition":"train",
         "audio":_reference(audio),"rhythm":_reference(rhythm),"visual":_reference(visual)}
    manifest={"schema_version":1,"campaign_root":str(tmp_path),
              "split_manifest":_reference(split),"records":[row]}
    return manifest,r


def test_source_bus_sigmoid_rate_and_absolute_crop_alignment(tmp_path):
    manifest,r = source_fixture(tmp_path)
    store=SourceStore(manifest,max_ram_bytes=2**24)
    audio=store.audio("r0")
    assert audio.shape == (4000,94)
    np.testing.assert_allclose(audio[:,92],.5)
    np.testing.assert_allclose(audio[:,93],.75)
    actual=store.visual("r0")
    transforms=fit_transforms([actual],palette_size=4)
    transforms["audio"]={"mean":[0]*94,"std":[1]*94}
    data=TrainingData(store,[actual],transforms,"S_none_P0")
    b=data.batch([("r0",0)])
    assert b["audio40"][0,0,64] == 17
    assert b["audio40"][0,4,64] == pytest.approx(17.1,abs=1e-5)
    assert b["accent"].shape == (1,2560)
    assert torch.equal(b["accent"][:,::4],b["accent"][:,1::4])


def test_historical_partition_identity_drift_and_test_rejected(tmp_path):
    manifest,_=source_fixture(tmp_path)
    manifest["records"][0]["partition"]="test"
    with pytest.raises(ContractError,match="TEST/result access"):
        SourceStore(manifest)
    manifest["records"][0]["partition"]="val"
    with pytest.raises(ContractError,match="split drift"):
        SourceStore(manifest)
    manifest["records"][0]["partition"]="train"
    manifest["records"][0]["canonical_id"]="invented"
    with pytest.raises(ContractError,match="identity drift"):
        SourceStore(manifest)


def test_source_hash_and_ram_cap_fail_closed(tmp_path):
    manifest,_=source_fixture(tmp_path)
    store=SourceStore(manifest,max_ram_bytes=100)
    with pytest.raises(ContractError,match="RAM cap"):
        store.audio("r0")
    manifest["records"][0]["audio"]["sha256"]="bad"
    with pytest.raises(ContractError,match="changed source"):
        SourceStore(manifest).audio("r0")


def test_pose_active_offset_and_coach_average_valid_only(tmp_path):
    manifest,_=source_fixture(tmp_path)
    motion=tmp_path/"motion.npz"; validity=tmp_path/"validity.npz"
    times=np.arange(1000)/10
    from lightsem.model.light_layers.training import POSE_NAMES
    np.savez(motion,times=times,meta=json.dumps({"start_s":7.}),
             **{"ch__"+key:times+i for i,key in enumerate(POSE_NAMES)})
    valid=np.ones((1000,4),bool);valid[100,1]=False
    np.savez(validity,times_s=times+7,valid=valid)
    manifest["records"][0]["coaches"]=[{"start_s":7.,"motion":_reference(motion),"validity":_reference(validity)}]
    actual=SourceStore(manifest).visual("r0",include_pose=True)
    assert actual.pose[0,0] == 10  # absolute17 minus active offset7
    assert actual.pose_valid[0,0]
    assert not actual.pose_valid[0,1]
    del manifest["records"][0]["coaches"][0]["validity"]
    with pytest.raises(ContractError,match="explicit gap/cut/tracking"):
        SourceStore(manifest).visual("r0",include_pose=True)


def test_reduced_smoke_requires_original_selection_exclusion_ledger():
    manifest={"records":[{"recording_id":"r0","partition":"train"}]}
    with pytest.raises(ContractError,match="original12"):
        validate_smoke_selection(manifest)
    manifest["smoke_selection"]={"original_recording_ids":[f"r{i}" for i in range(12)],
        "exclusions":{f"r{i}":"explicit_missing_pose" for i in range(1,12)},"replacement_records":False}
    validate_smoke_selection(manifest)


def test_preregistration_fail_closed_and_fixed_non_smoke_schedule(tmp_path):
    with pytest.raises(ContractError,match="exact preregistration"):
        verify_preregistration(None,{},"S_VR_P0")
    with pytest.raises(ContractError,match="6000"):
        TrainConfig(steps=30,batch=1).validate(False)
    TrainConfig(steps=30,batch=1).validate(True)
    with pytest.raises(ContractError,match="recurrence"):
        TrainConfig(recipe="S_VR_P0_R1").validate(False)
    assert len(implementation_hashes()) == 5


def test_off_phase_requested_crop_rejected_and_config_truthful():
    r=record(offset=17.03)
    with pytest.raises(ContractError,match="pooling phase"):
        r.validate()
    for field,value in (("cue_threshold",.6),("auxiliary_weight",.5),("precision","fp32"),("compile",True)):
        with pytest.raises(ContractError,match="constant"):
            replace(TrainConfig(),**{field:value}).validate(False)


def test_transform_payload_hash_not_merely_trusted(tmp_path):
    from lightsem.model.light_layers.training import provenance
    manifest,r=source_fixture(tmp_path)
    path=tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    store=SourceStore(manifest)
    transforms=fit_transforms([r],palette_size=4)
    provenance(path,store,TrainConfig(),transforms)
    transforms["palette_prototypes"][0][0] += .01
    with pytest.raises(ContractError,match="payload"):
        provenance(path,store,TrainConfig(),transforms)


def test_one_update_synthetic_cpu_smoke_writes_model_only(tmp_path,monkeypatch):
    import lightsem.model.light_layers.training as training
    manifest,r=source_fixture(tmp_path)
    rows=[]
    for i in range(12):
        row=dict(manifest["records"][0]);row.update(recording_id=f"r{i}",canonical_id=f"song-r{i}")
        rows.append(row)
    split=Path(manifest["split_manifest"]["path"])
    split.write_text(json.dumps({"recordings":{row["recording_id"]:
        {"canonical_id":row["canonical_id"],"partition":"train"} for row in rows}}))
    manifest.update(records=rows,split_manifest=_reference(split))
    store=SourceStore(manifest)
    records=[store.visual(row["recording_id"]) for row in rows]
    transforms=fit_transforms(records,palette_size=4,
        audio_rows=(("train",store.audio(row["recording_id"])) for row in rows))
    monkeypatch.setattr(training,"build_light_layers",lambda cfg,architecture:tiny_model())
    report=training.train_one(store,records,transforms,tmp_path/"run",
        TrainConfig(recipe="S_none_P0",steps=1,batch=1),{"synthetic":True},smoke=True,device="cpu")
    assert report["status"] == "complete" and report["smoke"]
    assert report["best_step"] is None
    assert not report["optimizer_checkpoints"]
    saved=torch.load(tmp_path/"run"/"final.pt",weights_only=False)
    assert set(saved) == {"model_state_dict","metadata","resume_supported"}
    assert not saved["resume_supported"]
    with pytest.raises(ContractError,match="already exists"):
        training.train_one(store,records,transforms,tmp_path/"run",
            TrainConfig(recipe="S_none_P0",steps=1,batch=1),{},smoke=True,device="cpu")


def test_extractor_v1_quantized_histograms_and_actual_pts(tmp_path):
    manifest,r=source_fixture(tmp_path)
    path=tmp_path/"extracted.npz";receipt=tmp_path/"complete.json"
    receipt.write_text(json.dumps({"status":"complete"}))
    actual=r.times+.01*np.sin(np.arange(640))
    novelty=r.novelty()
    np.savez(path,requested_times_s=r.times,source_times_s=actual,
        rich96=r.rich.astype(np.float16),hs_histogram=r.hs.astype(np.float16),
        color_valid=r.color_valid,retained_fraction=r.retained.astype(np.float16),
        chromatic_fraction=r.chromatic.astype(np.float16),value_valid=r.value_valid,
        value_summary=np.repeat(r.values[:,None],4,axis=1).astype(np.float16),
        novelty_times_s=actual,novelty_scores=novelty.scores.astype(np.float16),
        novelty_eligible=novelty.eligible)
    manifest["records"][0].update(visual=_reference(path),visual_complete=_reference(receipt),
                                 visual_schema="extractor_v1",cuts_status="unannotated")
    row=SourceStore(manifest).visual("r0")
    np.testing.assert_array_equal(row.times,r.times)
    np.testing.assert_array_equal(row.source_times,actual)
    np.testing.assert_array_equal(row.novelty().times_s,actual)
    np.testing.assert_allclose(row.hs.sum(1),1)


@pytest.mark.parametrize("architecture",["shared","independent"])
def test_independent_checkpoint_load_does_not_narrow_twice(tmp_path,architecture):
    from dataclasses import asdict
    from lightsem.model.light_layers.model import build_light_layers
    from lightsem.model.light_layers.training import load_checkpoint
    config=LightLayersConfig(visual_aux_dim=96)
    model=build_light_layers(config,architecture=architecture)
    path=tmp_path/"independent.pt"
    metadata={"reference_model_config":asdict(config),"model_config":asdict(model.cfg),
              "architecture":architecture,"norm_sha256":"example",
              "parameter_match":asdict(model.parameter_match) if architecture == "independent" else None}
    torch.save({"model_state_dict":model.state_dict(),"metadata":metadata,"resume_supported":False},path)
    loaded,meta=load_checkpoint(path,expected_bindings={"norm_sha256":"example"})
    assert loaded.cfg.d_model == model.cfg.d_model
    if architecture == "independent":
        assert loaded.reference_cfg.d_model == 128
    assert all(torch.equal(model.state_dict()[key],value) for key,value in loaded.state_dict().items())
    assert not loaded.training
    with pytest.raises(ContractError,match="Checkpoint binding"):
        load_checkpoint(path,expected_bindings={"norm_sha256":"changed"})


def test_missing_visual_frames_never_supervise_zero_placeholders_or_legacy(tmp_path):
    manifest,r=source_fixture(tmp_path)
    missing=slice(300,310)
    r.frame_valid=np.ones(640,bool);r.frame_valid[missing]=False
    r.color_valid[missing]=False;r.value_valid[missing]=False
    r.rich[missing]=0;r.hs[missing]=0;r.values[missing]=0
    r.retained[missing]=0;r.chromatic[missing]=0
    novelty=r.novelty()
    visual=tmp_path/"gap_visual.npz";receipt=tmp_path/"gap_complete.json"
    receipt.write_text(json.dumps({"status":"complete"}))
    activity=np.ones(640,bool);activity[300:311]=False
    np.savez(visual,requested_times_s=r.times,source_times_s=r.times,frame_valid=r.frame_valid,
        mask_activity_known=activity,rich96=r.rich.astype(np.float16),hs_histogram=r.hs.astype(np.float16),
        color_valid=r.color_valid,value_valid=r.value_valid,retained_fraction=r.retained.astype(np.float16),
        chromatic_fraction=r.chromatic.astype(np.float16),value_summary=np.repeat(r.values[:,None],4,axis=1),
        novelty_times_s=r.times,novelty_scores=novelty.scores.astype(np.float16),novelty_eligible=novelty.eligible)
    legacy=tmp_path/"legacy.npz"
    np.savez(legacy,times=np.arange(640)/10,offset_s=r.times[0],y=np.ones((640,6),np.float32))
    manifest["records"][0].update(visual=_reference(visual),visual_complete=_reference(receipt),
        visual_schema="extractor_v1",cuts_status="unannotated",legacy_visual=_reference(legacy))
    loaded=SourceStore(manifest).visual("r0",include_legacy=True)
    assert not loaded.legacy_valid[missing].any()
    assert not loaded.mask_activity_known[300:311].any()
    transforms=fit_transforms([loaded],palette_size=4)
    targets=make_targets(loaded,transforms)
    for key in ("color_mask","accent_mask10","palette_mask","legacy_mask","rich_mask"):
        assert not targets[key][missing].any(),key
    assert transforms["fit_validity"][0]["frame_valid"] == 630
    assert transforms["aux"]["legacy"]["counts"] == [630]*6
    invalid=record();invalid.frame_valid=np.ones(640,bool);invalid.frame_valid[300]=False
    with pytest.raises(ContractError,match="zero placeholders"):
        invalid.validate()


def observer_scope_fixture(tmp_path):
    from lightsem.model.light_layers.training import OBSERVER_RECIPES,json_hash
    capture_rows=[]
    for i in range(131):
        rid=f"r{i}"
        capture_rows.append({"recording_id":rid,"canonical_id":f"song{i}","partition":"train" if i<100 else "val",
            "source_path":str(tmp_path/f"source_{rid}.mp4"),"source_sha256":"a"*64,
            "pose_path":str(tmp_path/f"pose_{rid}.npz"),"pose_sha256":"b"*64,
            "audio_bus_path":str(tmp_path/f"audio_{rid}.npz"),"audio_bus_sha256":"c"*64,
            "rhythm_path":str(tmp_path/f"rhythm_{rid}.npz"),"legacy_visual_path":str(tmp_path/f"legacy_{rid}.npz")})
    capture=tmp_path/"capture.json";capture.write_text(json.dumps({"inputs":capture_rows}))
    records=[]
    for row in capture_rows[9:]:
        rid=row["recording_id"]
        receipt=tmp_path/f"complete_{rid}.json";receipt.write_text(json.dumps({"status":"complete","input":row}))
        records.append({"recording_id":rid,"canonical_id":row["canonical_id"],"partition":row["partition"],
            "cuts_status":"unannotated","enrollment_input_sha256":json_hash(row),
            "audio":{"path":row["audio_bus_path"],"sha256":row["audio_bus_sha256"]},
            "rhythm":{"path":row["rhythm_path"],"sha256":"d"*64},
            "legacy_visual":{"path":row["legacy_visual_path"],"sha256":"e"*64},
            "source_video":{"path":row["source_path"],"sha256":row["source_sha256"]},
            "source_pose":{"path":row["pose_path"],"sha256":row["pose_sha256"]},
            "visual_complete":_reference(receipt)})
    protocol={"id":"observer-proxy-20260922-v1","campaign_id":"synthetic-campaign",
        "separate_from_original_quality_claims":True,"primary_scope":"Synthetic observer scope, not clean truth",
        "capture":{"cut_policy":"Unknown cuts INCLUDED IN OBSERVER target, never called cut-clean"},
        "student":{"recipes":list(OBSERVER_RECIPES),"seeds":[0,1,2],"max_fits":15,
            "steps":6000,"batch":4,"optimizer":"AdamW","lr":.0003,"weight_decay":.01,
            "warmup":300,"clip":1,"precision":"BF16 mixed/FP32 losses","compile":False,
            "state_dropout":.5,"main_loss_weights":[1,1,1],"auxiliary_weight":.1,
            "context":"64s student feature window; FULL SOURCE RECORD; NOT a64s raw-audio or causal-context claim; Legacy6 fullrecord-relative"},
        "selection":{"manifest":str(capture),"sha256":sha256(capture),"train_groups":100,
            "historical_dev_groups":31,"original_dev_groups":40,"original_dev_coverage":.775,
            "original80percentgate":"FAILED; not waived/relabelled"}}
    path=tmp_path/"protocol.json";path.write_text(json.dumps(protocol))
    identity={"id":"observer-proxy-identity-exclusions-v1","study_kind":"historical_observer_proxy_v1",
        "campaign_id":"synthetic-campaign","capture_manifest_path":str(capture),"capture_manifest_sha256":sha256(capture),
        "observer_protocol_path":str(path),"observer_protocol_sha256":sha256(path),
        "fit_counts_expected":{"train":91,"val":31},"immutable_enrollment":{"train":100,"val":31},
        "capture_continues_unchanged":True,"exclusions":{
            row["recording_id"]:{"canonical_id":row["canonical_id"],"partition":"train",
                "reason":"synthetic unresolved identity","source_reasons":["unresolved_audio_similarity"]}
            for row in capture_rows[:9]}}
    ledger=tmp_path/"identity.json";ledger.write_text(json.dumps(identity))
    manifest={"study_kind":"historical_observer_proxy_v1","campaign_id":"synthetic-campaign",
        "unknown_cuts_policy":"included_in_observer_target","observer_protocol":_reference(path),
        "identity_exclusions":_reference(ledger),"video_input_manifest":_reference(capture),
        "records":records,"mask_mode":"sam2","fit_auxiliaries":["rich","legacy"]}
    return manifest,protocol,path


def test_observer_scope_explicit_and_original_gate_not_waived(tmp_path):
    from lightsem.model.light_layers.training import validate_run_scope,OBSERVER_RECIPES
    manifest,_,_=observer_scope_fixture(tmp_path)
    with pytest.raises(ContractError,match="explicit --observer-proxy"):
        validate_run_scope(manifest,TrainConfig())
    for recipe in OBSERVER_RECIPES:
        for seed in (0,1,2):
            scope=validate_run_scope(manifest,TrainConfig(recipe=recipe,seed=seed),observer_proxy=True)
            assert scope["original_80percent_gate"] == "FAILED; not waived/relabelled"
            assert scope["original_dev_coverage"] == .775
            assert not scope["raw_audio_64_second_context_claim"]
            assert not scope["pure_feature_dimension_causal_claim"]
            assert "complete active recording" in scope["legacy_visual_provenance"]
    manifest["study_kind"]="original_light_layers_v1"
    with pytest.raises(ContractError,match="original scope"):
        validate_run_scope(manifest,TrainConfig())
    with pytest.raises(ContractError,match="historical_observer_proxy_v1"):
        validate_run_scope(manifest,TrainConfig(),observer_proxy=True)


@pytest.mark.parametrize("change,pattern",[
    ("bad_hash","hash mismatch"),("bad_id","id mismatch"),
    ("manifest_cuts","unknown_cuts_policy"),("payload_cuts","cut policy payload"),
    ("context","feature provenance"),("selection","original31/40"),
    ("enrollment","enrollment changed"),("pose_aux","never pose")])
def test_observer_scope_fails_closed_on_scope_drift(tmp_path,change,pattern):
    from lightsem.model.light_layers.training import validate_run_scope
    manifest,protocol,path=observer_scope_fixture(tmp_path)
    if change == "bad_hash": manifest["observer_protocol"]["sha256"]="f"*64
    elif change == "bad_id": protocol["id"]="different-protocol"
    elif change == "manifest_cuts": manifest["unknown_cuts_policy"]="pretend_clean"
    elif change == "payload_cuts": protocol["capture"]["cut_policy"]="Known clean shots only"
    elif change == "context": protocol["student"]["context"]="64s raw causal audio only"
    elif change == "selection": protocol["selection"]["original_dev_coverage"]=.80
    elif change == "enrollment": manifest["records"][0]["recording_id"]="replacement"
    elif change == "pose_aux": manifest["fit_auxiliaries"].append("pose")
    if change in ("bad_id","payload_cuts","context","selection"):
        path.write_text(json.dumps(protocol));manifest["observer_protocol"]=_reference(path)
    with pytest.raises(ContractError,match=pattern):
        validate_run_scope(manifest,TrainConfig(),observer_proxy=True)


def test_observer_scope_never_allows_pose_or_smoke_overrides(tmp_path):
    from lightsem.model.light_layers.training import validate_run_scope
    manifest,_,_=observer_scope_fixture(tmp_path)
    with pytest.raises(ContractError,match="five frozen P0"):
        validate_run_scope(manifest,TrainConfig(recipe="S_V6_P1"),observer_proxy=True)
    with pytest.raises(ContractError,match="non-smoke"):
        validate_run_scope(manifest,TrainConfig(steps=30,batch=1),smoke=True,observer_proxy=True)


def test_observer_preregistration_binds_full_scope_and_all_five_recipes(tmp_path):
    from lightsem.model.light_layers.training import validate_run_scope,OBSERVER_RECIPES,OBSERVER_SCOPE_BINDINGS
    manifest,_,_=observer_scope_fixture(tmp_path)
    scope=validate_run_scope(manifest,TrainConfig(),observer_proxy=True)
    bindings={key:scope[key] for key in OBSERVER_SCOPE_BINDINGS}
    bindings.update({key:"a"*64 for key in ("manifest_sha256","source_sha256","split_sha256",
        "config_sha256","norm_sha256","vocab_sha256","transforms_sha256")})
    from lightsem.model.light_layers.training import json_hash
    from dataclasses import asdict
    bindings["config_sha256"]=json_hash(asdict(TrainConfig()))
    registration={"registered_before_fit":True,"vault_experiment_id":"synthetic-only",
        "study_kind":manifest["study_kind"],"unknown_cuts_policy":manifest["unknown_cuts_policy"],
        "bindings":bindings,"recipes":{recipe:{"config_sha256":[json_hash(asdict(TrainConfig(recipe=recipe,seed=seed)))
                                                             for seed in (0,1,2)]}
                                         for recipe in OBSERVER_RECIPES}}
    path=tmp_path/"preregistration.json";path.write_text(json.dumps(registration))
    assert verify_preregistration(path,bindings,"S_VR_P0") == sha256(path)
    registration["unknown_cuts_policy"]="clean_truth"
    path.write_text(json.dumps(registration))
    with pytest.raises(ContractError,match="scope/cut policy"):
        verify_preregistration(path,bindings,"S_VR_P0")
    registration["unknown_cuts_policy"]=manifest["unknown_cuts_policy"]
    registration["bindings"]=dict(bindings);registration["bindings"].pop("observer_protocol_sha256")
    path.write_text(json.dumps(registration))
    with pytest.raises(ContractError,match="scope binding"):
        verify_preregistration(path,bindings,"S_VR_P0")
    registration["bindings"]=bindings
    path.write_text(json.dumps(registration))
    with pytest.raises(ContractError,match="original quality scope"):
        verify_preregistration(path,{k:v for k,v in bindings.items() if k != "study_kind"},"S_VR_P0")


def test_identity_quarantine_and_additional_receipted_exclusions(tmp_path):
    from lightsem.model.light_layers.training import validate_run_scope,json_hash
    manifest,_,_=observer_scope_fixture(tmp_path)
    scope=validate_run_scope(manifest,TrainConfig(),observer_proxy=True)
    assert scope["usable_train_groups"] == 91 and scope["usable_dev_groups"] == 31
    assert len(scope["identity_quarantined_recording_ids"]) == 9
    assert scope["original_inventory_recordings"] == 131
    additional=[]
    for rid in ("r9","r100"):
        row=next(r for r in manifest["records"] if r["recording_id"] == rid)
        receipt=tmp_path/(rid+"_exclusion.json");receipt.write_text(json.dumps({"recording_id":rid,"reason":"synthetic capture integrity"}))
        additional.append({key:row[key] for key in ("recording_id","canonical_id","partition","enrollment_input_sha256")})
        additional[-1].update(pre_fit=True,not_based_on_target_or_model_scores=True,basis="capture_integrity",
                              reason="synthetic capture integrity",receipt=_reference(receipt))
        manifest["records"].remove(row)
    manifest["capture_exclusions"]=additional
    scope=validate_run_scope(manifest,TrainConfig(),observer_proxy=True)
    assert scope["usable_train_groups"] == 90 and scope["usable_dev_groups"] == 30
    assert scope["original_dev_coverage"] == .775
    assert scope["usable_dev_fraction_of_original40"] == .75
    assert len(scope["excluded_recording_ids"]) == 11
    additional[0]["not_based_on_target_or_model_scores"]=False
    with pytest.raises(ContractError,match="pre-fit reason"):
        validate_run_scope(manifest,TrainConfig(),observer_proxy=True)


def test_observer_source_binding_and_identity_ledger_hash_rejected(tmp_path):
    from lightsem.model.light_layers.training import validate_run_scope
    manifest,_,_=observer_scope_fixture(tmp_path)
    original=manifest["records"][0]["audio"]["path"]
    manifest["records"][0]["audio"]["path"]="/different/audio.npz"
    with pytest.raises(ContractError,match="source binding changed"):
        validate_run_scope(manifest,TrainConfig(),observer_proxy=True)
    manifest["records"][0]["audio"]["path"]=original
    manifest["identity_exclusions"]["sha256"]="bad"
    with pytest.raises(ContractError,match="ledger path/hash"):
        validate_run_scope(manifest,TrainConfig(),observer_proxy=True)


def test_donor_report_names_reuse_not_bijective_permutation():
    from collections import Counter
    records=[record("a"),record("b"),record("c")]
    transforms=fit_transforms(records,palette_size=4)
    audio=np.zeros((4000,94),np.float32);audio[:,91]=.5
    store=SimpleNamespace(audio=lambda rid:audio)
    data=TrainingData(store,records,transforms,"S_VR_P0_Ashuf")
    stats=data.donor_statistics()
    reuse=Counter(data.donors.values())
    assert stats["unique_donor_count"] == len(reuse)
    assert stats["recipient_count"] == 3
    assert stats["maximum_recipient_reuse"] == max(reuse.values())
    assert stats["policy"] == "conditional_tempo_nearest_donor_not_bijective_permutation"
    assert data.donors == deterministic_donors(records,{"a":120,"b":120,"c":120})
