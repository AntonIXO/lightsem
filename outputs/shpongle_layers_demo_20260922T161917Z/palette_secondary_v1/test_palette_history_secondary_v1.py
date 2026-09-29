"""New sidecar seam tests only. Synthetic CPU; no campaign data or GPU."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

DRIVER = Path(__file__).with_name("run_palette_history_secondary_v1.py")
spec = importlib.util.spec_from_file_location("palette_secondary_driver", DRIVER)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def prototypes():
    p = np.zeros((32,72),float)
    for i in range(32): p[i,(i%12)*6+(i//12+3)] = 1
    return p


def record(*, valid=None, offset=37., source=None, name="r", partition="train"):
    p = prototypes()
    phase = (np.sin(np.arange(640)/30)+1)/2
    hs = (1-phase[:,None])*p[0]+phase[:,None]*p[6]
    if valid is None: valid = np.ones(640,bool)
    hs[~valid] = 0
    return SimpleNamespace(canonical_id=name,recording_id=name,partition=partition,
        times=offset+np.arange(640)/10,source_times=source,color_valid=valid.copy(),
        frame_valid=np.ones(640,bool),cuts=np.zeros(640,bool),hs=hs)


def calibration(p):
    tracks=[]
    for i,(step,base,amp) in enumerate(((10,0.,.10),(20,.2,.15),(40,.45,.2),(100,.7,.25))):
        mix=base+amp*np.arange(640)/639
        hist=(1-mix[:,None])*p[0]+mix[:,None]*p[4]
        cues=np.arange(0,640,step)
        choices=((0,1),(0,4),(1,8),(0,8))[i]
        ids=np.array(choices)[np.arange(len(cues))%2]
        tracks.append(m.pm.PaletteTrack(f"train{i}",f"r{i}",np.arange(640)/10,hist,
            np.ones(640,bool),cues/10,ids,m.pm.palette_prototypes_hash(p)))
    return m.pm.fit_palette_scales(tracks,p,partition="TRAIN")


def labels(name="r"):
    requested=37.+np.arange(640)/10
    y=np.zeros(640);y[300]=y[340]=1
    return m.ev.FrozenLabels(name,name,"dev",requested,requested.copy(),y.copy(),y.copy(),
                            np.ones(640,bool),np.ones(640,bool))


class FakeModel(torch.nn.Module):
    def __init__(self,broken=False):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.broken=broken
        self.cfg=SimpleNamespace(palette_size=32)
    def palette_logits(self,context,ids):
        output=torch.full((*ids.shape,32),-100.)
        choice=torch.where(ids==32,0,(ids+1)%32)
        output.scatter_(-1,choice[...,None],100.)
        return output*float("nan") if self.broken else output


def test_common_grid_source_offset_gap_cut_and_raw_hs():
    source=37+np.arange(640)/10
    source[200:]+=.2
    r=record(source=source)
    r.frame_valid[40]=False;r.cuts[120]=True
    hs,valid=m.common_observer(r)
    assert not valid[[40,120,199,200]].any()
    assert np.all(hs[~valid]==0)
    t=m.reference_track(r,[0,100,200,400],prototypes())
    assert np.array_equal(t.times_s,np.arange(640)/10)
    assert np.array_equal(t.hs_histograms,hs)
    s=m.pm.summarize_palette_track(t,prototypes())
    assert s.eight_second_pair_count<560
    assert s.censored_residence_segments>1


def test_interval_mean_not_invalid_first_frame_or_point_token():
    r=record();r.color_valid[0]=False;r.hs[0]=0
    r.hs[1:100]=prototypes()[6]
    t=m.reference_track(r,[0,100],prototypes())
    assert t.token_ids[0]==6
    assert not t.valid[0]


def test_unsupported_placeholder_cannot_affect_metric_or_calibration():
    p=prototypes();valid=np.ones(640,bool);valid[100:200]=False
    t=m.reference_track(record(valid=valid),[0,100,200,300],p)
    assert t.token_ids[1]==0
    altered=replace(t,token_ids=t.token_ids.copy());altered.token_ids[1]=31
    a=m.canonical_track(t,p);b=m.canonical_track(altered,p)
    assert m.pm._track_hash(a)==m.pm._track_hash(b)
    assert m.serial(m.pm.summarize_palette_track(a,p))==m.serial(m.pm.summarize_palette_track(b,p))
    np.testing.assert_array_equal(m.active_tokens(a),m.active_tokens(b))
    cal=calibration(p)
    assert m.serial(m.pm.compare_palette_tracks(a,a,p,cal))==m.serial(m.pm.compare_palette_tracks(b,b,p,cal))


def test_degenerate_train_rule_fails_without_record_repair():
    p=prototypes();r=record();r.hs[:]=p[0]
    t=m.reference_track(r,[0],p)
    with pytest.raises(ValueError,match="degenerate"):
        m.pm.fit_palette_scales([t],p,partition="TRAIN")
    with pytest.raises(ValueError,match="TRAIN"):
        m.pm.fit_palette_scales([t],p,partition="DEV")


def test_pairing_persistence_donor_and_no_reference_api():
    model=FakeModel().eval();context=torch.zeros(1,640,2);cues=np.array([0,100,300])
    got=m.paired_conditions(model,context,context+1,cues)
    for draw,row in got.items():
        assert row["own_state"][100]==1 and row["null_state"][100]==0
        assert np.all(row["persistence"]==row["own_state"][0])
        np.testing.assert_array_equal(row["uniform_noise"],np.random.default_rng(draw).random(3))
        np.testing.assert_array_equal(row["tempo_donor_own"],row["own_state"])
    import inspect
    assert "reference" not in inspect.signature(m.paired_conditions).parameters
    with pytest.raises(TypeError):m.paired_conditions(model,context,context,cues,reference=record())


@pytest.mark.parametrize("bad",["model_train","context_nan","logits_nan","float64"])
def test_rollout_fail_closed_on_nonfinite_and_mode(bad):
    model=FakeModel(broken=bad=="logits_nan").eval()
    context=torch.zeros(1,640,2)
    if bad=="model_train":model.train()
    if bad=="context_nan":context[0,0,0]=float("nan")
    if bad=="float64":context=context.double()
    with pytest.raises(m.tr.ContractError):m.safe_rollouts(model,context,[0,100])


def test_actual_model_context_four_draw_track_score_round_trip():
    torch.set_num_threads(2)
    from lightsem.experiments.light_layers.model import LightLayersConfig,build_light_layers
    cfg=LightLayersConfig(d_model=16,n_blocks=1,n_heads=4,attn_window=4,conv_kernel=3,dropout=0.)
    model=build_light_layers(cfg,architecture="shared").eval()
    x=torch.zeros(1,2560,94)
    with torch.no_grad():out=model(x)
    cues=np.array([0,80,160,240,300,340,400,500])
    conditions=m.paired_conditions(model,out["context10"],out["context10"],cues)
    p=prototypes();r=m.reference_track(record(),cues,p);cal=calibration(p)
    scored=m.score_draw(r,conditions[101],p,cal,labels(),cues,np.full(640,.75))
    assert set(scored)==set(m.CONDITIONS)
    assert all(s["temporal_mean_scaled"] is not None for s in scored.values())
    assert all(s["temporal_scaled"]["mean_intercue_drift_per_s"]>0 for s in scored.values())
    # labels never enter model/rollout; changing references only changes metric values
    other=record();other.hs[:]=p[0]
    rr=m.reference_track(other,cues,p)
    assert not np.array_equal(rr.token_ids,r.token_ids)
    again=m.paired_conditions(model,out["context10"],out["context10"],cues)
    for draw in m.DRAWS:
        for name in m.CONDITIONS:np.testing.assert_array_equal(conditions[draw][name],again[draw][name])


def test_emitted_events_differ_from_raw_cue_head_and_no_gap_crossing():
    lab=labels();cues=np.array([0,300,340]);color=np.full(640,.8);valid=np.ones(640,bool)
    held=np.zeros(640,int)
    score=m.emitted_score(lab,held,cues,color,valid)
    assert score["emitted_count"]==0 and score["average_precision"]==0
    held[300:]=1;held[340:]=2
    score=m.emitted_score(lab,held,cues,color,valid)
    assert score["average_precision"]==1 and score["emitted_count"]==2
    valid[299]=False
    assert m.emitted_score(lab,held,cues,color,valid)["emitted_count"]==1


def synthetic_rows(n=31,positive=False):
    dev=[dict(recording_id=f"r{i}",canonical_id=f"s{i}") for i in range(n)]
    rows=[]
    for role in m.ROLES:
      for seed in m.SEEDS:
       for r in dev:
        for draw in m.DRAWS:
         conditions={}
         for name in m.CONDITIONS:
            value=.5 if positive and name=="own_state" else 1.
            conditions[name]=dict(temporal_mean_scaled=value,temporal_raw={k:value for k in m.pm.COMPONENTS},
                temporal_scaled={k:value for k in m.pm.COMPONENTS},undefined_components=[],marginal_scaled=1.,occupancy_ratio=1.,diversity_ratio=1.,
                generated=dict(observed_valid_time_fraction=1.),timing=dict(coverage=1.,average_precision=.5,false_cues_per_minute=0.))
         rows.append(dict(role=role,seed=seed,draw=draw,**r,conditions=conditions))
    return rows,dev


def test_complete_report_holm2_family_intervals_null_does_not_pass():
    rows,dev=synthetic_rows()
    report=m.summarize(rows,dev)
    for role in m.ROLES:
        assert report["reports"][role]["secondary_pass"] is False
        assert set(report["reports"][role]["holm2"])=={"null_minus_own_temporal","tempo_donor_minus_own_temporal"}
        for contrast in report["reports"][role]["contrasts"].values():
            assert contrast["family97p5"]["confidence"]==.975
            assert contrast["descriptive95"]["confidence"]==.95
    assert report["original_palette_qualified"] is None
    assert report["promotion"] is False


def test_positive_secondary_still_cannot_pass_original_and_missing_support_undefined():
    rows,dev=synthetic_rows(positive=True)
    report=m.summarize(rows,dev)
    assert report["reports"]["selected"]["secondary_pass"] is True
    assert report["original_palette_qualified"] is None
    for row in rows:
        if row["recording_id"] in ("r0","r1"):
            for c in row["conditions"].values():c["generated"]["observed_valid_time_fraction"]=.1
    assert m.summarize(rows,dev)["reports"]["selected"]["secondary_pass"] is None


def test_incomplete_draw_duplicate_and_relabel_fail():
    rows,dev=synthetic_rows(2)
    with pytest.raises(m.tr.ContractError):m.check_inventory(rows[:-1],dev)
    with pytest.raises(m.tr.ContractError):m.check_inventory(rows+[rows[0]],dev)
    rows[0]["canonical_id"]="changed"
    with pytest.raises(m.tr.ContractError):m.check_inventory(rows,dev)


def test_storage_root_no_overwrite_and_free_floor(tmp_path,monkeypatch):
    monkeypatch.setattr(m.shutil,"disk_usage",lambda p:SimpleNamespace(free=2**34))
    root=tmp_path/"derived"
    m.write_json(root/"one.json",{"n":1},root)
    with pytest.raises(m.tr.ContractError):m.write_json(root/"one.json",{"n":2},root)
    with pytest.raises(m.tr.ContractError):m.write_json(tmp_path/"escape.json",{},root)
    monkeypatch.setattr(m.shutil,"disk_usage",lambda p:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(m.tr.ContractError):m.space(root,1)
    monkeypatch.setattr(m.shutil,"disk_usage",lambda p:SimpleNamespace(free=2**34))
    with pytest.raises(m.tr.ContractError):m.space(root,m.CAP)


def test_authorization_fails_before_any_real_input_read(tmp_path,monkeypatch):
    path=tmp_path/"registration.json";path.write_text(json.dumps({"protocol_id":m.PROTOCOL_ID}))
    monkeypatch.setattr(m.ev,"checked_ref",lambda *a:pytest.fail("Source access before authorization"))
    with pytest.raises(m.tr.ContractError,match="authorization"):
        m.load_authorized(path,"score")


def test_secondary_cell_metadata_cannot_be_relabelled_before_sources():
    receipts=[];entries=[];mapping={"r":"donor"}
    for role in m.ROLES:
      for seed in m.SEEDS:
        ref={"path":f"/{role}/{seed}.npz","sha256":f"{role}{seed}"}
        old=dict(recording_id="r",canonical_id="song",artifact=ref,target_sha256="truth")
        receipts.append(dict(checkpoint_role=role,seed=seed,records=[old],checkpoint={"sha256":f"weights{role}{seed}"}))
        entries.append(dict(role=role,seed=seed,recording_id="r",canonical_id="song",original_prediction=ref,
                            target_sha256="truth",donor_id="donor",cues_sha256="cue"))
    meta=m.validate_secondary_entries({"records":entries},receipts,{"donor_map":mapping})
    assert len(meta)==6
    entries[0]["original_prediction"]=entries[1]["original_prediction"]
    with pytest.raises(m.tr.ContractError,match="provenance"):
        m.validate_secondary_entries({"records":entries},receipts,{"donor_map":mapping})


def test_calibration_requires_exact_canonical_record_pairs():
    cal=calibration(prototypes())
    train=[dict(canonical_id=f"train{i}",recording_id=f"r{i}") for i in range(4)]
    m.validate_calibration_inventory(cal,train)
    with pytest.raises(m.tr.ContractError):
        m.validate_calibration_inventory(replace(cal,training_record_ids=cal.training_record_ids+(cal.training_record_ids[0],)),train)
    pairs=list(cal.training_record_ids);pairs[0]=("different",pairs[0][1])
    with pytest.raises(m.tr.ContractError):m.validate_calibration_inventory(replace(cal,training_record_ids=pairs),train)


def test_saved_fractional_ids_and_persistence_corruption_rejected_before_cast():
    cues=np.array([0,100]);draw=101
    z={f"{draw}__{name}":np.zeros(640,np.uint8) for name in m.CONDITIONS}
    z[f"{draw}__uniform_noise"]=np.random.default_rng(draw).random(len(cues))
    m.validate_saved_draw(z,draw,cues)
    z[f"{draw}__own_state"]=np.full(640,.2)
    with pytest.raises(m.tr.ContractError,match="before any conversion"):m.validate_saved_draw(z,draw,cues)
    z[f"{draw}__own_state"]=np.zeros(640,np.uint8)
    z[f"{draw}__persistence"][300]=1
    with pytest.raises(m.tr.ContractError,match="Persistence"):m.validate_saved_draw(z,draw,cues)


def test_passed_source_bound_cpu_receipt_required():
    reg={name:{"sha256":name} for name in ("driver","protocol","tests","clarification")}
    cpu=dict(status="passed",synthetic_only=True,data_access=False,cuda_used=False,tests_failed=0,tests_passed=20,
        driver_sha256="driver",protocol_sha256="protocol",tests_sha256="tests",clarification_sha256="clarification")
    m.validate_cpu_receipt(cpu,reg)
    for key,value in (("status","failed"),("data_access",True),("driver_sha256","stale"),("tests_failed",1)):
        with pytest.raises(m.tr.ContractError):m.validate_cpu_receipt({**cpu,key:value},reg)


def test_unsupported_interval_count_exposed():
    valid=np.ones(640,bool);valid[100:200]=False
    ref=m.reference_track(record(valid=valid),[0,100,200],prototypes())
    assert m.unsupported_intervals(ref)==1
    cal=calibration(prototypes())
    score=m.compact_score(m.pm.compare_palette_tracks(ref,ref,prototypes(),cal),ref)
    assert score["unsupported_interval_count"]==1


def test_zero_relative_denominator_undefined_no_epsilon_and_marginal_exact():
    rows,dev=synthetic_rows()
    for row in rows:
        for condition in row["conditions"].values():
            condition["temporal_mean_scaled"]=0.
            condition["temporal_raw"]={k:0. for k in m.pm.COMPONENTS}
            condition["temporal_scaled"]={k:0. for k in m.pm.COMPONENTS}
            condition["marginal_scaled"]=0.
    report=m.summarize(rows,dev)["reports"]["selected"]
    assert report["relative_reduction"] is None and report["secondary_pass"] is None
    assert report["gates"]["marginal"] is True
    for row in rows:row["conditions"]["own_state"]["marginal_scaled"]=1e-9
    assert m.summarize(rows,dev)["reports"]["selected"]["gates"]["marginal"] is False


def test_changing_noncue_token_is_not_hidden_by_metric_adapter():
    p=prototypes();ref=m.reference_track(record(),[0,100,300],p)
    ids=np.zeros(640,np.uint8);ids[1::2]=1
    with pytest.raises(m.tr.ContractError,match="outside fixed cues"):
        m.generated_track(ref,ids,p)


def test_constant_occupancy_collapse_blocks_secondary_even_with_benefit():
    rows,dev=synthetic_rows(positive=True)
    for row in rows:
        row["conditions"]["own_state"]["occupancy_ratio"]=.25
        row["conditions"]["own_state"]["diversity_ratio"]=.25
    report=m.summarize(rows,dev)["reports"]["selected"]
    assert report["secondary_pass"] is False
    assert report["gates"]["occupancy"] is False and report["gates"]["diversity"] is False


def test_native_vault_identity_and_protocol_binding_not_bare_marker():
    reg={"vault_experiment_id":"exp-test","protocol":{"path":"/protocol","sha256":"frozen"}}
    native={"start":{"id":"exp-test","hypothesis_id":"h-layers-20260922-palette",
        "external_run":str(m.DERIVED_ROOT),"config_note":json.dumps({"protocol":reg["protocol"]})}}
    m.validate_native_vault_receipt(native,reg)
    with pytest.raises(m.tr.ContractError):m.validate_native_vault_receipt({"status":"registered"},reg)
    native["start"]["config_note"]=json.dumps({"protocol":{"path":"/protocol","sha256":"stale"}})
    with pytest.raises(m.tr.ContractError,match="protocol"):m.validate_native_vault_receipt(native,reg)


def test_requested_clock_center_boundaries_and_preserved_precenter_state():
    lab=labels();lab.color[:]=0;lab.color[280]=1
    lab.source_times10_s[:]=lab.requested_times10_s+.15
    cues=np.array([0,279,280,360])
    ids=np.zeros(640,int);ids[279:]=1;ids[280:]=2;ids[360:]=3
    score=m.emitted_score(lab,ids,cues,np.full(640,.8),np.ones(640,bool))
    assert score["candidate_count"]==1 and score["emitted_count"]==1
    assert score["average_precision"]==1


def test_fragmented_eighty_percent_support_without_lag_is_undefined():
    p=prototypes();valid=np.ones(640,bool);valid[::9]=False
    ref=m.reference_track(record(valid=valid),[0,100,200,300,400,500],p)
    assert ref.valid.mean()>.8
    ids=np.zeros(640,np.uint8)
    score=m.pm.compare_palette_tracks(m.generated_track(ref,ids,p),ref,p,calibration(p))
    assert score.temporal_mean_scaled is None
    assert "eight_second_change" in score.undefined_components
    assert score.generated.eight_second_pair_count==0
    assert m.diversity(ref)["occupancy"] is not None


def test_context_sensitive_donor_route_and_raw_diversity_saved():
    class ContextModel(FakeModel):
        def palette_logits(self,context,ids):
            out=torch.full((*ids.shape,32),-100.)
            initial=(context[...,0]>0).long()*4
            choice=torch.where(ids==32,initial,(ids+1)%32)
            out.scatter_(-1,choice[...,None],100.)
            return out
    model=ContextModel().eval();context=torch.zeros(1,640,2);cues=np.array([0,100,300])
    got=m.paired_conditions(model,context,context+1,cues)
    assert got[101]["own_state"][0]==0 and got[101]["tempo_donor_own"][0]==4
    ref=m.reference_track(record(),cues,prototypes())
    score=m.score_draw(ref,got[101],prototypes(),calibration(prototypes()),labels(),cues,np.full(640,.8))
    assert score["own_state"]["generated_diversity"]["occupancy"]==3
    assert score["own_state"]["reference_diversity"]["effective_diversity"]>=1


def test_missing_component_and_silently_surviving_aggregate_rejected():
    rows,dev=synthetic_rows(2)
    del rows[0]["conditions"]["own_state"]["temporal_scaled"][m.pm.COMPONENTS[0]]
    with pytest.raises(m.tr.ContractError,match="component"):m.check_inventory(rows,dev)
    rows,dev=synthetic_rows(2)
    score=rows[0]["conditions"]["own_state"]
    score["temporal_raw"]["eight_second_change"]=None
    score["temporal_scaled"]["eight_second_change"]=None
    score["undefined_components"]=["eight_second_change"]
    with pytest.raises(m.tr.ContractError,match="Aggregate"):m.check_inventory(rows,dev)
