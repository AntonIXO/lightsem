"""Build the D7-amplitude/readability corrective portfolio.

The selected native checkpoint and fresh v8 bodies stay immutable. This
campaign applies one fixed, song-specific root-relative D7 amplitude
calibration, then a fixed per-output angular readability calibration derived
from D7 under mapper 01. It is an explicit corrective intervention, not a
retrained checkpoint or temporal fallback. Every corrected body and emitted
timeline is gated before rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_transfer_v2 import CATALOG, make_outputs
from lightsem.conductor.body_transfer_v2.diagnostics import measure
from lightsem.conductor.d7_amplitude_adapter import calibrate_to_d7, root_relative_range
from lightsem.dance.lighting import _load_p2_colors, _resample_body_full_interval
from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
from lightsem.experiments.native_repair.assessment import raw_metrics

ROOT = Path("/root/lightsem")
BASE = Path("/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8")
OUT = Path(os.environ.get("NATIVE_CORRECTED_ROOT", "/mnt/lightsem-data/experiments/lightsem/native_portfolio_corrected_20261012_v1"))
D7_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/bodies/D7_reference")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(f"preserving existing artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def world(path):
    with np.load(path, allow_pickle=False) as z:
        key = "world" if "world" in z.files else "body"
        return z[key].astype(float), z["time"].astype(float) if "time" in z.files else np.arange(len(z[key])) / 30.


def freeze():
    if (OUT / "PLAN.json").exists(): raise FileExistsError("corrective campaign already frozen")
    old = json.loads((BASE / "PLAN.json").read_text())
    tracks = []
    for track in old["tracks"]:
        song = track["id"]
        tracks.append({**track, "v8_body": str(BASE / "bodies" / song / "body.npz"),
            "v8_body_sha256": sha(BASE / "bodies" / song / "body.npz"),
            "d7_reference_body": str(D7_ROOT / song / "body.npz"),
            "d7_reference_body_sha256": sha(D7_ROOT / song / "body.npz")})
    sources = [ROOT / "scripts/motion/rebuild_native_portfolio_corrected_20261012.py",
        ROOT / "lightsem/conductor/d7_amplitude_adapter.py", ROOT / "lightsem/conductor/body_transfer.py",
        ROOT / "scripts/rendering/render_body_portfolio.py", ROOT / "configs/dance/fixed_c.json"]
    files = {str(p.resolve()): sha(p) for p in sources}
    for t in tracks:
        for key in ("audio", "audio_reference", "timeline", "base92", "colors", "color_report", "loudness", "v8_body", "d7_reference_body"):
            p = t[key]["path"] if isinstance(t[key], dict) else t[key]
            files[str(Path(p).resolve())] = sha(p)
    plan = {"schema_version":1,"id":OUT.name,"status":"d7_corrective_intervention","base_plan":str(BASE/"PLAN.json"),"base_plan_sha256":sha(BASE/"PLAN.json"),"baseline_config":old["baseline_config"],"baseline_config_sha256":old["baseline_config_sha256"],"checkpoint":old["checkpoint"],"muq":old["muq"],"stereo_conditioning":old["stereo_conditioning"],"tracks":tracks,"algorithms":old["algorithms"],"counts":old["counts"],"prefix_s":180,"review_start_s":60,"review_duration_s":120,"fps":40.,"body_fps":30.,"d4max":False,"d6":False,"motion_limits":False,"correction":{"name":"D7_amplitude_plus_per_output_angular_readability","adapter":"lightsem/conductor/d7_amplitude_adapter.py","body_gain_scope":"one fixed song-specific root-relative gain derived from D7 range","mapper_gain_scope":"one fixed gain per moving output derived from D7 mapper01 angle range","max_mapper_gain":6.0,"training_checkpoint_changed":False,"temporal_controller":False},"readability_gates":{"body_range_ratio_min":0.90,"body_range_ratio_max":1.15,"angle_range_ratio_min":0.80,"angle_range_ratio_max":1.20,"active_fraction_min":0.70,"target_decoded_angle_ratio_min":0.98},"files":files}
    OUT.mkdir(parents=True)
    save(OUT/"PLAN.json",plan); save(OUT/"source_manifest.json",{"files":files,"scope":"Correction source bindings"})
    print(json.dumps({"stage":"freeze","root":str(OUT),"plan_sha256":sha(OUT/"PLAN.json")},indent=2))


def correct_bodies():
    plan=json.loads((OUT/"PLAN.json").read_text()); rows={}
    for t in plan["tracks"]:
        song=t["id"]; source,source_time=world(Path(t["v8_body"])); ref,_=world(Path(t["d7_reference_body"])); corrected,cal=calibrate_to_d7(source,ref,max_gain=1.6)
        path=OUT/"bodies"/song/"body.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,time=source_time,world=corrected,source_body_sha256=sha(t["v8_body"]))
        m=raw_metrics(corrected,ref);base=raw_metrics(source,ref);range_ratio=root_relative_range(corrected)/max(root_relative_range(ref),1e-9);speed_ratio=m["raw_relative_speed_mean_m_s"]/max(base["raw_relative_speed_mean_m_s"],1e-9);jerk_ratio=m["raw_relative_jerk_p95_m_s3"]/max(base["raw_relative_jerk_p95_m_s3"],1e-9)
        passed=bool(m["finite"] and not m["constant_body"] and .90<=range_ratio<=1.15 and speed_ratio<=1.7 and jerk_ratio<=1.7)
        rows[song]={"body":str(path),"body_sha256":sha(path),"source_v8_sha256":sha(t["v8_body"]),"d7_sha256":sha(t["d7_reference_body"]),"calibration":cal,"raw_metrics":m,"source_metrics":base,"range_ratio":range_ratio,"speed_ratio_vs_v8":speed_ratio,"jerk_ratio_vs_v8":jerk_ratio,"pass":passed}
        if not passed: raise RuntimeError(f"corrected raw body rejected: {song}")
    save(OUT/"RAW_CORRECTED_SAFETY.json",{"status":"complete","rows":rows,"scope":"D7 calibrated corrected raw native body before mapper/DMX/video"});print(json.dumps({"stage":"correct_bodies","status":"complete"},indent=2))


def map_all():
    plan=json.loads((OUT/"PLAN.json").read_text()); safety=json.loads((OUT/"RAW_CORRECTED_SAFETY.json").read_text()); cfg=json.loads((ROOT/"configs/dance/fixed_c.json").read_text()); outs=make_outputs(**plan["counts"])
    for t in plan["tracks"]:
        song=t["id"]; body,_=world(OUT/"bodies"/song/"body.npz"); d7,_=world(Path(t["d7_reference_body"])); n=7200; time=np.arange(n)/40.; world40=_resample_body_full_interval(body,np.arange(len(body))/30.,time); d7_40=_resample_body_full_interval(d7,np.arange(len(d7))/30.,time)
        left,right,_,_,_=_load_p2_colors(Path(t["colors"]["path"]),Path(t["audio"]),None);left,right=left[:n],right[:n];lufs,_=load_verified_loudness(Path(t["loudness"]["path"]),Path(t["audio"]))
        d7_dmx,d7_patch,_,_,_ = compile_world(d7_40,time,outs,left,right,lufs,cfg["lighting"]["transfer"],mapper_id="01"); d7_angles=decode(d7_dmx,d7_patch)["angles_deg"][:,:13]; d7_range_by_output=np.ptp(d7_angles,axis=0).mean(1); d7_range=float(d7_range_by_output.mean())
        for a in plan["algorithms"]:
            aid=a["id"]; folder=OUT/"timelines"/aid/song;folder.mkdir(parents=True,exist_ok=True);base=dict(cfg["lighting"]["transfer"]);dmx0,patch0,_,_,_=compile_world(world40,time,outs,left,right,lufs,base,mapper_id=aid);angles0=decode(dmx0,patch0)["angles_deg"][:,:13]; range0=np.ptp(angles0,axis=0);gains=np.clip(d7_range_by_output/np.maximum(range0.mean(1),1e-6),1.,plan["correction"]["max_mapper_gain"]);base["readability_gains"]=gains.tolist();dmx,patch,intent,brightness,hard=compile_world(world40,time,outs,left,right,lufs,base,mapper_id=aid);controls=decode(dmx,patch);angles=controls["angles_deg"][:,:13];angle_range=np.ptp(angles,axis=0);active=(np.abs(np.diff(angles,axis=0))>.01).mean(0);ratio=angle_range.mean()/max(d7_range,1e-9);target_ratio=float(np.mean(angle_range/np.maximum(np.ptp(angles0,axis=0),1e-6)));gate=bool(plan["readability_gates"]["angle_range_ratio_min"]<=ratio<=plan["readability_gates"]["angle_range_ratio_max"] and np.all(active>=plan["readability_gates"]["active_fraction_min"]) and target_ratio>=plan["readability_gates"]["target_decoded_angle_ratio_min"])
            np.savez_compressed(folder/"timeline.npz",time=time,body=world40,universes=dmx,targets=intent["targets"],dimmer=intent["dimmer"],source_points=intent["source_points"],field_points=intent["field_points"]);(folder/"patch.json").write_text(json.dumps(patch,indent=2)+'\n');diag=measure(world40,intent,dmx,patch,fps=40.,start_s=0.,duration_s=180.);report={"status":"diagnostic","algorithm":aid,"song":song,"checkpoint_sha256":plan["checkpoint"]["sha256"],"body_sha256":sha(OUT/"bodies"/song/"body.npz"),"timeline_sha256":sha(folder/"timeline.npz"),"patch_sha256":sha(folder/"patch.json"),"audio_sha256":t["audio_sha256"],"p2_sha256":sha(t["colors"]["path"]),"loudness_receipt_sha256":sha(t["loudness"]["path"]),"frames":n,"fixtures":len(outs),"hard_checks":hard,"diagnostics":diag,"correction":{"body":safety["rows"][song]["calibration"],"readability_gains":gains.tolist(),"d7_mapper01_angle_range_mean_deg":float(d7_range),"uncorrected_angle_range_mean_deg":float(range0.mean()),"corrected_angle_range_mean_deg":float(angle_range.mean()),"corrected_angle_ratio_to_d7":float(ratio),"target_to_decoded_ratio":target_ratio,"active_fraction_min":float(active.min()),"readability_gate_passed":gate},"d4max":False,"d6":False,"motion_limits":False,"scope":"D7-calibrated corrective intervention; not retrained native checkpoint"};save(folder/"report.json",report);print(f"MAPPED {aid} {song} gate={gate}",flush=True)
    print(json.dumps({"stage":"map","pairs":30},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("freeze","correct_bodies","map"));a=p.parse_args();{"freeze":freeze,"correct_bodies":correct_bodies,"map":map_all}[a.stage]()

if __name__=="__main__":main()
