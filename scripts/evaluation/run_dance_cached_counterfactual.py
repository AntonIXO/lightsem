#!/usr/bin/env python3
"""Cached-only native dance donor/time-shift counterfactuals.

The runner uses authenticated fixed-C timeline/Base92/stereo caches only.  It
keeps the checkpoint, native noise draw, clock and non-audio recipe fixed while
replacing the cached audio feature stream with the other song or a circular
time shift.  It stops at the raw native body and never maps or renders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluation import run_dance_feature_causal as base


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_song(audio_hash_label: str, timeline_path: Path, base92_path: Path, config: dict, n: int):
    timeline, base92, identity = base._load_inputs(None, timeline_path, base92_path, config, cached_only=True)
    n = min(n, len(base92))
    target_time = np.arange(n, dtype=np.float64) / base.FPS
    extra, names = base.model_features(timeline, target_time=target_time) if hasattr(base, "model_features") else (None, None)
    if extra is None:
        from lightsem.experiments.dynamic_dance.audio import model_features
        extra, names = model_features(timeline, target_time=target_time)
    if list(names) != base._expected_names() or extra.shape != (n, base.EXTRA_DIM):
        raise ValueError(f"Feature schema mismatch for {audio_hash_label}")
    stereo, stereo_identity = base._load_cached_stereo_context(timeline, config)
    if stereo is not None:
        stereo = stereo[:n]
        if stereo.shape[0] != n:
            raise ValueError(f"Stereo cache coverage mismatch for {audio_hash_label}")
    identity = {**identity, "stereo": stereo_identity, "frames": int(n)}
    return timeline, np.asarray(base92[:n], dtype=np.float32), np.asarray(extra, dtype=np.float32), stereo, target_time, identity


def _prepare(model, norms, base92, extra, stereo, stereo_norms, n, noise):
    native, plan, _, _, stereo_binding = base._prepare_generation(
        model, norms, base92, extra, stereo, stereo_norms, n, noise)
    from lightsem.experiments.motion_study.geometry import joints_from_features
    world = joints_from_features(native)[0].cpu().numpy().astype(np.float32)
    native_np = native[0].cpu().numpy().astype(np.float32)
    return native_np, world, plan, stereo_binding


def _run_target(args, config, config_path, target_name, target_spec, donor_spec, checkpoint):
    import torch
    from lightsem.dance.baseline import load_model
    from lightsem.experiments.native_dynamics.evaluation import cpu_noise, setup

    target_timeline, target_base, target_extra, target_stereo, times, target_identity = _load_song(
        target_name, target_spec[0], target_spec[1], config, args.max_frames)
    _, donor_base, donor_extra, donor_stereo, _, donor_identity = _load_song(
        donor_spec[0], donor_spec[0], donor_spec[1], config, args.max_frames)
    n = len(target_base)
    donor_base, donor_extra = donor_base[:n], donor_extra[:n]
    if target_stereo is not None:
        target_stereo = target_stereo[:n]
    if donor_stereo is not None:
        donor_stereo = donor_stereo[:n]
    model, norms, checkpoint_identity = load_model(config, device="cpu", checkpoint=checkpoint)
    setup()
    noise = cpu_noise(model.flow, max(int(config["noise_reference_frames"]), n), args.draw_seed)[:, :n].contiguous()
    noise_np = noise.numpy().copy()
    stereo_norms = norms[2] if model.config.stereo_dim else None
    arms = {
        "factual": (target_base, target_extra, target_stereo, "own_song"),
        "donor_audio": (donor_base, donor_extra, donor_stereo, "other_song_cached_audio"),
        "time_shift": (np.roll(target_base, args.shift_frames, axis=0),
                        np.roll(target_extra, args.shift_frames, axis=0),
                        None if target_stereo is None else np.roll(target_stereo, args.shift_frames, axis=0),
                        f"circular_shift_{args.shift_frames}_frames"),
    }
    out = Path(args.out).resolve() / target_name
    out.mkdir(parents=True, exist_ok=False)
    factual_world = None
    factual_motion = None
    rows = []
    for arm, (b, e, s, intervention) in arms.items():
        native, world, plan, stereo_binding = _prepare(model, norms, b, e, s, stereo_norms, n, noise)
        finite = bool(np.isfinite(native).all() and np.isfinite(world).all())
        motion = base._body_metrics(world) if finite else {"finite": False}
        if arm == "factual":
            factual_world, factual_motion = world.copy(), motion
        ratios = base._ratios(motion, factual_motion) if factual_motion is not None else {}
        failures = [name for name, ratio in ratios.items() if ratio is not None and ratio > ({"speed":1.5,"acceleration":2.0,"jerk":2.0}[name])]
        if not finite: failures.append("nonfinite")
        repeats=[]
        for _ in range(max(1,args.repeat_count)-1):
            rep, _, _, _ = _prepare(model, norms, b, e, s, stereo_norms, n, noise)
            repeats.append({"exact_native": bool(np.array_equal(rep, native)),
                            "max_abs_native": float(np.max(np.abs(rep-native)))})
        effect = 0.0 if arm == "factual" else base._effect(factual_world, world)
        rel_effect = effect / max(float(np.sqrt(np.mean((factual_world-factual_world[:,:1])**2))), 1e-12)
        metadata={"schema_version":1,"kind":"DanceCachedCounterfactualArm","song_label":target_name,
                  "intervention":intervention,"target_audio":target_identity,"donor_audio":donor_identity,
                  "checkpoint":checkpoint_identity,"draw_seed":args.draw_seed,"noise_sha256":base.array_sha(noise_np),
                  "raw_safety":{"status":"rejected_raw_safety" if failures else "raw_body_ready",
                                 "failures":failures,"ratios_vs_factual":ratios,"limits":{"speed":1.5,"acceleration":2.0,"jerk":2.0}},
                  "metrics":{"root_relative_articulation_rms_vs_factual":effect,"standardized_effect_vs_factual":rel_effect,
                              "motion":motion,"native_effective_rank":base._rank(native)},
                  "noise_floor":{"repeat_count":args.repeat_count,"repeats":repeats,"pass":all(x["exact_native"] for x in repeats)},
                  "scope":"Cached-only raw native 30 Hz donor/time-shift counterfactual; no mapper/DMX/video"}
        result_path=out/arm
        result_path.mkdir()
        if finite:
            from lightsem.experiments.dynamic_dance.contracts import DanceResult
            DanceResult(time=times,world=world,contacts=native[:,-4:],native=native,metadata=metadata).save(result_path/"body.npz")
        (result_path/"RECEIPT.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
        rows.append({"arm":arm,"intervention":intervention,"root_relative_articulation_rms":effect,
                     "standardized_effect":rel_effect,"rank":metadata["metrics"]["native_effective_rank"],
                     "raw_safety":metadata["raw_safety"],"exact_repeat":metadata["noise_floor"]["pass"],
                     "receipt":str(result_path/"RECEIPT.json")})
    report={"schema_version":1,"status":"diagnostic_only","qualification_status":"unqualified_cached_audio_counterfactual",
            "song":target_name,"rows":rows,"target_identity":target_identity,"donor_identity":donor_identity,
            "checkpoint":checkpoint_identity,"draw_seed":args.draw_seed,"shift_frames":args.shift_frames,
            "scope":"Cached-only donor/time-shift raw native body; no waveform opening, mapper or DMX"}
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shpongle-timeline",required=True,type=Path);p.add_argument("--shpongle-base92",required=True,type=Path)
    p.add_argument("--cure-timeline",required=True,type=Path);p.add_argument("--cure-base92",required=True,type=Path)
    p.add_argument("--baseline",type=Path,default=base.DEFAULT_CONFIG);p.add_argument("--checkpoint",type=Path)
    p.add_argument("--out",required=True,type=Path);p.add_argument("--max-frames",type=int,default=1800)
    p.add_argument("--repeat-count",type=int,default=2);p.add_argument("--draw-seed",type=int,default=7001)
    p.add_argument("--shift-frames",type=int,default=450)
    a=p.parse_args();
    if a.out.exists(): raise FileExistsError(a.out)
    config, config_path=base._load_config(a.baseline.resolve())
    target_specs={"shpongle":(a.shpongle_timeline.resolve(),a.shpongle_base92.resolve()),"cure_for_me":(a.cure_timeline.resolve(),a.cure_base92.resolve())}
    a.out.mkdir(parents=True)
    reports=[]
    for target, donor in (("shpongle","cure_for_me"),("cure_for_me","shpongle")):
        reports.append(_run_target(a,config,config_path,target,target_specs[target],target_specs[donor],a.checkpoint))
    (a.out/"PORTFOLIO_RECEIPT.json").write_text(json.dumps({"schema_version":1,"status":"diagnostic_only","songs":reports,
        "scope":"cached-only donor/time-shift raw native counterfactual; no waveform regeneration, mapper, DMX or video"},indent=2,sort_keys=True)+"\n")
    print(json.dumps(reports,indent=2))

if __name__=="__main__": main()
