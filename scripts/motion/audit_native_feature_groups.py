#!/usr/bin/env python3
"""Measure native-body sensitivity to authenticated audio feature groups.

Every group is mean-ablated in both fine and coarse audio tensors with the
same song, stereo context, solver noise and checkpoint. This is a diagnostic
of the current conditioning path, not evidence of choreography or feature
quality.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.dance.baseline import load_config, sha
from lightsem.dance.pipeline import accelerator_lock, music_input, write_json
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.evaluation import cpu_noise, setup
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from scripts.motion.audit_native_pairing import canonical_inputs


GROUPS = {
    "muq_1024": (0, 1024),
    "source_16": (1024, 1040),
    "mixture_4": (1040, 1044),
    "events_sections_17": (1044, 1061),
    "all_audio": (0, 1061),
}


def rms(value):
    return float(torch.sqrt(value.float().square().mean() + 1e-12).cpu())


def generate(protocol_path: Path):
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol["files"].items():
        if sha(name) != digest:
            raise ValueError("Frozen input changed: " + name)
    out = Path(protocol["output"])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    setup()
    config, _ = load_config()
    state = torch.load(config["checkpoint"]["path"], weights_only=False, map_location="cpu")
    model = NativeDynamicsModel(MusicConfig(**state["base_config"]), state["normalizers"], Config(**state["config"]))
    load = model.load_state_dict(state["model"], strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise ValueError(f"Checkpoint load mismatch: {load}")
    model.cuda().eval().requires_grad_(False)
    started = time.monotonic()
    report = {"status": "running", "protocol_sha256": sha(protocol_path), "songs": {}}
    with accelerator_lock(), torch.inference_mode():
        for song, audio in protocol["songs"].items():
            music, base, binding, stereo = music_input(Path(audio), out, config)
            args, kwargs = canonical_inputs(music, base, stereo, state, config, "cuda")
            noise = cpu_noise(model.flow, max(config["noise_reference_frames"], len(base)), config["draw_seed"])
            noise = noise[:, :len(base)].contiguous().cuda()
            baseline_native, _ = model.generate(*args, noise, **kwargs)
            baseline_world = joints_from_features(baseline_native)[0]
            baseline_root_relative = baseline_world - baseline_world[:, :1]
            song_report = {"binding": binding, "noise_sha256": sha_bytes(noise), "groups": {}}
            for group, (start, end) in GROUPS.items():
                fine, coarse, valid, frame_indices = (tensor.clone() for tensor in args)
                fine[..., 92 + start:92 + end] = 0.
                coarse[..., start:end] = 0.
                candidate, _ = model.generate(fine, coarse, valid, frame_indices, noise, **kwargs)
                world = joints_from_features(candidate)[0]
                rel = world - world[:, :1]
                delta = rel - baseline_root_relative
                song_report["groups"][group] = {
                    "feature_range": [start, end],
                    "fine_zeroed_sha256": sha_bytes(fine),
                    "coarse_zeroed_sha256": sha_bytes(coarse),
                    "root_relative_delta_rms_m": rms(delta),
                    "root_relative_candidate_rms_m": rms(rel),
                    "world_delta_rms_m": rms(world - baseline_world),
                    "native_delta_rms": rms(candidate - baseline_native),
                    "finite": bool(torch.isfinite(candidate).all() and torch.isfinite(world).all()),
                }
            report["songs"][song] = song_report
    report.update(status="complete", elapsed_s=time.monotonic() - started,
                  checkpoint=config["checkpoint"]["path"], checkpoint_sha256=sha(config["checkpoint"]["path"]),
                  groups=GROUPS, conditioning_note="1061 audio values -> width-32 pointwise adapter; planner self-attention is separate")
    write_json(out / "REPORT.json", report)
    write_json(out / "REPORT.sha256.json", {"sha256": sha(out / "REPORT.json")})


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    generate(parser.parse_args().protocol)
