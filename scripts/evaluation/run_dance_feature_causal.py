#!/usr/bin/env python3
"""CPU-only fixed-C feature interventions with deterministic noise repeats.

This runner is deliberately narrower than the native generation CLI.  It
reuses the selected checkpoint, frozen MuQ/Base92 caches, stereo context and
native 32-step solver, but changes only one declared feature family at a time.
It writes a new artifact root and never overwrites an existing receipt.

The output is a mechanism diagnostic.  It stops at the raw 30 Hz body and
does not map, interpolate, render or apply D4/D6.  A downstream raw-health
audit must run before any mapper/DMX use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/dance/fixed_c.json"
DEFAULT_ARMS = ("full", "zero_muq", "zero_base92")
SOURCE_NAMES = ("drums", "bass", "other", "vocals")
SOURCE_FIELDS = ("rms", "onset", "flux", "sustain")
MUQ_DIM = 1024
BASE92_DIM = 92
EXTRA_DIM = 1061
NATIVE_DIM = 139
FPS = 30.0


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _strict_hash(path: Path, expected: str | None, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if expected is not None and sha(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch: {path}")


def _load_config(path: Path) -> tuple[dict, Path]:
    from lightsem.dance.baseline import load_config

    config, resolved = load_config(path)
    if any(config["addons"][name]["enabled"] for name in ("d4max", "d6")):
        raise ValueError("Causal feature runner requires D4/D6 disabled in the selected config")
    return config, resolved


def _load_inputs(audio: Path, timeline_path: Path, base92_path: Path, config: dict):
    from lightsem.dance.baseline import validate_music, sha as baseline_sha
    from lightsem.dance.pipeline import validate_base92_clock
    from lightsem.experiments.dynamic_dance.contracts import MusicTimeline

    timeline = MusicTimeline.load(timeline_path)
    validate_music(timeline, audio, config)
    _validate_audio_binding(timeline, base92_path)
    with np.load(base92_path, allow_pickle=False) as values:
        if "x30" not in values or "time30" not in values:
            raise ValueError("Base92 cache must contain x30 and time30")
        base = np.asarray(values["x30"], dtype=np.float32)
        base_time = np.asarray(values["time30"], dtype=np.float64)
    validate_base92_clock(base, base_time, timeline.time)
    audio_hash = baseline_sha(audio)
    cached = None
    for key, row in config.get("cached_tracks", {}).items():
        if row.get("audio_sha256") == audio_hash:
            cached = row
            break
    if cached is not None:
        _strict_hash(timeline_path, cached["timeline"]["sha256"], "timeline")
        _strict_hash(base92_path, cached["base92"]["sha256"], "Base92")
    return timeline, base, {
        "audio_path": str(audio),
        "audio_sha256": audio_hash,
        "timeline_path": str(timeline_path),
        "timeline_sha256": baseline_sha(timeline_path),
        "base92_path": str(base92_path),
        "base92_sha256": baseline_sha(base92_path),
        "cached_track": next((k for k, row in config.get("cached_tracks", {}).items()
                               if row.get("audio_sha256") == audio_hash), None),
    }


def _validate_audio_binding(timeline, base92_path: Path) -> dict:
    """Validate the Base92 receipt without importing the Torch trainer."""
    receipt_path = base92_path.with_suffix(".json")
    digest_path = receipt_path.with_suffix(".json.sha256.json")
    if not receipt_path.is_file() or not digest_path.is_file():
        raise ValueError(f"Authenticated Base92 receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if sha(receipt_path) != json.loads(digest_path.read_text()).get("sha256"):
        raise ValueError("Base92 receipt metadata hash mismatch")
    if receipt.get("kind") != "Base92" or receipt.get("npz_sha256") != sha(base92_path):
        raise ValueError("Unauthenticated or changed Base92 inference features")
    waveform = timeline.metadata.get("analyzed_waveform_sha256")
    if not isinstance(waveform, str) or len(waveform) != 64 or waveform != receipt.get("analyzed_waveform_sha256"):
        raise ValueError("Base92 and MusicTimeline analyze different or unknown waveforms")
    if timeline.metadata.get("intervention") != receipt.get("intervention"):
        raise ValueError("Base92 and MusicTimeline intervention provenance differs")
    declared = timeline.metadata.get("base92", {}).get("npz_sha256", timeline.metadata.get("base92_sha256"))
    if declared is not None and declared != sha(base92_path):
        raise ValueError("Base92 differs from the timeline paired-feature receipt")
    return receipt


def _expected_names() -> list[str]:
    from lightsem.experiments.dynamic_dance.audio import MIXTURE_FIELDS, SECTION_LABELS

    return ([f"muq_{i}" for i in range(MUQ_DIM)]
            + [f"{source}_{field}" for source in SOURCE_NAMES for field in SOURCE_FIELDS]
            + [f"mixture_{name}" for name in MIXTURE_FIELDS]
            + [f"{event}_{side}_s" for event in ("beat", "downbeat", "section")
               for side in ("previous", "next")]
            + [f"section_{label}" for label in SECTION_LABELS])


def _apply_arm(base: np.ndarray, extra: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray, dict]:
    base = np.asarray(base, dtype=np.float32).copy()
    extra = np.asarray(extra, dtype=np.float32).copy()
    if base.ndim != 2 or base.shape[1] != BASE92_DIM or extra.ndim != 2 or extra.shape[1] != EXTRA_DIM:
        raise ValueError("Unexpected fixed-C feature dimensions")
    if arm == "full":
        changed = []
    elif arm == "zero_muq":
        extra[:, :MUQ_DIM] = 0.0
        changed = ["muq_0:1024"]
    elif arm == "zero_base92":
        base[:, :] = 0.0
        changed = ["base92_0:92"]
    elif arm == "zero_sources":
        extra[:, MUQ_DIM:MUQ_DIM + 16] = 0.0
        changed = ["drums", "bass", "other", "vocals"]
    elif arm.startswith("zero_source_"):
        source = arm[len("zero_source_"):]
        if source not in SOURCE_NAMES:
            raise ValueError(f"Unknown source arm: {arm}")
        start = MUQ_DIM + SOURCE_NAMES.index(source) * len(SOURCE_FIELDS)
        extra[:, start:start + len(SOURCE_FIELDS)] = 0.0
        changed = [source]
    else:
        raise ValueError(f"Unknown arm {arm!r}")
    return base, extra, {"arm": arm, "replacement": "normalized TRAIN mean (zero)", "changed_groups": changed}


def _pool_extra(extra: np.ndarray) -> np.ndarray:
    bins = (len(extra) + 14) // 15
    pad = bins * 15 - len(extra)
    padded = np.pad(extra, ((0, pad), (0, 0)))
    counts = np.minimum(15, len(extra) - np.arange(bins) * 15)[:, None]
    return padded.reshape(bins, 15, EXTRA_DIM).sum(1) / counts


def _prepare_generation(model, norms, base: np.ndarray, extra: np.ndarray, stereo: np.ndarray | None,
                        stereo_norms, n: int, noise: Any):
    import torch
    from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS

    base_mean, base_std = (x.detach().cpu().numpy() for x in norms[0])
    extra_mean, extra_std = (x.detach().cpu().numpy() for x in norms[1])
    base_norm = (base - base_mean) / base_std
    extra_norm = (extra - extra_mean) / extra_std
    bins = (n + 14) // 15
    planner_bins = max(CONTEXT_BINS, bins)
    coarse_raw = _pool_extra(extra)
    coarse_norm = (coarse_raw - extra_mean) / extra_std
    fine = torch.from_numpy(np.concatenate((base_norm, extra_norm), axis=1)).float()[None]
    coarse = torch.zeros((1, planner_bins, EXTRA_DIM), dtype=torch.float32)
    coarse[:, :bins] = torch.from_numpy(coarse_norm).float()[None]
    valid = torch.zeros((1, planner_bins), dtype=torch.bool)
    valid[:, :bins] = True
    indices = torch.arange(n, dtype=torch.long)[None]
    stereo_fine = stereo_coarse = stereo_binding = None
    if model.config.stereo_dim:
        if stereo is None or stereo_norms is None:
            raise ValueError("The selected stereo checkpoint requires authenticated stereo context")
        stereo_mean, stereo_std = (x.detach().cpu().numpy() for x in stereo_norms)
        stereo_norm = (stereo - stereo_mean) / stereo_std
        stereo_pool = _pool_extra(stereo) if stereo.shape[1] == EXTRA_DIM else None
        if stereo_pool is None:
            sbins = (len(stereo) + 14) // 15
            spad = np.pad(stereo, ((0, sbins * 15 - len(stereo)), (0, 0)))
            stereo_pool = spad.reshape(sbins, 15, stereo.shape[1]).sum(1) / np.minimum(15, n - np.arange(sbins) * 15)[:, None]
        stereo_coarse_raw = stereo_pool
        stereo_coarse_norm = (stereo_coarse_raw - stereo_mean) / stereo_std
        stereo_fine = torch.from_numpy(stereo_norm).float()[None]
        stereo_coarse = torch.zeros((1, planner_bins, model.config.stereo_dim), dtype=torch.float32)
        stereo_coarse[:, :bins] = torch.from_numpy(stereo_coarse_norm).float()[None]
        stereo_binding = {"stereo_context_sha256": array_sha(stereo), "stereo_dim": int(stereo.shape[1])}
    with torch.inference_mode():
        native, plan = model.generate(fine, coarse, valid, indices, noise,
                                      stereo_context=stereo_fine, coarse_stereo=stereo_coarse)
    return native, plan, base_norm, extra_norm, stereo_binding


def _rank(value: np.ndarray) -> int:
    x = np.asarray(value, dtype=np.float64)
    if len(x) > 2048:
        x = x[np.linspace(0, len(x) - 1, 2048).astype(int)]
    x = x - x.mean(0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    return int(np.sum(singular > max(float(singular[0]) if len(singular) else 0.0, 1e-12) * 1e-8))


def _body_metrics(world: np.ndarray) -> dict[str, Any]:
    from lightsem.eval.training_health import motion_metrics

    return motion_metrics(world, fps=FPS)


def _effect(full: np.ndarray, candidate: np.ndarray) -> float:
    full_rel = full - full[:, :1]
    cand_rel = candidate - candidate[:, :1]
    return float(np.sqrt(np.mean((cand_rel - full_rel) ** 2)))


def _ratios(candidate: dict, baseline: dict) -> dict[str, float | None]:
    keys = ("speed_p95", "acceleration_raw", "jerk_raw")
    result = {}
    c_speed = candidate.get("speed_p95")
    b_speed = baseline.get("speed_p95")
    result["speed"] = float(c_speed / b_speed) if c_speed is not None and b_speed else None
    for name in ("acceleration", "jerk"):
        c = candidate.get(name + "_raw", {}).get("p95")
        b = baseline.get(name + "_raw", {}).get("p95")
        result[name] = float(c / b) if c is not None and b else None
    return result


def run_song(args, config, config_path):
    import torch
    from lightsem.dance.baseline import load_model, sha as baseline_sha
    from lightsem.dance.pipeline import load_stereo_context
    from lightsem.experiments.dynamic_dance.audio import model_features
    from lightsem.experiments.dynamic_dance.contracts import DanceResult
    from lightsem.experiments.native_dynamics.evaluation import cpu_noise, setup
    from lightsem.experiments.motion_study.geometry import joints_from_features

    if args.device != "cpu":
        raise ValueError("This runner is CPU-only; GPU execution is deliberately unsupported")
    audio = Path(args.audio).expanduser().resolve()
    timeline_path = Path(args.timeline).expanduser().resolve()
    base92_path = Path(args.base92).expanduser().resolve()
    timeline, base, input_identity = _load_inputs(audio, timeline_path, base92_path, config)
    n = len(base)
    target_time = np.arange(n, dtype=np.float64) / FPS
    extra, names = model_features(timeline, target_time=target_time)
    if list(names) != _expected_names() or extra.shape != (n, EXTRA_DIM):
        raise ValueError("Native feature schema differs from the fixed 1061-D contract")
    config_path = Path(config_path).resolve()
    model, norms, checkpoint_identity = load_model(config, device="cpu", checkpoint=args.checkpoint)
    stereo, stereo_identity = load_stereo_context(audio, timeline, config)
    if stereo is not None and stereo.shape != (n, model.config.stereo_dim):
        raise ValueError("Authenticated stereo context does not match the Base92 clock")
    setup()
    seed = int(args.draw_seed)
    noise = cpu_noise(model.flow, max(int(config["noise_reference_frames"]), n), seed)[:, :n].contiguous()
    noise_np = noise.numpy().copy()
    arms = tuple(dict.fromkeys(args.arms))
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {out}")
    out.mkdir(parents=True)
    baseline_world = None
    baseline_motion = None
    arm_receipts = []
    for arm in arms:
        arm_root = out / arm
        arm_root.mkdir()
        arm_base, arm_extra, intervention = _apply_arm(base, extra, arm)
        native, plan, base_norm, extra_norm, stereo_binding = _prepare_generation(
            model, norms, arm_base, arm_extra, stereo, norms[2] if model.config.stereo_dim else None,
            n, noise)
        native_np = native[0].cpu().numpy().astype(np.float32)
        world_t = joints_from_features(native)[0].cpu().numpy().astype(np.float32)
        contacts = native_np[:, -4:]
        finite = bool(np.isfinite(native_np).all() and np.isfinite(world_t).all())
        motion = _body_metrics(world_t) if finite else {"finite": False}
        if arm == "full":
            baseline_world, baseline_motion = world_t.copy(), motion
        ratios = _ratios(motion, baseline_motion) if baseline_motion is not None else {}
        safety_limits = {"speed": float(args.max_speed_ratio), "acceleration": float(args.max_accel_ratio), "jerk": float(args.max_jerk_ratio)}
        safety_failures = [name for name, ratio in ratios.items()
                           if ratio is not None and ratio > safety_limits[name]]
        if not finite:
            safety_failures.append("nonfinite")
        status = "rejected_raw_safety" if safety_failures else "raw_body_ready"
        repeat_deltas = []
        for repeat in range(max(1, int(args.repeat_count)) - 1):
            repeat_native, _, _, _, _ = _prepare_generation(
                model, norms, arm_base, arm_extra, stereo, norms[2] if model.config.stereo_dim else None,
                n, noise)
            rep = repeat_native[0].cpu().numpy().astype(np.float32)
            repeat_deltas.append({"repeat": repeat + 1,
                                  "max_abs_native": float(np.max(np.abs(rep - native_np))),
                                  "exact_native": bool(np.array_equal(rep, native_np))})
        metadata = {
            "schema_version": 1,
            "kind": "DanceFeatureCausalArm",
            "song_label": args.label,
            "audio": input_identity,
            "baseline_config": str(config_path),
            "baseline_config_sha256": baseline_sha(config_path),
            "checkpoint": checkpoint_identity,
            "draw_seed": seed,
            "noise_sha256": array_sha(noise_np),
            "noise_reference_frames": int(noise_np.shape[1]),
            "feature_schema_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
            "feature_names": names,
            "intervention": intervention,
            "stereo": stereo_identity,
            "stereo_binding": stereo_binding,
            "raw_safety": {"status": status, "failures": safety_failures,
                            "ratios_vs_full": ratios, "limits": safety_limits},
            "metrics": {"root_relative_articulation_rms_vs_full": 0.0 if arm == "full" else _effect(baseline_world, world_t),
                        "motion": motion, "native_effective_rank": _rank(native_np)},
            "noise_floor": {"repeat_count": int(args.repeat_count), "repeats": repeat_deltas,
                             "pass": all(row["exact_native"] for row in repeat_deltas)},
            "scope": "Raw native 30 Hz body only; no mapper, interpolation, DMX or video",
        }
        if finite:
            result = DanceResult(time=target_time, world=world_t, contacts=contacts, native=native_np,
                                 metadata=metadata)
            result.save(arm_root / "body.npz")
        else:
            write_npz(arm_root / "raw_output.npz", time=target_time, native=native_np,
                      world=world_t, contacts=contacts)
        write_npz(arm_root / "conditioning.npz", time=target_time, base92=arm_base,
                  extra=arm_extra, base92_normalized=base_norm, extra_normalized=extra_norm,
                  noise=noise_np,
                  planner_conditioning=np.asarray(plan["conditioning"][0, :((n + 14) // 15)].cpu()))
        write_json(arm_root / "RECEIPT.json", metadata)
        arm_receipts.append({"arm": arm, "status": status, "path": str(arm_root),
                             "receipt_sha256": sha(arm_root / "RECEIPT.json"),
                             "body_sha256": sha(arm_root / "body.npz") if (arm_root / "body.npz").exists() else None})
    receipt = {
        "schema_version": 1,
        "protocol": "dance-feature-causal-fixed-c-cpu-20261027-v1",
        "status": "complete",
        "song": args.label,
        "device": "cpu",
        "arms": arm_receipts,
        "input_identity": input_identity,
        "baseline_config": {"path": str(config_path), "sha256": baseline_sha(config_path)},
        "checkpoint": checkpoint_identity,
        "draw_seed": seed,
        "repeat_count": int(args.repeat_count),
        "no_gpu": True,
        "downstream": "blocked until raw safety/collapse audit; no mapper/DMX/video",
    }
    write_json(out / "CAUSAL_RECEIPT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--base92", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="song")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                        help="comma-separated arms: full,zero_muq,zero_base92,zero_sources,zero_source_<name>")
    parser.add_argument("--draw-seed", type=int, default=7001)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--max-speed-ratio", type=float, default=1.5)
    parser.add_argument("--max-accel-ratio", type=float, default=2.0)
    parser.add_argument("--max-jerk-ratio", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="validate pinned inputs without loading weights or writing artifacts")
    args = parser.parse_args(argv)
    if args.repeat_count < 1 or args.draw_seed < 0:
        raise SystemExit("repeat-count must be >=1 and draw-seed must be nonnegative")
    args.arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if not args.arms:
        raise SystemExit("at least one arm is required")
    if "full" not in args.arms:
        raise SystemExit("the full arm is required as the matched raw-body reference")
    config, config_path = _load_config(args.baseline.resolve())
    if args.dry_run:
        _load_inputs(args.audio.resolve(), args.timeline.resolve(), args.base92.resolve(), config)
        print(json.dumps({"status": "validated", "device": "cpu", "arms": args.arms,
                          "baseline": str(config_path)}, sort_keys=True))
        return
    result = run_song(args, config, config_path)
    print(json.dumps({"status": result["status"], "song": result["song"],
                      "arms": result["arms"], "out": str(Path(args.out).resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
