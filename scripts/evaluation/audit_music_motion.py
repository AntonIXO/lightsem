#!/usr/bin/env python3
"""Run the Music--Motion Causal Audit on saved native bodies.

The command is CPU-only and read-only with respect to source artifacts.  It
expects one NPZ per audio pair with ``time``, ``onset`` and ``beats`` arrays;
``beats_per_bar`` is optional.  A manifest can contain multiple song pairs so
the diagonal/off-diagonal conditioning test is computed in one invocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lightsem.eval.music_motion import (
    actor_calibration,
    audio_predictive_gain,
    classify,
    conditioning_matrix,
    latent_metrics,
    motion_profile,
    synchrony_report,
)
from lightsem.eval.training_health import motion_metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_body(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        for key in ("body", "world", "native", "positions"):
            if key in z.files:
                return np.asarray(z[key])
    raise ValueError(f"{path} has no body/world/native/positions array")


def _load_audio(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        if not {"time", "onset", "beats"}.issubset(names):
            raise ValueError(f"{path} must contain time, onset and beats arrays")
        result: dict[str, Any] = {key: np.asarray(z[key]) for key in ("time", "onset", "beats")}
        if "downbeats" in names:
            result["downbeats"] = np.asarray(z["downbeats"])
        if "beats_per_bar" in names:
            result["beats_per_bar"] = int(np.asarray(z["beats_per_bar"]).reshape(-1)[0])
        return result


def _load_latent(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        for key in ("latent", "z", "slots", "features", "representation"):
            if key in z.files:
                return np.asarray(z[key])
    raise ValueError(f"{path} has no latent/z/slots/features/representation array")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "lightsem.music_motion.manifest.v1":
        raise ValueError("unsupported music-motion manifest schema")
    pairs = payload.get("pairs")
    if not isinstance(pairs, dict) or len(pairs) < 1:
        raise ValueError("manifest.pairs must contain at least one song")
    base = path.parent
    return payload, {
        str(label): {
            "body": _resolve(base, row["body"]),
            "audio": _resolve(base, row["audio"]),
            "reference_body": _resolve(base, row["reference_body"])
            if row.get("reference_body") else None,
            "reference_bodies": [_resolve(base, value) for value in row.get("reference_bodies", [])],
            "latent": _resolve(base, row["latent"]) if row.get("latent") else None,
        }
        for label, row in pairs.items()
    }


def build_report(manifest_path: Path) -> dict[str, Any]:
    config, pairs = _manifest(manifest_path)
    fps = float(config.get("fps", 30.0))
    n_null = int(config.get("n_null", 200))
    seed = int(config.get("seed", 0))
    bodies, audios = {}, {}
    per_song = {}
    reference = {}
    actor_reports = {}
    for label, row in pairs.items():
        body = _load_body(row["body"])
        audio = _load_audio(row["audio"])
        bodies[label], audios[label] = body, audio
        per_song[label] = synchrony_report(body, audio, fps=fps, n_null=n_null, seed=seed)
        per_song[label]["raw_health"] = motion_metrics(body, fps=fps)
        profile = motion_profile(body, fps)
        per_song[label]["audio_predictive_gain"] = audio_predictive_gain(
            audio["time"], audio["onset"], profile["time"], profile["energy"],
            n_shuffle=int(config.get("predictive_n_shuffle", 20)), seed=seed)
        if row["latent"] is not None:
            per_song[label]["latent"] = latent_metrics(_load_latent(row["latent"]))
        ref_paths = list(row["reference_bodies"])
        if row["reference_body"] is not None and row["reference_body"] not in ref_paths:
            ref_paths.insert(0, row["reference_body"])
        if ref_paths:
            ref_bodies = [_load_body(path) for path in ref_paths]
            reference[label] = synchrony_report(ref_bodies[0], audio,
                                                fps=fps, n_null=n_null, seed=seed)
            actor_reports[label] = actor_calibration(body, ref_bodies, audio, fps=fps)
    cross = conditioning_matrix(bodies, audios, fps=fps, seed=seed)
    # The classifier uses a conservative aggregate: the weakest per-song
    # null-z and bar score must pass, and the cross-song diagonal must pass.
    weakest = min(per_song.values(), key=lambda row: (
        row["beat_alignment"].get("z") if row["beat_alignment"].get("z") is not None else -np.inf))
    cls = classify(weakest, cross, **config.get("thresholds", {}))
    report = {
        "schema": "lightsem.music_motion.v1",
        "protocol": config,
        "inputs": {label: {"body": str(row["body"].resolve()),
                            "audio": str(row["audio"].resolve()),
                            "reference_body": str(row["reference_body"].resolve())
                            if row["reference_body"] is not None else None,
                            "reference_bodies": [str(path.resolve()) for path in row["reference_bodies"]],
                            "latent": str(row["latent"].resolve()) if row["latent"] is not None else None}
                   for label, row in pairs.items()},
        "per_song": per_song,
        "conditioning": cross,
        "actor_reference": reference or None,
        "actor_calibration": actor_reports or None,
        "classification": cls,
        "interpretation": (
            "This is a raw-body musicality diagnostic. It does not qualify a model, "
            "prove choreography, or override the mandatory raw safety gate."
        ),
    }
    return _jsonable(report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    report = build_report(args.manifest.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out.resolve()),
                      "status": report["classification"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
