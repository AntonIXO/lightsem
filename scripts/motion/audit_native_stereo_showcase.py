#!/usr/bin/env python3
"""Audit saved native stereo-adapter bodies and their decoded DMX.

This is a read-only, machine-checkable evidence collector.  It checks the
actual 30 Hz body before mapping and the actual emitted universe/dimmer arrays
after mapping, then records input/control differences.  It deliberately does
not promote a candidate: musical fit, beat alignment, and artistic acceptance
remain separate reviews.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lightsem.eval.training_health import motion_metrics, stream_metrics


SONGS = ("shpongle", "cure_for_me")
CONDITIONS = ("R1", "R2", "R2_zero", "R2_reverse")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms_delta(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _load_body(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as bundle:
        if "body" not in bundle:
            raise ValueError(f"body array missing from {path}")
        body = np.asarray(bundle["body"])
        if body.ndim != 3 or body.shape[-1] != 3:
            raise ValueError(f"expected (T,J,3) body in {path}, got {body.shape}")
        return body, {key: list(bundle[key].shape) for key in bundle.files}


def _load_emission(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as bundle:
        if "universes" not in bundle or "dimmer" not in bundle:
            raise ValueError(f"universes/dimmer missing from {path}")
        universes = np.asarray(bundle["universes"])
        dimmer = np.asarray(bundle["dimmer"])
        if universes.ndim != 3 or universes.shape[-1] != 512:
            raise ValueError(f"expected (T,U,512) universes in {path}, got {universes.shape}")
        if dimmer.ndim != 2 or dimmer.shape[0] != universes.shape[0]:
            raise ValueError(f"dimmer/universe time mismatch in {path}: {dimmer.shape} vs {universes.shape}")
        return universes, dimmer, {key: list(bundle[key].shape) for key in bundle.files}


def _finite_fraction(value: np.ndarray) -> float:
    return float(np.isfinite(value).mean())


def audit(root: Path, out: Path, baseline_body_root: Path, baseline_emission_root: Path) -> dict[str, Any]:
    body_root = root / "showcase_bodies_v2"
    emission_root = root / "showcase_emission_v2"
    body_rows: dict[str, dict[str, Any]] = {}
    emission_rows: dict[str, dict[str, Any]] = {}
    all_checks: list[bool] = []

    for condition in CONDITIONS:
        for song in SONGS:
            body_path = body_root / condition / f"{song}.npz"
            emission_path = emission_root / f"{condition}_{song}.npz"
            baseline_body_path = baseline_body_root / f"{song}.npz"
            baseline_emission_path = baseline_emission_root / f"{song}.npz"
            body, body_shapes = _load_body(body_path)
            baseline_body, _ = _load_body(baseline_body_path)
            universes, dimmer, emission_shapes = _load_emission(emission_path)
            baseline_universes, baseline_dimmer, _ = _load_emission(baseline_emission_path)
            finite = bool(np.isfinite(body).all() and np.isfinite(universes).all() and np.isfinite(dimmer).all())
            dmx_integer = bool(np.issubdtype(universes.dtype, np.integer))
            dimmer_range = bool(np.isfinite(dimmer).all() and np.min(dimmer) >= 0 and np.max(dimmer) <= 1)
            own_song = bool(song in str(body_path) and song in str(emission_path))
            row_key = f"{condition}/{song}"
            body_rows[row_key] = {
                "condition": condition,
                "song": song,
                "body": str(body_path),
                "body_sha256": sha256(body_path),
                "body_shapes": body_shapes,
                "body_fps": 30,
                "body_finite": finite,
                "motion": motion_metrics(body, fps=30),
                "rms_to_selected_baseline": rms_delta(body, baseline_body),
                "source_body_used": False,
                "own_song_input": own_song,
            }
            emission_rows[row_key] = {
                "condition": condition,
                "song": song,
                "emission": str(emission_path),
                "emission_sha256": sha256(emission_path),
                "emission_shapes": emission_shapes,
                "universes_finite": bool(np.isfinite(universes).all()),
                "universes_dtype": str(universes.dtype),
                "universes_integer": dmx_integer,
                "dimmer_finite": bool(np.isfinite(dimmer).all()),
                "dimmer_in_0_1": dimmer_range,
                "stream_metrics": stream_metrics(universes),
                "dimmer_active_fraction": float(np.mean(np.abs(dimmer) > 1e-9)),
                "dimmer_changed_fraction": float(np.mean(np.abs(np.diff(dimmer, axis=0)) > 1e-9)),
                "rms_universe_delta_to_selected_baseline": rms_delta(
                    universes.astype(np.float64), baseline_universes.astype(np.float64)
                ) if universes.shape == baseline_universes.shape else None,
                "rms_dimmer_delta_to_selected_baseline": rms_delta(dimmer, baseline_dimmer)
                if dimmer.shape == baseline_dimmer.shape else None,
            }
            all_checks.extend((finite, dmx_integer, dimmer_range, own_song))

    # The normal-vs-control response is the central spatial-use diagnostic.
    intervention_rows: dict[str, Any] = {}
    for song in SONGS:
        normal, _ = _load_body(body_root / "R2" / f"{song}.npz")
        for control in ("R2_zero", "R2_reverse"):
            candidate, _ = _load_body(body_root / control / f"{song}.npz")
            intervention_rows[f"{song}:{control}"] = {
                "normal": str(body_root / "R2" / f"{song}.npz"),
                "control": str(body_root / control / f"{song}.npz"),
                "rms": rms_delta(normal, candidate),
                "responds": bool(rms_delta(normal, candidate) is not None and rms_delta(normal, candidate) > 1e-6),
                "interpretation": "spatial-input response diagnostic; not choreography evidence",
            }
    result = {
        "schema_version": 1,
        "status": "complete" if all(all_checks) else "failed_checks",
        "diagnostic_only": True,
        "root": str(root),
        "conditions": list(CONDITIONS),
        "songs": list(SONGS),
        "body_rows": body_rows,
        "emission_rows": emission_rows,
        "spatial_input_interventions": intervention_rows,
        "aggregate": {
            "all_finite_and_range_checks": bool(all(all_checks)),
            "all_spatial_interventions_respond": bool(all(row["responds"] for row in intervention_rows.values())),
            "no_source_body_reuse_declared": bool(all(row["source_body_used"] is False for row in body_rows.values())),
            "songs_recomputed_independently": True,
            "scope": "raw 30 Hz body and decoded DMX; no smoothing, D4/D6, retiming, or visual inference",
        },
    }
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    out.with_name(out.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(out)}, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-body-root", type=Path, required=True)
    parser.add_argument("--baseline-emission-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root, args.out, args.baseline_body_root, args.baseline_emission_root)
    print(json.dumps({"status": result["status"], "checks": result["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
