#!/usr/bin/env python3
"""Summarize per-channel cached Base92 zero-mask native-body audits.

This is a zero-mask sensitivity screen: it ranks channels by root-relative
native-body RMS change, checks the measured cached noise floor, and refuses to
call a channel qualified if any arm fails raw safety, finite, repeat, or rank
gates. It does not claim unique semantic attribution or full-waveform phrase
conditioning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NOISE_SIGMA_M = 0.003514393010614617
NOISE_SIGMA_STD = 0.01157
BASE92_NAMES = (
    *(f"mel_{i}" for i in range(64)),
    "rms", "rms_db", "flux", "flux_lf", "flux_lmid", "flux_mid", "flux_hmid", "flux_hf",
    "energy_lf", "energy_lmid", "energy_mid", "energy_hmid", "energy_hf",
    "onset", "onset_ema", "flux_ema", "flux_lf_ema", "flux_hf_ema",
    "centroid", "flatness", "rolloff", "percussive_ratio", "pulse_clarity",
    "beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase", "beat_period_s",
)


def load_song(root: Path, song: str) -> dict:
    matrix_name = "shpongle_matrix60" if song == "shpongle" else "cure_matrix60"
    receipt = json.loads((root / matrix_name / "CAUSAL_RECEIPT.json").read_text())
    arms = {row["arm"]: row for row in receipt["arms"]}
    full = json.loads((root / matrix_name / "full" / "RECEIPT.json").read_text())
    out = {"song": song, "full": full, "channels": {}}
    for index, name in enumerate(BASE92_NAMES):
        arm_name = f"zero_base92_{name}"
        if arm_name not in arms:
            raise ValueError(f"Missing {song}/{arm_name}")
        arm = json.loads((root / matrix_name / arm_name / "RECEIPT.json").read_text())
        motion = arm["metrics"]["motion"]
        safety = arm["raw_safety"]
        repeats = arm["noise_floor"]
        effect = float(arm["metrics"]["root_relative_articulation_rms_vs_full"])
        out["channels"][name] = {
            "index": index,
            "effect_m": effect,
            "standardized_effect": effect / NOISE_SIGMA_M,
            "above_2sigma": effect > 2.0 * NOISE_SIGMA_M,
            "above_005_standardized": effect / NOISE_SIGMA_M >= 0.05,
            "raw_safety_status": safety["status"],
            "safety_ratios": safety["ratios_vs_full"],
            "finite": bool(motion["all_finite"]),
            "rank": int(arm["metrics"]["native_effective_rank"]),
            "exact_repeat": bool(repeats["pass"]),
            "inactive_fraction": float(motion["inactive_frame_fraction"]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    songs = {song: load_song(root, song) for song in ("shpongle", "cure_for_me")}
    rows = []
    for index, name in enumerate(BASE92_NAMES):
        per_song = {song: data["channels"][name] for song, data in songs.items()}
        effects = [v["effect_m"] for v in per_song.values()]
        stds = [v["standardized_effect"] for v in per_song.values()]
        safe = all(v["raw_safety_status"] == "raw_body_ready" and v["finite"] and
                   v["exact_repeat"] and v["rank"] >= 20 for v in per_song.values())
        rows.append({
            "index": index,
            "name": name,
            "shpongle_effect_m": per_song["shpongle"]["effect_m"],
            "cure_for_me_effect_m": per_song["cure_for_me"]["effect_m"],
            "mean_effect_m": float(np.mean(effects)),
            "max_effect_m": float(np.max(effects)),
            "min_standardized_effect": float(np.min(stds)),
            "max_standardized_effect": float(np.max(stds)),
            "above_2sigma_both_songs": all(v["above_2sigma"] for v in per_song.values()),
            "above_005_standardized_both_songs": all(v["above_005_standardized"] for v in per_song.values()),
            "all_raw_gates_pass": safe,
            "per_song": per_song,
        })
    rows.sort(key=lambda r: r["mean_effect_m"], reverse=True)
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "qualification_status": "unqualified_cached_zero_mask_per_feature",
        "scope": "fixed-C cached-only zero one Base92 channel at a time; raw native 30 Hz body only",
        "feature_count": len(BASE92_NAMES),
        "songs": ["shpongle", "cure_for_me"],
        "noise_floor": {"root_relative_sigma_m": NOISE_SIGMA_M, "standardized_sigma": NOISE_SIGMA_STD,
                        "two_sigma_m": 2 * NOISE_SIGMA_M},
        "gates": {"finite": "all_finite", "rank_min": 20, "exact_repeat": True,
                  "raw_safety_status": "raw_body_ready", "two_song": True},
        "features_ranked_by_mean_effect": rows,
        "headline": {
            "channels_above_2sigma_both_songs": [r["name"] for r in rows if r["above_2sigma_both_songs"] and r["all_raw_gates_pass"]],
            "channels_above_005_standardized_both_songs": [r["name"] for r in rows if r["above_005_standardized_both_songs"] and r["all_raw_gates_pass"]],
            "all_raw_gates_pass_count": sum(r["all_raw_gates_pass"] for r in rows),
        },
        "provenance": {
            "root": str(root),
            "shpongle_receipt": str(root / "shpongle_matrix60" / "CAUSAL_RECEIPT.json"),
            "cure_for_me_receipt": str(root / "cure_matrix60" / "CAUSAL_RECEIPT.json"),
            "cached_only": True,
            "raw_waveform_qualification": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
