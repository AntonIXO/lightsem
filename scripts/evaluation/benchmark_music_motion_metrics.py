#!/usr/bin/env python3
"""Benchmark candidate music--motion metrics on known synthetic mechanisms.

This is a metric sanity check, not model evidence.  It creates paired songs
with known mechanisms (accent-conditioned body, BPM-only metronome, reused
body, root-only drift, and latent collapse) and records which diagnostics
separate them.  The script is intentionally CPU-only and writes a new JSON
receipt rather than changing any checkpoint or training artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lightsem.eval.music_motion import (
    actor_calibration,
    audio_predictive_gain,
    conditioning_matrix,
    feature_family_predictive_gain,
    latent_metrics,
    motion_profile,
    synchrony_report,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _audio(duration: float, *, pattern: tuple[float, ...], phase: float = 0.0):
    beats = np.arange(0.5, duration - 0.1, 0.5)
    at = np.arange(0.0, duration, 0.02)
    onset = np.zeros_like(at)
    for i, beat in enumerate(beats):
        phrase = 0.65 + 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * (i / 16.0) + phase))
        amp = pattern[i % len(pattern)] * phrase
        onset += amp * np.exp(-0.5 * ((at - beat) / 0.035) ** 2)
    return {"time": at, "onset": onset, "beats": beats, "beats_per_bar": 4}


def _body_from_audio(audio, *, fps: float = 30.0, scale: float = 0.08,
                     constant_amplitude: bool = False, root_only: bool = False):
    duration = float(audio["time"][-1])
    t = np.arange(int(duration * fps), dtype=float) / fps
    body = np.zeros((len(t), 22, 3), dtype=float)
    body[:, :, 1] = 1.0
    if root_only:
        drift = 0.3 * np.sin(2 * np.pi * 0.7 * t)
        body[:, :, 0] = drift[:, None]
        return body
    for i, beat in enumerate(audio["beats"]):
        pulse = np.exp(-0.5 * ((t - beat) / 0.050) ** 2)
        amp = 1.0 if constant_amplitude else float(np.interp(beat, audio["time"], audio["onset"]))
        # Integrating an event-shaped velocity gives a raw position trajectory
        # whose root-relative speed peaks at the declared musical event.
        body[:, 1, 0] += scale * amp * pulse
        body[:, 2, 2] += 0.5 * scale * amp * pulse
        body[:, 7, 0] += 0.7 * scale * amp * np.roll(pulse, int(round(0.08 * fps)))
    return body


def _latent(seed: int, *, collapsed: bool = False):
    rng = np.random.default_rng(seed)
    if collapsed:
        return np.ones((720, 8, 16), dtype=float)
    return rng.normal(size=(720, 8, 16))


def _summarize(pair, audio, fps=30.0):
    report = synchrony_report(pair, audio, fps=fps, n_null=120, seed=7)
    profile = motion_profile(pair, fps)
    report["predictive"] = audio_predictive_gain(
        audio["time"], audio["onset"], profile["time"], profile["energy"],
        n_shuffle=24, seed=7)
    return report


def build_report() -> dict:
    duration = 24.0
    audio_a = _audio(duration, pattern=(1.0, 0.15, 0.75, 0.25), phase=0.0)
    audio_b = _audio(duration, pattern=(0.25, 0.95, 0.20, 0.80), phase=1.7)
    true_a = _body_from_audio(audio_a)
    true_b = _body_from_audio(audio_b)
    mechanisms = {
        "accent_conditioned": {"bodies": {"a": true_a, "b": true_b}, "expected": "responsive"},
        "bpm_only_metronome": {
            "bodies": {"a": _body_from_audio(audio_a, constant_amplitude=True),
                       "b": _body_from_audio(audio_b, constant_amplitude=True)},
            "expected": "beat_only",
        },
        "reused_body": {"bodies": {"a": true_a, "b": true_a.copy()}, "expected": "shared_prior"},
        "root_only": {"bodies": {"a": _body_from_audio(audio_a, root_only=True),
                                  "b": _body_from_audio(audio_b, root_only=True)},
                       "expected": "root_failure"},
    }
    audios = {"a": audio_a, "b": audio_b}
    results = {}
    rng = np.random.default_rng(19)
    for name, item in mechanisms.items():
        per_song = {label: _summarize(body, audios[label])
                    for label, body in item["bodies"].items()}
        matrix = conditioning_matrix(item["bodies"], audios, fps=30.0, n_null=0, seed=7)
        results[name] = {
            "expected_mechanism": item["expected"],
            "conditioning": matrix,
            "per_song": {
                label: {
                    "bas": row["beat_alignment"].get("bas"),
                    "bas_null_z": row["beat_alignment"].get("z"),
                    "beat_bin_null_z": row["beat_bins"].get("null_z"),
                    "bar_cosine": row["bar_structure"].get("cosine_mean"),
                    "phrase_recurrence_null_z": row["phrase_structure"].get("recurrence_null_z"),
                    "phrase_novelty_null_z": row["phrase_structure"].get("novelty_null_z"),
                    "predictive_gain": row["predictive"].get("incremental_gain"),
                    "predictive_shuffle_z": row["predictive"].get("shuffle_gain_z"),
                    "active_fraction": row["motion"].get("active_fraction"),
                }
                for label, row in per_song.items()
            },
        }
    actor_refs = [true_a,
                  true_a + 0.002 * rng.normal(size=true_a.shape),
                  true_a + 0.003 * rng.normal(size=true_a.shape)]
    actor_outlier = true_a.copy()
    actor_outlier[:, 7, 0] += 0.05 * np.sin(np.arange(len(actor_outlier)) / 13.0)
    results["actor_calibration"] = {
        "actor_like": actor_calibration(true_a, actor_refs, audio_a),
        "actor_outlier": actor_calibration(actor_outlier, actor_refs, audio_a),
        "expected": "distributional_calibration_only",
    }
    results["latent_geometry"] = {
        "healthy": latent_metrics(_latent(1)),
        "collapsed": latent_metrics(_latent(1, collapsed=True)),
        "expected": "latent_rank_separates_geometry_only",
    }
    simple = np.column_stack([audio_a["onset"], np.sin(2 * np.pi * audio_a["time"] / 0.5),
                              np.cos(2 * np.pi * audio_a["time"] / 0.5)])
    advanced = np.column_stack([simple, rng.normal(size=(len(simple), 29))])
    profile = motion_profile(true_a, 30.0)
    results["feature_families"] = {
        "simple_base92_like": feature_family_predictive_gain(
            audio_a["time"], simple, profile["time"], profile["energy"],
            pca_components=3, n_shuffle=12, seed=7),
        "advanced_muq_layer_like": feature_family_predictive_gain(
            audio_a["time"], advanced, profile["time"], profile["energy"],
            pca_components=8, n_shuffle=12, seed=7),
        "interpretation": "synthetic layer-like comparison only; no claim about real MuQ layer ranking",
    }
    return {
        "schema": "lightsem.music_motion.metric_benchmark.v2",
        "protocol": {
            "duration_s": duration,
            "bpm": 120,
            "fps": 30,
            "null_draws": 120,
            "songs": ["a", "b"],
            "note": "synthetic metric sanity check; no model or product evidence",
            "benchmark_version": 2,
        },
        "results": results,
        "interpretation": [
            "BAS must be read with its metronome and circular-shift null.",
            "Beat-bin and within-bar metrics are required to reject a constant BPM metronome.",
            "The conditioning matrix is required to reject reused bodies/shared priors.",
            "Root-relative motion is required to reject root-only movement.",
            "Latent effective rank diagnoses geometry and cannot establish musical conditioning.",
            "Phrase recurrence/novelty is secondary evidence and needs nulls.",
            "Actor calibration is distributional and requires multiple references.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(build_report()), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out.resolve()), "status": "synthetic_metric_benchmark"}))


if __name__ == "__main__":
    main()
