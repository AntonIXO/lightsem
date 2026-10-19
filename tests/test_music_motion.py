"""Contract tests for the Music--Motion Causal Audit."""
from __future__ import annotations

import numpy as np

from lightsem.eval.music_motion import (
    audio_predictive_gain,
    actor_calibration,
    phrase_structure_alignment,
    bar_structure_alignment,
    classify,
    conditioning_matrix,
    motion_events,
    motion_profile,
    latent_metrics,
    feature_family_predictive_gain,
    synchrony_report,
)


def _pair(duration: float = 24.0, fps: float = 30.0):
    t = np.arange(int(duration * fps), dtype=float) / fps
    beats = np.arange(0.5, duration - 0.1, 0.5)
    at = np.arange(0.0, duration, 0.02)
    onset = np.zeros_like(at)
    for b in beats:
        onset += np.exp(-0.5 * ((at - b) / 0.035) ** 2)
    body = np.zeros((len(t), 5, 3), dtype=float)
    body[:, :, 1] = 1.0
    # Each beat contains a short kinematic accent.  The second joint is
    # root-relative so root drift cannot create the event.
    for b in beats:
        pulse = np.exp(-0.5 * ((t - b) / 0.055) ** 2)
        body[:, 1, 0] += 0.08 * pulse
        body[:, 2, 2] += 0.04 * pulse
    audio = {"time": at, "onset": onset, "beats": beats, "beats_per_bar": 4}
    return body, audio


def test_motion_profile_is_root_relative_and_frozen_body_is_inactive():
    body = np.zeros((240, 5, 3), dtype=float)
    body[:, :, 1] = 1.0
    body[:, :, :] += np.arange(len(body))[:, None, None] * np.array([1.0, 0.0, 0.0])
    profile = motion_profile(body)
    assert profile["finite"] is True
    # Pure root translation is removed by the root-relative representation.
    assert profile["speed_p95"] == 0.0
    frozen = np.repeat(body[:1], len(body), axis=0)
    assert motion_profile(frozen)["active_fraction"] == 0.0


def test_synchrony_reports_beat_null_and_bar_structure():
    body, audio = _pair()
    report = synchrony_report(body, audio, n_null=80, seed=7)
    assert report["beat_alignment"]["n_events"] > 0
    assert report["beat_alignment"]["z"] > 0.0
    assert report["beat_bins"]["count"] > 10
    assert report["bar_structure"]["bar_count"] > 3
    assert report["bar_structure"]["cosine_mean"] > 0.5


def test_cross_song_diagonal_detects_reused_body():
    body_a, audio_a = _pair()
    body_b = body_a.copy()
    # A different song has accents on the half-beat offset.  Reusing body_a
    # cannot earn a diagonal advantage over both audio streams.
    audio_b = dict(audio_a)
    audio_b["beats"] = audio_a["beats"] + 0.25
    audio_b["time"] = audio_a["time"]
    audio_b["onset"] = np.interp(audio_b["time"] - 0.25, audio_a["time"], audio_a["onset"], left=0.0, right=0.0)
    reused = conditioning_matrix({"a": body_a, "b": body_b}, {"a": audio_a, "b": audio_b}, n_null=0)
    assert reused["diagonal_margin"] < 0.03


def test_classification_does_not_call_beat_locking_musicality():
    body, audio = _pair()
    report = synchrony_report(body, audio, n_null=50)
    result = classify(report, None)
    assert result["status"] in {"unqualified_musicality", "beat_locked_only"}
    assert "conditioning_matrix_missing" in result["reasons"]


def test_latent_metrics_separates_constant_channels_from_finite_rank():
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(120, 3, 8))
    latent[:, :, 0] = 0.0
    report = latent_metrics(latent)
    assert report["finite"] is True
    assert report["effective_rank"] > 1.0
    assert report["constant_channel_fraction"] > 0.0


def test_audio_predictive_gain_beats_circular_audio_shift_on_synthetic_signal():
    duration = 36.0
    at = np.arange(0.0, duration, 0.02)
    mt = np.arange(0.0, duration, 0.1)
    audio = np.sin(2 * np.pi * 1.7 * at) + 0.2 * np.sin(2 * np.pi * 0.31 * at)
    motion = np.sin(2 * np.pi * 1.7 * (mt - 0.5))
    report = audio_predictive_gain(at, audio, mt, motion, n_shuffle=8, seed=2)
    assert report["status"] == "measured"
    assert report["incremental_gain"] > report["shuffle_gain_mean"]


def test_feature_family_probe_accepts_high_dimensional_layer_features():
    duration = 36.0
    at = np.arange(0.0, duration, 0.1)
    mt = at.copy()
    rng = np.random.default_rng(8)
    informative = np.sin(2 * np.pi * 0.7 * at)
    features = np.column_stack([informative, rng.normal(size=(len(at), 31))])
    motion = np.sin(2 * np.pi * 0.7 * (mt - 0.3))
    report = feature_family_predictive_gain(at, features, mt, motion,
                                            pca_components=8, n_shuffle=6, seed=3)
    assert report["status"] == "measured"
    assert report["feature_dim"] == 32
    assert report["pca_components"] == 8


def test_phrase_structure_and_actor_calibration_are_distributional():
    body, audio = _pair(duration=24.0)
    # Give every fourth bar a different accent pattern and mirror it in the
    # body so phrase novelty is a real signal rather than a constant loop.
    for i, beat in enumerate(audio["beats"]):
        amp = 1.0 + 0.6 * np.sin(i / 7.0)
        pulse = np.exp(-0.5 * ((np.arange(len(body)) / 30.0 - beat) / 0.055) ** 2)
        body[:, 1, 0] += 0.03 * amp * pulse
        audio["onset"] += amp * np.exp(-0.5 * ((audio["time"] - beat) / 0.035) ** 2)
    profile = motion_profile(body)
    phrase = phrase_structure_alignment(audio["time"], audio["onset"], profile["time"], profile["energy"], audio["beats"])
    assert phrase["status"] == "measured"
    assert phrase["bar_count"] >= 4
    references = [body, body + 0.002 * np.random.default_rng(1).normal(size=body.shape)]
    calibrated = actor_calibration(body, references, audio)
    assert calibrated["status"] == "measured"
    assert calibrated["reference_count"] == 2
