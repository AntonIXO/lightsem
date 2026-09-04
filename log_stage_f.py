#!/usr/bin/env python3
"""Log the Stage F negative result to the LIGHTSEM vault.

Uses log_to_vault.Client, which pins VIBESCIENCE_VAULT=/root/lightsem-vault so
records cannot land in any other project's vault.
"""
import json

from log_to_vault import PROBLEM, Client

res = json.load(open("data/cache/macro_results.json"))
ceil = json.load(open("data/cache/ceiling_probe.json"))
base = json.load(open("data/cache/baselines.json"))

import numpy as np

test_now = float(np.mean(res["test_r2"][0]))
ar = float(np.mean(ceil["persistence only (AR)"]))
ar_audio = float(np.mean(ceil["AR + audio"]))
audio_only = float(np.mean(ceil["audio only"]))
delta = ar_audio - ar
curve = ceil["shift_curve"]

c = Client()
try:
    c.initialize()

    claim = (
        "A causal temporal encoder over openl3 audio embeddings can predict the "
        "RPMC-L2 stage-lighting scene vector (6-dim HSV summary), i.e. a trained "
        "macro layer can replace the hand-written rule table by regressing "
        "audio -> absolute lighting state."
    )
    evidence = f"""Refuted by a marginal-contribution test, not by a weak model.

SETUP: 140 songs, 126,332 frames @10 Hz, split BY SONG (99/23/18). Target is
the per-song-CENTRED scene vector (35-65% of raw variance is between-song
venue palette, which is given, not predicted). Metric R2_within on held-out songs.

TRAINED MODEL (2,359,769 params, causal, 40 epochs, RTX 5070):
  test R2_within (now)   = {test_now:+.4f}
  best epoch             = 0 (before learning anything)
  train loss 0.1554 -> 0.0418 while val R2 fell -0.004 -> -0.445 (memorisation)

BASELINES (all below the zero point):
  per-song mean oracle   = {float(np.mean(base['per_song_mean_oracle'])):+.4f}  <- zero point
  best fair linear       = {float(np.mean(base['ridge_centred_1frame'])):+.4f}

CEILING PROBE (the decisive measurement):
  audio only             = {audio_only:+.4f}
  persistence only (AR)  = {ar:+.4f}
  AR + audio             = {ar_audio:+.4f}
  audio's marginal gain  = {delta:+.4f}   <- NEGATIVE

Lighting is ~{ar*100:.0f}% predictable from its own past 0.1-4.0 s, with NO audio.
Adding 512 audio dims on top of that does not help.

TIME-SHIFT CONTROL: correlation of audio vs light, y shifted within-song:
  lag -30s {curve['mean_abs_r'][0]:.5f} | lag 0s {curve['mean_abs_r'][5]:.5f} | lag +30s {curve['mean_abs_r'][10]:.5f}
Peak at lag {curve['peak_lag_s']:+d}s, MINIMUM near zero. Real frame-level coupling
must peak at lag 0. This is slow co-drift over minutes, not coupling.

RETRACTION: the earlier data-audit finding of "20-36x audio-light correlation
vs shuffled control" is withdrawn as evidence. The shuffle destroyed
within-song structure, so it tested the wrong null. The lag-shift control is
the correct one and it fails."""

    r = c.call("rule_out", {
        "problem_id": PROBLEM,
        "claim": claim,
        "method": (
            "Marginal-contribution test on per-song-centred targets, held-out songs. "
            "(1) Ridge fits for audio-only, persistence-only (y's own lags 0.1-4.0 s), "
            "and AR+audio; the audio claim stands or falls on AR+audio minus AR, not "
            "on audio's absolute score. (2) A 2.36M-param causal transformer trained "
            "40 epochs as the strong-model control. (3) A lag-shift control: audio-light "
            "correlation recomputed with y shifted +-30 s WITHIN each song, since real "
            "frame-level coupling must peak at lag 0 while slow co-drift does not. "
            "Split is by song (hash), so no venue appears in both train and test."
        ),
        "evidence": evidence,
        # rule_out() exposes ONLY topic_tags. 'matched-baseline' is registered
        # on the 'problem' axis and there is no problem_tags parameter here, so
        # it cannot be attached to a rule-out at all. Worth proposing upstream
        # (/root/vibescience-mcp): rule_out should accept problem_tags like the
        # hypothesis/experiment paths do, since a rule-out IS a failure mode.
        "topic_tags": ["audio-to-light"],
    })
    print("rule_out ->", json.dumps(r)[:400])

    r2 = c.call("register_diagnostic", {
        "name": "r2_within_song",
        "description": (
            "Variance of the lighting scene vector explained AFTER removing each "
            "song's own mean, on held-out songs. Global R2 is inflated because "
            "35-65% of target variance is between-song venue palette identity, "
            "which a lighting model is given rather than predicts. Zero point is "
            "the per-song-mean oracle."
        ),
        "unit": "R2",
    })
    print("diagnostic ->", json.dumps(r2)[:300])
finally:
    c.close()
