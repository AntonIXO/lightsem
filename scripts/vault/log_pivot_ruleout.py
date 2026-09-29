#!/usr/bin/env python3
"""Log the motion->sound->light pivot-chain rule-out to the LIGHTSEM vault."""
import json

import numpy as np

from scripts.vault.log_to_vault import PROBLEM, Client

p = json.load(open("data/cache/pivot_probe.json"))
ar = float(np.mean(p["persistence_only"]))
mot = p["motion_band_marginal"]
best = p["best_band_marginal"]
frac = p["slow_band_variance_fraction"]

c = Client()
try:
    c.initialize()
    r = c.call("rule_out", {
        "problem_id": PROBLEM,
        "claim": (
            "Using sound as a PIVOT modality rescues the mapping: learn motion -> sound "
            "on AIST (which has motion+audio) and sound -> light on RPMC-L2 (which has "
            "audio+light), then compose them to get motion -> light. This is proposed as "
            "a way around the absence of any dataset pairing motion with stage lighting."
        ),
        "method": (
            "Attack the chain at its SECOND link, which is shared with the already-refuted "
            "direct mapping, so link 1 never has to be built. By the data processing "
            "inequality, motion -> sound -> light makes motion conditionally independent of "
            "light given sound; the chain's ceiling is therefore what the "
            "MOTION-RECOVERABLE PART of sound explains about light. Prior AIST measurement "
            "(scripts/diagnostics/probe1_pose_vs_audio.py, 263 sequences) shows motion and audio share only a "
            "slow envelope: R2 = 0.0019 at 0.1 s but r = 0.65 at 8 s smoothing. So link 1 "
            "was given an ORACLE: assume motion recovers the audio envelope PERFECTLY at "
            "timescale w and nothing faster, then band-split audio (centred uniform filter) "
            "and measure each band's marginal R2_within over a persistence (AR) baseline on "
            "held-out songs. Deliberately generous: the smoothing is CENTRED so it uses the "
            "future, and the oracle is perfect within band; a real causal motion->sound "
            "model gets strictly less than this bound."
        ),
        "evidence": f"""The bound is at or below zero at EVERY timescale, so the chain cannot work.

Persistence baseline (lighting's own past 0.1-4.0 s, no audio): R2_within {ar:+.4f}

Marginal contribution of each audio band OVER persistence, held-out songs:
  full audio (no split)   {p['bands']['full audio (no split)']['marginal']:+.4f}
  slow band  w=1s         {p['bands']['slow band, w=1s']['marginal']:+.4f}
  slow band  w=2s         {p['bands']['slow band, w=2s']['marginal']:+.4f}
  slow band  w=4s         {p['bands']['slow band, w=4s']['marginal']:+.4f}
  slow band  w=8s         {mot:+.4f}   <- the motion-recoverable band, ORACLE link 1
  slow band  w=16s        {p['bands']['slow band, w=16s']['marginal']:+.4f}
  fast band  w=1s         {p['bands']['fast band, w=1s']['marginal']:+.4f}   <- best of ANY band
Best marginal over all timescales: {best:+.4f}. Every value is NEGATIVE.

Note the ordering: the slow bands are the WORST. The band motion can actually
reach ({frac['8.0']*100:.0f}% of audio variance at 8 s) hurts more than the full signal
({mot:+.4f} vs {p['bands']['full audio (no split)']['marginal']:+.4f}), and it degrades monotonically as the window widens
(1s {p['bands']['slow band, w=1s']['marginal']:+.4f} -> 16s {p['bands']['slow band, w=16s']['marginal']:+.4f}). The pivot bottleneck selects for
exactly the component of audio that is least useful about light.

This is an ORACLE bound on link 1, so it is not a statement about how good a
motion->sound model could be made. Even a perfect one inherits {mot:+.4f}.

Root cause is a composition-of-ceilings problem, not two independent bugs:
  link 1 (motion->sound) passes only a slow envelope  [AIST: 0.0019 @0.1s, r=0.65 @8s]
  link 2 (sound->light) has zero marginal signal at all timescales  [this probe]
Cross-modal transfer moves only REDUNDANT information (multimodal-ai lecture 08),
so a pivot chain can only pass information present in BOTH intersections. Here
that intersection is empty.""",
        # tags are a REGISTERED vocabulary; 'cross-modal' is the existing term
        "topic_tags": ["audio-to-light", "cross-modal"],
    })
    print("rule_out ->", json.dumps(r)[:300])
finally:
    c.close()
