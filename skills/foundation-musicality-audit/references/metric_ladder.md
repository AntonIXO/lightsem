# Metric ladder and decision rules

## Tier 0: raw safety and mechanism separation

Run before any musical score. Use the repository training-health audit on the
saved raw body and stream. Reject nonfinite values, raw motion explosions,
invalid provenance and unsafe ratios to the selected baseline. Keep motion,
RGB, brightness, event and decoded angular streams separate. Root-relative
motion is required for musicality; root drift alone is not articulation.

Distinguish:

* representation collapse: latent effective rank/cosine/sample diversity;
* output collapse: constant or duplicate head/stream;
* inactivity: no emitted values or no events;
* saturation: channel range clipping;
* input bypass: no response to declared audio/limb interventions.

## Tier 1: simple musical evidence

Use frozen audio arrays and raw body energy:

* audio onset/RMS/flux and beat phase;
* root-relative joint-speed energy and limb-specific speeds;
* BAS/BeatAlign, always next to the metronome upper reference and
  circular-shift null;
* circular phase concentration and signed phase offset;
* beat-bin rank correlation and null-z;
* within-bar accent/energy cosine and rank correlation;
* continuous onset/energy lag scan.

Simple metrics are useful because they are inspectable and cheap. They are not
enough for a verdict: a periodic metronome can score well on BAS and bar shape.

## Tier 2: foundation representation and feature families

For each saved latent tensor, report effective rank, constant-channel fraction,
temporal delta RMS and pairwise cosine. Do this on native latent and any
learned loss projector separately.

For each frozen audio family, use the same blocked ridge probe:

`future motion ~ motion history` versus
`future motion ~ motion history + audio family`.

`feature_family_predictive_gain` standardizes and fits PCA on each training
fold only. Run it for Base92, MuQ layer 9, each available MuQ hidden layer,
MuQ+Base92 and explicit ablations. Report circular feature-time shift nulls.
This is an incremental predictive proxy, not an unbiased conditional mutual
information estimator. It is secondary until it beats a real actor/calibration
ceiling and remains stable across songs.

## Tier 3: causal conditioning

For bodies/heads generated from songs `i` and audio `j`, compute all `M[i,j]`,
not only the diagonal. The evidence matrix combines BAS, beat-bin accent
correlation and within-bar cosine while retaining each raw component.

Require:

* two own-song recomputations with own body/audio provenance;
* same-BPM donor and beat-jitter/time-shift controls;
* fixed weights, seed/noise, metadata, mapper and clock;
* root-relative response to audio swap, not only RGB changes;
* positive own-song diagonal margin for each song.

Different hashes or nonzero RMS are response evidence only. They do not prove
that the response is musical.

## Tier 4: structure, style and retention

`phrase_structure_alignment` is the first advanced check. It compares
beat-synchronous bar recurrence and phrase novelty with a circular bar-shift
null. Add it only after Tier 1–3 are finite and stable; a one-bar loop must not
pass because BAS is high.

`actor_calibration` uses descriptor nearest-neighbour distances and a real
leave-one-out floor. Use actor bodies as a held-out distribution, never as a
framewise target. Compare generated event density, phase, bar profiles, holds,
limb-energy proportions, recurrence and raw dynamics against actor percentile
bands and shuffled/donor controls. Two actor references are the minimum; a
single actor body is unqualified.

For product delivery, repeat timing/structure descriptors on decoded moving-head
angles and emitted events. Report body→requested→decoded lag and retention;
mapper loss cannot be blamed on the foundation without this seam audit.

## Which ideas are worth keeping

The synthetic benchmark in the repository currently supports this order:

1. composite own-song diagonal evidence;
2. root-relative raw health and latent geometry;
3. beat-bin/bar metrics with explicit nulls;
4. feature-family probes for simple versus MuQ layers;
5. predictive gain as a secondary diagnostic;
6. phrase/style/actor calibration after enough independent songs.

BAS alone, absolute audio-to-motion correlation, output hashes, and a learned
critic without split-by-song calibration are not promotion metrics.
