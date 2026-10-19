---
name: foundation-musicality-audit
description: Audit a new audio foundation model for raw safety, collapse, real music conditioning, and musical structure across simple features and frozen encoder layers. Use before comparing checkpoints or delivering a foundation-model demo.
---

# Foundation Musicality Audit

Use this skill for an audio-conditioned foundation model whose saved outputs
include latent representations, native body/aim heads, or learned light heads.
It is primarily a pre-promotion audit for the foundation model. Use the old
native body-to-DMX path only as a downstream retention check.

The audit must preserve three separate artifacts per own song:

1. foundation latent before output heads;
2. raw native body/aim before mapper, interpolation, smoothing, D4/D6 or DMX;
3. decoded/emitted streams after the mapper.

Never turn a failed raw body into a demo with an oscillator, reused body, new
soundtrack, post-hoc smoothing, retiming, fixture caps or arbitrary gain.
Run the existing raw health audit first:

```bash
bash run.sh scripts/evaluate_training.py \
  --manifest <training-health-manifest.json> \
  --output <new-health-report.json>
```

Then run the music–motion audit with a manifest containing at least two
independently recomputed own-song bodies and matching audio arrays (`time`,
`onset`, `beats`):

```bash
bash run.sh scripts/evaluation/audit_music_motion.py \
  --manifest <music-motion-manifest.json> \
  --out <new-mmca-report.json>
```

The canonical implementation is `lightsem.eval.music_motion`. Its synthetic
metric benchmark is `scripts/evaluation/benchmark_music_motion_metrics.py`;
run it after changing metrics, not after observing a candidate, and preserve
the resulting receipt under a new protocol path.
The repository idea-by-idea implementation table is
`docs/research/2026-10-19-music-motion-framework/IDEA_STATUS.md`.

## Metric ladder

Read [references/metric_ladder.md](references/metric_ladder.md) when choosing
checks. In brief:

* **Simple, mandatory:** finite/raw speed–acceleration–jerk, root-relative
  activity, limb/fixture attribution, onset/beat phase, BAS beside a
  metronome and circular-shift null, beat-bin accent coupling, within-bar
  shape, exact duplicates and inactivity.
* **Foundation representation:** latent spectral effective rank and constant
  channel fraction; compare native latent and any learned projector; use
  `feature_family_predictive_gain` with train-only PCA for Base92, each frozen
  MuQ layer, concatenated MuQ+Base92, and ablated feature groups.
* **Conditioning:** score every body against every song audio. Require an
  own-song diagonal advantage over wrong-song and same-BPM donor audio while
  holding seed/noise, weights, non-audio inputs and clock fixed. Add
  motion-history-only and circular audio-shift controls.
* **Advanced, only after the simple layer is stable:** beat-synchronous phrase
  recurrence/novelty alignment, actor split-half percentile calibration and
  body→decoded-DMX retention. MDSC-like style calibration remains optional
  until a frozen encoder and split-by-song calibration set exist.

Do not use a single weighted musicality score. The machine verdict is the
weakest blocking component. A high BAS with no diagonal margin is
`unconditioned_or_shared_prior`; a high BAS with no beat-bin/bar/phrase
evidence is `beat_locked_only`. A healthy latent rank never overrides a
collapsed output head, and an active output never overrides raw safety.

## Simple versus advanced feature comparisons

Compare feature families on the same song split and probe:

* motion-history only;
* simple Base92 onset/RMS/flux/beat phase;
* MuQ layer 9 (the pinned project layer);
* other saved MuQ hidden layers, one at a time;
* MuQ+Base92;
* shuffled-time and same-BPM donor controls.

For each family record feature hash, timestamp contract, train-only PCA hash,
blocked folds, history baseline MSE, augmented MSE, gain, circular-shift null,
and whether the native latent/body/output response changes. A layer that wins a
probe but fails audio swap, own-song diagonal or raw output safety is not useful
evidence. A layer result is unqualified if its extraction protocol, split or
checkpoint is not frozen.

## Required provenance and verdicts

Keep checkpoint, code/protocol, MuQ revision/layer, Base92/WAV, normalizer,
split/group, seed/noise, feature-family, body, mapper, renderer and song hashes.
Every control must state which input changed. Missing evidence is
`unqualified`, never zero.

Use the existing statuses:

* `failed_safety`;
* `collapsed_motion` or `output_collapsed`;
* `unconditioned_or_shared_prior`;
* `beat_locked_only`;
* `unqualified_musicality`;
* `diagnostic_music_aligned`.

`diagnostic_music_aligned` is an engineering result. It is not artistic
acceptance until actor calibration, phrase evidence and blind human review are
available. Read [references/manifest_and_receipt.md](references/manifest_and_receipt.md)
for the manifest fields and receipt rules.
