# Audit protocol

The audit is a post-training measurement protocol. It never edits the run it
reads. Create a new report path for every code, cache, checkpoint, seed, source,
or threshold change; never overwrite a sealed receipt.

## Inputs

The canonical command is:

```bash
bash run.sh scripts/evaluate_training.py \
  --manifest audit_manifest.json \
  --output reports/training_health.json
```

The manifest uses schema `lightsem.training_health.manifest.v1` (see
`docs/training_health_manifest.schema.json`). `body` is a raw native body NPZ;
`stream` is the raw emitted stream/DMX NPZ. `samples` names at least two
song-specific bodies when diversity or cross-song checks are being judged;
`interventions` names counterfactual outputs produced with one declared input
changed. `provenance` points to a JSON receipt. Relative paths resolve beside
the manifest. Optional `limits` make raw speed/acceleration/jerk and activity
thresholds explicit; do not invent them after seeing a candidate. A plot is
optional and is written only to a new path. With `--compare-baseline`,
`--comparison-plot` produces a shared raw overlay and signed difference plot.

Generate an independent baseline report with the same settings and compare the
candidate with `--compare-baseline baseline_report.json`. The comparison is
paired by gate, not by a larger hash or a single attractive metric. The two
reports must state the same sampling rate, output layout, mapping, renderer,
noise, non-audio inputs, and threshold policy. The only intended difference is
the checkpoint or explicitly registered training recipe.

## Required evidence

For a qualification claim, require checkpoint, code/protocol, feature/audio,
normalizer, split/group, seed/noise, mapper, and rendering hashes. Keep
Shpongle and AURORA — Cure For Me as separate inputs with separate saved body,
DMX, and soundtrack provenance. A reused body with a new soundtrack is a
companion-controller test, not a new song-conditioned generation.

Evaluate raw native outputs before interpolation, fixture caps, DMX encoding,
post-hoc smoothing, retiming, D4/D6 pacing, or video rendering. Keep motion,
color, brightness, and emitted streams as separate components. A downstream
visual pass cannot repair a failed generator.

## Controls and decision order

1. Reject nonfinite values, invalid shapes, missing files, hash mismatches, and
   incomplete provenance. Missing coverage is `unqualified`, never zero.
2. Measure raw position range, speed, acceleration, jerk, inactive frames,
   root/support proxies, and high-frequency residuals. A stationary or
   uniformly slowed candidate is not a jitter repair.
3. Measure representation/sample diversity, exact duplicate streams, output
   inactivity, and saturation separately. A zero event count alone is not
   representation collapse.
4. Recompute the audio-dependent path for both songs while holding weights,
   random noise, non-audio inputs, mapping, and rendering fixed. Use frozen-body,
   limb perturbation, shifted/shuffled/donor, loudness, and same-input controls
   where the run claims conditioning. Require assigned outputs to respond to an
   assigned limb and extra outputs to preserve source gesture timing.
5. Compare candidate and baseline gates. A candidate that loses a gate the
   baseline passed is a regression even if another metric improves. A passing
   health report is still diagnostic until the declared musical/quality gates
   have evidence.

The machine status vocabulary is `collapsed`, `failed_safety`, `unqualified`,
and `diagnostic_only` for a single report. A comparison adds
`qualified_regression`, `qualified_improvement`, or
`unchanged_or_unmeasurable` only when the paired gate evidence supports it.
