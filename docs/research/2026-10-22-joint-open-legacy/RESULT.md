# Joint open-source training of the legacy pipeline — 2026-10-22

## Protocol

Warm-start the frozen legacy `foundation_v3_jerk_safe` checkpoint and train the old downstream decoder jointly with:

- canonical BeatSaver MuQ+Base92 appearance targets;
- RPMC-L2 HSV/value histogram appearance proxies;
- Roskilde released `[T,60]` abstract lighting tensors through a shared-latent auxiliary head.

RPMC/Roskilde use a source-specific 36-D MIR encoder. The canonical MuQ+Base92 path, legacy context, fixture decoder, RGB/brightness/event heads and bounded motion projection remain in the pipeline. No pan/tilt/DMX target is constructed.

Seed `0`, 900 steps, same canonical 80/20 song-group split.

Vibescience experiment: `exp-joint-open-source-legacy-training-20261022`

## Result

Verdict: **directional_only**.

The legacy pipeline's canonical appearance improved:

- baseline composite error: `0.4529533607`
- joint treatment: `0.4446344103`
- gain: `0.0083189504`
- preregistered MDE: `0.005`

Components:

- brightness L1: `0.4049016132 -> 0.3851338284`
- RGB L1: `0.3351288613 -> 0.3334364048`
- event BCE: `0.6188296076 -> 0.6153330`

Own-song RGB swap also increased:

- `0.0375576653 -> 0.0439277068`

Fresh raw treatment audit passed:

- Shpongle: speed/acceleration/jerk p99 `0.0105324 / 0.0133682 / 0.1163336`
- Cure For Me: `0.0104623 / 0.0154630 / 0.1402740`
- limits: `0.18 / 0.50 / 1.50`

## What failed

The open-source objectives themselves did not transfer under this mixed recipe:

- RPMC appearance proxy:
  - constant baseline: `0.2682523`
  - treatment: `0.3291277`
- Roskilde abstract-light MSE:
  - corrected same-normalization mean baseline: `0.2477778`
  - treatment: `1.2639479`

Therefore this is not evidence that the old pipeline learned RPMC/Roskilde lighting semantics. The canonical improvement is a real partial result, but the open-source branches need source isolation or redesigned target heads before another joint arm.

No renderer call was made because the blocking open-source skill gate failed. No authored physical-light or choreography claim follows.

## Artifacts

- Corrected receipt: `/mnt/lightsem-data/experiments/lightsem/foundation_joint_open_legacy_20261022_v1_corrected/TRAINING.json`
- Checkpoint: `/mnt/lightsem-data/experiments/lightsem/foundation_joint_open_legacy_20261022_v1_corrected/model.pt`
- Fresh raw audit: `/mnt/lightsem-data/experiments/lightsem/foundation_joint_open_legacy_20261022_v1_corrected/raw_audit/RAW_AUDIT.json`
- Training runner: `/root/lightsem/scripts/experiments/run_joint_open_legacy.py`
- W&B analysis: https://wandb.ai/antonix-itmo/lightsem/runs/7p5y7ggk
