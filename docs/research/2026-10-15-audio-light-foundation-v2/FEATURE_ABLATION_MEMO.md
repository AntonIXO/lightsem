# Feature utility and bounded-context audit — 2026-10-15

## Scope
Diagnostic only. Fixed eight-row corpus, six train/two held-out songs, seed 0, 30 Hz, 384-frame windows, 400 updates, width 96 and AdamW budget held fixed. No render, native body, DMX, post-hoc smoothing, or promotion.

## Inputs and target limits
`data.py` concatenates authenticated 1024-D MuQ layer 9 and 92-D Base92. Base92 timing occupies columns 87–91 of its group. Body-derived aim/speed/acceleration are auxiliary proxies, not authored pan/tilt labels. Eight fixture groups repeat cache groups when fewer than eight exist.

## Pre-run registration
Vibescience problem: `lightsem-audio-to-light-feature-utility-and-context-contract`. Hypothesis: `on-the-fixed-eight-song-split-and-one-seed-edec41`. Experiment: `light-feature-utility-context-20261015-v1`. The full feature intervention uses equal-width zeroed controls: full, MuQ-only, Base92-only, and no-clock. Zeroing is an intervention, not a natural missingness claim.

## Redundancy measurement
`/mnt/lightsem-data/experiments/lightsem/audio_light_feature_redundancy_20261015.json` and `_CKA.json`. Over 40,351 aligned frames: MuQ–Base92 mean absolute channel correlation 0.074 (p95 0.199), so the groups are not redundant at raw channel level; Base92 internal off-diagonal mean absolute correlation 0.552 (p95 0.924), much higher than MuQ 0.048 (p95 0.121), indicating substantial within-Base92 redundancy. Linear CKA was 0.107 for full MuQ vs Base92 and 0.028 for MuQ vs Base92 timing subset; these are representation-similarity diagnostics, not predictive utility.

## Literature boundary
Skip-BART (Zhao et al., arXiv:2506.01482v2; official repo `RS2002/Skip-BART`) is verified as offline single-light HSV hue/value generation from OpenL3 at 10 Hz. Its aligned skip path, masked continuous-audio reconstruction and circular hue/slew-aware decoding are transferable hypotheses only. They do not establish moving-head aim, arbitrary fixtures, physical DMX, or raw safety. SeqLight (arXiv:2605.03660v1; `RS2002/SeqLight`) uses a synthetic color-mixing environment and venue-specific decomposition; it does not establish LightSem fixture dynamics.

## Initial measured feature results
The completed 400-update ablation is stored under `/mnt/lightsem-data/experiments/lightsem/audio_light_feature_ablation_20261015_v1/SUMMARY.json`. Values must be read from the receipt; headline interpretation is deferred until all controls and the bounded-context arm finish.


## Completed 400-update controls

| arm | aim MAE | RGB MAE | brightness MAE | event BCE | cross-song aim corr | raw jerk p99 |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.3417 | 0.3940 | 0.4213 | 0.2127 | 0.9970 | 2592 |
| MuQ-only | 0.3131 | 0.3912 | 0.3826 | 1.0633 | 0.9491 | 15017 |
| Base92-only | 0.3406 | 0.3943 | 0.3946 | 0.2182 | 0.9962 | 2552 |
| no-clock | 0.3434 | 0.3968 | 0.4091 | 0.2265 | 0.9972 | 1711 |

The full arm beats Base92-only only trivially on RGB/event and loses on brightness, aim, song differentiation, and jerk. MuQ-only has lower aim MAE but substantially worse raw jerk and event fit; it is not a safe movement solution. Removing explicit clock features does not remove the shared movement trajectory.

The bounded bidirectional halo-16 arm completed at `/mnt/lightsem-data/experiments/lightsem/audio_light_context_bidir_halo16_20261015_v1/RESULTS.json`: aim MAE 0.3431, RGB 0.3972, brightness 0.4905, event BCE 0.2128, cross-song correlation 0.9979, jerk p99 2071. It reduced raw jerk versus full by ~20%, but worsened aim/RGB/brightness and increased cross-song correlation. It is diagnostic-only, not a repair or promotion candidate.

Vibescience closeout: experiment `light-feature-utility-context-20261015-v1` was closed `supports` only because the preregistered multi-diagnostic direction matched overall; its interpretation explicitly records the recipe-conditional negative movement findings. This must not be summarized as “full features improved the model.”


## Official Skip-BART repository demo

A faithful external-repository demo was generated from `/root/Skip-BART` commit `88e56d1c07460276273fbbed777d248239262ced` using the official trained checkpoint and OpenL3 512-D embeddings from `/root/shpongle.flac`. Receipt: `/mnt/lightsem-data/experiments/lightsem/skipbart_repo_demo_20261015/DEMO_RECEIPT.json`; video: `skipbart_shpongle.mp4`.

The output is the repository’s single-light HSV sequence at 10 Hz, 400x400, 1024 frames / 102.4 s. Generated controls contain 82 unique hue/value pairs; hue range 115–122 and value range 41–64. This is an upstream offline HSV baseline, not LightSem output. It uses no MuQ, Base92, fixture metadata, body, native DMX, or LightSem renderer. Compatibility deviations and the upstream `*_gen.npy` bug are recorded in the receipt.
