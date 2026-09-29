# Native music-generator audit

2026-09-10. Read-only production audit. No GPU training, TEST access, vault writes, or production edits. All new results below are **unpreregistered diagnostics**, not a new scientific experiment or promotion gate.

## Bottom line

There is no observed disconnected-audio, reversed-flow-time, wrong-flow-sign, or per-chunk-noise bug in the four inspected modules. The concrete defect is **objective/physical-output mismatch**: coordinate and coordinate-derivative MSE do not enforce a coherent planted foot or bound integrated root drift. CPU counterexamples prove this mismatch.

The trained model does use audio. A broader CPU probe finds a severe native-target generalization gap, especially for regression. Calling this complete “conditioning collapse” would misidentify the failure: changing audio strongly changes predictions, but the change is usually not useful on held-out songs. This is consistent with overfitting and target ambiguity; it does not isolate their causes or establish that stronger audio fusion will fix them.

The frozen study remains failed: flow ES .8796 versus regression 1.1934; flow correct-audio gains versus donor 2.34% and shift .50%, both below 5% with intervals spanning zero. These distributional numbers do not repair the physical failure (flow excess sliding .882 m/s).

## Sources and integrity

Read `CLAUDE.md`, motion `PLAN.md`, `RESULTS.md`, `NEXT_REVISION.md`, and `REVISION_NATIVE_REFERENCE.md`. Inspected `music_model.py`, `music_training.py`, `music_data.py`, `music_evaluation.py`, plus audio preparation, geometry, native codec, sampler and shared runtime.

All four requested modules are byte-identical to the frozen study snapshot at:
`/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/source`.
The probes load the existing final checkpoints on CPU through `bash run_torch.sh`; they do not call training/evaluation entry points or alter frozen artifacts.

## 1. Concrete loss failures, tested

`music_model.py:99–105` minimizes mean normalized native-coordinate squared error plus .05 mean TRAIN-scaled coordinate-derivative squared error. There is no FK geometry, support consistency, trajectory displacement, or contact/feet coupling in this objective. Rotations get 132/139 coordinate slots; root velocity has two and contact labels four. Per-channel standardization does not create joint-geometry supervision.

**Root-bias counterexample:** add constant +.1 m/s horizontal velocity to the first 32 seconds of TRAIN recording001, leaving rotations, height and contacts unchanged.

- Native objective increases only .00108144; derivative contribution is effectively zero.
- Terminal root displacement changes by 3.19667 m.
- Foot sliding increases .03208 → .10883 m/s; contact coverage remains .65599.
- Root travel increases 6.0960 → 7.6605 m.

The derivative term cannot see a constant velocity bias. Small local coordinate error accumulates into large world-space displacement. The existing full/chunk test cannot detect this because both paths integrate the same erroneous velocity.

**Rotation-gauge counterexample:** multiply all rotation6D values by two in that same target.

- FK joint coordinates are exactly unchanged (maximum absolute difference 0 m).
- The native objective is 63.1582.

Gram–Schmidt decoding removes positive row scale, but the loss pays heavily to reproduce it. This does not mean canonical rotation targets are wrong. It proves that native-coordinate objective values cannot be interpreted as physical-motion error. Conversely, it leaves potentially serious geometric errors cheap. The two counterexamples identify an actual training/evaluation metric mismatch, not a proven single cause of weak music conditioning.

## 2. Audio is present; useful generalization is weak

CPU checkpoint diagnostic: first960 frames of every eligible recording, 125 TRAIN recordings/107 songs and33 VAL recordings/28 songs. Equal recording weights within each canonical song, then equal song weights. Each donor is the first canonical-distinct eligible recording in sorted order. This donor recipe is diagnostic, not the frozen evaluation donor selection. Same noise and state for all audio interventions. No ODE generation or motion-summary energy score is used here.

Entries are native training-objective losses, not ES. “Mean predictor” is the zero normalized output, i.e. the fixed TRAIN native-coordinate mean. “Zero audio” supplies the TRAIN-mean audio feature vector to the trained network; it is NOT an unconditional model trained without audio.

| Model / split / flow time | Correct | Donor | Zero audio | Mean predictor / zero field |
|---|---:|---:|---:|---:|
| Regression TRAIN | .32687 | 1.06661 | .84836 | .76949 |
| Regression VAL | 1.14289 | 1.19647 | .88865 | .81529 |
| Flow TRAIN t=0 | .72781 | .84006 | .83455 | 2.05670 |
| Flow VAL t=0 | .86767 | .88859 | .88450 | 2.10759 |
| Flow TRAIN t=.5 | .59919 | .62464 | .61157 | 2.05670 |
| Flow VAL t=.5 | .68441 | .69509 | .68145 | 2.10759 |

For flow the last column is a zero vector field, not a mean-motion generator. At t=.5 the state contains true target information, so these are teacher-forced vector-field diagnostics, not conditional generation tests.

Regression improves objective over the mean predictor by57.5% on TRAIN. On VAL it is worse by40.2%, and worse in all28 song groups. Paired difference +.32760 has a diagnostic10,000-resample song-bootstrap interval [.27623,.38045]. This interval covers only the fixed first-window probe, not all original validation windows.

Changing correct to donor audio changes the regressor output by1.34× its correct-output RMS on TRAIN and1.41× on VAL. For the flow at t=0, the corresponding field changes are .282× and .275×. These are mean per-record relative RMS values, song-weighted. **The audio path is not numerically dead.** Flow native-loss improvement from correct versus donor audio shrinks from13.36% TRAIN to2.35% VAL at t=0. That exposes poor useful conditional generalization, not merely insufficient signal magnitude.

A separate four-record time sweep included t=.1,.9,1. The donor benefit generally diminishes when the teacher-forced state already contains most of the target. This is expected for flow matching and is not a time-direction bug. Do not use it to justify changing the time distribution without a new matched experiment.

## 3. Other implementation checks

- Audio and motion normalization are applied in the same order at training and evaluation. Both use the same aligned cached Base92 arrays. Equal array length is checked. This audit does not independently prove waveform/motion synchronization.
- Flow interpolation `(1-t)*noise+t*target`, desired field `target-noise`, and positive Euler updates from noise at0 to data at1 are consistent.
- Stationary smoothed noise has unit pointwise variance and one global sequence state. Euler updates occur globally at each step, including chunked execution. No noise restart, trajectory blending or repeated root integration occurs in `generate`.
- Evaluation reuses the same draws across correct/donor/shift conditions. Regression intentionally gets one deterministic draw. The energy-score distinct-pair denominator is correct.
- The short-recording exclusion is explicit in `MusicResident`: n_frames>=960. It loses three TRAIN and three VAL recordings. Its scientific limitation is already documented; it is not a hidden sampler indexing error.
- The nominal receptive field is957 frames (31.9s), while training crops have960. Only four crop positions have the complete nominal receptive field without padded boundaries. Full-track inference has many more fully supported positions. This is a real train/full-track context mismatch, but it cannot by itself explain the failed fixed-window conditioning gate, which uses the same960-frame window size. No measured effective receptive-field claim is made.
- There is a single audio/state1×1 input projection followed by shared temporal blocks. There is no independent audio encoder, repeated audio fusion, cross-attention, or modality-specific normalization. This is a weak inductive bias, not proof that audio amplitudes are too small. Direct sensitivity measurements argue against a simple disconnected/zero-scale explanation.
- For one fixed diagnostic draw, zero-output noise objective was1.29482 versus .53597 for the target; its derivative component was5.574 versus .609. The smoothing kernel still makes noise cancellation a substantial part of the task. This is a scale observation, not evidence that a different noise schedule would pass conditioning.

## 4. Smallest defensible intervention

**Do not start with “increase audio gain” or more ODE steps.** The trained network already reacts strongly to audio, and stronger sensitivity is not the failed endpoint. More steps did not fix the separate incumbent jitter experiment; that historical result does not rule out every sampler for this model.

For the smallest directly justified physical intervention, keep the native139 representation, checkpoint identity rules, sampler exposure, audio architecture and flow objective, and add a separately weighted differentiable physical loss on a reconstructed clean motion estimate:

`clean_normalized = state + (1 - t) * predicted_velocity`

Denormalize, decode rotations, integrate root once, then apply:

1. FK joint/velocity error against the training reference.
2. Planted-foot velocity using reference contact masks at both adjacent frames. Keep generated-contact coverage and generated-contact/foot-consistency gates at evaluation, so predicting zero contacts cannot evade the loss or the gate.
3. A root-displacement/bias diagnostic over multiple horizons. If adding a root penalty, supervise reference travel rather than shrinking all velocity toward zero.

This changes the flow-matching optimization target; it is a new experimental arm, not a supposedly equivalent implementation repair. Use a new identity and frozen weight choices. It may improve physical motion without improving music conditioning.

For conditioning, the minimum next diagnostic is song-held-out predictability/lag testing on the same physical summaries, with donor and within-track shifts. Stronger local/multiscale audio fusion should be a separate matched ablation only after data alignment and predictable target content are established. Keep the existing proposed baseline/physical-only/audio-only/combined matrix; do not combine all changes and attribute the result to audio architecture.

## 5. Validated tests and acceptance tests still needed

Executed CPU tests in `outputs/dance_generator_loss_test.py`:

- Synthetic stationary skeleton: reference-masked foot-speed MSE exactly0.
- Same skeleton translated at .1m/s: foot-speed MSE .0100000.
- Gradient with respect to that velocity bias is finite and .2000000, so the proposed support path actually supplies a correcting gradient.
- Exact clean-estimate identity at t=0,.5,.99,1: normalized maximum error <=2.39e-7.

Executed probes and raw results:

- `outputs/dance_generator_probe.py`, `.json`, `.log`: four-record time sweep and loss counterexamples.
- `outputs/dance_generator_cohort_probe.py`, `.json`, `.log`: all eligible first-window checkpoint diagnostic.
- `outputs/dance_generator_cohort_summary.json`: song-weighted aggregates.
- `outputs/dance_generator_loss_test.py`, `.json`, `.log`: synthetic differentiability checks.

Commands all used `CUDA_VISIBLE_DEVICES="" bash run_torch.sh outputs/<script>.py`. Exit codes were0. Runtime was approximately3.1s,26.4s and.8s respectively. No parameters were optimized.

Before training a revised arm, still test: finite physical gradients on real examples including near-degenerate rotations; masks excluding padded/invalid frames and derivative pairs; no reference input in generation; active travel retained; planted/sliding/contact-free counterexamples; correct full/chunk support after any audio encoder change; exact checkpoint replay and matched crop exposure. Then retain all existing conditioning and independent physical gates. The CPU tests establish the proposed loss path's mechanics, not a successful trained intervention.
