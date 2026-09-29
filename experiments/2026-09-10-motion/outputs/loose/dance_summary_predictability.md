# Base92 → physical-summary predictability and lag diagnostic

2026-09-10. **Unpreregistered exploratory CPU probe**, not a new neural-model experiment or promotion test. Local analysis choices were written before loading paired arrays. No TEST arrays, GPU, checkpoint loading, neural training, production edits, or vault writes.

## Answer

**There is a small song-held-out linear signal in within-record motion dynamics. It explains little of their variation. There is no persuasive evidence here for a single audio-motion offset.**

At zero lag, VAL within-record standardized MSE is **0.78914**, versus **0.80844** for the oracle per-record target-constant control: **2.39% lower error**. This is a fraction of within-record variance explained, not music fit, dance quality, conditional energy score, or a passing frozen conditioning gate. Most summary variation remains unexplained. The fixed ridge also fits TRAIN weakly: **0.76240 versus 0.79920**, a 4.60% reduction.

The descriptive best aggregate lag is +0.5 s, but its improvement over zero lag is only **0.00083 MSE (0.11%)**, with exploratory paired song-bootstrap interval **[-0.00200, 0.00397]**. Song-specific best offsets vary from -4 to +4 s. **Do not shift the corpus based on this result.**

## Instrument and controls

Read `CLAUDE.md`, `docs/research/2026-09-10-motion/PLAN.md`, `outputs/dance_investigation_generator.md`, and the current native audio preparation/data/summary functions. The manifest aligns waveform t=0 with motion frame 0; it does not fit an offset. Base92 is extracted causally at 40 Hz and interpolated onto native 30 Hz timestamps.

- Existing native audio manifest only. TRAIN: **125 recordings / 107 canonical songs / 35,190 half-second bins**. VAL: **33 recordings / 28 canonical songs / 8,049 bins**. Canonical groups are disjoint by assertion.
- Full recordings with at least 960 frames, matching the native training eligibility threshold. Excludes TRAIN 006/177/189 and VAL 171/190/192. Findings do not cover these six short recordings.
- Unmodified `motion_summary(features)` computes 18 physical-summary channels at 2 Hz, with its existing log1p transforms for speeds/energies. Native FK, not cached source joints, supplies geometry. Base92 uses matching nonoverlapping 15-frame means. The first/last eight half-second bins are excluded at **every lag**, so support never changes across offsets.
- One 92-input linear ridge per offset. Fixed penalty **lambda=0.1** in weighted mean standardized squared error plus squared coefficient norm. No feature, lambda, context, seed, or validation model selection. Means, scales, and coefficients fit on TRAIN only. Target standardization is the shared TRAIN global summary SD (floor .01); input SD floor .01.
- Primary model: demean audio and target separately within each recording. This removes record-level identity/constant cues. VAL target means enter **only scoring/its oracle constant**, never model fitting or prediction. Centering full-record audio is an offline diagnostic, not a causal deployed interface.
- Complementary raw model: global TRAIN-standardized audio → uncentered summary, fitted only at zero lag. Its **audio-track-constant** control feeds the same model each recording's mean audio. Its **TRAIN-mean** baseline predicts the fixed TRAIN target mean.
- Donor: next sorted canonical-distinct recording **within the same split**, deterministic, with centered features linearly resampled to target support length. It is an artificial diagnostic donor, not the frozen native evaluation donor recipe. Within-track null: separately circular-shift the correct predicted series by 25%, 50%, and 75% of its support length; average their errors, not their predictions. Linear per-bin prediction makes this equivalent to shifting input features. No recurrent/derivative computation crosses the wrap.
- Every aggregate uses equal canonical-song weight, equal recording weight within song, and equal bin weight within recording. Uncertainty uses 10,000 paired canonical-song resamples, seed 20260910. Intervals are exploratory, unadjusted, and conditional on this one fixed probe. Not frame-iid intervals.

## Zero-lag errors and nulls

All entries below are MSE averaged over 18 TRAIN-standardized channels; lower is better. Within-record and raw scores share the same target scale but answer different questions.

| Model / control | TRAIN | VAL |
|---|---:|---:|
| Within-record ridge, correct audio | .76240 | .78914 |
| Oracle per-record target constant | .79920 | .80844 |
| Within-record ridge, donor audio | .79757 | .82126 |
| Within-record ridge, within-track shifts | .84423 | .86342 |
| Raw ridge, correct audio | .90288 | 1.05171 |
| Raw ridge, per-record constant audio | .92646 | 1.06173 |
| Fixed TRAIN target mean | 1.00000 | 1.05969 |

For the primary within-record model, correct VAL audio reduces MSE versus:

| Comparator | Relative reduction | Comparator − correct MSE, exploratory 95% interval | Songs improved |
|---|---:|---:|---:|
| Oracle per-record target constant | 2.39% | .01929 [.00060, .04077] | 16/28 |
| Donor audio | 3.91% | .03212 [.01462, .05075] | 18/28 |
| Within-track shifts | 8.60% | .07427 [.04232, .11044] | 24/28 |

The within-track null is much worse than the constant predictor. Its 8.60% contrast does **not** mean the model explains 8.60% of motion variation. Long shifts disrupt slow nonstationary envelopes as well as local timing. Donor resampling also changes feature timing. Neither contrast isolates beat alignment or perceptual music fit.

The raw model improves VAL MSE over the fixed TRAIN mean by only **0.75%**: 1.05171 versus 1.05969, paired difference .00798 with exploratory interval **[-.03522, .05408]**, 14/28 songs improved. Per-track constant audio is nearly as poor (1.06173). Audio does not reliably predict absolute summary levels with this fixed probe. The oracle motion-track mean is unavailable to an audio-only deployed predictor; do not compare its favorable number as if it were deployable.

## VAL spread and absolute-error baselines, all channels

Values use the existing summary coordinate units: metres for heights/spreads/extensions, dimensionless lean/contact; **log1p(m/s)** for speeds and **log1p((m/s)^2)** for energies. These are half-second means of transformed values, not transformed half-second means. `SD` is song-weighted global spread; `within SD` is the square root of song-weighted within-record variance. MAEs are song-weighted. Centered-ridge MAE scores within-record deviations (equivalently adding the oracle target mean to predictions), not raw inference levels.

| Summary | SD | Within SD | TRAIN-mean MAE | Track-constant MAE | Centered-ridge MAE | Within MSE reduction |
|---|---:|---:|---:|---:|---:|---:|
| root_speed | 0.1487 | 0.1299 | 0.1141 | 0.0985 | 0.0971 | +1.37% |
| root_height | 0.1231 | 0.1062 | 0.0684 | 0.0596 | 0.0604 | -0.94% |
| body_speed | 0.2040 | 0.1690 | 0.1651 | 0.1318 | 0.1262 | +6.80% |
| upper_speed | 0.2268 | 0.2002 | 0.1838 | 0.1564 | 0.1503 | +7.96% |
| lower_speed | 0.2397 | 0.1921 | 0.1950 | 0.1513 | 0.1482 | +2.11% |
| left_hand_speed | 0.3650 | 0.3380 | 0.2999 | 0.2733 | 0.2639 | +7.02% |
| right_hand_speed | 0.3794 | 0.3511 | 0.3091 | 0.2832 | 0.2739 | +6.44% |
| left_foot_speed | 0.3144 | 0.2718 | 0.2503 | 0.2075 | 0.2050 | +0.57% |
| right_foot_speed | 0.3227 | 0.2873 | 0.2643 | 0.2190 | 0.2171 | +0.23% |
| hand_spread | 0.2704 | 0.2469 | 0.2182 | 0.1961 | 0.1962 | +0.02% |
| hand_height | 0.2859 | 0.2557 | 0.2097 | 0.1946 | 0.1945 | -0.01% |
| foot_spread | 0.1862 | 0.1715 | 0.1467 | 0.1332 | 0.1335 | -0.04% |
| torso_lean | 0.1756 | 0.1548 | 0.1156 | 0.1041 | 0.1049 | -0.43% |
| arm_extension | 0.0842 | 0.0760 | 0.0685 | 0.0605 | 0.0606 | -0.91% |
| contact_fraction | 0.2711 | 0.2260 | 0.2227 | 0.1862 | 0.1842 | +0.31% |
| vertical_speed | 0.1296 | 0.1073 | 0.1050 | 0.0823 | 0.0793 | +6.07% |
| upper_energy | 0.4300 | 0.3827 | 0.3470 | 0.2958 | 0.2862 | +5.70% |
| lower_energy | 0.4297 | 0.3423 | 0.3400 | 0.2510 | 0.2496 | -0.13% |

The small signal is concentrated in speed/energy envelopes, especially upper-body/hand/body/vertical speed. Shape, height, lean, contacts and lower energy show near-zero or negative gains. This is a descriptive all-channel breakdown, not a separately validated channel-selection result.

## Predeclared lag sweep

Positive lag means **motion at t predicted from audio at t+lag**. Each row refits the same fixed ridge on TRAIN at that offset; VAL is never fitted. All offsets were specified before measuring: -4, -2, -1, -.5, 0, +.5, +1, +2, +4 seconds. The centered target-constant baseline is .79920 TRAIN / .80844 VAL at every offset.

| Audio lag (s) | TRAIN correct MSE | VAL correct MSE | VAL donor MSE | VAL shift MSE | VAL zero-lag − this-lag MSE |
|---|---:|---:|---:|---:|---:|
| -4.0 | 0.77007 | 0.79639 | 0.80671 | 0.85557 | -0.00725 [-0.01886, +0.00296] |
| -2.0 | 0.76828 | 0.79335 | 0.81193 | 0.85027 | -0.00421 [-0.01203, +0.00305] |
| -1.0 | 0.76685 | 0.79163 | 0.81647 | 0.85381 | -0.00249 [-0.00752, +0.00246] |
| -0.5 | 0.76508 | 0.79105 | 0.81862 | 0.85791 | -0.00190 [-0.00545, +0.00148] |
| +0.0 | 0.76240 | 0.78914 | 0.82126 | 0.86342 | +0.00000 [+0.00000, +0.00000] |
| +0.5 | 0.76236 | 0.78831 | 0.82071 | 0.86259 | +0.00083 [-0.00200, +0.00397] |
| +1.0 | 0.76388 | 0.79025 | 0.82013 | 0.85994 | -0.00111 [-0.00490, +0.00273] |
| +2.0 | 0.76778 | 0.79244 | 0.81959 | 0.85696 | -0.00329 [-0.00830, +0.00182] |
| +4.0 | 0.77350 | 0.79877 | 0.81759 | 0.84499 | -0.00963 [-0.01929, -0.00098] |

Descriptive per-song best lag counts: {'-4': 3, '-2': 5, '-1': 1, '-0.5': 2, '0': 4, '0.5': 5, '1': 5, '4': 3}. All best-lag counts and the aggregate minimum are selected descriptions, not confirmatory tests. At +.5 s, only 16/28 songs improve versus zero lag. No nonzero offset has a positive zero-lag improvement interval excluding zero. Causal feature smoothing, half-second averaging, anticipated movement, and choreography can all affect this lag curve. This is **not independent waveform/video synchronization verification**, and does not exclude finer offsets, per-record offsets, nonlinear predictability, or useful longer-context features.

## Implication before stronger audio training

A modest held-out linear dynamics signal exists, so the audio/summary pair is not wholly uninformative. However, this instrument does not show that stronger audio fusion will recover coherent motion or materially improve the failed native generator. Keep physical-coherence fixes and audio-fusion ablations separate. Do not add audio gain or alter dataset alignment on this evidence. If alignment remains suspect, inspect independent source synchronization anchors rather than optimizing an offset on this VAL sweep.

The fixed narrow probe does not rule out nonlinear/multiscale models. Its weak TRAIN fit also limits any information-ceiling claim. These MSE diagnostics cannot be substituted for the frozen generative energy-score controls, physical gates, or visual acceptance.

## Artifacts and execution

- Local premeasurement plan: `outputs/dance_summary_probe_20260910T213429Z_plan.json`
- Script: `outputs/dance_summary_probe_20260910T213429Z.py`
- Complete per-record/per-song/channel results, lag scores and bootstrap intervals: `outputs/dance_summary_probe_20260910T213429Z.json`
- Log: `outputs/dance_summary_probe_20260910T213429Z.log`
- New extracted diagnostic arrays and fitted coefficients: `/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/diagnostics/dance_summary_probe_20260910T213429Z`
- Manifest SHA256: `1c49d667e489ced893cef1ef94a1e36c0ec5ab072b28c702772d491c9a5fd98f`
- Summary-source SHA256: `491fc6b2028bd81ab15bb433e9a2554b9263df661c7cafd04a851d71f881b85d`

Command: `CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash run_torch.sh outputs/dance_summary_probe_20260910T213429Z.py > outputs/dance_summary_probe_20260910T213429Z.log 2>&1`

Exit 0; shell runtime 8.08 s; diagnostic runtime 7.09 s. TRAIN/VAL motion and audio hashes matched the manifest before extraction. Canonical split disjointness, paired lengths, feature dimensions and finite arrays were asserted. Production/frozen inputs were read only. Only new diagnostic artifacts and this report were written.
