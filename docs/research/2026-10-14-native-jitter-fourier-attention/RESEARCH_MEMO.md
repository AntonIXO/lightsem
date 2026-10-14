# Native jitter / spectral / attention research memo (2026-10-14)

This memo records the incremental Deep Research sweep and arXiv verification for the LightSem native audio -> body -> DMX problem. It does not promote a method.

## Findings that change the experiment order

- The active descriptor is wide (1061 audio values plus 80 plan and optional 128 stereo values), but the native flow sends the 1061 stream through a 32-wide pointwise adapter. A wide input therefore does not imply rich temporal or multiplicative use of the features.
- The current whole-song planner uses self-attention over a motion-derived plan. That is not the same as decoder cross-attention from motion queries into an audio token sequence. A matched-capacity local cross-attention arm is the right attention test.
- The most relevant motion papers put constraints on decoded clean motion and transitions, not only the flow vector field: EDGE (Tseng, Castellon, Liu, 2022), DiffDance (Qi et al., 2023), Lodge (Li et al., 2024), and FlowMDM (Barquero, Escalera, Palmero, 2024). FlowMDM explicitly reports Peak Jerk and Area Under the Jerk as abrupt-transition diagnostics; this supports a raw jerk audit, not a claim that its loss transfers unchanged.
- DisCoRD (Cho et al., 2024) uses rectified-flow decoding from discrete motion tokens to continuous motion and reports smoother motion, but its representation/training task differs from LightSem. It is a design reference, not evidence for this pipeline.
- Joint Time-Frequency Scattering (Andén, Lostanlen, Mallat, 2018) supports fixed multiscale time-frequency probes. SALSA-Lite (Nguyen et al., 2021) supports normalized inter-channel phase-difference features for spatial audio, but its microphone-array setting is not physical localization for a stereo music master.
- The earlier LightSem Fourier/PLP results remain negative or inconclusive for their exact recipes. The new Fourier arm therefore uses complex local coordinates at two scales over source flux and mixture descriptors, with a new native training intervention.

## Falsifiable study claims

1. Local complex Fourier context -> lower held-out rest/slowdown prediction error and lower raw quiet-interval high-frequency residual than the matched native baseline, while active-motion amplitude and raw safety gates do not regress.
2. Local audio-query cross-attention -> larger song-conditioned root-relative articulation and decoded angular DMX response than matched additive/FiLM conditioning, with no increase in quiet jerk. The claim fails if nearest-additive projection explains the same response.
3. Decoded endpoint velocity/acceleration/contact/rest objectives -> lower raw quiet jerk and longer correctly timed holds without a stationary-body collapse or active-passage false rests.

## Proposed frozen-feature CPU probes before further GPU work

Use song-level splits and the complete current descriptor as the comparator. Fit only TRAIN statistics. Predict body activity slope, rest onset/offset, rest duration, and emitted-light hold state. Compare current descriptor, local complex-STFT residuals, CQT/register-energy plus chroma, unfolded tempogram with pulse confidence, restricted joint scattering, and stereo IPD/reliability. Keep constant-grid, time-shift, source-shuffle, loudness-rescale, and quiet-but-active controls.

## Verified primary sources

- [EDGE, arXiv:2211.10658](https://arxiv.org/abs/2211.10658): transformer diffusion with music features, physical plausibility, beat alignment, diversity and user evaluation.
- [DiffDance, arXiv:2308.02915](https://arxiv.org/abs/2308.02915): cascaded music-to-dance diffusion, audio-motion alignment and geometric losses.
- [Lodge, arXiv:2403.10518](https://arxiv.org/abs/2403.10518): coarse-to-fine long dance generation, dance primitives and foot refinement.
- [FlowMDM, arXiv:2402.15509](https://arxiv.org/abs/2402.15509): blended positional encodings, pose-centric cross-attention, Peak Jerk and Area Under the Jerk.
- [DisCoRD, arXiv:2411.19527](https://arxiv.org/abs/2411.19527): discrete-to-continuous rectified-flow motion decoding.
- [Joint Time-Frequency Scattering, arXiv:1807.08869](https://arxiv.org/abs/1807.08869): fixed wavelet/modulus multiscale audio representation.
- [SALSA-Lite, arXiv:2111.08192](https://arxiv.org/abs/2111.08192): normalized inter-channel phase-difference spatial audio features.
- [HTS-AT, arXiv:2202.00874](https://arxiv.org/abs/2202.00874): hierarchical audio transformer with localized event representations.
- [FlowerDance, arXiv:2511.21029](https://arxiv.org/abs/2511.21029): MeanFlow, physical consistency constraints and channel-level cross-modal fusion; newest result is treated as unconfirmed until code/data are verified.
- [Complex-domain onset detection, DOI 10.1109/LSP.2004.827951](https://doi.org/10.1109/LSP.2004.827951): phase-prediction residuals for musical onset evidence.

All papers were verified through the arXiv MCP where an arXiv identifier exists. No claim above establishes choreography or emitted-light quality without LightSem's raw-body and decoded-DMX gates.

## Pilot corrections

The first implementation draft requested a 16 Hz component from a 30 Hz
descriptor grid. That exceeds the 15 Hz Nyquist limit and is excluded from
selection; the implementation now uses 14 Hz as its highest 0.5 s-window
component. The first run remains preserved as flawed exploratory evidence.

The separate active-baseline quiet-speed pilot reduced quiet residuals but also
reduced high-energy mean speed to 0.873 (Shpongle) and 0.815 (Cure For Me) of
the baseline. It is rejected as a uniformly slowed jitter fix. A follow-up
must use an excess-only quiet penalty plus an active-motion preservation gate.

The corrected Fourier candidate passed raw safety and source-dependence gates,
so it received labeled full-song diagnostic renders. Canonical health receipts
are `native_local_fourier_context_20261014_v5/health/*_candidate_v5.json`.
Shpongle raw p95 jerk improved (507.64 versus 581.92 baseline); Cure For Me
regressed (852.64 versus 683.04), so no promotion was earned. A separate
waveform phase-prediction probe is stored beside the renders and remains a
frozen CPU diagnostic.

## Corrected pairing and feature-group evidence

The first Cure comparisons were re-run after discovering that the experimental
generator drew only `n` noise frames while production draws the fixed
`noise_reference_frames=8320` field and takes its prefix. The corrected protocol
is `native_pairing_correction_20261014_v4`; its identity arm is exactly the
saved production body for both songs. The Fourier arm remains mixed after this
correction: quiet jerk is 0.700x baseline for Shpongle and 1.534x for Cure For
Me. The previous unpaired Cure numbers are withdrawn as causal evidence.

The matched mean-ablation report is
`native_feature_group_audit_20261014_v2/REPORT.json`. With the selected
checkpoint, removing all 1024 MuQ channels changes root-relative body RMS by
0.0463 m and 0.0401 m on Shpongle/Cure. Removing the 16 source channels changes
0.00142/0.00159 m; the four mixture channels 0.00139/0.00122 m; and the 17
event/section channels 0.00341/0.00276 m. The result is a property of this
checkpoint and readout, not a claim that MuQ is the only useful representation.
The direct audio path compresses 1061 channels through a width-32 pointwise
projection, while planner attention is self-attention over audio-derived plan
bins. A real audio-query cross-attention test must expose local audio tokens to
motion-state queries and compare against a matched additive/FiLM control.

The new low-energy root-relative jerk objective was tested with zero-initialized
gated adapters. Scale 0.2 produced quiet-jerk ratios 0.970/0.982 and active
speed ratios 0.986/0.986; scale 0.8 produced 0.920/0.933 and 0.951/0.951
(Shpongle/Cure). Both are raw-safe, but Cure still has about 230 m/s3 quiet
jerk and the active slowdown is measurable. No arm is promoted or rendered.

## Local motion-query fusion campaign

The matched local-fusion campaign added nine native-frame audio support points
(`-60,-30,-15,-5,0,5,15,30,60`) to a motion-state query. Attention and FiLM
use the same width and trainable parameter count; hybrid concatenates their
hidden states. The branch is zero-initialized, so the incumbent is exact before
training. The attention arm preserved active speed at 0.991/0.992 and produced
quiet-jerk ratios 0.997/0.984 (Shpongle/Cure). FiLM produced 0.956/0.958 active
speed and 0.949/0.946 quiet jerk. A hybrid arm produced 0.959/0.961 active
speed and 0.951/0.950 quiet jerk. These responses are not feature-count
placebos: zeroing only the local fine audio descriptor while holding planner,
stereo, and noise fixed changed root-relative body RMS by 0.0045--0.0108 m.

A strict hybrid follow-up used active-preservation weight 8 and a 0.70 quiet
jerk target. It remained raw-safe, but active speed was still 0.955/0.958 and
quiet jerk 0.946/0.942. The branch is therefore a useful diagnostic fusion
candidate, not a qualified jitter repair or replacement baseline.

The strict hybrid passed the required pre-render raw and source-response gates,
so it was rendered as an explicitly diagnostic-only artifact with the canonical
P2/LUFS/mapper-01 path. Full own-song videos, with actual emitted universes and
no fixture speed caps, are:

- `native_local_audio_fusion_hybrid_v2_full_20261014_v3/shpongle/shpongle_local_hybrid_strict_full.mp4`
- `native_local_audio_fusion_hybrid_v2_full_20261014_v3/cure_for_me/cure_for_me_local_hybrid_strict_full.mp4`

Both decode at 30 Hz with own-song AAC. Post-render health reports have finite,
raw-dynamics, stream, diversity, input-sensitivity, and provenance checks
passing, while the delivery remains diagnostic-only because active-motion
preservation and artistic choreography are not qualified.
