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

## Smoothed relative-gate result

The follow-up added a 31-frame native moving average to the **energy control
signal only**, while leaving generated 30 Hz motion raw. With the corrected
local intervention audit, quiet-jerk ratios became 0.874/0.878 and active-speed
ratios 0.983/0.986 for Shpongle/Cure; raw jerk safety ratios were 0.942/0.926.
Baseline and local-bypass identity deltas stayed below `1e-5` m. This is the
first two-song candidate that improves the quiet-jerk metric while preserving
active motion within the fixed engineering gate. It remains diagnostic-only
until a held-out/confirmation study and choreography review; it is not silently
promoted over fixed C.

Full diagnostic videos:

- `native_local_audio_fusion_smoothed_full_20261014_v4/shpongle/shpongle_local_hybrid_smoothed_full.mp4`, SHA256 `e76154482afa9e08e446314ea96c071a9bb897f7fd46fd81a48b358a933e8e71`.
- `native_local_audio_fusion_smoothed_full_20261014_v4/cure_for_me/cure_for_me_local_hybrid_smoothed_full.mp4`, SHA256 `d2e5d622486cebba8409ce19ddbfa5b587771c53d75eea9cd7ad55605e74c5f4`.

The videos were regenerated after commit `945dc0e` with a renderer-side raw gate
that recomputes finite values and speed/acceleration/jerk ratios from the saved
30 Hz body, binds the audit to the candidate and selected-baseline body hashes,
checkpoint hash, and own-song audio hash, and rejects before mapper/video on any
mismatch. Both own-song renders passed this gate, decoded cleanly at 1280x1024,
30 Hz with AAC, and passed the autonomous health manifest with
`healthy_checks_passed`. The frame diagnostics still show a noisy raw body; the
engineering improvement is real but this is not evidence of artistic
qualification or a fully repaired dancer.

## Energy-gated local fusion follow-up

The final native repair arm gates the local hybrid fusion by the TRAIN
30th-percentile normalized mixture-RMS and raises its bounded contribution to
0.8. Under the same canonical noise/context pairing, active speed is preserved
at 0.996/0.997 of baseline for Shpongle/Cure. Quiet jerk changes to 0.932/0.996
of baseline: the gate removes the cross-song uniform slowdown, but Cure still
does not improve materially. The arm remains diagnostic-only, with mixed
song-level evidence rather than a promoted fix.

It passed raw safety, local-audio input response (0.00506/0.01038 m
root-relative zero-local RMS), mapper source dependence, actual DMX finite/
duplicate checks, and post-render health audits. Full own-song diagnostic videos:

- `native_local_audio_fusion_energy_full_20261014/shpongle/shpongle_local_hybrid_energy_full.mp4`
- `native_local_audio_fusion_energy_full_20261014/cure_for_me/cure_for_me_local_hybrid_energy_full.mp4`

## Gate calibration correction

The first local-audio intervention also changed legacy audio adapters because it
zeroed the entire fine descriptor. The corrected audit now replaces only the
new module's cached keys/values and requires both baseline and local-bypass
identity. Its local response is smaller but causal; all incumbent bypass deltas
are below `1e-5` m.

The whole-recording relative-q30 gate was then trained with one seed. It did
activate per-song as intended, but produced quiet-jerk ratios 0.898/1.057 and
active-speed ratios 0.987/0.989 (Shpongle/Cure). This refutes the simple
threshold-calibration explanation for Cure and remains an unrendered failed
diagnostic. The global energy-gated arm is still the best mixed result, not a
qualified two-song repair.

## Raw temporal repair campaign (2026-10-15)

The first end-to-end raw temporal arm lowered quiet jerk to `0.490/0.503` of
the matched baseline, but active speed fell to `0.752/0.765`; this is the
uniform-slowdown failure and was rejected before mapping. A localized quiet
adapter preserved more motion (`0.946/0.946` active speed) but remained above
the quiet-jerk target (`0.914/0.929`).

The combined smoothed-local-hybrid plus quiet-adapter arm was the strongest
trade-off: Cure For Me passed the frozen engineering gates and Shpongle missed
active speed by `0.00090` (`0.94910`). A continuation then regressed active
speed on both songs (`0.94520/0.94884`). The final bounded arm restored active
speed (`0.98038/0.98287`) and raw safety, but quiet jerk remained just above
the pre-registered target (`0.86265/0.86395` versus `0.85`). These arms remain
diagnostic-only; none is mapped or rendered. Their sealed protocols and
verdicts are under `/mnt/lightsem-data/experiments/lightsem/native_raw_temporal_repair_20261015_v{1,2,3,4,5}/`.

This sequence establishes the current trade-off rather than proving a solved
jitter repair: aggressive raw jerk reduction suppresses musical movement,
while bounded adapters preserve motion but leave residual high-frequency noise.
No threshold was lowered and no post-hoc smoothing, retiming, fixture cap, D4,
or D6 was used. The only newly rendered candidate remains the
renderer-authenticated smoothed diagnostic described above.

The final v6 bounded pass kept active speed at `0.979/0.982` and raw safety
below baseline, but quiet jerk remained `0.857/0.859`, just above the frozen
`0.85` gate. The bounded sweep is closed: no candidate satisfies both quiet
jerk and active-speed gates on both songs, and no additional threshold or
post-hoc processing is allowed.

## Native temporal-adapter follow-up (2026-10-16)

To test a model-level remedy rather than post-hoc smoothing, a zero-initialized
depthwise five-tap temporal adapter was inserted inside the native flow vector
field. Its two-frame context halo is now part of chunked inference; full versus
chunked generation is exact in tests. Local attention/audio weights stayed
frozen. Scale `0.1` preserved active speed at `0.981/0.983` and raw safety, but
quiet jerk was `0.872/0.876`. A scale `0.2`, higher-jerk-weight follow-up kept
active speed `0.978/0.981`, yet quiet jerk remained `0.870/0.874`. Both arms
were rejected before mapping/rendering. The intervention is therefore a valid
native temporal module with no chunk-boundary artifact, but it does not solve
the residual low-energy jitter.

## Audio-energy latent-noise gate (2026-10-16)

A zero-init scalar gate was added inside native rollout to reduce the fixed
latent noise only in low-energy frames; this preserves the audio clock and is
not post-hoc body smoothing. The first arm was invalid because the historical
four-step truncated graph detached the initial noise, leaving the parameter at
exactly zero. The corrected arm backpropagated through all 32 native steps and
learned a finite nonzero scalar (`0.0009445`), but matched audits were
unchanged: active speed `0.983/0.986` and quiet jerk `0.874/0.878`. It is
rejected as a quiet-jerk repair; no render was authorized.

## Direct quiet-window Fourier objective (2026-10-16)

The spectral probe was promoted into a differentiable native training loss:
raw 30 Hz velocity FFT power above 6 Hz is measured in 1-second low-energy
windows, with no body smoothing or retiming. A temporal adapter trained only
with this Fourier loss had nonzero loss on every logged update and kept the
jerk-proxy loss disabled. Matched results were active speed `0.979/0.982`, raw
jerk safety `0.935/0.919`, and quiet jerk `0.870/0.874`. The Fourier objective
is therefore causal and safe but does not meet the frozen `0.85` quiet-jerk
gate; it remains diagnostic-only and was not rendered.

## Quiet-gated temporal Fourier/jerk fusion (2026-10-17)

A native temporal residual was gated by the normalized mixture-RMS quiet
signal inside the vector field, then trained with both the raw jerk excess and
the quiet-window velocity FFT loss. This lets the adapter act in rests without
an active-passage rate controller. Matched audits gave active speed
`0.983/0.985`, raw jerk safety `0.941/0.925`, and quiet jerk `0.873/0.868`
for Shpongle/Cure. The architecture is finite and raw-safe but misses the
frozen `0.85` quiet-jerk gate on both songs; it was not mapped or rendered.
