# Motion representation results

The clean-source prerequisite passed. All three neural codecs and all four linear
revision capacities failed. An explicitly uncompressed native reference subsequently
passed the fidelity checks and a fresh visual review. Both matched music models
subsequently failed conditioning and generated-motion quality gates. Second-seed
confirmation remains gated. No model outcome
was evaluated on TEST. One group inspected during early source-format work was later
found in the duplicate-closed test partition; its explicit exclusion from any future
pristine-test claim is recorded in `data/source_gate.json`.

All artifacts below are under
`/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910`.

## Source and diagnostic repair

The official FineDance public subset was imported with archive CRC, path, size and
SHA checks. Label, exact-audio and shifted audio-fingerprint duplicate groups were
closed before splitting. The resulting split has 136 TRAIN, 38 VAL and 28 TEST
recordings. Ten audio-length mismatches remain valid motion-codec inputs but are
excluded from paired-audio work. No estimated synchronization offsets were fitted.

Twelve fixed source clips cover holds, travel/turns and coordination. The front and
oblique sampled-frame review shows coherent articulation, including preserved quiet
introductions. Source review is an agent's sampled-frame assessment, not a human
rating of choreography or music fit. AIST++ optimized COCO17 joints from three
official TRAIN clips are a separate reference, with the published ignore list
excluded and timestamps at exact 60 Hz. They are not FineDance training inputs.
[Official AIST++ format](https://google.github.io/aistplusplus_dataset/download.html)

The old video comparison now renders every finite predicted pose continuously,
with source detection uncertainty shown separately. Historical predictions and
protocols remain unchanged. Review packs:

- [Corrected old comparison](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/delivery/legacy_pose/wnw4_ZfVmrA_c0.mp4)
- [Clean source, quiet introduction](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/delivery/source/holds_067_0.mp4)
- [Source gate and provenance](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/data/source_gate.json)
- [Separate AIST++ reference](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/delivery/aist_reference/report.json)

## Fixed 20,000-update comparison

All arms used seed 0, the same equal-song crop sampler, batch 32, 16-second crops,
BF16 training with FP32 geometry, compilation and prefetch. Each was evaluated at its
actual saved initialization and final scheduled checkpoint. Results below give
fixed-window / full-recording values, averaged by canonical song group.

| Codec | Joint error / body height | Velocity NRMSE | Contact F1 | Excess foot sliding, m/s |
|---|---:|---:|---:|---:|
| Frame24 | .02545 / .05653 | .31159 / .34037 | .97213 / .97676 | .08819 / .08024 |
| Temporal64 | .02288 / .05813 | .32472 / .34092 | .96417 / .97030 | .07329 / .06799 |
| Temporal128 fallback | .02280 / .05334 | .32213 / .33924 | .96418 / .97042 | .07249 / .06670 |
| Required | <=.02 both | <=.15 both | >=.90 both | <=.03 both |

The temporal codecs also lose slightly more than 10% of reference energy in the
fixed windows. Recognizable poses and accurate contact labels do not establish
accurate dynamics or planted feet. Full-track integration exposes accumulated root
offsets in the fixed travel clips.

The primary full-recording velocity difference against Frame24 is +.00055 for
Temporal64 (paired 32-song 95% bootstrap interval [-.01980, .01562]) and -.00113 for
Temporal128 ([-.02048, .01371]). Neither establishes superiority. Their fixed-window
velocity differences are worse: +.01312 [.00511, .02110] and +.01053 [.00202, .01905].

Vibescience computed `refutes` for the Temporal64 experiment and `directional_only`
for the prespecified Temporal128 fallback. These labels and failed independent
quality gates are preserved in `/root/lightsem-vault`; the tiny fallback directional
change does not authorize promotion.

- [Initial-arm review, travel](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/delivery/codec_frame24_temporal64/travel_027_4200.mp4)
- [Fallback review, coordination](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/delivery/codec_frame24_temporal128/coordination_057_1200.mp4)
- [Complete selection and gates](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/codec/final_selection.json)
- [Computed scientific records](/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/codec/vibescience_results.json)

## Failure localization and revision

Post-result oracle-root diagnostics show that supplying the true root does not repair
the remaining limb-velocity error. This substitution is solely a diagnostic; it is
not a candidate reconstruction or a playback input.

The frozen 16-second quiet/stationary cohorts both contain zero windows. Their true
boolean therefore provides no quiet-motion evidence. Additional two-second checks
find 83 quiet windows and excess motion in the original codecs. This coverage flaw
is explicitly recorded, without retroactively changing the original gates.

The accepted plan calls for a documented revision after failure. The distinct
[native-frame codec revision](REVISION_PHASE_CODEC.md) tests a structured linear
representation with explicit short-hold coverage. It has a new hypothesis, source
snapshot and output root. Original runs are not extended or overwritten.

## Linear phase revision: failed

The four capacities share one exact, equally song-weighted TRAIN covariance fit.
Root and contact values are explicit, counted latent channels; the decoder does not
receive a separate true trajectory. Every candidate includes native-frame temporal
bands. All capacities were evaluated before selection.

| Linear candidate | Values/second | Full joint error / height | Full velocity NRMSE | Full energy ratio | Excess sliding, m/s |
|---|---:|---:|---:|---:|---:|
| Native frame PCA24 | 930 | .08883 | 1.09912 | 1.43060 | .57531 |
| Phase64 | 690 | .07022 | 1.79245 | 4.02046 | .77726 |
| Phase128 | 1,170 | .02230 | 1.14242 | 2.18786 | .56619 |
| Phase256 | 2,130 | .00479 | .75050 | 1.61026 | .40052 |

All four retain contact labels and root travel, but fail temporal and physical
fidelity. The largest capacity often looks close in sampled poses, illustrating
why still-frame accuracy cannot replace velocity and foot-sliding checks. It also
adds up to .145 m/s of excess speed in the prespecified short quiet windows.

The fixed primary Phase128 comparison against the original Temporal128 worsens full
velocity NRMSE by .80318, with a 32-song paired 95% interval [.73900, .87579].
Vibescience computed `refutes`. No alternate capacity replaced the primary after
outcomes, and no compression candidate was promoted.

- [All phase gates and capacities](/mnt/lightsem-data/experiments/lightsem/motion_representation_phasecodec_20260910/codec/final_selection.json)
- [Larger-capacity review](/mnt/lightsem-data/experiments/lightsem/motion_representation_phasecodec_20260910/delivery/phase128_phase256/coordination_057_1200.mp4)
- [Computed phase verdict](/mnt/lightsem-data/experiments/lightsem/motion_representation_phasecodec_20260910/codec/vibescience_results.json)

## Uncompressed native reference: engineering acceptance

The documented [native revision](REVISION_NATIVE_REFERENCE.md) retains every one of
the 139 channels at 30 Hz, or 4,170 values per second. It has no learned compression,
fit, or training seed. The music generator must generate every root, rotation and
contact channel itself.

Every VAL feature is preserved exactly. Window and full-recording joint/velocity
errors are zero, contact F1 and energy/travel ratios are one, and the 83 quiet plus
157 stationary two-second windows have zero excess motion. These expected identity
results are an engineering check, not a learned-model success.

A strict shortcut for transferring the previous source visual review failed because
recomputed joints differed by up to 7.15e-7 metres, exceeding its 5e-7 limit. That
result was retained. A separately frozen, fresh 12-clip source/native review then
passed; sampled poses, travel, support-leg changes and floor transitions agree.
The numerical motion gates were not changed.

- [Fresh native comparison](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/native139/coordination_057_1200.mp4)
- [Exhaustive native audit, including failed review-transfer shortcut](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/codec/acceptance.json)
- [Final engineering acceptance with fresh review](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/codec/final_acceptance.json)

## Music experiment: failed qualification

The paired dataset contains 128 TRAIN recordings in 109 canonical groups and 36 VAL
recordings in 31 groups. The 32-second crop sampler actually uses 125 TRAIN recordings
in 107 groups, and scores 33 VAL recordings in 28 groups. Three shorter recordings
in each partition do not supply a complete crop; this is a limitation of this recipe.
Exact Base92 features are sampled at fixed physical
timestamps; audio-duration mismatches remain excluded. Both models have 3,845,515
parameters and receive 30,000 identical crop draws in batches of 32, with 32-second
contexts. Both completed the full schedule in approximately ten minutes each.

The primary score uses 18 fixed physical motion summaries at 2 Hz, normalized on
TRAIN. It is the probabilistic energy score across six flow draws, with deterministic
regression as the matched control. All 119 validation windows, donor recordings,
contiguous within-track shifts, random seeds, thresholds and bootstrap rules were
frozen before training. Direct audio lighting controls are disabled for these tests.
Generated motion has separate energy, travel, contact, floor, sliding, high-frequency,
full-sequence continuity and visual gates. Results:

| Model | Initial correct ES | Final correct ES | Donor improvement, 95% CI | Shift improvement, 95% CI | Excess foot sliding |
|---|---:|---:|---:|---:|---:|
| Regression | 1.5244 | 1.1934 | 2.58% [-2.45%, 7.10%] | 1.64% [-1.15%, 4.31%] | .618 m/s |
| Flow | 2.4276 | .8796 | 2.34% [-.15%, 4.64%] | .50% [-.60%, 1.44%] | .882 m/s |

The six-draw flow improves distributional score over the deterministic regressor,
but both wrong-music comparisons fail the 5% requirement and their intervals include
zero. Contact coverage remains populated while feet slide, so contact labels cannot
substitute for physical support. The flow's kinetic energy is 1.85 times reference.
Full/chunk agreement passes, while generated travel can leave the fixed stage view.
All 12 sampled-frame music reviews were completed with the incumbent alongside.

Vibescience computed `directional_only` for the flow claim. No music model qualifies,
and no second seed, TEST access, or new-model promotion is authorized by these results.

- [Frozen music protocol](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/protocol.json)
- [Discarded TRAIN runtime preflight](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/engineering.json)
- [Music comparison including the incumbent at 8 and 32 steps](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/music/seed_0/holds_067_0.mp4)
- [Complete music gates and intervals](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/selection_seed_0.json)
- [Computed music verdict](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/vibescience_results.json)

The playback loudness path uses FFmpeg's EBU R128 scanner: 400 ms momentary and
3-second short-term measurements, delivered in 100 ms metadata frames. Their
mapping to brightness and activity was then evaluated separately below.
[FFmpeg filter documentation](https://ffmpeg.org/ffmpeg-filters.html#ebur128)

## Incumbent integration: 32 steps did not fix jitter

The same incumbent checkpoint, full audio and noise seed 101 were evaluated at 8
and 32 Euler steps on all 12 review recordings. Pose spectral power above 6 Hz
increased from .002303 to .002517 of AC power, a 9.32% increase. The paired-song
95% interval for the proposed reduction was [-13.64%, -4.64%]. Total pose velocity
energy increased 38.33%, outside the frozen 80–125% preservation interval.
Vibescience computed `refutes`; retain the incumbent's 8-step setting.

- [Integration statistics](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/review/integration_statistics.json)
- [Computed integration verdict](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/review/integration_vibescience.json)

## Renderer calibration: fixed role gains qualify on optical ramps

The legacy whole-rig bulb response to 20%→80% dimmer is only 1.067 times its
dark-subtracted brightness. Five initial shared exposure scales all failed the
joint visibility/response gates. Their complete curves remain recorded; the
selected-primary endpoint was undefined because no candidate qualified.
Vibescience correctly computed `inconclusive` for that selection-based primary.

A distinct, preregistered revision fixed exposure at .4 and static radiance gains
for beams/washes/flashes/projectors at 1/.3/.03/.2. Every original gate was retained.
Its 20%→80% bulb response is 3.367 for a single head, 3.398 for six heads and 3.686
for the whole rig. All ramp monotonicity, quiet visibility, beam visibility and
clipping gates pass. Vibescience computed `supports` and the independent optical
gates pass. This is an opt-in renderer profile, separate from music qualification.

The updated legacy default is pixel-identical to the actual archived renderer
across all 63 ramp frames. Fixed profile operation removes the time-varying
total-intensity normalization; role gains and exposure stay constant throughout
the track. This calibration is specific to the canonical rig and recorded camera.

- [Original failed scalar study](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/exposure/report.json)
- [Fixed-role protocol and results](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/exposure_roles/report.json)
- [Whole-rig ramp comparison](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/exposure_roles/whole_rig_all_scales.png)

## Full Shpongle lighting: useful brightness control, no overall qualification

All five 277.3-second variants have complete DMX, synchronized audio/video and
5 Hz rendered observations. Native flow seed 0 supplies six fixed full-track
draws. The ordinary mapper uses one body for hands/legs/torso fixture pairs;
the independent ablation uses one draw per head. No source motion enters playback.

The three fixed intervals are quiet 0–20 s, active 70–90 s and ending 250–277 s.
Brightness-off changes only dimmer bytes; geometry, color, zoom and generated
motion are byte-identical to the hybrid input. The renderer ablation uses exactly
the same DMX with legacy exposure.

| Variant | Quiet median speed, °/s | Active median speed, °/s | Active/quiet bulb luminance |
|---|---:|---:|---:|
| Incumbent | 40.872 | 51.028 | 2.152 |
| Shared body + audio dynamics | 2.360 | 2.657 | 3.354 |
| Same motion, brightness control off | 2.360 | 2.657 | .568 |
| Six independent bodies + audio dynamics | 1.978 | 2.399 | 3.347 |
| Shared hybrid DMX, legacy exposure | 2.360 | 2.657 | 1.273 |

Quiet speed falls 94.23%, exceeding the required 60% reduction. Active median
2.657°/s and p95 7.929°/s fall below the frozen 3°/s and 8°/s minima. The emitted
quantized trajectory stays below 30°/s speed and 80°/s² acceleration throughout
(observed maxima 28.637 and 79.377). Sampled views show a restrained but overly
similar fan arrangement; convincing regional choreography was not established.

Mean bulb luminance rises from .03287 in the quiet intro to .11022 in the active
passage. Quiet visibility, entry contrast and ending-glare gates pass, with zero
white clipping in the 5 Hz full-track observations. The dimmer-only ablation
supports the direct-audio brightness effect; legacy exposure compresses it.
However, the new whole scene is substantially darker than the incumbent, and
the native generator remains unqualified. These are diagnostic previews, not a
replacement recommendation. No ending-position feature or scripted fade was used.

Vibescience computed `directional_only` for both coherence and dynamics. The
coherence claim fails active/visual/regional gates; the dynamics subcomponent
passes its lighting gates but fails the shared upstream motion prerequisite.
All outcomes and calibration updates are recorded only in `/root/lightsem-vault`.
Review is the primary agent's sampled-frame assessment, not human approval or
a claim of continuously watching every movie.

- [Full incumbent versus experimental Shpongle comparison](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/shpongle_comparison.mp4)
- [Full-track audio, DMX, speed and luminance figure](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/shpongle_dynamics.png)
- [Complete observations and gates](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/lighting/observations.json)
- [Computed final lighting records](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/vibescience_results.json)
- [Full Shpongle skeleton review](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/skeletons/shpongle.mp4)
- [Historical source comparison: wnw4](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/skeletons/label_corrected/wnw4_ZfVmrA_c0.mp4)
- [Historical source comparison: PeYhi](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/skeletons/label_corrected/PeYhiDbAWkk_c0.mp4)
- [Historical source comparison: 3WjC](/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/skeletons/label_corrected/3WjC-P1nbfk_c0.mp4)

The implementation passes 11 physical/indexing/quantization counterexample tests.
Source snapshots and input hashes were reverified, and all review movies have the
expected video and audio streams. See [reproduction](REPRODUCE.md) for commands
and [the next proposed revision](NEXT_REVISION.md) for the specific unresolved work.


## User-rejected spatial collapse: correction follow-up

**The previous Shpongle hybrid was spatially collapsed, not merely an overly
similar fan.** Its pacing/brightness improvements did not satisfy the product.
Original computed results above remain unchanged. All-frame geometric evidence,
new calibrated failure controls and two mapper revisions are documented in
[SPATIAL_CORRECTION.md](SPATIAL_CORRECTION.md).

Revision01 widens the footprint but fails movement. Revision02 improves median
first-surface span1.817->6.490m and active translation-removed4s extent.454->1.130m
against rejected hybrid (.372->1.130m against revision01). Photometry and source
motion are exactly held fixed. Full movies rendered and decoded, all555half-second
samples reviewed, with dense key-interval sampling and all-frame pixel analysis.
**Still not a complete fix:** visible reconvergence194–201s and a limited
floor-facing fan composition remain. Visual gate fails; both new computed verdicts
are `directional_only`. Music conditioning/foot-support failures remain unchanged;
no seed1, TEST, generator promotion or human acceptance was earned.

Full synchronized before/after:
`/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_02/shpongle_before_after.mp4`.
