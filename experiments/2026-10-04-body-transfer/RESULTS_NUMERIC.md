# Body transfer portfolio: V3 numerical results and interpretation

**V3 has 30 independently verified timelines. Fourteen methods pass the unchanged
engineering gate on both songs; corrected method 03 fails on both and remains a
negative diagnostic. All 30 full dark-stage videos are delivered and verified.**

The 15 mappings use each song's own accepted body and
preserve the P2 RGB bytes. The source body, mapping and rendering clocks agree.
The primary audit re-evaluated the frozen mapper and reproduced every saved
target, brightness value and semantic field exactly, then decoded DMX independently.
It verified 523 frozen input bindings and all 30 outputs of 7,200 ticks and 63
fixtures. Maximum emitted speed/acceleration are 89.853°/s and 354.985°/s² under
the declared 90°/s and 360°/s² simulation caps.

## What changed

The earlier mapper omitted direct head response and contained two identical
beam/laser desired-target pairs. All new methods respond to the six tested
anatomical regions after quantization; all constant-body controls produce
constant commands. Within each default installation, its 13 moving-angle streams
are distinct and its moving desired-target streams have no exact duplicates.
The 30 whole-installation DMX variants are also distinct. This does **not** imply
unique or active brightness channels: 03 has duplicated complete bulb-command
streams and 08 has seven constant moving-output dimmers, detailed below. Adding
fixtures preserves the IDs, positions and source roles of existing fixtures.

V2 originally passed all 30 crossing/sensitivity comparisons, but subsequent
semantic checks found that fixture offsets broke 03's absolute skeletal shape
and inverted 08's proximity behavior. V3 preserves the V2 artifacts and repairs
those two methods in a new frozen campaign. The other 26 saved timeline files
are byte-identical across revisions. Original V2 outcomes remain historical;
they do not validate the repaired mechanisms.

| Current V3 result | Shpongle crossing | Cure crossing | Unchanged engineering gate |
|---|---:|---:|---|
| V1 reference | 0.5818 | 0.5686 | Comparator |
| 03 Articulated sculpture | 0.7791 | 0.7829 | **Fail / fail** |
| 08 Proximity bridges | 0.2507 | 0.2488 | Pass / pass |
| Other thirteen methods | 0.0853–0.1185 | 0.0864–0.1195 | All pass |

The gate remains an absolute crossing drop of at least 0.10, source response no
worse than V1, and the hard validity checks. All 30 retain six-region emitted
source response, compared with V1's five of six. V3 records 28 `supports` and
two `refutes` outcomes, the latter for method 03. This is a comparison of each
anatomical method plus its capability mapping; it does not isolate an individual
paper's contribution. Crossings are measured in a declared geometric projection,
not as volume overlap or artistic quality.

## What the 03/08 repairs establish

**03 now preserves its declared absolute FK structure before mechanical fitting.**
All 22 expanded skeleton points share one fixed rotation, translation and uniform
bounding scale, including the 13 roles sampled by the moving outputs. Independent
verification found pair-distance ratios consistent with that common scale to
less than `1e-14`. The scale depends on the supplied body, never elapsed time or
output count; source attribution includes its group-wide dependencies.

This restores structural meaning but produces a dense central ray arrangement.
Crossings increase by 0.1973 / 0.2143 relative to V1, so 03 is **not a qualified
lighting improvement**. Desired points are not luminous endpoints: even perfectly
aimed rays continue to the stage boundary. In the review crop, ideal boundary-hit
pair distances have median ratios of 2.43 / 2.45 to intended pair distances.
Mechanical fitting adds separate errors: the 95th-percentile desired-point miss
from the actual finite ray is 0.135 / 0.159 m. These observations locate geometric
and actuator distortion separately from preservation of the internal skeleton.

**08 now preserves source proximity through its capability mapping.** It solves
instantaneous graph consensus for wrists, torso and head, then applies one common
spatial transform to that group and to the corresponding garland field. During
source wrist separations at or below 0.10 m, the median closest approach of the
two emitted finite rays changes from **1.842 to 0.00080 m** for Shpongle and
**1.847 to 0.00051 m** for Cure. The 95th-percentile remaining V3 approach error
is 0.0244 / 0.0010 m. Across the full review crop, source wrist distance versus
emitted-ray closest-distance correlation changes from −0.412 / −0.339 to
0.802 / 0.745.

There are only 13 / 5 qualifying contact frames, totaling 0.325 / 0.125 seconds;
these are correlated observations rather than independent contact trials. The
structural report retains the sparse contact-only correlations and every error
distribution. V3 08 passes the original fixed-brightness crossing gate, while
brightness-weighted crossing remains 0.543 / 0.527. Improved proximity and a lower
unweighted crossing fraction do not settle the visible-composition tradeoff.

## Morphology and timing are separate from geometric spread

The **V3 ancillary timing analysis** reruns the exact V2 measurement source
(`7362637febc33ff8294cec2d1cec785605c6d42d247027ec50ad062bf07cdf4a`) on a new
82-file input freeze, before computing the new values. All 32 timelines
(30 candidates and two V1 references) were measured. Inputs and primary verdicts
remained unchanged, and the 13 unchanged methods reproduce their previous
cross-song results exactly. The original V2 reports remain intact.

Temporal self-distance agreement compares the source body with emitted ray
directions after removing fixed offsets and normalizing each trajectory's RMS.
The values below are **current V3 measurements**, with the same V1 reference.

| Method | Shpongle body/direction recurrence | Cure body/direction recurrence |
|---|---:|---:|
| V1 reference | 0.710 | 0.693 |
| 03 V3 articulated sculpture | 0.917 | 0.920 |
| 08 V3 proximity bridges | 0.916 | 0.907 |
| 10 Coherent deformation | 0.940 | 0.936 |
| 11 Balanced transport | 0.940 | 0.938 |
| 02 Bend planes | 0.498 | 0.600 |
| 07 Observer-frame interactions | 0.572 | 0.467 |

These values describe a specific coordinate-dependent proxy. Orientation and
interaction representations deliberately change some invariances, so the
proxy is not a universal ranking of artistic quality. The historical V2 result
for 03 was 0.922 / 0.922 and did not detect its broken structural readout. The
new V3 temporal evidence complements the dedicated structural audit; it does
not reverse 03's failed crossing gate. Every method remains available for review
with its revision and diagnostic status visible.

For corrected 03, body-to-requested-direction recurrence is 0.919 / 0.937 and
requested-to-emitted recurrence is 0.979 / 0.959. For corrected 08 these are
0.921 / 0.933 and 0.988 / 0.973. Body-to-emitted recurrence falls near zero when
source timing is shifted by a quarter-window with the wrap boundary excluded
(−0.021 to 0.014 across these four cases), or reversed (−0.072 to −0.039).
Constant-output controls are undefined in every case, rather than rewarded.

Velocity structure is less well preserved than position recurrence suggests:

| V3 method | Body velocity → requested velocity | Body velocity → emitted velocity | Requested velocity → emitted velocity |
|---|---:|---:|---:|
| 03, Shpongle / Cure | 0.923 / 0.966 | 0.566 / 0.719 | 0.612 / 0.671 |
| 08, Shpongle / Cure | 0.929 / 0.964 | 0.584 / 0.703 | 0.649 / 0.716 |

The existing frozen per-timeline V3 diagnostics also record motor-angle holds
and attenuation. At the fixed 1°/s hold threshold, requested/emitted hold-mask
IoU is 0.810 / 0.819 for 03 and 0.790 / 0.784 for 08. Both have a best pooled
velocity lag of zero at 25 ms resolution, yet their emitted/requested angular
RMS ratio at 2–4 Hz is only 0.543 / 0.544 for 03 and 0.776 / 0.737 for 08.
At 4–8 Hz the ratios are 0.308 / 0.270 and 0.559 / 0.412 respectively. These are
motor-coordinate proxies; true 3D ray errors are reported separately. Zero lag
does not establish preserved fast gestures, and a hold-mask score is not an
annotated musical-phrase evaluation.

The same-protocol **V3 waveform audit** shows that **10 and 11 are close visual relatives**:
their centered direction similarity is 0.979 / 0.972 and angular-velocity
similarity is 0.971 / 0.970. Different mathematical mechanisms did not produce
strongly different motion in this case. Both experiments are retained, but they
should not be counted as two independent visual discoveries. The full pairwise
matrices and source-derived correlation limits are in
`records/ANCILLARY_SIMILARITY_REPORT.json`. Exact duplicates and high correlation
remain distinct findings. The current report lives in V3 and preserves the
unchanged 10/11 values exactly; the original V2 report is retained separately.
Most methods share the same fixture anchors and spatial bases, which explains
part of their common visual style.

The structural repairs change the waveform relationships: 03/08 direction
similarity changes from V2's 0.934 / 0.924 to V3's −0.020 / −0.108. In V3,
08's closest method by signed direction similarity is 11 at 0.542 / 0.512;
03/11 similarity is −0.404 / −0.527. Signed negative similarity can describe
opposing spatial gestures, and this metric does not fit rotations between
layouts. These changes establish different measured waveforms, not improved
choreography, independent source information or an artistic winner.

Method 05 has a pronounced tradeoff: its requested-to-emitted recurrence is
approximately 0.668 / 0.644, and its moving-output dimmers are within one byte
of the configured 0.8 ceiling for 28.9% / 34.0% of output frames. Method 15
also reaches that ceiling frequently, about 16.5% / 16.8%. These are saturation
relative to the chosen brightness transfer, distinct from full-scale DMX
saturation. The limits and brightness response were not altered after inspection.
Methods 09 and 15 have substantially smaller movement than the orientation
methods; static pictures can therefore look similar despite distinct trajectories.
These ancillary results apply to unchanged 05/09/15 and reproduce in V3. The
repaired 03/08 measurements are separate; no new brightness controller was added.

## Exact fixture duplicates and inactive brightness

Readback of the saved quantized commands confirms the dimmer-duplicate groups
already recorded in each timeline's diagnostics. The following counts describe
the actual 60–180 second review crop, separately from whole-variant hashes:

| Method / song | Constant dimmer outputs | Actual DMX level | Exact complete-fixture duplicates |
|---|---|---:|---|
| 03 / Shpongle | 11 edge bulbs: 001–003 and 043–050 | 6/255 = 2.35% | Bulbs 001–003 form one identical dimmer+RGB group; 043–050 form another |
| 03 / Cure | 12 edge bulbs: 001–006 and 045–050 | 6/255 = 2.35% | Bulbs 001–006 form one identical dimmer+RGB group; 045–050 form another |
| 08 / both songs | Beams 007–008 and lasers 001–005 | 20/255 = 7.84% | None: their pan/tilt streams vary and differ |

For 03, these edge bulbs have no intensity dynamics after quantization and stay
at the nominal 0.025 dimmer floor. Their shared side-specific P2 colors still
change, so a constant dimmer is not a completely time-constant fixture. Within
each listed side group, however, the entire four-channel dimmer/RGB sequence is
identical. Over the full 0–180 seconds, 8 Shpongle bulbs (001–003, 046–050) and
11 Cure bulbs (001–005, 045–050) remain at that floor; the larger review-crop
counts must not be presented as full-track counts. The reported 53 unique dimmer
streams for Shpongle means 10 redundant streams beyond a representative of the
11-member group, not 10 inactive bulbs.

For 08, the seven dimmers remain at the nominal 0.08 floor throughout the full
180 seconds on both songs. Its proximity/support activity descriptor is zero
for these assigned roles. This is an inactive brightness response while their
angles continue to follow the body; all 13 moving-angle streams remain distinct.
P2 RGB also changes within each review crop. This brightness limitation must
remain visible alongside 08's improved contact geometry and passing crossing
gate. These are observations of the saved output, not a new test or altered
controller.

## Cross-song and color limits

Beat-index alignment uses authenticated cached detections, with an explicit
limitation: Shpongle uses a 90 BPM constant grid, while Cure uses a local tracker
near 116 BPM. Matching 179 intervals covers different elapsed durations and
does not match musical phrases. No independent similar-BPM third song is included.
The actual aligned support is 60.17–179.42 seconds for Shpongle and 60.05–153.10
seconds for Cure. Both V2 and V3 reports retain unaligned and aligned
source/body/mapper/emitted comparisons with timing-null controls. Corrected V3
03/08 have aligned cross-song emitted-direction cosines of −0.024 / −0.011,
respectively. Their complete per-stage and per-fixture results are in the V3
report; dimmer and RGB comparisons are separate from motion.
Near-zero aligned cross-song similarities do not establish meaningful music
conditioning or artistic differentiation. The accepted body generator was not
retrained or requalified in this campaign.

P2 is unchanged. In the review crop, Shpongle has 137 saved color cues,
44 palette-token changes and seven occupied palettes; Cure has five cues,
one token change and two palettes. Actual RGB changes form 44 / 1 contiguous
transition runs. Cure remains unchanged for 112.4 seconds after its fade settles.
This color limitation remains separate from motion and brightness evaluation.

## Evidence and delivery

The current artifact root is
`/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v3`.

- `COMPARISON.json` and `COMPARISON.md`: all 30 numerical comparisons.
- `records/PRIMARY_NUMERIC_VERIFICATION.json`: independent identity, intent and emitted-command audit.
- `records/STRUCTURAL_READOUT_VERIFICATION.json`: 03 shape/optical/mechanical distortion and 08 V2-to-V3 actual contact-ray comparison.
- `records/V3_REPAIR_PROTOCOL.json`: declared semantic repairs, fixed mapping and unchanged gates.
- `records/ANCILLARY_TIMING_PROTOCOL.json` and `ANCILLARY_TIMING_REPORT.json`: identical-method V3 rerun, current 03/08 recurrence, cross-song controls and brightness concentration.
- `records/ANCILLARY_SIMILARITY_PROTOCOL.json` and `ANCILLARY_SIMILARITY_REPORT.json`: same-protocol V3 method/fixture waveform comparisons, including repaired 03/08 and unchanged 10/11 similarity.
- `timelines/<method>/<song>/report.json`: frozen motor-angle holds, extrema, attenuation and separate true 3D ray errors.
- `timelines/<method>/<song>/ATTRIBUTION.json`: explicit source-joint dependency support.
- `records/VIBESCIENCE_VERDICTS.json`: computed narrow engineering verdicts; no choreography approval.
- `records/PRESENTATION_STATUS.json`: explicit per-method labels, including 03's negative diagnostic and 10/11 similarity.
- `records/PRIMARY_DARK_SMOKE_ADMISSION.json`: actual/fixed-brightness contact-sheet inspection admitting the labeled portfolio for full rendering.

The separate V2 root, `/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2`,
preserves `records/P2_COLOR_PROVENANCE_AUDIT.json`, both `ANCILLARY_TIMING_*`
records, both `ANCILLARY_SIMILARITY_*` records and `CPU_VISUAL_REVIEW.json`.
Its complete first-pass evidence is retained, including six audited schematic
previews (01, 08, 15 on both songs). Those earlier videos do not represent V3 08.

The V3 repair protocol records 296 passing implementation-admission tests.
Stable fixture IDs/roles and distinct appended mount positions support arbitrary
output counts; they do not increase the source dance's independent information.
The production compilation path remains whole-trajectory fitting followed by
16-bit angle quantization and DMX packing, with final decoding used for audits
and rendering. These are simulated limits and virtual laser rays, not measured
hardware performance. The renderer's 16-moving/128-bulb limit is independent of
the numerical mapper's output-count support.

**All 30 V3 dark-stage videos are rendered and delivery-verified:** 120 seconds,
3,600 frames, 1920×1080 at 30 fps, AV1 video and the authenticated song audio.
The videos total 603,978,838 bytes. Both songs retain their own visible source
body. The [review page](http://127.0.0.1:8773/?campaign=v3-complete) includes the
30 candidates, two historical baselines, diagnostic labels and downloads.

`records/DELIVERY_INDEPENDENT_VERIFICATION_V1.json` binds the complete media,
frozen sources, full-decode receipts, uniform frame timing, decoded reference
audio, all 32 HTTP Range/download routes, and archived presentation source.
Its 120 decoded-frame comparisons have maximum full-image mean absolute
encoding error 0.828/255. The primary visually inspected the largest-error and
lowest-correlation examples; these are integrity checks, not artistic scores.
Browser playback was not automated. Combined retained V2/V3 artifacts remain
under the 2 GiB budget, with more than 20 GiB free. Artistic acceptance remains open.
