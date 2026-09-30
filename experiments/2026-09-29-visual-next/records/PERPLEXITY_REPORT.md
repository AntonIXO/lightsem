# LightSem Visual-Learning Study

The highest-leverage next intervention is **not detector replacement**. First isolate shot cuts and stabilize masks temporally; then test whether richer representations preserve chromatic layout and temporal order; finally prove that fixture-wise color output has an oracle advantage before spending GPU time on audio learning.

> **Bibliographic limitation:** the supplied historical excerpt names many exclusions but does not enumerate all 81 references as a machine-readable bibliography. The ten papers below are absent from the visible exclusion text, but “incremental beyond all 81” still requires one final title/DOI deduplication against the complete vault bibliography before registration.

## Fixed scope

The following remain frozen:

- Baseline C: independent Rich96 `I_VR_P0`, seed0, rhythm companion02, generated palette history ON, causal 300 ms smoothstep transitions.
- Beam geometry, motion, aiming mechanics, cue times, dimmers, brightness, palette history and fades.
- Music remains the playback input; whole-track processing is allowed.
- Generated pose remains an internal spatial-style source. No pose retraining, retiming or motion-generator replacement belongs in this study.
- HistoricalDEV remains selection-exposed and cannot supply fresh confirmation.
- Shpongle remains an exposed demonstration track, not a selection or confirmatory unit.
- No conclusion about observer effects, independence, Rich96 benefit or palette memory is upgraded beyond the existing evidence.
- No new model becomes a product candidate merely because it improves proxy labels or oracle reconstruction.

The direct recommendation is a four-stage chain:

1. **Target audit and cleaning**
2. **Color-faithful temporal representation**
3. **Fixture-color oracle**
4. **Independent rendered-light evaluation**

Failure at any stage stops the dependent stages.

## Ten primary papers

None of these ten papers directly establishes better music-driven stage lighting. They contribute computer-vision or audiovisual methods whose transfer to LightSem is a falsifiable hypothesis; ColorVideoVDP is the closest to rendered-light evaluation, but it is still a full-reference perceptual metric rather than stage-light evidence.

| # | Primary paper and source | LightSem intervention → diagnostic → expected direction | Reproducibility, resources and transfer risk |
|---|---|---|---|
| 1 | **“TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection” — Tomáš Souček, Jakub Lokoč.** arXiv:2008.04838; [official code](https://github.com/soCzech/TransNetV2). | Add hard-cut and gradual-transition probabilities to target construction → manually audited cut precision/recall and color-event false positives within ±0.5 s of cuts → cut-aware exclusion should preferentially remove camera/editing events rather than legitimate within-shot lighting changes. | Pretrained inference is released. The authors warn that reproducing training involves tens of gigabytes of source data and hundreds of gigabytes after export, so LightSem should use inference only.[1] Transfer risk: music-video editing differs from the benchmark domains; a local cut audit is mandatory. |
| 2 | **“Tracking Anything with Decoupled Video Segmentation” — Ho Kei Cheng, Seoung Wug Oh, Brian Price, Alexander Schwing, Joon-Young Lee.** ICCV 2023; [official code](https://github.com/hkchengrex/Tracking-Anything-with-DEVA). | Feed existing per-frame SAM2 hypotheses into a temporal fusion/propagation layer instead of replacing the detector → flow-warped mask disagreement, leakage audit and target-label stability → temporal fusion should reduce segmentation flicker while retaining actual background changes. | Code and pretrained models are released. DEVA explicitly warns that false-positive detections can be amplified temporally and that growing memory can increase false positives, so unmatched-track deletion and maximum missed detections must be audited.[2] This is a stability intervention, not evidence that DEVA understands lighting. |
| 3 | **“Putting the Object Back into Video Object Segmentation” — Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, Alexander Schwing.** CVPR 2024; [official code](https://github.com/hkchengrex/Cutie). | Propagate audited person/UI masks from selected anchor frames → temporal mask consistency, identity retention through occlusion and false removal of colored background regions → a persistent object representation may outperform independent SAM2 frames on long shots. | Pretrained models and an inference example at 480-pixel internal resolution are released under MIT.[3] It requires an initial mask and therefore does not replace object discovery; detector errors and bad anchors remain failure modes. |
| 4 | **“RAFT: Recurrent All-Pairs Field Transforms for Optical Flow” — Zachary Teed, Jia Deng.** ECCV 2020; [official code](https://github.com/princeton-vl/RAFT). | Warp masks and explicit color-region maps between adjacent frames → warped IoU, boundary disagreement, residual chroma difference and camera-motion magnitude → flow compensation should distinguish moving-camera color displacement from true regional color evolution. | Official pretrained models are available; the alternate correlation implementation uses less GPU memory but is reported as slower.[4] Flow fails at cuts, occlusions, particle effects and illumination changes, so it must be disabled across detected transitions and never serve as semantic ground truth. |
| 5 | **“Mask-Free Video Instance Segmentation” — Lei Ke, Martin Danelljan, Henghui Ding, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu.** CVPR 2023; [official paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ke_Mask-Free_Video_Instance_Segmentation_CVPR_2023_paper.html). | Add a temporal nearest-neighbour patch-consistency diagnostic to masks → agreement of corresponding appearance patches across frames → instability concentrated away from cuts should fall under a successful temporal-mask intervention. | The paper uses temporal KNN patch consistency as supervision without mask annotations.[5] LightSem should borrow the diagnostic concept, not train its full VIS system; concert-video illumination changes may violate appearance constancy. |
| 6 | **“A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation” — Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, Alexander Sorkine-Hornung.** CVPR 2016; [official paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.html). | Report region overlap and boundary quality separately, plus degradation through a shot → mask-region score, contour score and temporal decay → a stable method should improve both boundaries and late-shot retention, not merely smooth masks. | DAVIS contains 50 Full-HD sequences selected to include occlusion, motion blur and appearance change.[6] Its object-centric evaluation transfers only partially: LightSem also needs explicit UI/person leakage and false-removal-of-background-lighting labels. |
| 7 | **“DINOv2: Learning Robust Visual Features without Supervision” — Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski.** arXiv:2304.07193; [official code](https://github.com/facebookresearch/dinov2). | Concatenate frozen spatial patch features with explicit Lab/hue/mask channels → controlled recoloring, spatial-swap and downstream target probes → patch features may add region/layout information, but explicit channels must carry the chromatic signal. | The smallest released ViT-S/14 has 21M parameters, with pretrained code and weights under Apache 2.0.[7] DINOv2 was trained for broadly transferable visual semantics, not color fidelity; invariance to appearance can be actively harmful here. |
| 8 | **“VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training” — Zhan Tong, Yibing Song, Jue Wang, Limin Wang.** NeurIPS 2022; arXiv:2203.12602; [official code](https://github.com/MCG-NJU/VideoMAE). | Compare ordered frozen video features against frame-shuffled and time-reversed inputs with identical dimensionality → prediction of audited region births, disappearances and composition changes → a real temporal representation should beat order-destroying controls. | Official models are released, but the project is mostly CC-BY-NC 4.0 and its reported models target action recognition.[8] Its 90–95% tube masking is not evidence of color fidelity; use it only after a small color/order probe, not as the default extractor. |
| 9 | **“ColorVideoVDP: A Visual Difference Predictor for Image, Video and Display Distortions” — Rafał K. Mantiuk, Param Hanji, Maliha Ashraf, Yuta Asano, Alexandre Chapiro.** SIGGRAPH 2024, DOI: [10.1145/3658144](https://doi.org/10.1145/3658144); [official code](https://github.com/gfxdisp/ColorVideoVDP). | Compare global-color and fixture-wise oracle renders under a fixed display model → temporal distortion maps and color-aware perceptual difference → a useful spatial-color output should visibly reduce composition/color errors, not just histogram error. | The implementation is MIT and supports GPU or slower CPU execution.[3] It requires calibrated display geometry, luminance and viewing conditions; incorrect photometry yields invalid predictions, and unrelated compositions can produce extreme scores.[3] It is a diagnostic, never a human-preference substitute. |
| 10 | **“Synchformer: Efficient Synchronization from Sparse Cues” — Vladimir Iashin, Weidi Xie, Esa Rahtu, Andrew Zisserman.** ICASSP 2024; [official code](https://github.com/v-iashin/Synchformer). | Apply a frozen synchronizability/offset probe only to emitted renders → correct-versus-shifted, rate-matched and donor controls → the candidate should not gain sync score merely by producing frequent flashes. | Pretrained models are released, but original training used large natural audiovisual datasets and substantial multi-GPU infrastructure; AudioSet pretraining reportedly ran for 12 days and did not saturate.[3] Natural sound-source motion is far from abstract stage lighting, so this is a transfer diagnostic and cannot be the primary endpoint. |

## Study A: Target cleaning

### Core question

Does the current failure come primarily from **what gets segmented**, or from **how independently segmented masks fluctuate over time and how cuts are interpreted**?

The initial study should not replace YOLO. Detector replacement changes object discovery, mask prompts and coverage simultaneously, while the observed problems—cuts, camera motion and framewise flicker—are temporally defined. DEVA’s own limitations also show why temporal propagation must be guarded: persistent false detections can be amplified rather than corrected.[2]

### Arms

Use the same decoded frames, existing YOLO boxes and SAM2 masks:

| Arm | Cut handling | Temporal mask handling | Detector |
|---|---|---|---|
| A0 | None | Current stateless SAM2 | Current YOLO |
| A1 | Cut exclusion/downweighting | Current stateless SAM2 | Current YOLO |
| A2 | None | Temporal propagation/stability | Current YOLO |
| A3 | Cut exclusion/downweighting | Temporal propagation/stability | Current YOLO |
| A4 | Same as best A1–A3 | Alternative detector | Conditional only |

A4 is not run unless A3 leaves unacceptable, manually demonstrated object/UI leakage. It must reuse the same segmentation and cut logic so it is a detector-only contrast.

### Audit labels

Sample frames and transitions from TRAIN-family songs, stratified by:

- High and low cut probability
- High and low estimated camera motion
- High and low mask instability
- Large and small Rich96/color-novelty changes
- Person/UI-present and apparently clean regions

Annotate:

- Hard cut, gradual transition or no cut
- Camera movement without meaningful background-light change
- Actual background/stage-light change
- Person leakage
- UI/text leakage
- Mask flicker
- False removal of a genuine colored background region
- Ambiguous/unjudgeable

The independent statistical unit is the **source-song/audio-identity family**, not the frame, crop, coach or event. Events may supply within-unit observations but cannot inflate inferential sample size.

### Registered hypotheses

**H1a — cut awareness.** Relative to A0, A1 will increase manually audited precision of color-change targets by at least **0.05 absolute**, while reducing recall of genuine within-shot lighting changes by no more than **0.03 absolute**.

**H1b — mask stability.** Relative to A0, A2 will reduce median flow-warped mask disagreement by at least **20%** and person/UI leakage by at least **0.05 absolute**, without increasing false removal of genuine background-light regions by more than **0.03 absolute**.

**H1c — combined intervention.** A3 will outperform both A1 and A2 on audited label precision without exceeding either arm’s false-removal rate by more than **0.02 absolute**.

**H1d — detector necessity.** Detector replacement proceeds only if, after A3, at least **5% of audited eligible frames** still contain detector-attributable person/UI leakage and a blinded review confirms that the errors originate in missing or incorrect boxes rather than propagation, cuts or SAM2 boundaries.

These are **design thresholds**, not predicted effects. Confidence intervals must be clustered by song family. Pilot data cannot be described as powered confirmation.

### Failure controls

- Disable propagation across detected cuts.
- Compare masks after camera-motion warping; raw frame-to-frame IoU would punish legitimate motion.
- Include a deliberately shifted-flow control.
- Include a no-person/UI subset to detect excessive smoothing.
- Keep an “uncertain cut” band rather than forcing every transition into cut/non-cut.
- Preserve the original masks and targets so apparent cleaning cannot silently delete difficult examples.
- Inspect whether temporal propagation freezes real color-region entrances or exits.

### Stop rule

Stop target-cleaning work and do not retrain the light model if:

- Audited agreement is inadequate to distinguish genuine light changes from editing/camera changes.
- A3 misses both H1a and H1b thresholds.
- Improvements come predominantly from deleting supervision rather than improving precision at retained coverage.
- Eligible audited coverage remains below the already-failed 80% requirement.

## Study B: Richer representation

### Representation ladder

The smallest defensible comparison is:

1. **R0:** current Rich96.
2. **R1:** higher-resolution explicit color layout—masked Lab/hue histograms, occupancy and region moments on a fixed spatial grid.
3. **R2:** R1 plus framewise frozen DINOv2-S/14 patch features.
4. **R3:** R1 plus frozen temporally ordered video features.
5. **R2-capacity control:** R1 plus a fixed random projection with the same added dimension as R2.
6. **R3-shuffle:** identical R3 features after within-window frame-order permutation.
7. **R3-reverse:** identical R3 features with reversed temporal order.

R1 must exist independently of foundation features. Otherwise a gain cannot be attributed to visual abstraction rather than simply exposing more explicit spatial color data.

### Color-risk probes

Before any audio model training, evaluate each frozen representation on synthetic transformations of real eligible frames:

- Hue rotation with luminance and mask fixed
- Saturation scaling
- Swapping left/right or upper/lower colored regions
- Translating a colored region while retaining its color
- Preserving layout while replacing colors
- Preserving colors while scrambling region layout
- Reversing the temporal order of region growth or disappearance

Diagnostics:

- Linear-probe reconstruction of masked Lab/hue summaries
- Region-centroid and occupancy reconstruction
- Sensitivity to hue distance
- Correct identification of spatial swaps
- Ordered-versus-shuffled prediction of the next audited visual change
- Song-identity prediction before and after song-mean residualization

### Registered hypotheses

**H2a — explicit spatial color.** R1 will reduce song-family macro error on held-out audited spatial color summaries by at least **15% relative** to Rich96, without increasing global hue-distribution error.

**H2b — semantic-feature increment.** R2 must improve the same endpoint by at least **5% relative** over R1 and by at least **3% relative** over the matched random-capacity control.

**H2c — chromatic preservation.** On controlled recoloring, R1+encoder features must retain at least **90% of R1’s hue-change sensitivity**. Failure means the encoder may be retained only as a separate layout channel, never as a replacement for explicit color channels.

**H2d — temporal ordering.** R3 must beat both R3-shuffle and R3-reverse by at least **0.03 macro AP** on audited region-change events. Beating a static representation without beating order-destroying controls is classified as a dimension/context gain, not evidence of temporal understanding.

**H2e — identity leakage.** No representation advances if its gain disappears under song-family-held-out evaluation or is accompanied by materially increased source-song identification without a corresponding order-sensitive visual gain.

### Stop rule

Do not connect the representation to the audio network if:

- Controlled recoloring is weak or non-monotonic.
- Spatial swaps cannot be distinguished.
- R2 fails the matched-capacity control.
- R3 fails both temporal-order controls.
- Feature extraction cannot be bounded to chunked processing on the 12 GB GPU.
- The required license is incompatible with intended product use.

A failed VideoMAE probe does not refute temporal visual representation generally. It only rejects that frozen feature recipe for this chromatic target.

## Study C: Fixture-color oracle

### Why this precedes audio learning

The current output broadcasts one RGB value to all color-enabled fixtures. No target or encoder can teach spatial color composition through that bottleneck. The first question is therefore not whether audio predicts multicolor staging; it is whether a constrained multicolor fixture output can produce a measurably better render while everything else is fixed.

### Oracle arms

| Arm | Color capacity | Other DMX |
|---|---|---|
| O0 | Current single global RGB | Exact baseline C |
| O1 | Two fixed fixture-region colors | Exact baseline C |
| O2 | Four fixed fixture-region colors | Exact baseline C |
| O3 | Same capacity as O2, randomly permuted region-to-fixture mapping | Exact baseline C |
| O4 | Same capacity as O2, donor-song color composition | Exact baseline C |

Use a fixed, preregistered fixture partition based on emitted stage footprint—for example left/right and floor/back-wall roles—not a learned free permutation. Assignment can minimize masked Lab transport cost plus a continuity penalty, but cannot:

- Add, remove or move cues
- Modify dimmers
- Change beam geometry or mechanics
- Change palette-history state
- Change the 300 ms transition law
- Touch any non-RGB DMX slot

### Diagnostics

- Spatial Lab optimal-transport distance between eligible source regions and rendered fixture regions
- Quadrant/region hue-distribution error
- Region occupancy and dominant-color error
- Temporal assignment-switch count
- ColorVideoVDP under one frozen, documented display model
- Exact binary equality of every non-RGB DMX slot
- Exact cue-onset equality
- Exact target-color-arrival delay of 300 ms
- Fixture-gamut and transition-rate violations

ColorVideoVDP is suitable only after the display model is fixed because it explicitly models display luminance, geometry, viewing distance and ambient conditions.[3]

### Registered hypotheses

**H3a — capacity.** O2 will reduce song-family median spatial Lab transport error by at least **20% relative** to O0.

**H3b — genuine mapping.** O2 will outperform both O3 and O4 by at least **10% relative** on the same endpoint.

**H3c — broad benefit.** At least **80% of independent song families** will show lower spatial color error under O2 than O0.

**H3d — invariants.** All non-RGB DMX values, cue onsets, dimmers, geometry and mechanics must remain exactly equal; RGB transitions must still reach their targets after 300 ms.

**H3e — stability.** Oracle improvement fails if it requires more than one unprompted fixture assignment swap between adjacent existing color cues. Fixture colors may transition at existing cues, not jitter continuously.

### Stop rule

Stop before audio learning if:

- O2 misses H3a or H3b.
- Benefits occur only on Shpongle or another previously exposed selection track.
- Spatial assignments collapse to near-identical fixture colors.
- The oracle gains proxy similarity but loses blinded designed-appearance preference.
- Any frozen DMX invariant changes.
- Improvement depends on framewise color jitter rather than stable cue-held compositions.

This stop rule is deliberately harsh: if privileged visual access cannot exploit fixture-wise colors, an audio model is unlikely to justify the additional complexity.

## Study D: Emitted-light evaluation

### Independent labels

The evaluation target must describe the **rendered light**, not agreement with the same video-derived teacher used for training. Create separate annotation instruments:

**Music fit**

- Accent is appropriately timed
- Color transition is appropriately timed
- Change rate is suitable for the passage
- The light responds to meaningful musical structure rather than every beat

**Designed appearance**

- Color composition looks intentional
- Fixture groups form a coherent palette
- Spatial balance is attractive
- Transitions are smooth and readable
- The result avoids accidental flicker, muddy mixtures and all-fixtures sameness

The source-video target may be displayed in a separate diagnostic task, but never alongside product-preference trials. Otherwise raters may score imitation rather than good stage lighting.

### Objective controls

Every sync endpoint must include:

- Circular time shift
- Other-song/donor audio
- Rate-matched metronomic events
- Event-count-matched random events
- Current baseline C
- No-change or cue-held reference where applicable

Synchformer predicts audiovisual offset from sparse 0.64-second segments, but its published models were trained on natural audiovisual datasets. Therefore it can flag gross offset sensitivity, but cannot decide whether abstract stage light “fits” music.[3]

### Human comparison

Use blinded, randomized, multi-track paired comparisons:

- Baseline C versus the candidate
- Separate questions for music fit and designed appearance
- No method names
- Loudness-normalized audio
- Identical render camera and exposure
- First presentation order randomized
- Repeated pair only for reliability estimation, never counted as a new independent song unit
- Shpongle excluded from selection and confirmation
- All versions of the same song or shared audio identity assigned to one family and one split

### Human gate

The confirmatory product gate remains **pending** because no external panel, variance estimate or power calculation exists.

A provisional decision target—not a claim of achievable effect—is:

- Candidate preference at least **60%** on designed appearance
- Candidate preference at least **55%** on music fit
- Two-sided 95% song-family-clustered interval excludes 50% for designed appearance
- No evidence of a music-fit loss larger than **5 percentage points**
- Acceptable rater reliability under a threshold fixed only after the annotation pilot

Do not fix confirmatory sample size from these percentages alone. Run an annotation-only pilot, estimate between-song and between-rater variance, then register the power calculation and final sample size before viewing candidate comparisons.

## Data roles

| Data role | Permitted use |
|---|---|
| TRAIN families | Target audits, feature probes, representation fitting and internal cross-validation |
| Pilot-development subset | Threshold debugging, extraction failures, resource measurement and annotation-instrument refinement |
| HistoricalDEV | Regression comparison only; explicitly selection-exposed, never fresh confirmation |
| Shpongle | Demonstration and regression rendering only; never method selection or hypothesis testing |
| Sealed TEST | Unread until intervention, metrics, exclusions, sample size and stop rules are frozen |
| Fresh independent labels | Final rendered-light music-fit and designed-appearance evaluation |

All clips, crops, coaches, remasters, duplicates and derived versions sharing audio identity belong to the same family. Splits must be materialized as immutable manifests before extraction.

## Ordered pilot program

### Pilot 0: Freeze and audit

- Record parent commit `ff342ff`, checkpoint SHA256, baseline configuration and full fixture/DMX hashes.
- Produce deterministic baseline renders and cue manifests.
- Hash audio identities and quarantine cross-split duplicates.
- Mark Shpongle and historicalDEV as exposed.
- Freeze the ten-paper deduplication against the complete 81-reference vault bibliography.

**Pass:** deterministic reproduction and exact manifest agreement.  
**Stop:** any inability to reproduce C with history and 300 ms fades intact.

### Pilot 1: Small label audit

- Use a stratified subset of TRAIN song families only.
- Annotate cuts, camera motion, genuine lighting change, leakage and flicker.
- Compare A0–A3; do not download an alternative detector.
- Measure annotation time and disagreement for later power planning.

**Pass:** H1a or H1b, with no corresponding false-removal failure.  
**Stop:** unreliable labels or coverage below the retained eligibility gate.

### Pilot 2: Frozen feature probe

- Run R0, R1 and DINOv2-S/14 only.
- Extract frames in chunks; retain compact float16 features and diagnostics, not duplicate videos.
- Perform recoloring, spatial-swap and random-capacity controls.
- Add a temporal encoder only if R2 passes.

**Pass:** H2a–H2c.  
**Stop:** color invariance, identity leakage or resource overflow.

### Pilot 3: Temporal-order probe

- Add one bounded frozen temporal candidate.
- Compare ordered, shuffled and reversed sequences with identical dimensionality.
- Do not train the audio model.

**Pass:** H2d plus chromatic preservation.  
**Stop:** gains survive order destruction or appear only within repeated crops from the same song.

### Pilot 4: Fixture oracle

- Implement O0–O4 on existing cues.
- Keep all non-color outputs byte-identical.
- Render a small blinded set not containing Shpongle.
- Evaluate composition metrics and designed appearance.

**Pass:** H3a–H3e and no human appearance warning.  
**Stop:** no privileged-input oracle benefit.

### Pilot 5: Annotation instrument

- Compare baseline C with oracle-derived candidates solely to validate the rating questions.
- Do not use these ratings as product confirmation.
- Estimate rater and song-family variance.
- Freeze final eligibility, exclusion, preference threshold and power calculation.

**Pass:** raters can distinguish music fit from designed appearance with acceptable repeat reliability.  
**Stop:** the questions collapse into source imitation, generic brightness preference or exposure artifacts.

### Conditional Pilot 6: Audio learning

This stage is outside the present research/preregistration task. It becomes eligible only if target cleaning, color representation, temporal ordering, fixture oracle and annotation gates all pass. Any later audio model must be compared with baseline C while retaining the exact spatial-motion and output invariants.

## Resource controls

- No foundation-model training.
- No full-video caches or dataset duplication.
- No model-weight download before a written size, license and storage plan.
- Maintain at least 1 GiB free on both relevant filesystems.
- Decode source video into temporary frame chunks and delete them after feature extraction.
- Store explicit color summaries separately from frozen embeddings so encoders can be changed without rebuilding masks.
- Use the smallest released candidate first; DINOv2-S/14 is 21M parameters.[7]
- Measure wall time, peak VRAM and output size on one short clip before corpus extraction.
- Stop or reduce resolution/window size before GPU allocation approaches the 12 GB hardware limit.
- Run GPU stages sequentially.
- Runtime and VRAM remain unmeasured until the bounded smoke test; literature hardware figures are not LightSem measurements.

## Interpretation rules

- Better mask consistency does not prove more accurate light targets.
- Better proxy-target AP does not prove better emitted light.
- Better spatial reconstruction does not prove music fit.
- Better synchronization scores do not prove designed appearance.
- Better designed appearance does not prove synchronization.
- A frozen encoder gain does not prove temporal understanding unless it beats order-destroying controls.
- A fixture oracle gain establishes output capacity, not audio predictability.
- Failure of one detector, propagation model or frozen encoder is recipe-scoped.
- One informed preference on Shpongle remains useful product feedback but not blinded generalization evidence.
- No result promotes a new model or reopens baseline C unless a separately registered product decision study is completed.
