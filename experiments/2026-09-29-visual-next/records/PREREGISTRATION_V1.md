# Visual learning after baseline C — prospective protocol V1

Status: prospective specification; NATIVE_REGISTRATION.json records native registration.
All hypotheses and gates must precede any new study output.
This document is a research preregistration, not a report of completed experiments.
Baseline C remains the user-selected product reference. No training, feature-model
inference, annotation, oracle optimization, TEST reading or new rendering was
performed in this research task.

## 1. Fixed context and scope

Parent: master@ff342ff; exact baseline and source hashes are in INPUTS.json.
C = independent Rich96 I_VR_P0 selected seed0, rhythm companion02, generated
palette history ON,300ms causal smoothstep transitions. Poses remain the source
of beam spatial style; audio controls movement time. The new studies address
visual supervision/color output and do not replace, retime or retrain motion.
Whole-track audio processing is allowed; no online-causality claim is made.

Historical selected DEV is exposed. Shpongle is an exposed demonstration and
is excluded from statistical method selection. Original31/40 source coverage and
29/30 qualified-null failures remain failed; the proposed pilot cannot repair
them by changing denominators. A successful pilot is conditional on its enrolled
source families and is not fresh unseen-song confirmation or product promotion.

The first Perplexity call timed out at300s. The same56411-character query was
successfully run through the same configured research MCP with a900s client
limit. The answer,498 source results and raw receipts are preserved. All10
recommended papers and7 supplementary primary papers were downloaded and read
in the stated scopes; none establishes a stage-light benefit for these proposals.

## 2. Cohort, annotation and execution prerequisites

ANNOTATION_ENROLLMENT.json prospectively fixes12TRAIN calibration families by
SHA256(salt+canonical_id) and all31 historicalDEV assessment families, each using
its existing64s source window. All43 source files exist; their full hashes must
be rechecked before extraction. No outcome-dependent replacements. The full91
TRAIN groups remain the training population for H3; no DEV transforms/fitting.
All remasters, duplicate audio, coaches and crops sharing canonical audio identity
remain one family. An unresolved cross-split identity conflict stops execution.

Two independent human annotators, blind to method predictions, mark source-video
content color changes, hard/gradual edits, camera-only changes, people/UI leakage,
ambiguous intervals and genuine color changes coincident with edits. Cut-coincident
color changes are NOT automatically labeled false. Judge content from silent video;
music-fit judgments belong to a separate later task. Disagreements are adjudicated
before any scored method outputs are inspected. At offsets0,7,14,21,28,35,42,49,56,63s,
independently trace retained-background masks, foreground and fixed UI regions.

Calibration agreement requirements: event matching F1>=.80 at±.2s and median
foreground IoU>=.85 between annotators; ambiguous time<=20%. These are instrument
eligibility requirements, not measured values. Freeze the manual/independent
assessment masks and events before scoring. At least30/31DEV families,>=80% of
each retained family's source duration, and>=10 event-bearing families are needed
for event inference. Mask comparisons require>=30DEV families with>=8/10 valid
independent mask frames each. Annotator failures, missing support or missing
model weights are unmet prerequisites, not evidence of zero effect.

All algorithms, feature transforms, code/weight hashes, software environment,
source timestamp mapping, grouping and statistical scripts must be frozen in a
new execution manifest BEFORE any new predictions. A future implementation may
fill addresses/hashes and implement this protocol; changing a method, threshold,
metric or cohort rule requires a versioned amendment before outcomes, not a silent
implementation choice. The annotation, implementation and model-acquisition
prerequisites are currently PENDING. No noise sigma or powered MDE is asserted.

## 3. Common inference, controls and statistics

Primary unit: canonical source-song family. Average frame/event observations
within a family; average paired training seeds within a family before aggregation.
Do not use frames, events, crops or repeated ratings as independent sample units.
Use10000 paired family bootstrap draws (seed20260929); report95% intervals and
paired sign-flip p values. The primary family is fixed to four hypotheses. Interim reports use conservative
Bonferroni-adjusted p=min(1,4p); once all four are evaluated, also report Holm.
Unexecuted hypotheses remain unmeasured, never assigned fabricated test results.
Use the conservative four-test adjusted p for early gate decisions; never select
a favorable subset post hoc. H3's additional control contrasts form a separately
reported Holm family. Pilot intervals are not a prospective power calculation.

Event matching: one-to-one maximum-cardinality matching within±.2s, then minimum
total timing distance, deterministic timestamp tie-breaking. Candidate peaks use
the existing .2s NMS; frozen .5 output threshold for F1/rate summaries. AP groups
tied scores and remains undefined for reference-nonevent families. Nonevent and
empty-output cases retain false-cue/minute and recall reporting; no undefined AP
is replaced by zero. Support is fixed from source/reference quality and shared
across arms, not chosen by predictions. Thresholds/scalers fit TRAIN only.

Real-data comparisons require constant/persistence, event-rate-matched random,
beat-only and circular-shift controls where relevant. H3 additionally uses
recipient-rhythm-preserving donor audio as a limited intervention, not a complete
causal isolation. Twenty deterministic admissible circular shifts per group must
have genuinely common source/reference support; failure is not repaired by
relaxing support after scoring. Numerical tie/endpoint behavior must be tested.

Before scoring, validate instruments on known hue changes at constant luminance,
luminance changes at fixed chromaticity, dark/achromatic frames, static backgrounds
with moving masks, camera translations, abrupt edits, color flashes, duplicated
frames, hue wraparound, ties and missing intervals. Report perturbation recovery
separately from real-image qualification. Self-distance must be0; constant output
and wrong spatial assignment must be detectable by the relevant instrument.

Relative improvements require a positive baseline denominator; zero/degenerate
references remain undefined or fail eligibility, never receive an epsilon floor.

Thresholds below are committed engineering targets, NOT literature effect-size
estimates and NOT powered MDEs. A sign-only native verdict never overrides failed
blocking gates. No hypothesis is started/closed in this task.

## 4. H1 — edit-aware color targets

ID: h-visual-next-cut-targets-20260929. Compare A1 against A0 using identical
cached YOLO prompts, stateless SAM2 masks, HS targets and source frames.
A0 is the existing chromatic observer. A1 uses released TransNetV2 at its fixed
.5 threshold and vetoes candidate color events within±.2s of predicted edit
intervals. It does not remove those times from the evaluation denominator.
Gradual-transition intervals use the released inference convention, which must
be pinned in the execution manifest. No local detector training or threshold sweep.

Primary: macro-family F1 against the independent CONTENT-color event annotations,
including genuine edit-coincident positives. Prediction: A1-A0>=+.05 absolute.
Blocking guards: lower95% paired interval>0, four-test adjusted primary p<.05; genuine
content-event recall loss<=.03 absolute; false-cue/minute must not increase; source
and annotation support unchanged. A veto that deletes real musical color changes
fails, even if its remaining event precision looks attractive. Report edit-only,
within-shot and edit-coincident strata; none becomes primary after inspection.

## 5. H2 — stable masks that improve background color observation

ID: h-visual-next-temporal-masks-20260929. New temporal recipe, linked as a
successor of h-layers-20260922-mask; the old SAM2-vs-box claim stays unchanged.
Primary comparison A2 vsA0: Cutie-base pretrained propagation at480px shorter side
from SAM2 union-of-registered-people anchors versus the original stateless SAM2.
Keep original person boxes, fixed UI exclusions and image geometry identical.
Refresh anchors every2s and reset at the SAME frozen edit boundaries in both
propagation controls. New-object omissions remain errors, not automatic exclusion.
Nearest-neighbor resize of binary masks back to the original decoded geometry.

Controls: anchor-mask zero-order hold with the same refresh/reset schedule;
original box mask as historical context; deliberately time-shifted mask/flow
controls; frozen-background/moving-mask positive/negative fixtures. DEVA is a
literature-supported alternative, not an unregistered substitution in this arm.
A3 (cut veto +A2 masks) is a secondary factorial interaction arm only; it cannot
replace either isolated primary contrast. No YOLO replacement in this protocol.

Primary: existing layers-mask-color-error, Jensen-Shannon DISTANCE (sqrt divergence,
base2) between extracted72-bin HS distribution and independent retained-background
reference; macro over the fixed frames, then equally over families. Prediction:
>=15% relative and>=.02 absolute reduction from A0. Primary lower95% paired benefit
interval>0 and four-test adjusted primary p<.05; improve over held-mask control; independent foreground
IoU must not drop by>.02; false removal of genuine background regions may not rise
by>.03 absolute; content-event false-cue rate cannot increase by>5%. Empty predicted
background on reference-valid frames is maximal color error, never a dropped row.
Reference invalidity is fixed independently for all arms. Flow-warped agreement,
boundary quality and late-shot decay are secondary; smoothing alone is not success.

## 6. H3 — richer color/temporal supervision for the audio student

ID: h-visual-next-color-temporal-20260929; successor of the original visual-aux
hypothesis with a different representation and independent human event endpoint.
The student still receives audio only; video features are TRAINING targets.
No image input, true palette history or future target labels at playback.

First qualify representations on the12TRAIN calibration families. R0=Rich96.
R1=4x4 grid of the existing72-bin HS histogram plus mean saturation, chromatic
occupancy and retained fraction per cell. R2 adds frozen DINOv2 ViT-S/14 last-layer
patch features pooled within the same cells. Resize whole frames to224x224 using
bilinear interpolation; use official input normalization, no crop/color jitter.
R3 adds R2 differences at .1s and .5s lags. Exact source timeline is10Hz; boundaries
without required past support are invalid, never padded. Each vector is projected
to96 components using TRAIN-only deterministic PCA then TRAIN-global coordinate
normalization, preserving the student's96-output auxiliary size. No per-song
normalization using DEV or whole future targets. Record pre-PCA and deployed
post-PCA color retention; a loss-only projection cannot hide information loss.

Controls: Rrand adds an UNTRAINED DINOv2 of identical architecture to R1 (seed17).
R3-shuffle and R3-reverse keep the current frame and labels fixed but permute/reverse
the five preceding10Hz frames used for differences; shuffle seed is the hash of
canonical ID plus20260929. R2 controls static features/context. These are fixed
arms, not a grid from which the best method/seed is reported.

Controlled recoloring/layout probes are eligibility checks, not primary quality
claims: explicit-summary reconstruction is tautological if the same summaries
are supplied as input. Require>=90% of R1's baseline-corrected hue/layout probe
skill to survive projection, monotone response to increasing controlled color
changes, and no collapse relative to the random representation. Publish these
checks, including dark/flat-background position controls. DINOv3, VideoPrism,
VideoMAE and V-JEPA2.1 are reviewed alternatives; switching teacher needs a new
preregistration rather than a post-result fallback.

Only after annotation/instrument/representation gates, fit all7 arms (R0,R1,
Rrand,R2,R3,R3-shuffle,R3-reverse) with3 paired seeds0/1/2 from matched independent
student initialization. Keep the original91TRAIN groups,94D audio bus,64s feature
context/central8s supervision, main targets, data draws, optimizer,6000 updates,
batch4, lr3e-4, weight decay.01, warmup300, gradient clip1, auxiliary weight.1 and
state dropout.5 identical. Use FINAL6000-step checkpoints in every arm, without DEV
checkpoint selection. Score untrained initializations as an instrument/capacity
control. C's accepted checkpoint is a separately named product anchor, not the
matched training comparator. Main targets are common: H1's cut veto is applied
to every arm only if H1 passes; otherwise the original observer labels remain
and this condition is declared before fitting. Original stateless masks stay fixed
in every H3 arm, isolating representation from H2. Source labels remain proxies;
assessment uses the separate human event annotations.

Primary: independent content-color event AP, R3 vsR0, target>=+.03. Blocking:
lower95% paired interval>0 and four-test adjusted primary p<.05; R3 must exceed Rrand,R2,R3-shuffle and
R3-reverse each by>=.01 AP with positive paired intervals and corrected control
p<.05. Report R1 separately even if it beats R0 while R3 fails. Correct-audio AP
must exceed each beat-only, donor and admissible shift control by>=.02; nonevent
false cues may not increase>5%; all21fits and all controls required. All mask,
support, annotation and source gates apply. No primary R2 on directly supplied
features, no selected-best seed, and no claim of unseen-song or human preference.
Event times may differ in this prediction study; the accepted C rendering is
untouched. Any later emitted-light trial requires a separate bound delivery.

## 7. H4 — a fair fixture-color capacity oracle

ID: h-visual-next-fixture-oracle-20260929. This is an output-capacity reframe,
not a claim that audio predicts the reference picture or that it looks better.
Use the existing C DMX trajectory from110–174s (64s) as ONE fixed rig/motion
context for each reference family's64s color sequence. Conclusions generalize
only over the reference families conditional on that fixed trajectory/camera.
Shpongle audio is not a statistical test unit here. No new pose inference.

Candidate render/color histograms can be reused across reference families because
the trajectory and allowed palette schedule are identical; the resource smoke
must prove the enumeration fits its fixed cap before any assessment optimization.
All oracle arms receive the SAME privileged reference regional color information
and the SAME palette candidates. Primary comparator O0 is an optimized SINGLE
GLOBAL-color oracle, not unprivileged baseline C. C is a separate engineering
anchor. O2 uses left/right groups. O4 uses left-heads, right-heads, left-other and
right-other RGB fixtures, by fixture x-coordinate<0 vs>=0; the center projector
belongs to right-other. Freeze exact fixture names/addresses before optimization.

Keep C's latent palette-token/history sequence, cue times, dimmers, all479non-RGB
bytes and300ms transition law exact. At each actual C palette-change/adoption time (not repeated cue opportunities), permissible
colors comprise C's emitted target RGB and three chromatic representatives of
the SAME selected TRAIN palette prototype (weighted farthest-point initialization
and deterministic weighted k-means,10iterations, lowest-bin-index ties; modes
ordered by weight then index). Palette prototypes with<3 occupied distinct bins
repeat available colors explicitly. Each group chooses one permissible color,
held until the next cue and blended with the fixed fade. Enumerate all choices
for O0/O2/O4; optimize mean per-region chromatic histogram divergence over the
cue interval using the same fixed1Hz frames, plus .01 times the fraction of
changed group assignments. Equal target access and same palette restrictions in
O0 prevent confounding capacity with privileged information. No adaptive hue
retuning, beam moves or dimmer compensation.

Controls: fixed cyclic permutation of O4 group assignments; canonical-donor
reference ordering fixed before outcomes; an unrestricted per-pixel palette
quantization floor reported but never subtracted to fabricate improvement.
Optimization uses1Hz frames; primary scoring uses the disjoint half-second-offset
frames. Freeze source/reference mask support and a neutral-RGB rendered visibility
mask before comparing colors. The renderer, camera and tone mapping are identical.
The four regional72-bin HS distributions use the existing chromatic eligibility
thresholds. Empty predicted chromatic mass on reference-valid support scores the
maximal regional distance; darkness is not a free exclusion. Source-invalid regions
are fixed for all arms. Report source-gamut and geometry limits separately.

Primary: macro-family mean of four-region JS DISTANCE on independent scoring
frames. Prediction: O4 error at least20% lower than O0; lower95% benefit interval>0,
four-test adjusted primary p<.05;>=80% of eligible reference families improve; O4 beats both permutation
and donor by>=10% relative; sampled mean displayed luma difference from O0<=5%,
clipping increase<=.1percentage point. All non-RGB/cue/history/fade checks exact.
If equal-luma guards or palette coverage fail, report failure instead of changing
the renderer or broadening the palette. The original C global RGB and model state
are preserved as the product reference irrespective of the oracle result.

## 8. Resources, dependencies and product gates

Serial execution on one RTX5070 12GB. Pilot cap8GPU-hours total: H1/H2 masks/cuts2h,
H3 representation extraction1h and student fits3h, H4 oracle/render2h. A stage
cannot borrow another stage's allowance after a failure without amendment. Peak
reserved VRAM<=10GiB, per-process RAM<=8GiB, new study cache/checkpoints<=3GiB on
root, no scientific cache growth on the nearly full/mnt volume; both filesystems
retain>=1GiB free. Research PDFs are a separate<=512MiB cache. No foundation-model
training, full decoded-video cache or silent model substitution. A bounded
16-frame smoke test pins runtime/VRAM and sources before corpus work. Runtime
numbers are unmeasured until that test; an OOM abort is not a hypothesis refutation.

Annotation/source/instrument validity are prerequisites for every affected test.
H1 and H2 are isolated and can fail independently. H4 needs valid reference masks
and targets, not a successful foundation encoder. H3's preliminary feature checks
precede its conditional audio fits. H4 failure blocks learning a fixture-wise
output under that recipe, not all audio/color research. No sealed TEST access.

The original human-preference-winrate diagnostic requires a separate blinded
multi-track panel (>=20raters). A later product trial must have independent music
fit and designed-appearance questions, equal audio/camera/exposure, brightness
controls, randomized order and song/rater clustering. Sample size/power and rater
reliability thresholds must be registered from an annotation-only pilot before
candidate comparisons. No such panel or power estimate exists now. C remains the
baseline; numerical proxy/oracle wins cannot change it automatically. No MDE is
entered in native gates because power has not been established.
