# Implementation readiness V1 — pose auxiliary, recurrence, and video-pose validation

Date: 2026-09-28. **Read-only review; no new experiment or scientific verdict.**

## Decision

1. **Do not start new fits now.** Pose training machinery exists, but its data qualification and evaluation path are incomplete. R1 is not implemented. The original source-coverage gate still fails. The scientific-artifact cap cannot hold a complete matched family. Separately authorized full demos can use the root media filesystem after their own byte/free-space preflight.
2. **Do not rerun the old 2D/mocap experiment merely because its vault record is open.** Its saved report already records G3 failure for the measured ten-clip recipe. Reconcile that existing evidence administratively, with explicit scope and the 12-planned/10-measured discrepancy.
3. Next actionable engineering is a versioned pose-validity/provenance builder and a non-observer evaluation contract for pose arms. Implement R1 only after defining repeat relationships, a qualified repeat cohort, and a failure-capable metric. Neither work changes the accepted dancer.
4. Before execution, obtain explicit storage/cap authorization and freeze source, tests, data, comparators, seeds, statistics, and native successful vault receipts. A new directory is not a budget waiver.
5. After **each major emitted-LIGHT change**, deliver a new full Shpongle render even when numbers worsen. Use the frozen accepted motion and protected DMX bytes. A failed data gate is not permission to train an invalid arm just to make a video.

## Scope and evidence

System-of-record input: `VAULT_INVENTORY_V1.json`, queried 2026-09-28T18:16:46.988875Z. This review did not call the vault. It read source, protocols, prior result reports, completed-fit metadata and file sizes. It did not run imports, tests, scoring, inference, extraction, downloads or training. No sealed TEST input was opened. No source or completed artifact was edited. Only these readiness documents were created.

The current `training.py`, `model.py`, `evaluation.py`, and `palette_metrics.py` under `lightsem/experiments/light_layers/` are byte-identical to `source_snapshots/observer_fit_v1` and the frozen Sep22 source hashes. The JSON companion records the hashes. Source presence is not test execution or proof of scientific qualification.

Relevant vault records:

| Hypothesis | Snapshot state | Evidence state |
|---|---|---|
| `h-layers-20260922-pose-aux` | proposed | No linked pose fit; observer protocol explicitly excludes pose |
| `h-layers-20260922-recurrence` | proposed | No R1 fit; trainer explicitly rejects it |
| `descriptors-extracted-from-ordinary-video-with-3fc5b9` | testing | Open `exp-007-descriptors-extracted-from-936a48`; saved G3 failure already present |

## 1. Pose recipes: implemented vs missing

### Implemented in the frozen code

- `training.py:69,76–81`: exact channels `vert_bounce_amp`, `hand_spread`, `qom`, `head_speed`; all three requested recipes `S_V6_P1`, `S_VR_P1`, `S_VR_Pshuf` are recognized.
- `model.py`: a four-output auxiliary linear head. It emits only in training mode. The only observed forward input is 94-D audio. Pose is not an inference input.
- `training.py:583–608`: hash-checked coach motion loading; recorded active-span offset added once; descriptor interpolation to the visual grid; explicit per-channel validity; cuts, finite values and interpolation support intersected. Coaches are averaged equally among valid observations for each channel/frame, not turned into independent song samples.
- `training.py:325–405,415–435`: TRAIN-only normalization, common color/intensity/palette targets, masked pose standardization. Partial TRAIN pose availability is rejected.
- `training.py:612–722`: fixed different-canonical, same-partition, tempo-nearest donor for `Pshuf`; recipient audio and main lighting targets stay unchanged, donor pose **and donor validity masks** are substituted. Donor reuse is allowed: this is not a bijective permutation.
- `training.py:800–828`: center8s loss within a 64s feature window; pose loss weight0.1. Main loss is accent BCE + color BCE + palette CE. Generated palette history and 0.5 state dropout already exist, without gold history at evaluation.
- Seeds0/1/2, 6000 updates, batch4, AdamW lr0.0003, weight_decay0.01, 300-update warmup/checkpoints, BF16 with FP32 losses, no compile, 60-minute fit ceiling. Matched common audio, transforms and target frames are possible.

### Missing or not qualified

- No reviewed producer/receipt establishes per-frame, per-channel **gap/cut/identity validity** for the target coaches. Finite descriptors, positive boxes, longest track, and SAM2 completeness are not that qualification.
- The original LIGHT target has no completed independent quality qualification here. Source DEV coverage remains31/40=77.5%, below80%. Successful observer captures do not repair this gate.
- `evaluation.py:29–37` hard-codes five P0 recipes and eight observer contrasts. Its prediction/index/scope contracts do not admit P1/Pshuf. The existing scorer cannot simply score these new arms.
- The completed observer scope deliberately blocks P1/R1 (`training.py:990–1009`). Do not remove that guard or reuse its registration as permission. A separately frozen implementation/protocol is needed; if the proposed test remains only an authored-video observer proxy, name it as such rather than closing the original LIGHT hypothesis.
- Frozen trainer source binding does not itself cover a new validity producer, repeat extractor or new evaluation tests. Add all executable dependencies and exact test/receipt bindings prospectively.
- Pair P1/Pshuf with identical architecture/initialization/task exposure. P1 adds only516 active parameters to its P0 counterpart (4×129; richP1=1,242,306, legacyP1=1,230,696 by static arithmetic), but report the difference rather than claiming identical total capacity. Verify common backbone initialization and sampler draws in future synthetic controls.
- **Potential control confound:** Pshuf transfers donor masks. Unequal valid auxiliary frame counts can make the comparison partly a missingness/regularization intervention. Freeze and report the mask policy before fitting. If a matched-support policy changes the recipe, preregister the revision; do not fix it after outcomes.

## 2. Required data qualification — pose and common LIGHT target

These requirements precede fitting; they are not satisfied by the old observer study.

1. Preserve Stage11 canonical music-family grouping before recordings, coaches, windows and augmentation. Keep historical TRAIN/DEV membership and identity quarantine. No new split salt. No TEST reads or cross-split aliases. RPMC is not a confirmatory substitute without its original35-performance mapping.
2. Freeze source audio/video/pose/motion/annotation/feature/model/config hashes, source PTS, requested-grid timestamps, extraction version, coach identity and exposure ledger. All fitted thresholds, normalization and palette vocabulary use TRAIN only. DEV checkpoint selection remains exposed development, not fresh confirmation. Fresh confirmation would need a separately frozen cohort of at least40 unexposed canonical songs.
3. At least80 usable TRAIN and30 DEV canonical families; at least80% of the prespecified DEV population must remain eligible. For the original40 this means at least32, not31. Do not silently replace excluded songs or rename31 as a fresh denominator. Revised enrollment requires a prospective superseding design and keeps the historical failure.
4. Every included coach needs a descriptor NPZ with `times`, `meta.start_s`, and all four `ch__...` arrays; a validity NPZ with absolute `times_s` and boolean `valid[T,4]` in `POSE_NAMES` order; exact hashes in the manifest. All arms use the same available songs, core target frames, transforms and source exposure. `fit_auxiliaries` includes rich, legacy when required, and pose for the common prepared manifest.
5. Independently verify coach identity/ID swaps, camera cuts and keypoint gaps. Mark contamination over the **whole descriptor/filter support**, not merely the missing frame. Bounce amplitude uses a 2s window; smoothing and derivatives can carry contamination into neighboring frames. Do not count interpolated long gaps as observations. Validate finite, strictly increasing validity timestamps, no unsupported nearest-neighbor validity leakage, and coverage after masking.
6. Verify derivative precision and absolute timing before reuse: source FPS/PTS, active-span offset exactly once, coordinate scale/orientation, confidence gating, gap policy and descriptor computation dtype. Casting an already quantized/incorrect derivative cache to float32 does not repair it. Precision/identity error thresholds and acceptable per-channel support were not fully specified in the old high-level plan; freeze explicit criteria **before** screening or fitting, rather than inventing a retrospective pass.
7. Qualify a common color-only and separate intensity/accent instrument with independent annotations/controls. If using the SAM2 target, its mask study calls for at least120 independently annotated frames across30 canonical videos, plus separately labeled known-background composites and moving-mask/frozen-background negatives. Synthetic masks alone support only a synthetic claim. Unknown cuts included by the observer protocol are not clean LIGHT labels.
8. Video intensity novelty at10Hz is not fine beat truth. Keep independent fine beat labels separate. Pose footfalls and absolute KE must not become beat or dimmer truth.
9. Freeze positive/negative/shift/tie/no-event metric controls; retain quiet/no-event songs in false-cue/coverage reports. Use paired canonical-song aggregation, seeds0/1/2, 10,000 bootstrap replicates and the predeclared Holm family. A short-lived all-zero loss must not pass a quality gate.

**Original pose acceptance:** rich P1 must gain at least0.01 accent AP over **both** rich P0 and Pshuf with corrected positive paired evidence; color AP loss no more than0.01 and palette error increase no more than5%. Auxiliary-loss improvement alone cannot meet the claim. Legacy P1-vs-P0 is a prespecified secondary contrast.

## 3. Recurrence R1: substantial new code and data contract

### What exists is not R1

- `lightsem/audio/structure.py` computes a chroma/timbre recurrence matrix for segmentation, then returns **boundaries**. The internal cluster labels are not preserved as validated repeat IDs. Functional `intro/build-up/drop/...` labels are explicitly unvalidated heuristics. Two sections labeled `drop` are not necessarily the same motif.
- Current palette history embeds the immediately previous generated token. There is no keyed repeated-section memory, persistent recurrence state, trained retrieval/update rule, or R1 state dropout.
- `S_VR_P0_R1` is absent from `RECIPES`; `TrainConfig.validate` says “R1 recurrence is blocked.” No R1 fit queue, checkpoint schema, prediction path or repeat-relation metric was found in the frozen light-layer implementation.
- `palette_metrics.py` measures temporal residence/change/drift/lag and marginal histograms. It does not implement the registered repeated-section relationship error.

### Work needed before a fit

- Audio-only repeat-ID producer with algorithm/checkpoint/hash provenance; section boundaries, recurring-family IDs, confidence/invalidity and audio clock. Never derive IDs from target colors. Human listening/audited audio-derived labels can qualify identity; no licensed new corpus or pretrained model is assumed available.
- A preregistered definition of “two qualifying returning sections,” minimum duration/separation, repeat/nonrepeat matching, valid target support, and any audio repeat-detection accuracy gate. The original plan leaves these operational details unfinished.
- At least20 eligible canonical songs with the required returns, within the common student qualification (>=30 DEV and>=80% enrollment), and enough TRAIN recurrent examples to fit the mechanism. Current one64s-window/center8s observer enrollment does **not** establish that support. Audit metadata first; if longer/full-song feature coverage or chronological multi-window training is required, bind it before extraction and budget it explicitly.
- Explicit train/infer memory state keyed by audio repeat IDs, reset at song boundaries, generated-memory writes, trained dropout/null-memory exposure and defined gradient/truncation policy. No target history at inference or copying a target palette into memory. Full-song generation must carry memory across chunk seams.
- Define how recurrence memory is trained when previous occurrences lie outside64s; the current random crop sampler cannot silently provide prior-song state. A sequential sampler, deterministic generated prefix, or another explicit solution changes exposure/runtime and needs a matched no-memory baseline.
- Implement `layers-repeat-palette-error`: TRAIN-scaled absolute error in **pairwise joint-HS palette distances** for repeated pairs and matched nonrepeat pairs against the same reference relationships. Constant color must fail when the reference relationships differ. Freeze distance, scaling, zero-spread policy, pairing, aggregation and eligibility before outcomes.
- Compare R1 with no-memory `S_VR_P0` and with **shuffled repeat assignments** using fixed cue inventory/vocabulary and paired generation noise. Merely renaming ID symbols bijectively preserves all equivalence classes and is a useless shuffle. The shuffle must break which sections share memory while preserving relevant counts/structure. Original high-level plan specifies an ID ablation, not a separate trained shuffled-R1 recipe; define that explicitly before adding fits.

**Original R1 acceptance:** >=15% relative and>=0.02 absolute reduction versus both no-memory and shuffled-ID controls, corrected positive paired evidence; >=20 qualifying songs; diversity>=70% reference; color AP loss<=0.01; marginal palette error increase<=5%; matched nonrepeat and constant-palette checks. Passing an “always same chorus color” rule is not the registered mechanism.

## 4. Old 2D video vs mocap: answer already available, narrowly

`outputs/stage0a/pose_validation.json` and the vault gate agree: **mean core correlation0.627, below the registered0.8 threshold, on10 clips**. This is enough to preserve/report the measured recipe's G3 failure without another extraction. It does not establish all video pose is unusable or that a pose training auxiliary cannot help LIGHT. No shift/donor null or median-MAE baseline was recorded, so this correlation report has weaker calibration than a modern validity study; none is fabricated here.

For the four proposed auxiliary channels, old per-clip macro mean correlations are bounce amplitude0.794, hand spread0.812, qom0.682, head speed0.598 **against the old0.8 transfer threshold**. Only hand spread clears that point-estimate threshold. Calling all four “validated>0.8” is false. These AIST camera results do not certify Just Dance identity/cut/gap quality or prove downstream utility.

Static hazards in `run_pose_validation.py`:

- The plan requested12, but selection allows one clip per genre; the saved sample is10. Preserve that discrepancy. Adding two convenient clips now is not completion of a previously frozen twelve-clip cohort.
- Saved experiment specifies YOLO11x/ByteTrack; current CLI defaults YOLO11m. An unqualified rerun would change the recipe.
- Driver hard-codes60fps although the source metadata is59.94fps and the helper returns FPS. It pairs prefixes by array index/min length, not an explicit checked common timestamp grid.
- `pose2d.to_skeleton` fills all confidence gaps, including long gaps, and records fractions rather than a per-frame validity mask. `TRANSFERABLE` is a static name list, not evidence: it includes footfall/yaw/symmetry channels that the saved result shows weakly transferred.
- Outputs overwrite a fixed `pose_validation.json`; auto-download is built in. Neither is acceptable for an immutable new study.

**Smallest useful follow-up, only if a changed representation/qualification claim needs it:** first a synthetic timestamp/projection/known-motion/gap/precision control suite; then a small preregistered paired set using existing allowed TRAIN/DEV video+mocap, fixed YOLO checkpoint, exact PTS and immutable output paths. Include projected-mocap vs full3D as a geometry ceiling if calibrated camera projection is available; otherwise state that it is unavailable. Compare real alignment to time-shift/other-clip controls and report per-channel spread and constant-predictor error with errors, not correlations alone. No downloads, new dancer training, or relabeling this as the old experiment. A small feasibility set cannot certify a broad corpus claim.

## 5. What completed evidence can and cannot answer without rerun

- **Old G3:** measured ten-clip broad transfer recipe failed its threshold. Administrative closure can cite the saved gate with narrow scope and disclosed enrollment; this review performs no closure.
- **Observer15-fit study:** pipeline and matched P0 recipes executed. It is not pose or recurrence evidence. Original source coverage31/40<80% and paired-null support29<30 remain failures. All eight primary contrast intervals cross zero. Do not reuse its scores as clean-LIGHT baselines after changing labels/support/cohort.
- **Palette secondary:** selected own-history temporal error0.859022 is worse than trained-null0.830589; final0.750180 vs null0.755757 does not establish the required effect. This only answers the fixed previous-token observer-proxy setup. It does not test long-range repeated-section memory. Its retrospective vault-binding repair remains disclosed.
- Existing feature/checkpoint artifacts can be reused as immutable inputs only where their exact provenance, task scope, target definition, cohort, transforms and support match the new registered question. No assertion that existing P0 weights are a matched control for newly qualified LIGHT labels. If all those are identical in a separately registered observer-only pose study, reuse may be defensible, but it still cannot complete the original clean-LIGHT hypothesis.
- Existing source-box completeness proves neither pose accuracy nor camera-cut correctness. A completed capture or training smoke is not a quality result.

## 6. Smallest failure-capable next work, in dependency order

| Step | Proposed deliverable/test | Can fail because | Blocker class |
|---|---|---|---|
| A | Reconcile old G3 report and planned/measured sample | Existing gate already fails | Record interpretation; no GPU needed |
| B | Versioned source-only pose-validity/annotation schema and manifest validator | Missing/unsorted timestamps, wrong offsets, bad IDs/cuts, no valid channel support | Actionable code; human annotation needed for real qualification |
| C | Synthetic pose controls: translated/slow periodic motion, float precision, cut/gap contamination, coach permutation/duplication, donor exclusion, masked gradients, main-target invariance, no pose inference input | Invariance/gradient/timing/masking breaks | Actionable code; tests not run in this review |
| D | Frozen TRAIN-only12-record functional smoke, no DEV/TEST, minimal steps; stop if channels/gradients degenerate | Real target data fail validity or gradients | Labels + space + prospective bindings |
| E | Complete primary pose family P0/P1/Pshuf×3 seeds on one qualified cohort | Registered lighting effect or guards fail | Data + full-family storage/fit slots; not just GPU |
| F | Add secondary legacy P0/P1×3, no result-based omission | Secondary contrast fails | Same prerequisites; additional6 fits if old baseline not matched |
| G | R1 schema + synthetic A-B-A/C-B-C sections, return beyond64s, song reset/chunk seams, null/drop/shuffle, nonrepeat and constant-color controls | Memory absent/leaky; rename-only shuffle; constant reward; no training gradient | Substantive code plus repeat/target qualification |
| H | TRAIN-only12-record R1 smoke, then R1/no-memory×3 with paired ID ablations after cohort qualification | <20 repeats, memory cannot learn/use returns, guards fail | Repeat data, matched exposure, space and revised budget |

A synthetic memory learnability smoke should force distinct palettes/relationships at returns and include misleading/nonrepeat sections; a constant/flat target cannot show that memory works. It is an engineering instrument, not a scientific outcome. Freeze the test before running it. Do not expend full27-fit resources to discover that the loader has no valid pose or the repeat metric rewards constancy.

## 7. Matched budget and storage decision

**Current supplied accounting:** 952,976,108 scientific-campaign bytes; cap floor(0.95×2^30)=1,020,054,732. Remaining **67,078,624 bytes=63.97MiB**. Parent clarifies `/mnt` free1.254GB and root free6.31GB; maintain the mandatory1GiB floor on each output filesystem. The scientific cap governs new data/checkpoints/results; full-video delivery is separately authorized and needs sufficient media space. Recheck both filesystems and reserve peak bytes before any later run. A separate scientific-data directory is not a cap waiver.

Completed metadata: 15 fits,205,362,821 persistent bytes, summed fit time3,199.49s; queue wall time3,254.13s (54.24min). Rich shared P0 mean191.81s and13,679,636 bytes per fit; these are prior timings, not a guarantee for pose/R1. Existing reservation formula is `parameters*4*3 + steps*2048 + 8MiB`: rich P0 reserves35,578,088 bytes; pose adds only6,192 reservation bytes. Final/best plus logs use about13.7MB/fit, but transient/preflight reserve is about35.6MB. Pose validity arrays for122 records×640frames×(4 bool+float64 time) alone are about0.94MB before per-coach multiplicity, descriptors and provenance; this is not a complete artifact estimate.

**Fit-count trap:** 15 existing full fits already count against27, leaving12 slots. Three pending pose recipes×3 + R1×3 uses all12 only by assuming the old P0 baselines can be reused. They are not automatically matched to a new qualified target/cohort.

- Primary pose trio including a new P0:9 fits; nominal29min, about123MB persistent before data/predictions/demo.
- Full pose primary+legacy secondary:15 fits; nominal48min, about205MB.
- Add R1×3 while reusing that **new** P0 only if exposure/context/cohort truly match:18 fits; nominal58min, about246MB. Total original+new=33 fits: exceeds27.
- If R1 changes chronological exposure/context and needs its own new no-memory baseline:21 fits, about287MB plus memory-specific growth. A separately trained shuffled-ID model would add another3 if adopted prospectively.

These counts are planning alternatives, not fit authorization. A meaningful complete primary family does not fit63.97MiB even if one isolated fit might. Do not start a stranded partial family. Reconcile the global fit ledger and seek a prospective cap change rather than hiding runs in another study directory. Keep one student GPU job at a time, <=60min/fit and <=36GPUh overall unless explicitly changed. A practical future allowance is30–60min for the9-fit pose primary,1–2h for full pose plus small R1 fits **if the64s workload survives**; R1 full-song sequential work is unmeasured and requires its own bounded smoke. Old pose extraction took387.5s total for10 clips; that is historical runtime, not a reason to rerun or a quote for a different resolution/model.

### Required full-demo budget

The existing full Shpongle audit records277.3s,8319frames,1920×1080 AV1 with source audio. Its MP4 is **223,607,757 bytes** and render time193.53s. Reserve approximately250–350MB per retained full demo on root for media/audit output, plus encoder/transient peaks; three are approximately0.75–1.05GB. Root free6.31GB leaves about5.24GB above the1GiB floor, so this scale of versioned full-video delivery is feasible after current preflight. These explicitly requested media files need not be charged against the scientific0.95GiB cap. New scientific predictions/features/checkpoints/DMX evidence still require explicit accounting; moving those to a media directory is not a scientific-cap bypass.

Scientific storage remains the immediate fit blocker:9 matched pose fits need approximately123MB persisted (about56MB more than existing headroom) before qualification data, predictions and roughly22MB additional transient fit reserve. Full18-fit pose/R1 coverage needs approximately246MB, about179MB beyond headroom before those additions. A reasonable prospective scientific allowance is an additional0.25–0.5GiB for a bounded continuation, subject to exact repeat-feature/prediction estimates. `/mnt` currently has only about180MB above its1GiB floor, so cap authorization alone does not create enough physical space for the larger continuation. Provision more capacity or explicitly authorize archival/removal of specified non-study material; never delete immutable study artifacts or quietly place scientific data in a new dirname. Media/root preflight remains a separate pass.

Do not delete or overwrite old runs, checkpoints, media, or failed receipts. Code/doc-only work can proceed within its tiny explicit reservation. Fits/extraction remain blocked on scientific qualification, fit-count and storage gates. A new full demo using an explicitly authorized major LIGHT change can proceed in its versioned root media location when its own byte/free-space and inference/artifact preflights pass; it is not automatically blocked by the0.95GiB scientific cap.

## 8. Major-change demo and release contract

Before each changed-light fit/integration, declare which candidate checkpoint/seed/role will be rendered (for example fixedseed0 selected role), rather than choosing a winner after seeing numbers. After each major LIGHT change, render the **whole** Shpongle track even for a negative result. Preserve each numbered demo and its audio/checkpoint/code/DMX hashes. A 20s preview or contact sheet does not substitute for the full requested deliverable.

The user's Sep22 feedback is binding: light movement is good; missing color transitions and beats must be learned by the model, not repaired with manual rules or called generalization from this single demo. The companion body and pan/tilt/zoom/gobo/strobe and every other protected byte remain identical to the accepted stream; only the declared RGB and bounded nonstrobe dimmer channels may change. No dancer retiming/retraining. A fixed safety/visualization adapter is not evidence that the model learned color or beat control. Record finite output/flash bounds and exact byte parity; native1080p AV1, audio and full decode; use the unchanged renderer/camera/exposure for comparison. Human audiovisual review is separate from proxy scores; no default promotion follows from this review, the experiment, or a demo. If storage blocks the required demo, report that block before declaring a major product iteration complete.

## Primary evidence paths

- `docs/research/2026-09-28-end-to-end/VAULT_INVENTORY_V1.json`
- `docs/research/2026-09-22-color-beat/CAMPAIGN_DESIGN.json`
- `docs/research/2026-09-22-color-beat/preregistration_payload.json`
- `docs/research/2026-09-22-color-beat/OBSERVER_PROXY_PROTOCOL_V1.json`
- `docs/research/2026-09-22-color-beat/OBSERVER_FIT_DEFINITION_V1.json`
- `docs/research/2026-09-22-color-beat/OBSERVER_RESULTS_V1.md` and `OBSERVER_RESULTS_SUMMARY_V1.json`
- `docs/research/2026-09-22-color-beat/PALETTE_SECONDARY_RESULTS_V1.md`
- `outputs/stage0a/pose_validation.json`; `run_pose_validation.py`; `lightsem/motion/pose2d.py`
- `lightsem/experiments/light_layers/{model,training,evaluation,palette_metrics}.py`
- `lightsem/audio/structure.py`
- Original campaign `observer_fits_v1/*/complete.json` (resource metadata only)
- `outputs/shpongle_layers_demo_20260922T161917Z/shpongle_experimental_full_1080p_av1_{render,audit}.json`
