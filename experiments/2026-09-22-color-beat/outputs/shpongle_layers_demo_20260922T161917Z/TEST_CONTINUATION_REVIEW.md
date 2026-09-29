# Testing continuation review — 2026-09-22

**Recommendation:** complete the prespecified but unreported **palette-history secondary comparison**, using the six existing `S_VR_P0` checkpoints (selected/final × seeds 0/1/2). Do not train again. First freeze the missing track-construction and calibration definitions, add the narrow integration tests below, then preregister before reading new scientific results. This is an exposed-DEV **observer-distribution diagnostic**, not a LIGHT-quality or human-preference test.

This review ran no inference, scoring, GPU job or test suite. It read frozen code/protocols and artifact metadata. It did not open held-out TEST raw inputs, change frozen sources, or write the vault.

## What is actually available

Root `A=/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z`.

- `observer_training_manifest_v1.json`: the frozen 91 TRAIN / 31 historical DEV inventory, with nine TRAIN identity exclusions. Preserve it exactly.
- `observer_train_transforms_v1.json`: fitted audio normalization, thresholds and **32 × 72** TRAIN palette prototypes. It contains **no temporal/marginal IQR calibration**.
- `observer_evaluation_binding_checkpoints_v1.json`: 30 bound checkpoints. The narrow palette question needs only `02_seed0_S_VR_P0_{selected,final}`, `07_seed1_S_VR_P0_{selected,final}`, `12_seed2_S_VR_P0_{selected,final}`. Paths are `observer_fits_v1/{03_seed0,08_seed1,13_seed2}_S_VR_P0/{best,final}.pt`.
- All **930 prediction NPZs** under `observer_parallel_predictions_v1` have the same 34-member schema. They include argmax `generated_palette_ids`, `previous_ids`, `generated_color_cues`, color/accent curves and clocks, for model/donor. They do **not** include `context10`, full mark probabilities, null-state draws, or draws 101–104. Archive member names were checked, not scientific outputs scored.
- Consequently, existing predictions can freeze the cue inventory, but **new forward inference is necessary** to recover context and sample the four paired draws. It need not use GPU and does not need retraining. Do not pretend the saved argmax trajectory is the prescribed stochastic comparison.
- Frozen `evaluation.py`, `palette_metrics.py`, `model.py`, `training.py`, and `targets.py` still match the registered hashes. No palette calibration/score artifact was found in this campaign.

## Frozen mechanics to preserve

Sources: `OBSERVER_FIT_DEFINITION_V1.json`, `observer_evaluation_schema_v1.json`, `PALETTE_METRIC_DEFINITION_V1.json`, and `evaluation.palette_rollouts`.

1. Use the same 64-second, 2560 × 94 normalized audio-feature window and 640-step requested 10 Hz palette grid. This is offline full-source feature construction, **not causal or 64-second raw-audio context**. Keep source PTS separately. Never use visual/pose/reference tokens as model input.
2. Use FP32 inference, `model.eval()`, disabled autocast/TF32, existing weights/vocabulary/normalizer. No head recalibration or checkpoint reselection.
3. Fixed common audio-only cues, beginning at index 0 for display initialization; initialization is not a color event. Generated cues use the frozen 0.5 threshold / 0.5-second rollout spacing, **not** the event-AP scorer's 0.2-second candidate NMS. Do not use reference event counts to choose cues.
4. `palette_rollouts(..., draw_seeds=(101,102,103,104))`: inverse-CDF sampling with exactly paired uniform noise at each cue. Own-state feeds the last generated token; null-state feeds ID 32 at every cue. Both hold between cues and can reselect the same mark. Both branches were trained through whole-sequence state dropout 0.5. No teacher forcing.
5. Selected checkpoints remain the primary secondary report; final checkpoints are a separate companion. Average paired draws/recordings/training seeds within canonical song; 10,000 paired song bootstraps, seed 0. Draws and checkpoint roles are not extra independent samples.
6. Frozen palette math: 36-projection circular joint-HS sliced W1; temporal components are log1p residence, changing-mark prototype distance, mean raw-HS intercue drift, and **exact 8-second** HS change. Use all four TRAIN-IQR-scaled errors, equally weighted, plus the separately scaled marginal error. Expose censoring, no-change flags, coverage and pair counts. Never average surviving components, add epsilon scales, or refit on DEV.
7. Use **full 64-second trajectories** for this secondary instrument. An 8-second center crop cannot provide an 8-second lag pair; silently scoring center8 would make the aggregate undefined. The resulting scope differs from the center8 event report and includes window-edge context limitations.

## Missing prerequisites — freeze before any new scoring

There is currently no palette CLI or end-to-end track adapter. The module functions alone are not an executable study. Parent must prospectively bind these choices, rather than label them already frozen:

- **Cue inventory:** recommend `0 ∪ saved model__generated_color_cues` from each matching `S_VR_P0` checkpoint/seed/recording. Hash the resulting six inventories before rollout. Each own/null/control comparison uses exactly that inventory. No best-seed or best-role choice. Cross-recipe palette comparisons should wait; they need a separately fixed common `S_VR_P0` cue policy.
- **Reference construction:** `PaletteTrack` needs one reference token per common cue, normalized observed HS at every valid frame, relative regular-grid timestamps, and explicit zero placeholders only on invalid frames. Freeze whether reference cue tokens use point assignment or an observed-interval summary; define invalid-at-cue handling so a zero placeholder is never interpreted as palette 0. Preserve the frozen KMeans squared-Euclidean assignment if point assignment is chosen. This is not currently specified by the metric definition. Do not replace the raw reference HS trajectory with prototype holds; that would erase real drift and artificially favor held outputs.
- **Calibration construction:** create a content-hashed `PaletteCalibration` from the exact 91 TRAIN records only. Freeze the TRAIN reference cue rule, validity and assignment rule first; one reference per song/record, not repeated by checkpoint or draw. The module's hierarchical inverse-CDF IQR weighting is already fixed. Every TRAIN record must define all four components, and every component/marginal IQR must be positive. On failure, stop with a blocked/undefined receipt; no posthoc record dropping or metric repair. Existing normalizer/vocabulary fitting is not this calibration.
- **Common support:** freeze the exact external mask (color/frame validity and registered gaps/cuts policy) and the source-PTS-to-requested-grid adapter. No bridging gaps or model-dependent abstention. Freeze whether the secondary requires at least 80% valid full-window time and at least 30/31 DEV groups; those are a proposed conservative secondary support rule, **not an already passed or inherited palette rule**. All arms/draws need identical support and defined lag/drift components. Original 80%-of-40 qualification remains failed regardless.
- **Controls:** mandatory self-floor, separately reported quantized-reference floor, constant persistence, null-state independent cue-held sampling, distinct-canonical donor, and diversity/occupancy. Freeze diversity/occupancy formulas and the donor intervention precisely. The existing tempo donor map is many-to-one and replaces non-rhythm columns 0:86 while retaining 86:94; it is **not a shuffled permutation or full rhythm-preserving intervention**. It is usable as a separately named limited donor control, but cannot silently satisfy the original unspecified shuffled-palette control. Preserve cue/noise/dimmer equality.
- **Statistics/gates:** prospectively declare the secondary contrast family and correction. Original palette thresholds remain ≥15% **and** ≥0.02 normalized error reduction vs independent cue-held, positive corrected paired CI and better than shuffled donor; diversity/occupancy ≥70% reference, marginal-error degradation ≤5%, color-AP decline ≤0.01. The observer protocol calls this secondary/exploratory, so report all components and intervals without upgrading it to the original qualified LIGHT claim. Fixing raw color-head cues makes head-AP equality an architectural invariant, not proof that emitted color changes retain timing.

The useful immediate work is the new adapter/registration plus TRAIN calibration. If calibration fails, that is a substantive instrument qualification result; do not spend inference compute to produce an undefined aggregate.

## Only new blocking integration tests worth adding

The registered CPU receipt already passed **254 tests**, including 21 palette-metric tests. Existing tests already cover state sensitivity, own/null paired noise, no teacher history, common cues, TRAIN-only fitting, weighted calibration, all four components, gaps, degeneracy, hash tampering, checkpoint loading, protected DMX bytes and mask/color synthetic controls. Do not rerun the whole suite as scientific progress.

Add synthetic tests in a **new sidecar**, without editing frozen modules:

1. Real-model/adapter round trip: checkpoint metadata → FP32 context → fixed saved cue inventory → four own/null draws → `PaletteTrack` → scalar report. Compare saved color cues/logits on replay before accepting regenerated context; fail on provenance mismatch. Prove that changing reference labels cannot change rollouts.
2. Reference/support adapter: nonzero source offset, irregular source PTS, invalid cue start, internal gap, known cut, and all-invalid interval. Test explicit no-bridge behavior and non-teacher-forced reference quantization. This seam is not covered by the separate module tests.
3. Rollout fail-closed validation: nonfinite contexts/logits/probabilities and out-of-range IDs must fail. `palette_rollouts` itself lacks the explicit finite-logit check present in `predict_window`; a new wrapper must guard it. Test actual model eval/precision setup, not only the existing deterministic SpyModel test.
4. Complete-report inventory: missing draw/seed/role, altered cues/vocabulary/calibration, wrong partition, different arm validity, and output overwrite all fail. Require a measured storage reserve before creating anything. Synthetic null/no-effect and constant/flicker controls must not manufacture an aggregate pass.

**Implementation interface needed (does not exist yet):** a new `run_palette_history_secondary_v1.py` with `schema`, `calibrate`, `predict`, `score` phases; immutable registered JSON inputs; exclusive output creation; explicit CPU device; separate selected/final outputs; no training entry point. Keep code outside frozen source snapshots. Its registration must bind original artifacts plus the new definition, adapter/tests, cue inventories and calibration. The current `run_light_layers_evaluate.py` only offers `schema`, `prepare-baselines`, `predict`, `score` for the event study; running its `score` again does not evaluate palettes.

Native invocation template **after implementation and registration**, not a current runnable command:

```text
CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -m pytest -q <new-sidecar-test-path>
CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u <new-runner> calibrate --registration <new-frozen-binding> --output <new-exclusive-path>
CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u <new-runner> predict --registration <new-frozen-binding> --device cpu --output <new-exclusive-path>
CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u <new-runner> score --registration <new-score-binding> --output <new-exclusive-path>
```

## Why this is not R1 or the other original tests

Palette history is **last-generated-token state at successive cues**; repeated selection can persist across cues. R1 is audio-derived **repeated-section identity/memory** across returning sections, with no-memory and shuffled-repeat-ID controls and ≥20 songs with two qualifying returns. The model has no section-ID/memory input and the trainer does not even list `S_VR_P0_R1`. R1 needs its own qualified repeat cohort, implementation and future fitting authorization. Existing P0 checkpoints cannot test it. Pose P1/Pshuf arms likewise were not fitted by the completed 15-fit observer campaign.

### Independent label search (metadata first; no label values opened)

- **Mask:** no independently annotated human-mask audit was located in this campaign, repository docs/outputs, or the Just Dance metadata search. The original visual smoke payload explicitly says independent mask annotations are absent. Saved SAM/box summaries and audit PNGs are outputs, not independent truth. Original ≥120 frames / ≥30 videos, real-mask IoU and moving-mask nuisance gates remain blocked. A synthetic composite can only earn a synthetic claim. Do not use another automatic mask model as truth.
- **Beat labels exist:** `.../night_expression_20260909/rhythm/guitarset/annotation/` contains **360 `.jams` files**, with local `audio_mono-mic/`. This is not a newly untouched cohort. `rhythm/beat_this_reference/` already contains **358 saved predictions and `report.json`** from the earlier final0 benchmark. BeatThis final/small trained on GuitarSet comping, with related musical content in other versions. Repeating that benchmark would not pass the original unseen-song overlap gate or be a new substantive test.
- **Raveform candidate independent annotations exist:** `/mnt/lightsem-data/raveform/raveform/structures/beats/` contains **1423 `.beat.csv` files**, with `structures/segments.json`. The local research review identifies this as the expert metrical/structure subset. The bundle also has 68,078 files in `beats/`; do not assume those broad files are independent expert truth. No audio files are present in the Raveform bundle. Next permissible action is an annotation-provenance/audio-availability/canonical-overlap metadata join, explicitly excluding sealed TEST; no label scoring yet. Need ≥20 independently grouped annotated songs, exact-audio timing/precision/rights, and checkpoint training-overlap clearance before a new rhythm comparison. Do not fabricate labels with another tracker.

## Resource bound

Observed metadata: campaign **952,976,108 bytes**; cap `int(.95 * 2**30)` = **1,020,054,732**, leaving **67,078,624 bytes**. `/mnt` free **1,279,082,496 bytes** (decimal 1.279 GB); retaining 1 GiB leaves **205,340,672 bytes**. The campaign cap, not mount slack, is tighter. Recheck immediately before every phase because the demo may change free space.

Use one CPU process, bounded SourceStore RAM, and stream one recording/context at a time. Six checkpoints ×31 records =186 recipient forwards; an audio-donor condition needs up to186 additional forwards. CPU wall time is unmeasured; record it honestly and bound the job, rather than promising GPU-like timing. No GPU authorization is needed for a CPU run.

Persist compact uint8 token IDs and scalar metrics, not derived HS/context tensors. Four 640-frame conditions × four draws ×186 cells take **1,904,640 token bytes**; worst-case float64 cue uniforms take **3,809,280 bytes**. With previous IDs, hashes, calibration and compact reports, a conservative **32 MiB reservation** is feasible now, subject to actual preflight. Reconstruct prototype HS in RAM. Saving all30 contexts alone would take **304,742,400 bytes**; duplicating full generated HS for two branches/four draws would take **1,371,340,800 bytes** and is forbidden by current limits. No source copies, new masks, renders, deletion, overwrite or cap bypass.

## Conclusions that must not change

**Original source coverage remains 31/40 =77.5%, below80%. Paired-null support remains 29/30.** New palette secondary work cannot repair either completed event-study failure. It cannot qualify the original nine hypotheses, establish clean LIGHT truth, claim unseen confirmation, or authorize default promotion. Preserve all old results and failures, even if a new secondary contrast looks favorable.
