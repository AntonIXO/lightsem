# Frozen structure-intensity protocol candidate V1

**PROPOSAL ONLY. Not registered, authorized, or run.**

Scope: row8 only. The parent must review and bind this protocol before any scientific input read or model call. The JSON companion is the detailed machine-readable specification:
`PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V1.json`.

## 1. Question and claim limit

Measure the missing drop/breakdown intensity contrast for the three exact surviving historical JD checkpoints. This is a **new retrospective draw** on exposed legacy material. It is not a replay of saved predictions, fresh TEST evidence, or a rescue of failed BAS/shuffle results.

The instrument is the historical model and `ConductorHybrid`, with the canonical14-fixture rig, packed DMX, and decoded mean `dimmer * max(HSV_value, 1e-6)`. It measures a DMX intensity proxy under cached heuristic role labels. It does not measure physical luminance, video brightness, human preference, or independently annotated musical sections. Fixture encoding quirks remain part of this instrument.

Original gate, unchanged:

> G-struct-contrast: rendered light intensity ratio drop/breakdown within 1.3x of the corpus 1.68-2.53

Proposed explicit interpretation: `R ∈ [1.68/1.3, 2.53*1.3] = [1.2923076923, 3.289]`. The old wording did not settle aggregation or CI treatment. The parent must bind this interpretation before outcomes. Report the point estimate and95% CI, plus separate point-in-band and CI-entirely-in-band flags. Neither flag closes the original hypothesis.

The published1.68–2.53 is a **movement** contrast from another corpus. The JD role builder explicitly calls its labels unvalidated. The originally promised independent Oslo-boundary validation remains absent. Do not silently replace it with this heuristic-label measurement.

## 2. Exact models and lineage

| Arm | Checkpoint | SHA256 |
|---|---|---|
| none | `light_trunk_jd_none_smooth.pt` | `9420bfb431fa8fea0d7b463284945665a9b860c94f3cfef1cc3f5c3a0b7e04ad` |
| struct | `light_trunk_jd_struct_smooth.pt` | `af2aefcd49f8164b5c175b53d672d67d5af870af0ffde51f08c126f2f3628a63` |
| shuf | `light_trunk_jd_shuf_smooth.pt` | `4e043491aae00067f69b2463ae5163baaefa9c8493ef730f5fa2e62107c6308e` |

All reside under `/mnt/lightsem-data/ckpt/`. They were byte-hashed, not deserialized. Only `none_smooth` also has a matching independent Sep7/Sep8 checksum receipt. Other hashes pin the current surviving bytes, not a historical signed run manifest.

Historical source anchor: `e11900dd2917ca0f97b9ba8a38cd32e997bfdd4d`. Use historical `light_trunk.py` blob `1d2bbb9c04805cb2353d6c69babc86a4478a72d1`; current source adds an attention-mask cache. The JSON lists source hashes, historical blobs, all453 metadata rows, and calibration reports. Pin transitive dependencies before execution.

Use checkpoint `mu/sd`, block names, event thresholds, and smoothing without refit. Expected model: d256,6blocks,4heads, window160, conv15, A4/B3/C17, AR flow, causal trunk; input92 for none and100 for struct/shuf. Verify these after approval. Both unscale and conductor squash use the same hashed `lightsem/conductor/scalers.json` (`c3c076b4c3bd9d830ca2eaf4352304e5ab5f6d705ae564e3ba1db5ae0ef3c6d4`). Missing constants mean STOP, not recalibration.

Historical input normalizers and event prevalence used the whole tensor, including oldTEST. Preserve that limitation. `train_jd_none.json` says steps0 because the report was overwritten by eval-only reporting; the chain documents the existing12000-step checkpoint.

## 3. Recovered cohorts and exclusions

Only the `meta.npy` scalar JSON member was read from each historical tensor. No numeric tensor arrays were read. All three contain the same ordered453 clips /294 legacy song keys. Metadata reproduces352TRAIN,39VAL,62oldTEST, and exactly the original40 report clips in order.

- Original-exposed descriptive cohort:40 clips,31 recordings,29 current canonical families.
-13 clips /7 recordings /7 families have known historical TRAIN canonical aliases.
- Sensitivity cohort:27 clips /24 recordings /22 families without a known oldTRAIN canonical alias.
- All scored recordings and recovered original shuffled donors map to current Stage11 TRAIN. No current Stage11 TEST canonical alias was found.
- FineDance name-only screening has no substantive title match, but uninformative name `N`, abbreviations/transliterations, and absent cross-corpus authentication remain unresolved. No title hit is not audio dedup proof.

The parent must approve or exclude **every accessed recording**, including shuffled donors, before numeric payload reads. The donor union stays inside the original31 recordings. Use the recovered original mapping, never a new convenient donor. If exclusion breaks an arm, stop for a preregistered cohort amendment; do not replace it after predictions.

The sensitivity label means only “no known old gradient-TRAIN family alias.” It does not undo preprocessing exposure, oldTEST evaluation, donor exposure, or unknown cross-corpus identity.

## 4. Safe targeted loader

**Do not call `lightsem.model.light_data.load_tensor`.** Lines134–142 eagerly materialize every NPZ member. `run_stage1_eval.py` calls it and also reads reference poses and repeats completed metrics. Do not run that script unchanged, any trainer, calibrator, tensor builder, or structure builder.

The new eval-only harness must use the JSON metadata index. Its allowed scientific read union is:

- The exact3 checkpoints and hashed scaler JSON.
-31 explicit approved `cache/justdance_audio/<recording>.npz` files.
-31 explicit approved `cache/justdance_structure/<recording>.npz` files, including all donors.
- **Zero** numeric training tensors and **zero** reference-pose files.

`load_bus(explicit_path)` loads only that recording, not a corpus. It reads all its channel arrays, times, mel, beats, downbeats and metadata. For structure read only required arrays and metadata from approved paths. Check resolved paths and hashes; reject unknown symlinks. No glob loader or fallback.

`LightModel` must receive the explicit checkpoint, never the default. `load_scalers` raises `FileNotFoundError`; it does not run the builder mentioned in its error text. Preserve stop-on-missing. Audit imports and log every input path before execution. Old training lineage does not require rereading all453 training clips.

No new real-motion-mapped arm is included. Original G-struct-contrast does not require it, and it would need extra reference-pose access. This decision is made before results. Keep3 learned arms, beat-trigger, constant null, shuffle control, all cohort strata and label-validity limitations.

## 5. Exact clocks and roles

For each clip, use frozen tensor-metadata `T` and index `raw_start_s`. Let `i0 = round(raw_start_s*40)`.

- Features and real structure conditioning use indices `i0:i0+T`.
- Local frame clock is `i/40`.
- Target role and section assignment uses cached index `i0+i`, corresponding to `(i0+i)/40`, **not** `raw_start_s+i/40`.
- Record raw offset, rounded origin, and their difference. Its magnitude must be at most0.0125s, apart from1e-9 arithmetic tolerance.
- Preserve legacy beat filtering/subtraction by **raw** offset. Do not silently phase-correct this mismatch.
- Donor conditioning starts at donor index0. Outcome roles always come from the target recording.

Validate40Hz, finite/aligned features,64mel bands,28named scalar channels, and float32 clock agreement within1e-4s. Actual bus length must equal frozen `T`; no automatic truncation, audio padding, or retiming. A mismatch requires a pre-output eligibility failure and explicit amendment, not a favorable repair.

Preserve original structure transform:5role one-hot channels, energy, and `tanh(to_rise_s/30)`, `tanh(to_drop_s/30)`, with its documented NaN-to-zero handling. Preserve original conditioning-tail zeros if a donor is short and log the support. Padded conditioning is not observed role truth.

Outcome roles require valid, nonoverlapping cached segment coverage. Mark gaps invalid rather than intro. Before prediction require at least95% target-clip valid role coverage and at least80 retained frames each for drop and breakdown, using segment intersections of at least80frames. Freeze and publish all40 eligibility decisions. Missing support remains an inconclusive component; do not choose new songs.

Report the cached `oracle_beat` flag. These whole-track controls do not establish causal deployment.

## 6. New draws and fixed mapper

Use one new seed0 common-noise draw per clip. Reinitialize the same-device generator at seed0 for each of the3 arms. All must use C17 and the same noise-call shapes/order. Record runtime versions, deterministic flags, device, generator-state hashes and call shapes. Original RNG states/runtime were not saved: do not assert exact historical predictions.

Preserve chunk2400, full prime `6*(159+14)+1 =1039` frames,3Euler steps, clamp6, original AR state reset per chunk, and overlap-context sampling. No seed search or cheaper chunk substitution. One draw supports song-cluster uncertainty conditional on that draw, not style-seed or training-seed uncertainty.

Freeze historical `ConductorHybrid`, canonical14-fixture `club_rig`, and the exact conductor config in the JSON. Retain checkpoint thresholds/smoothing and external scalers. Produce packed DMX in RAM, decode scalar intensity, then discard DMX. No renderer, DMX cache, video, or per-section brightness matching.

Beat-trigger uses the same bus and rig,40Hz and delay0. Its existing per-clip audio percentile normalization is a fixed rule, not a newly fitted baseline.

## 7. Auditable outputs

Save, for every approved clip:

1. Four float32 scalar intensity vectors: none, struct, shuf, beat-trigger.
2. Shared int32 frame index, uint8 target role (`255` invalid), uint16 section ID (`65535` invalid).
3. Identity, clock, hash, source, old-fold/alias, eligibility and sampling metadata.
4. Per-section counts, sums and means for **all** roles/arms, plus rejected sections and reasons.

Compute section sums in float64 from saved float32 vectors, so the report can be reproduced from the archive. Verify finite intensity in[0,1] within1e-6; do not clip or replace failed outputs. No descriptors, pose, hidden activations or full DMX are saved.

## 8. Estimator, paired CIs and nulls

Within each eligible clip, divide retained drop mean by retained breakdown mean. Breakdown at or below1e-8 gives undefined ratio, not an epsilon-inflated value. Average log-ratios equally over coaches within recording, recordings within canonical family, then canonical families; exponentiate for geometric mean ratio.

Report original-exposed, no-known-oldTRAIN-alias sensitivity, and oldTRAIN-alias strata separately. Use matched clips/masks across arms. Report every undefined denominator and any complete-case attrition; never omit an arm or claim40 independent songs.

Use10000 paired canonical bootstrap resamples, seed20260928, two-sided percentile95% CIs. Under2 eligible families means undefined CI; under10 means low-support descriptive evidence. Primary output is `R_struct` with band comparison. All improvement/null contrasts are exploratory unless the parent binds multiplicity control before outcomes.

Required comparisons:

- Each arm versus constant positive-intensity ratio1 and the published1.68–2.53 range.
- Paired struct-minus-none, struct-minus-shuf, and struct-minus-beat-trigger log-ratio contrasts.
- Four fixed within-clip circular role shifts at rounded fractions0.2,0.4,0.6,0.8 of `T`, applied jointly to retained role/section/valid masks against saved intensity. Preserve mask counts. Report all4 shifts and their mean; do not count shifts as independent songs.

The circular-role null is an added score control. It does not replace the frozen shuffled-structure model or validate musical labels.

## 9. Budget and stop rules

Parent shared scientific headroom:33,524,192 bytes. Row8 hard cap:12,582,912 bytes (**12MiB**).

| Payload | Uncompressed bytes |
|---|---:|
|4float32 intensity traces over371,393 frames |5,942,288 |
|Shared int32 clock +uint8 role +uint16 section |2,599,751 |
|Headers, sections, identity, sources and logs allowance |4,040,873 |

No compression savings are assumed. Atomic writes must not duplicate a whole archive. No checkpoint copies. Preserve at least1GiB free on `/mnt`; before each phase require the floor plus remaining peak-write reservation.

With a separate1MiB row11 reservation, these two candidates use13MiB and leave19,892,704 bytes for all other science. The parent owns the combined allocation.

This **explicitly supersedes** the earlier row8 sums-only2MiB and joint4MiB estimates in `NEW_FROZEN_LEGACY_EVAL_NOTES_V1.json`, which now points here. It also fixes the no-reference-pose comparator choice before results.

No neural, linear/ridge, PCA/scaler, threshold, or smoothing fit is allowed. No completed BAS/FID/diversity,15event fits, palette186, poseG3, decoder sweep or student training is repeated.

Required before execution: parent binding and validated native receipt; full input+donor identity qualification; immutable source/input/checkpoint checks; new targeted harness read audit; fixed clock/role eligibility; separately approved runtime validation and smoke forward. None has been run by this proposal.

## 10. Runtime and memory admission — proposal, not measurement

The parent must check the remaining **36GPU-hour cumulative ledger**. The existing60min-per-fit rule is not permission for unlimited zero-fit evaluation.

- One GPU job at a time. No overlap with training, evaluation, or rendering.
- Separately approved engineering smoke: one12s/480frame clip, all3arms, **5min total**. It runs only after protocol binding and input clearance. Record timing, memory and reproducibility receipts, not section scores or favorable-seed selection.
- Each model arm: **60min maximum**. All3arms: **3GPU-job hours maximum**, plus the5min smoke.
- Whole-job host RAM: **4GiB ceiling**. VRAM: **8GiB ceiling**. These are proposed policy limits, not measured peaks or guarantees.

Static metadata workload for the original40clips:371,393 output frames;173 trunk chunks per arm;138187 repeated prime frames;509580 AR-sampled frames per arm. With3Euler steps this gives1528740 velocity calls per arm, or4586220 across the3models. This is an operation-count estimate, **not throughput measurement**.

A12s smoke has no full1039frame priming and does not establish the memory cost of a3439frame input chunk. Extrapolation must account for the fullchunk/priming workload and load/runtime overhead. If the time or memory bound is unresolved or exceeds the caps, **STOP**. Do not reduce the seed/cohort, omit a component, change chunking/precision, or start another unbudgeted job. A different plan needs a parent-approved amendment before outcomes.

## 11. Metadata-field exposure chronology

This proposal does **not** claim zero TEST label-field access. At18:48:43.037Z on2026-09-28, the child parsed the entire FineDance `split.json` into memory, including all203 entries and TEST label/style metadata. This was not a projected-field parser.

At18:48:52.275Z, it printed `entries[0]`: key/source ID001, canonical `fd-7a3e6b16cb75f48f`, partition **TRAIN**, label `frames=2927`, `name=feedback`, `style1=Street`, `style2=Jazz`, plus duration, path and hash metadata. Style fields were not used in identity matching, cohort selection or scientific analysis.

The name-only instruction arrived at18:54:40.584Z. The child disclosed the parse/display to its **immediate parent** at18:54:52.166Z, not directly to the root. The send receipt was not printed or bound; no receipt ID or delivery timestamp is invented. Subsequent alias screening used the fixed `NEW_IDENTITY_EXCLUSION_METADATA_V1.json` projection.

The JSON companion records exact rollout path,1-based lines26–29,71–73, entry IDs and timestamps. This chronology was checked from recorded output only; raw labels were not reopened. It establishes metadata-field exposure, not numeric TEST-array, waveform, pose, model-forward or scientific-metric exposure. Do not broaden or understate it. Future identity checks must use the fixed projection table, not reparse raw FineDance labels.

## Metadata field exposure correction — 2026-09-28

See `METADATA_FIELD_EXPOSURE_DISCLOSURE_V1.md/json`. Full FineDance split/manifest JSON was parsed, including embedded label metadata across partitions. The confirmed style display was **TRAIN entry001**. Therefore earlier blanket “no TEST label access” assurances are too broad for metadata; they are not asserted here. No numerical TEST array read, model call or new scientific metric is established by these events.

The child notified its immediate parent at18:54:52.204 UTC; the root was not explicitly informed then. The root correction was sent at19:09:38.443. Identity-only filtering applies going forward and does not erase the earlier access. Future lookups use only the already-created identity-only table.

## Qualified-gate boundary

This measures an **EMITTED-intensity proxy**, not image-rendered or calibrated photometry. Cached JD roles remain unvalidated against the original Oslo hand-marked requirement. Completing this numerical proxy does **not** complete the original qualified `G-struct-contrast` gate. The original negative BAS/shuffle result stays unchanged. The factor-interval interpretation is a proposal to bind before outcomes, not an invented post-result threshold.
