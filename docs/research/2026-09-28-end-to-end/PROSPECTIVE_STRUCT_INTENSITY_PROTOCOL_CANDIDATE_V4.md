# Structure-intensity protocol candidate V4

**Implementation only. Preparation has not run. No model, smoke, scoring or TEST access is authorized.**

This document and `PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V4.json` are the V4 pair. Every V3 file, plus V1 and V2, remains unchanged. V3 was held by static review before tests or payload execution. The JSON contains the frozen original40 clip inventory, historical lineage and full bindings. This is a diagnostic preparation tool, not a product or default change.

## 1. Claim and stage boundary

The proposed later measurement remains **whole-rig heuristic-role-conditioned EMITTED DMX-intensity contrast** for the three exact historical smooth checkpoints: none, struct and shuf, against beat-trigger and a constant-intensity null. There is no blinder-only, C-null or reduced-model substitute.

Original gate wording remains: “rendered light intensity ratio drop/breakdown within 1.3x of the corpus 1.68-2.53.” The proposed emitted proxy cannot complete rendered photometry or the original independent Oslo hand-marked section validation. The published range concerns movement, not a measured physical-light target distribution. The earlier BAS/shuffle negative results remain unchanged. The factor interval `[1.68/1.3, 2.53*1.3]` is a prospective descriptive interpretation, not a recovered historical threshold or a new gate closure.

The only implemented CLI stage is `prepare`. It has no model construction, forward, evaluation, ratio, bootstrap or GPU stage. It cannot start a smoke or study after a successful preparation receipt. Future scientific predictions must be separately preregistered before any model outputs, including engineering outputs.

## 2. Historical descriptive admission, not fresh TEST

Root prospectively accepts descriptive use of explicitly hash-bound **currentStage11TRAIN** recipient/donor files with no known currentStage11TEST alias. This permission does not establish authenticated cross-corpus separation. Unknown aliases, old whole-tensor preprocessing, old evaluation and donor exposure remain limitations. No freshTEST, independent-generalization or zero historical contamination claim is permitted.

Keep the original40 clips/31 recordings/29 canonical families; the27-clip/22-family no-known-oldTRAIN-alias sensitivity remains exposed. Preserve the13-clip/7-family oldTRAIN-alias stratum. Hashes authenticate the selected bytes; they do not prove that recordings are independent.

The exact62 allowed scientific input slots are31 audio-bus NPZs and31 target/donor structure NPZs. Root must fill their SHA256/size bindings and approve each slot before preparation. Every donor is the recovered original donor; never replace a donor or song. No TEST payload or raw corpus manifest is allowed. Only the frozen identity-only projection is read. The earlier whole FineDance label-bearing metadata parse remains disclosed in the JSON; future field restrictions do not erase it.

## 3. Explicit future preparation authorization

The preparer requires one root-generated authorization JSON bound to104 native synthetic checks (source authored, unrun) and:

- Exact V4 plan, preparer and test hashes and exact `argv`.
- Successful native synthetic-test receipt with exact source/plan/test bindings, explicit passed count, zero failures/errors, zero real-model forwards/checkpoint loads/TEST reads, and hash-bound log.
- A separately validated native preparation-registration receipt. Root must inspect its semantic success, not only process exit0.
- Every source and checkpoint reference in the plan, the projected identity table, and all62 approved input references.
- An unchanged original-campaign metadata inventory, all active reservations, and externally enforced preparation process-tree limits:900s wall,4GiB host RAM, no GPU.

CUDA must be hidden and bytecode writes disabled. Native tests require the exact published command string and exact104-test count, no skipped/expected-failure cases, and a hash-checked verbose log ending in plain OK. The native registration receipt remains a root-verified trust boundary, not an independent vault-semantic check by this driver.

There is no unsafe deserialization or missing-file repair fallback. The new packet directory must not already exist; partial failure requires root review, not automatic cleanup/resume/overwrite. Preparation stdout/stderr goes to a new internal `preparation_v4.log` capped at64KiB; every append, including final status, is cap-checked. Do not redirect preparation output to another uncounted log. The separate synthetic-test log is also counted in the source packet.

Preparation authority is not scientific evaluation authority. The parent must first test and bind this implementation. No source in this delivery has been imported, compiled or executed by its authors.

## 4. Source and CPU checkpoint qualification

The historical trunk was retrieved with read-only `git show 1d2bbb9c04805cb2353d6c69babc86a4478a72d1`. Its content Git blob was verified. It is stored under the new output directory, not checked out over the current trunk. Historical infer, hybrid, v0, rig, packing/decoding and JD conditioning references remain pinned. Current trunk attention-mask caching is not silently substituted. Existing source is referenced, not imported by preparation.

The future preparer uses `torch.load(..., map_location='cpu', weights_only=True)` with only a fixed NumPy reconstruction allowlist needed by the original checkpoint format. Unsupported serialization stops; `weights_only=False` is never used. It creates no model object and runs no project source. It checks exact checkpoint top-level fields, complete state-key/shape/dtype schema, input dimensions92/100, A4/B3/C17, config, finite float32 `mu/sd` with positive `sd`, ordered block names, and exact frozen thresholds/smoothing. Checkpoint tensor values are not passed to any model or scientific score.

A later full-study implementation must preserve the same checkpoint/scaler/source identities. `onset_percentile=88.0` belongs to the `ConductorHybrid` constructor, not the `ConductorConfig` dataclass. V4 separates it. Missing constants or incompatible checkpoint schema mean stop, not recalibration.

## 5. Exact recipient clocks and role support

For each frozen row, let `T=T_metadata`, `raw=start_s_metadata`, `i0=round(raw*40)`. Every later arm and beat-trigger must use exactly recipient indices `i0:i0+T`. Full-recording length is not clip length. Preserve the raw offset for legacy beat subtraction and save the rounded sample origin and half-frame roundoff separately.

Preparation keeps both40Hz absolute-clock and adjacent-step tolerances at1e-4s, with monotonicity, mel64xN, all28 scalar lengths/finites and float32 representability. Bus duration may differ fromN/40 by at most one frame+1e-4. Insufficient bus support fails qualification. No silent `min()`, start clamp, audio padding, truncation or timing repair is allowed. Frozen tensor T, not an inferred new duration, is authoritative; this is not a claim of recovered historical predictions.

### Producer duration meanings: a final-endpoint exception only

V4 pins `lightsem/__init__.py`, `audio/features.py`, `audio/structure.py` and `build_justdance_structure.py` as schema-source evidence. For the producer's22050Hz,512-sample centered STFT:

- `L` is the last feature-frame center. The cache saves `duration_s=L` and `N=round(L*40)` role samples at `k/40`.
- `D=len(y)/sr` is the waveform endpoint used for the final `Segment.end_s` by the segmenter. For an integer hypothetical sample count, `L=floor(len(y)/512)*512/22050`, so `0 <= D-L < 512/22050` (at most511 residual samples).
- `N/40` is a rounded grid stop. It is not proof of waveform end and may round either above or below L.

The existing structure-duration/grid half-frame+1e-4 bound applies to **L**, unchanged. All segment starts and intermediate ends stay within `L+1e-4`; recognized labels, role-array agreement, positive lengths, order and nonoverlap remain strict. Only the **last segment end** may extend through `L+512/22050+1e-9`. The1ns is arithmetic roundoff, not a blanket timestamp tolerance. This conservative metadata upper bound does not permit an intermediate overrun, shifted start, overlap, wrong label or a final endpoint beyond one hop.

This is **METADATA_CONSISTENCY_ONLY_NOT_WAVEFORM_END_VERIFICATION**. The preparer does not read a waveform or authenticate same-source waveform-duration provenance. Matching IDs or arbitrary segment maxima do not independently establish D. The target/donor receipt records L and the signed final-endpoint-minus-L, with that limited scope.

No cache bytes, bounds or labels are rewritten. No array is extended to D, no sample is resampled and no role gap/tail is filled. An allowed final endpoint beyond the cached array creates no extra observed scoring support. Existing gaps stay invalid255; the95%/80-frame criteria are unchanged.

The104 source-only tests retain all95 V3 test names and add9 boundary/command cases. Hypothetical sample-count fixtures derive L,D,N and role masks for exact-hop and511-sample residuals and both40Hz rounding directions. They include over-hop final endpoints, internal overruns, invalid final starts, overlaps, preserved gaps and nonempty wrong native commands. None has run; no real eligibility, pickle compatibility or OS limit enforcement follows from these fixtures.

Structure segments must have recognized labels, ordered nonoverlapping bounds and source-clock agreement with the stored role array. Gaps and source-tail absence are invalid role truth, never automatic intro. Outcome roles always come from the target at indices `i0+i`; donor conditioning starts at donor index0. Declared conditioning-tail zeros preserve the historical function but are not observed labels. Energy/distance NaNs use the frozen zero replacement and distance tanh; Inf fails. Missing counts and target/donor support are recorded.

Before any predictions, require at least95% valid target role coverage and at least80 retained frames each for drop and breakdown. A section intersection must contain at least80 frames to be retained. Save all40 decisions, every section intersection/rejection and every material/support failure. No replacement or automatic cohort reduction is permitted. An eligibility list is not permission to run it.

## 6. Readiness output and immutable storage

The sole scientific-workspace output is the new bounded `preparation_readiness_v4.json` under:

`/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/legacy_struct_intensity_v4`

It reports hashes, checkpoint metadata checks, all40 input/role eligibility decisions, resource receipts and explicit zero model/score/GPU counters. A successfully written receipt does not mean any clip is eligible or any study is authorized. Material failures remain in the40-row table. A process failure or partial packet requires root review. Parent must verify native exit0 and the actual receipt semantics. Final internal logging precedes receipt publication, and the receipt binds the finished log hash. It explicitly requires parent-verified native exit0; a receipt file alone is not proof of successful preparation.

The original campaign tree is never written. Hash-bound source/protocol/authorization metadata and logs are part of the packet even when stored on root. The12MiB cap includes all preservedV3 and newV4 preparer/tests/templates/handoffs/review/log files in the shared source directory, the preserved V3 JSON/MD and both V4 documents, authorization/native receipts, preparation output, later traces and all temporary files. Existing shared source is reused by hash; no weight/audio copies.

Combined accounting uses original campaign + entire new shared workspace + external V4 packet bytes + unused comparison reservation + remaining row8 reservation + all parent-bound other reservations. Ceiling: **1,020,054,732 bytes**. Comparison reservation remains32MiB. Require at least1GiB free plus all remaining row8 peak writes on both filesystems. No compression discount, alternate directory or symlink bypass. Final root admission must reconcile other active work.

## 7. Later scoring contract — not implemented or authorized

Save four float32 whole-rig traces and shared frame/role/section arrays, then compute float64 sums from those saved values. Preserve pack/decode quantization and the decoder's fixture quirks; intensity is mean decoded `dimmer*max(val,1e-6)`, not physical radiometry.

For each arm/clip, both retained drop and breakdown means must exceed1e-8 for a defined log-ratio. Otherwise record undefined with numerator/denominator/reason; no epsilon rescue. Average log-ratios equally across coaches within recording, recordings within canonical family, then families. Exponentiate. This is not a pooled-frame or pooled-section estimand.

Report each arm on its own defined support, including primary R_struct. Separately label the four-arm complete-case summary and each paired contrast's defined-support intersection. Do not silently replace primary support with paired attrition. Report clip/recording/family denominators and every exclusion for each table and cohort.

The four fixed circular role shifts at rounded fractions .2/.4/.6/.8 of T operate on the already retained masks. Preserve counts; do not reapply minimum-section tests to wrap fragments. Each shifted ratio needs positive means. A true-minus-mean-shift contrast needs true and all4 shifts defined. These are descriptive temporal controls, not an exchangeable randomization test or independent songs.

Future CIs use10000 fixed-seed20260928 paired canonical bootstrap resamples and the same nested aggregation. Fewer than2 eligible families gives undefined CI; fewer than10 is low-support descriptive. They condition on exposed weights, selected clips, donor map and one draw. They do not cover unknown aliases, label validity, selection, training/style uncertainty or prove independence. Improvement/null contrasts stay exploratory.

## 8. Later GPU admission — draft only, additional registration required

No GPU code or fallback is supplied. Before ANY model output, root must approve the later source/runtime, exact readiness/input bindings and scientific predictions. Enforce one GPU job and the remaining36GPU-hour project ledger.

Within a single separately registered engineering job capped at300s total, including loads/construction, validate all3 exact arms serially:

1. After construction, create/reset a private same-device seed0 generator. Never rely on a global seed set before construction.
2. Validate actual maximum trunk shape `[1,3439,in_dim]` with the historical dtype/no-grad execution and live buffers. Use an explicitly declared engineering-only synthetic input or approved input-derived padding; never score it. Include maximum C output/state storage and AR temporary peaks in the memory bound. A480-frame smoke alone is inadequate.
3. Time480 AR frames on the same model/runtime. Record exactly480 noise calls of shape `[1,1,17]` and1440 Euler velocity calls. No bulk-noise substitute. Record synchronized times, peaks and generator-state hashes. No ratios, selection or saved neural outputs.
4. Stop if any required coverage is missing before the300s ceiling. No second automatic smoke.

For the full prospective study, preserve chunk2400, prime1039, AR3Euler steps, clamp6, y_prev reset per chunk and sampling of all overlap frames. The documented workload is173 trunk calls and509580 AR-sampled frames per arm, including overlap:4,586,220 velocity calls across3 arms. These are operation counts, not throughput measurements.

The draft per-arm conservative admission estimate is:

`load_s + 2*(173*max_3439_trunk_s + 509580*AR480_s/480) + 300s`

The2x safety factor and300s mapper/packing/IO allowance are prospective engineering bounds, not measured guarantees. If a justified CPU/memory bound is missing, stop. Each arm bound must fit3600s and their sum10800s; engineering must fit300s separately. Host limit4GiB and VRAM8GiB include live buffers and mapper/read overhead. Root must accept the bound before a study; watchdogs still enforce actual ceilings.

Future inference must check finite values before sigmoid/clamp can conceal Inf. Keep all native clamps, impulse normalization, slew and uint8 coding. Record saturation without changing values. Record exact noise-call schedules and initial/final states per clip/arm. This is a new common-noise retrospective draw, never original RNG replay.

No reduced cohort, omitted arm/null, changed chunk/precision, C-null or blinder-only endpoint may rescue failed admission. A smaller new question needs its own protocol, not a completion claim for this one.

## V4 static-review delta

The final `STATIC_PREPARATION_REVIEW_V3.json/md` reports three concrete source-only issues: final-duration semantics, nonempty-only native command validation, and missing endpoint-producer source hashes. V4 addresses all three without running or changing V3. Native test receipt commands now require exact equality to `preparation_v4.native_synthetic_test_command`. Restricted CPU `weights_only=True` deserialization is unchanged and remains operationally untested.900s/4GiB remain root-bound external enforcement requirements; the synthetic suite is not an OS/cgroup test. No model/scientific/smoke code is added.
