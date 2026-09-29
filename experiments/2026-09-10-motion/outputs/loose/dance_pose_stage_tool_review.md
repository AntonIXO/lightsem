# Final independent static re-review — 2026-09-11T01:26:19.707430+00:00

**Disposition: B1 and B2 are resolved in the exact source revisions below for pre-freeze engineering review. No remaining blocker was found by this static review. Parent may now perform its final hash/test review and decide whether to freeze. This is not authorization to run the real study, and not a claim that all planned outputs fit the storage budget.**

This section supersedes the initial “do not freeze” disposition below. The original review is retained as historical evidence. No source file was edited by this reviewer and no CLI/test/workload was run by this reviewer.

## Final inspected hashes

| File | Raw-byte SHA-256 |
|---|---|
| `outputs/eval_pose_stage_study.py` | `53c596ae4ca075cd361c55bd744512bb7b37a41de5c93158aef95ba7929ab212` |
| `outputs/render_pose_stage_study.py` | `d2a1a79047ac13ac714600e4f56cf9566f5b0f94ba020d7d5f573ffd7fb61830` |
| `tests/test_pose_stage_study.py` | `ae9f0ee3b5bb2523158ae5412a42201dc54df55d2babfefb855c13af8733ab9a` |
| `tests/pose_stage_namespace_smoke.py` | `4164d678259f9cad64fed05bcb02ac8d78d1e602e176a46d3b087c03dd5bcad4` |
| `tests/test_pose_stage_renderer.py` | `84a163bd064b3e51f92c4fa325320518444bf737fc575bccb5e7c8adf84eabc7` |
| `outputs/dance_pose_stage_study_implementation.md` | `68b3e5b0286be15eedf6f6dc4de84e2e9e1cc86abce2a9acb18b9481d1403b63` |
| `outputs/pose_stage_renderer_implementation.md` | `8810818c42d9cf5f797f54df6434097e9c055e1e56d13349d9766924e116874d` |
| `outputs/pose_stage_synthetic_validation.txt` | `78849050835dce740b36f264d211145d8c0b95f2ff8cb5d2aa2add6e9c78dd79` |
| `outputs/pose_stage_pre_freeze_review_decisions.md` | `1a1f60155b54f12b8f0f82a0e7acd9fd53c361357f1461ca4ade4f97f1b3b396` |
| `outputs/pose_stage_study_plan.json` | `1d082d94781036439754be20c5c52ce02f0730a8f4ba9a9e6274c8b259ace0b6` |

All ten files were rehashed after inspection and remained unchanged. The registered plan is byte-identical to the initial review, and its canonical `jhash` remains `9879f1a2af77f7e17eae12fdcda52901c623e91c8713c465eaabfbc5dabb0e9f`. The raw plan digest and canonical digest remain distinct, valid instruments.

## B1 resolution: failure-capable evidence and terminal reporting

- Evaluator L381–397 now records operation phase, completed checks/gates/mechanics and file hashes on mapping failure. Resource interruption is distinct from a scientific gate verdict. Secondary receipt-write failure is attached to, rather than substituted for, the original exception.
- Evaluator L407–443 records mapping phases and per-arm numeric receipts before final map completion. Candidate failure cannot erase an already-written baseline numeric receipt. Completed mechanic/nonangle gate values remain in the failure operation record.
- Evaluator L504–538 supports complete/failed/resource-blocked/corrupt/incomplete/unrun items, preserves completed gates, inventories partial files and lists unassessed items plus preflight results. Missing coverage cannot produce an all-numeric pass.
- Renderer L372–528 tracks phases and assessed gates; actual decode failure remains false, resource-interrupted unassessed media becomes null, and already-assessed pixels remain unchanged. Failure receipt writes are best effort without masking the original exception.
- Renderer L587–656 aggregates false/true/unknown separately and produces a terminal report for completed, failed, resource-blocked, incomplete and unrun approved entries. It no longer requires all scheduled movies to have successful completion receipts. Bad review evidence does not erase measured pixel/media results.

**Instrument/claim scope:** these code paths and inspected synthetic regressions support a failure-capable reporting implementation. They do not prove recovery from every OS/storage failure, actual candidate failure outcomes, or full real-media coverage.

## B2 resolution: exact parent operating policy and guarded writes

- Evaluator L24–65 enforces a 1 GiB filesystem-free reserve and 1.75 GiB total study-artifact limit. Normal writes leave 4 MiB for bookkeeping; failure/terminal receipts may use that operating margin while still respecting the reserve and total cap. L81–102 guards serialized writes under the active study lock. Freeze and numeric-stage preflights are at L195–196 and L483–484.
- Renderer L24–110 implements the same reserve, cap and bookkeeping margin. It counts all files under the frozen study root, including partial files. It estimates combined retained video at the parent-approved 2 MiB/s (silent component 0 for this direct-mux implementation), decoded PNGs, sheets and metadata. This rate is explicitly an assumption, not a bitrate setting or compressed-size bound.
- Runtime checks occur at every encoded frame (L417), decoded frame (L269), decoded-audio 1 MiB block (L297), sheet write (L238), metric/hash group (L457), and final receipt (L481). Video encoding and audio muxing are one ffmpeg process/stream. The 32 MiB next-write/inflight allowance and external-writer limitation are documented; no raw RGB cache was added.
- Renderer L697–698 rejects any non-integer-one budget before taking the lock or creating a context; CLI L756 likewise restricts `--budget` to 1. Subsequent entry preflight counts actual retained bytes. There is no threshold, quality, coverage or output-location workaround.

### Re-review caught and fixed the cross-module receipt bug

Intermediate renderer `3af281683b8a4b56f1a5b1da2e3962f2c2ee0f885719264893e3e962159ea46d` tried an ordinary `started.json` write before its bookkeeping `failed.json` in one `try`. Under the real evaluator lock, a normal write could consume neither the reserved margin nor the failure path, so the failure receipt was skipped even when bookkeeping space remained. Earlier FakeEval tests did not measure this path.

Final renderer L659–694 independently attempts failure bookkeeping after an ordinary start-receipt rejection and records the secondary error. `tests/test_pose_stage_renderer.py` L242–276 now uses the **real evaluator lock/write implementation**, with mocked free-space and total-budget boundary states. It checks that no ordinary start file is written, the failure file is written, and the terminal report records resource-blocked/null coverage. L278–283 checks rejection of budget 2, 0, -1 and boolean True before any evaluator calls. The intermediate version is not cleared; only the final hash above is covered.

**Instrument/claim scope:** this closes the absence of parent-specified storage guards and the observed receipt-ordering defect. It is not a formal bound on external encoder buffering, concurrent external disk writes, or future compressed sizes. Guards can stop the study; stopping must remain a resource-blocked incomplete outcome.

## Storage feasibility remains conditional

The documented single full-Shpongle advisory estimate is **1,536,058,266 bytes**. It requires existing study bytes <=338,795,622 and filesystem free >=2,613,994,394 at that preflight. The fixed 20-second Shpongle diagnostic estimate is **178,553,856 bytes**. These are metadata estimates, not real output measurements.

The combined two-full-entry estimate exceeds the 1.75 GiB cap even with an empty study. Parent therefore required and code now enforces `--budget 1`. This does **not** prove sequential completion will fit: every later entry includes all earlier actual retained bytes and may block. On a resource block, preserve evidence and ask parent. No deletion, relocation, quality reduction or silent coverage change is authorized. Parent must recheck actual capacity before freeze/launch.

## Regression and unchanged scientific logic

The original scientific-alignment findings remain valid after diff review: Shpongle-first selection despite its last position in `items`; exact six-per-head mean ratios; exact-edge Welch integration and subband power partition; zero missing selected-window geometry with missingness separate from collapse; decoded-coordinate quiet cap 16.3488; strict six-of-eight travel/coordination improvement with holds retained for other gates; complete-record mapping; cached baseline byte parity and pre-override nonangle parity; authenticated new source namespace; global brightness matching; no acceptance, hardware or model-qualification claims.

Additional evaluator test sources cover unequal-head-radius aggregation, actual Shpongle-last ordering and the coordinate quiet ceiling. The inspected validation log ends with **“Ran 41 tests in 5.095s / OK.”** This is the builder's synthetic CPU test record; this reviewer inspected the source and log but did not independently execute the suite. Tests do not substitute for parent review or real-stage gates.

**No mapping, render, inference, GPU work, source-array load, VAL/TEST access, vault access, registration or freeze was performed by this reviewer.** Only this review artifact was written. Any later source/hash change requires another bounded diff review before relying on this disposition.

---

# Historical initial review (superseded disposition; exact original snapshot retained)

# Independent static review: pose-stage study tools

Review time: 2026-09-11T01:08:25.369162+00:00.
**Disposition: do not freeze/launch these inspected revisions until the two blockers below are resolved or explicitly mitigated by the parent.** This is a read-only code review, not an execution result or model qualification.

## Exact inspected instruments

All line references below refer to these complete file snapshots:

| File | SHA-256 of raw file bytes |
|---|---|
| `outputs/pose_stage_study_plan.json` | `1d082d94781036439754be20c5c52ce02f0730a8f4ba9a9e6274c8b259ace0b6` |
| `outputs/eval_pose_stage_study.py` | `4ab5bc19fcaddd183049b8fc72f8c36ea8daaed200731c5d05dd294b56d695c3` |
| `outputs/render_pose_stage_study.py` | `cb6ed22269094fc163865c5df17e0d841616e5d0e257e8d8b28a06cf95fdae69` |

The plan's separately computed canonical JSON hash is `9879f1a2af77f7e17eae12fdcda52901c623e91c8713c465eaabfbc5dabb0e9f`. It is correct for `jhash`; it is **not** the raw-file digest. Both are SHA-256 instruments. No mismatch was found. The registered plan, not its stale `plan_status` prose, is the review contract. Registration/vault receipts were not queried or written during this review.

The three files were rehashed before writing this report and remained unchanged. The builder was still working. Any later hash invalidates this review's coverage of the changed portions; parent must inspect the diff before freeze.

## Blockers

### B1. The reporting path cannot finish a negative media result

- Renderer `report()` immediately calls `verified_render()` for every scheduled entry (render L462–465). That requires a successful `render.json` and complete media (L232–257).
- A software decode, audio coverage, sample export, or encoder failure produces `failed.json` and rethrows (L400–403), without `render.json`. That is useful preserved local evidence, but `--stage report` then raises instead of recording a terminal failed study. It also fails if later entries are deliberately unrun after this failure.
- The success-only report hard-codes `all_approved_entries_rendered: True` (L473–479). It cannot distinguish failed, partial, unrun, and complete entries. Numeric stage/map receipts are linked but numerical gate failures and unassessed reference coverage are not summarized in a single terminal report.
- Numeric `map_item()` has no exception receipt around L333–360. After `started.json`, a baseline parity or numerical/IO exception leaves a blocking directory but no structured reason. `evaluate_stage()` L399 then cannot emit a stage receipt. This is **not** evidence that the primary or collapse gate failed; it is an execution/measurement failure.

Required resolution: a create-only failure-capable terminal report, with failed/partial/unrun coverage, available numerical gate values and receipt bindings, exact failure reasons, and explicit no-promotion/no-qualification. Preserve successful artifacts and failure artifacts without repairing them in place. A media failure must remain a failed media gate, not require successful media to report it. Add synthetic tests for a failed render plus an unrun scheduled entry and for a mapping exception.

### B2. No bounded disk/reserve policy with only 3.07 GiB free

Read-only filesystem metadata at review measured **3.0662 GiB free on `/mnt`**. No workload or source-array access was used for this measurement.

- Renderer streams raw frames into ffmpeg (L314–354), which is correct; it does not cache a raw RGB movie.
- However, libx264 CRF output has no storage bound (L78–84). Every sampled decoded PNG and sheet is retained (L131–150, L184–189), and pending entries are launched without a disk budget or minimum reserve (L510–531).
- `--budget` limits entry count, not bytes (L530, L540). A full Shpongle entry is itself long. An ENOSPC error can also prevent `failed.json` or the final receipt from being written.
- The all-pass schedule comprises 23,838 triptych frames. A raw triptych cache would be about **46.04 GiB** and is forbidden. Current streaming avoids this particular cost. This calculation is not an estimate of compressed storage, and does not prove the planned compressed outputs fit 3.07 GiB.

Required resolution before launch: record a conservative compressed-media/sample/freeze/map budget and leave a protected receipt/failure reserve. Enforce a pre-entry and preferably periodic free-space check, or secure enough storage under a parent-approved plan that does not alter the registered immutable inputs or output protocol. Retain streaming; do not add raw RGB caches. Do not claim disk safety from the lack of a raw cache alone.

## Static alignment checks

These are code findings, not measured pass results.

| Requirement | Finding and snapshot lines |
|---|---|
| Shpongle first even though listed last | `evaluate_stage` selects by key, not list position; references require a passing native receipt (eval L393–405). Native-fail schedule returns only 0–20, 70–90, 190–210 (L407–410). |
| Full records before clips | Full arrays and exact 30/40 Hz timelines validated (eval L206–209, L333–345); both mappings use `n/30` before geometry interval selection. Renderer crops only after verifying full arrays (render L54–65, L281–292). |
| Exact saved comparator, no reselection | Fixed `source_arm` selected from authenticated cached source arrays; full cached baseline must match recomputation byte-for-byte, with no repair path (eval L67–95, L211–217, L333–340). No new inference call was found in these tools. |
| Strict six per-head primary | Decoder requires six heads (eval L219–228); RMS is divided by each head's own temporal mean radius before averaging (L285–291). Grand ratio is explicitly supplemental. |
| New exact spectral edges | Welch Hann/400/linear/40 Hz with interpolated inclusive endpoints and trapezoid (eval L235–245). All three subbands are reported, not selected. This is not the old half-open-bin instrument. |
| Selected-window missingness | Primary never drops/interpolates samples (eval L237, L285–291). All-six cloud coverage must be finite with positive radii (L253–260). Both arms' primary/cloud validity enter measurement coverage (L313–324). Missing cloud geometry is explicitly `unmeasurable_coverage_not_collapse`. |
| Quiet ceiling | Exact decoded pan/tilt coordinate first differences, Euclidean norm, all heads, unweighted 0–20 seconds (eval L349–355). Hard cap 16.3488 at L323, with matched baseline and signed difference retained. No geodesic or lit-gated substitute. |
| Mechanics/global validity | Full-record decoded speed/acceleration (eval L230–233); global candidate valid >= .99 and 10-second reports rather than new per-block gates (L283–300, L318). |
| Surface/collapse scopes | Shpongle 70–90 or exact reference window (eval L344). Floor/wall >= .10 only for Shpongle (L322). First-hit radii, not midshaft radii, are used for cloud collapse (L281–292). |
| Pair diagnostics not collapse | Separate Euclidean pair distances, equivalent-circle ratios, valid/invalid counts, first-hit x span, floor-wall chart hull area/support/truss counts (eval L293–310). No pair-spacing gate. |
| Oracle six of eight | Strict greater-than in travel/coordination only; exact denominator 8 (eval L371–374). Holds remain in `all_item_gates`, not gain wins (L398–404). Reference-fail schedule uses the exact three reference windows plus Shpongle 70–90 (L411–416). |
| Nonangle parity before optics | Only four angle bytes per head excluded (eval L214–217); original baseline/candidate parity assessed before renderer's common zero-zoom override (eval L356; render L291–292). |
| Immutable provenance | Create-only hash-enveloped JSON and chmod, no repair (eval L27–58). Exact registered-plan equality/hash required, cached input hashes bound, original helper sources authenticated, implicit config defaults compared (L60–65, L124–160). Frozen files/inputs/runtime checked before activation (L168–187). |
| New frozen source namespace | Helpers load without calling their old `context()`; explicit fresh namespace activation checks each initially imported project module path/hash plus actual `pose_stage` path (eval L162–203). No helper context activation found in renderer. |
| Optical/render constraints | Shared EGL inside exclusive `gpu_lock`, one context per invocation (render L522–534). Frozen camera and fixed shared-body front/oblique panels, raw pre-caption pixel metrics, explicit source captions (L318–350, L384–396, L406–413). |
| Pixel controls | Candidate outside-bulb any-channel >=250 mean <.01/max <.05 (render L227–229, L395). Frozen helper also reports white clipping and bulb-only metrics. A failed pixel gate is stored, not silently dimmed away. |
| Global brightness control | Full Shpongle only; scalar per stream from full-record outside-bulb means; repeated original frame hashes checked; quantization max/mean residual and matched mean difference reported (render L264–273, L298–304, L328–371). No framewise matching. |
| Decode and sampled coverage | Full software video/audio decode and duration checks, chronological 1 Hz+last and fixed-fast frames, decoded-byte review sheets (render L37–51, L155–219). Machine decode is not observer proof. |
| No acceptance claims | Render receipts/report keep human acceptance null, pipeline/default promotion false (render L395–396, L453–458, L473–483). Observer independence/coverage are explicit attestations, not machine proof. No unavailable incumbent3D comparator is invented. |

## Resource and regression-test notes

- Geometry uses full-record 64-vertex cone-circle projection. For Shpongle the float64 ring coordinates alone are about 97.5 MiB; projection temporaries and two radius calls increase peak memory. This is bounded by a single recording, not all 13, but no measured peak-RSS result is available from this review. CPU solver memory was not benchmarked.
- Renderer retains hashes, scalar metrics, full DMX/world arrays, and only current image buffers in memory; it does not accumulate all rendered frames (render L282–313, L317–360).
- Synthetic test sources were read, not run. Existing tests cover exact-edge power partition, cached parity rejection, selected geometry invalidity, oracle holds exclusion, early-stop schedules, and fresh namespace/full synthetic mapping.
- Add regression cases for **unequal head radii** (mean of ratios vs ratio of means), native selection with **Shpongle last**, and quiet cap/coordinate-vs-geodesic discrimination. Current early-stop fixture places Shpongle first, unlike the registered plan (`tests/test_pose_stage_study.py`, `StageTests.plan`).
- Test helper copies presently equal the frozen helper bytes. Keep that explicit hash parity check in future tests instead of assuming mutable `outputs` helper copies remain equal.

Additional inspected raw SHA-256 hashes:

- Frozen support `reviewer.py`: `49b42bbac02c3c0ff2e951dfc3a9a29520ab710aef821851d9943017671842d5`.
- Frozen optics `run_dance_optics_full.py`: `62f43ef37437b9c036ce4e2ed20da7d60034200fbf3a762de842ae601ec4f823`.
- `tests/test_pose_stage_study.py`: `471ac1a1e05a59ebc186a1d9fdd86efa82f11a98e7e647f03e99c0d9476c4288`.
- `tests/test_pose_stage_renderer.py`: `fae7fa152f99c18be47104e7bda104427bcae4e00e079a51bd21020c2a70c79d`.
- `tests/pose_stage_namespace_smoke.py`: `c06a297d80d5b9cb333e453a331cc4f23a69a169612b639cedf4aa7154460621`.
- `tests/test_pose_stage_integration.py`: `9209beca356cdc3da596f0f1663690baba90f99f7e1c969f9a2a3da66eabc411` (partial inspection only).
- `lightsem/experiments/motion_study/lighting_map.py`: `55faff3b7f327b88b88db72a7a4e65fbd292520eda8b669b5bed7d703665a99d` (entry point inspected).
- `lightsem/experiments/stage11_common.py`: `1196cfda2e0a222d85ebd508f8c09636907611edd01a83d0624abef3465f64d3` (`gpu_lock` inspected).

## Scope of this review

No actual CLI job, test, mapping, render, inference, GPU inspection/workload, VAL/TEST access, source-array load, vault access, registration write, or freeze was run. Source data/meta and source code were not modified. The only artifact written is this review. Findings were sent to `pose-stage-study-tools` with a parent copy. Parent review of any fixes is required before freeze/launch.
