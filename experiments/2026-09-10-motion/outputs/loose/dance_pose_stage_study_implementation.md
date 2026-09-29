# Pose-stage study tooling implementation — 2026-09-11

## Status

Implementation and synthetic CPU validation only. The parent registered the final
plan while tooling was being tested. This worker did **not** freeze or execute the
real study. No saved-record candidate map, score, render, VAL/model run, GPU/EGL
context, training, vault write, commit, cleanup, or default change was made.
Metadata and helper source were inspected. Tests used synthetic arrays and files.

Authoritative plan: `outputs/pose_stage_study_plan.json`.

- File SHA256: `1d082d94781036439754be20c5c52ce02f0730a8f4ba9a9e6274c8b259ace0b6`.
- Canonical-JSON SHA256: `9879f1a2af77f7e17eae12fdcda52901c623e91c8713c465eaabfbc5dabb0e9f`.
- These are different hash instruments, not competing plan versions.
- The canonical hash is the evaluator's supported-plan binding. Freeze also binds
  the exact plan file bytes. Changed plans require explicit implementation review.
- The parent's existing registration was checked for exact plan/id equality using
  metadata only. It matches. This is not an independent vault audit.

## Files

- `outputs/eval_pose_stage_study.py`: registration validation, freeze, identity
  verification, complete-record mapping, geometry, staged gates and render schedule.
- `outputs/render_pose_stage_study.py`: conditional media, independent sampled
  review receipts and a create-only coverage report.
- `tests/test_pose_stage_study.py`: evaluator synthetic CPU tests.
- `tests/pose_stage_namespace_smoke.py`: fresh-process copied-source smoke and
  synthetic complete 20-second mapping/cache test.
- `tests/test_pose_stage_renderer.py`: renderer synthetic CPU tests.
- `outputs/pose_stage_renderer_implementation.md`: renderer details and review JSON
  schema. Existing production mapper files were not edited by this worker.

## Ambiguities resolved before implementation was finalized

Questions were sent to the parent before choosing semantics. The final registered
plan and `outputs/pose_stage_parent_decisions.md` resolve them:

1. Primary aggregation is the mean of six per-head RMS/mean-radius ratios.
2. NEW spectral integration uses piecewise-linear Welch PSD with interpolated exact
   band edges. The three adjoining subband powers sum to broadband power. Historical
   optics used half-open-bin trapezoids and is not numerically interchangeable.
3. Quiet uses decoded pan/tilt coordinate first-difference Euclidean speed, all six
   heads without brightness gating, matching `surface_revision.mechanics`. It does
   not use geodesic direction speed. Candidate ceiling is 16.3488 deg/s. The matched
   baseline and candidate-minus-baseline difference are both reported.
4. Full-record valid head-sample fraction is gated at .99. Every 10-second block is
   descriptive. Selected primary/cloud windows require zero missing geometry.
   Missing samples fail `measurement_coverage`; they are not called observed collapse.
5. Oracle gain is **6 of 8 travel/coordination windows**, not the superseded draft
   9 of 12. All four holds remain mapped/reported and subject to all other applicable
   gates. More motion in holds is not a success criterion.
6. Pair diagnostics use valid-pair denominators and explicit invalid counts.
   Equivalent camera-circle overlap is not true ellipse or illuminated-pixel overlap.
   Floor-wall intrinsic convex-hull area is reported only on all-floor/wall frames,
   with unsupported and truss-frame counts. There is no pair-spacing gate.
7. The parent registration contract is accepted. New frozen mapper imports must
   not be shadowed by an older support-review source activation.

## Freeze and source identity

The freeze command requires explicit output, plan and registration paths. Registration
must contain the exact hypothesis/experiment ids and exact plan payloads, plus
`registered_by: "parent"` and `registered_before_candidate_execution: true`.
This is a local parent attestation, not cryptographic proof of authorship or time.

Freeze validates complete support-review map receipts and referenced hashes. It
binds cached full-world/full-DMX files, the existing source loudness curve and audio.
Only the selected `reference` or Shpongle `prior_native_flow` keys are opened by the
mapper. No new inference or source/draw selection exists.

- Baseline explicit config must equal the cached support-review config.
- Every implicit `LightingConfig` default must equal the authenticated old mapper's
  literal dataclass defaults. Frozen renderer/shader/rig hashes must match optics.
- Complete baseline DMX is recomputed from the same cached world/loudness before
  candidate mapping. Byte parity is mandatory, not an approximate score or repair.
- All current project Python/JSON files, evaluator, renderer and wrappers are copied.
- Frozen support `reviewer.py` and optics `frozen/run_dance_optics_full.py` are copied
  into explicit new `source_helpers` bindings. Mutable output helpers are never the
  real-stage imports. Their old context-activation routines are not called.
- `activate_source` rejects already-imported project modules. It loads the new
  mapper/pose-stage module, verifies actual paths/hashes of imported `lightsem`
  modules under the new frozen source root, and records the bindings in map receipts.
- Runtime identity hashes installed numerical/render package files, interpreter,
  linked OS libraries, EGL/GL driver files, codec executables and font. Numerical/EGL
  environment settings are bound. This is an explicit file fingerprint, not a
  hermetic container or hardware-safety claim. Hashing can take time.

Every real CLI stage checks freeze, source, input, helper, runtime and registration
identity before work. Use `run_torch.sh` for the real frozen workflow because the
freeze binds that runtime. No real freeze or runtime inventory was executed here.

## Numeric instruments and sequence

Both arms map complete cached worlds before any selected time slicing. The saved
world must remain exactly unchanged. There is no FK/root reintegration in mapping.
Original non-angle DMX bytes must match before the common zero-zoom optical copy.

Mechanics uses every decoded full-record pan/tilt first/second difference and the
90 deg/s, 360 deg/s² limits. The existing numerical comparison allowance is 1e-8.

The geometric instrument reuses the frozen optics finite floor/wall/truss first-hit,
camera projection, angular half-peak and 64-vertex equivalent cone-circle helpers.
Broadband midshaft RMS/radius is new .5–4 Hz; every registered subband is reported.
All-head collapse uses **first-hit** camera spread and **first-hit** mean cone radius.
It records threshold fraction and longest contiguous run. Three overlapping pairs
at separated locations do not count as all-head collapse.

`native` maps and analyzes full saved Shpongle only. Its failure permits only the
fixed 0–20, 70–90, 190–210 diagnostic renders. `references` refuses to run unless
native gates pass. It maps all twelve full records and analyzes the fixed windows.
Its failure permits only holds083, travel027, coordination057 and Shpongle70–90
clips. Only all numeric gates permit all13 planned views and the additional full
Shpongle global brightness pair. Cached stage/map gate schemas and thresholds are
recomputed from authenticated summaries; cached `pass` flags are not trusted alone.

## Render, media and independent review

A single explicit project-exclusive GPU lock encloses one shared EGL context per
render invocation. Video encoding is CPU `libx264`. Panels are fixed cached body
(front+oblique), matched baseline and candidate. No recentering, pose padding,
per-frame brightness match, exposure change or source selection is available.

Uncaptioned raw pixel metrics and per-frame/whole-stream hashes precede captions
and encoding. Outside-bulb any-channel clipping uses strict mean<.01 and max<.05.
White clipping and source-bulb metrics are retained separately. Full Shpongle global
RGB dimming rerenders each original stream, requires every repeated raw hash to
match, and saves scalars and quantization residuals. No hidden brightness repair.

Every delivered frame and audio sample is software-decoded. Exact frame count,
video format/duration and audio coverage are checked. Audio duration tolerance is
80 ms for AAC frame accounting; source trim commands remain exact whole-frame
intervals and do not add padding. Chronological1Hz+last and fixed-fast images/sheets
come from decoded delivery. Independent review CLI requires all declared samples
and sheets, reviewer/coverage attestations, observations and limitations. Machine
decode and silent sheets never set music-fit acceptance or pipeline qualification.

All CLIs take the shared study lock. Outputs are create-only. Complete item/render
receipts are verified and skipped on restart. Partial/corrupt outputs block retry
in place rather than being deleted or silently repaired. Numeric mapping exceptions
write structured phase/item failure receipts with already-measured checks and
gates. Per-arm numeric receipts preserve completed measurements. Failure-recording
errors are attached as notes without replacing the original operation exception.
The create-only numeric `report` stage retains
completed, failed, corrupt, incomplete and unrun coverage without claiming all-pass.
The renderer report likewise records completed/failed/incomplete/unrun entries
and false versus unassessed gates. It does not require all media to succeed first.
A terminal report is a snapshot; do not create it early if work will continue. No actual candidate gate
or media outcome is reported by this implementation document.

## Synthetic validation

Command (from repository root):

```text
CUDA_VISIBLE_DEVICES="" bash run_torch.sh -c "import sys,unittest;sys.path.insert(0,'tests');unittest.main(module=None,argv=['unittest','test_pose_stage_study','test_pose_stage_renderer','-v'])"
```

**41 combined tests passed** (27 evaluator +14 renderer;5.095s test time, exit0).
Final command output: `outputs/pose_stage_synthetic_validation.txt`.
Coverage includes:

- Wrong hashes, corrupt/self-inconsistent receipts, incomplete caches, strict
  full-record timeline/shape/finite checks, create-only writes, lock contention.
- Missing registration/freeze, mismatched registration payload, mandatory baseline
  parity and exact non-angle controls, decoded mechanical violations.
- Invalid finite-stage geometry and missing window coverage, first-hit versus
  midpoint radius, global collapse versus pair merging, threshold/run duration.
- New broadband versus selected narrowband and exact-edge power partition.
- Fixed native/reference early-stop schedules and corrected oracle denominator.
- Fresh-process frozen namespace activation, rejection of mixed old imports and
  corrupted source binding, complete synthetic20s baseline/candidate map plus
  unchanged world and authenticated restart/corruption rejection.
- Actual six-frame synthetic CPU x264 plus full software video/audio decode,
  decoded review sheets, global dim/repeated hashes and residuals, media tampering,
  failed-repeat retention and missing review coverage.
- Four run-level review/report routing cases reject any project/GPU import.
- Exact reserve/total-budget boundaries, all artifact categories, guarded-write
  rejection before partial file creation, bookkeeping reserve, pre-GPU blocking,
  streamed resource interruption, resource-blocked terminal coverage and preservation
  of the original exception when receipt writing also raises an error.
- Real evaluator-lock cross-module reserve/cap tests: an ordinary `started.json`
  can be blocked while independent bookkeeping `failed.json` and the terminal
  report still succeed. Budget values other than integer1 are rejected before
  lock/context/GPU work.

Tests caught a generated constant-name typo, unsupported host ffmpeg `-vsync`, and
renderer CLI stage-variable shadowing. These were fixed. The shadowing defect was
not covered by the earlier lower-level tests; dedicated CLI-route tests now cover it.

Not measured: actual saved-record baseline parity, any real candidate geometry or
score, real-stage EGL rendering, GPU repeat determinism, full real audio alignment,
real clipping/media gates, independent visual acceptance or model quality.

## Parent-only next commands (not executed here)

**Pre-freeze storage policy:** `outputs/pose_stage_pre_freeze_review_decisions.md`
sets a1GiB output-filesystem free reserve and1.75GiB TOTAL study-artifact stop
threshold. These are infrastructure limits, not scientific gates. All frozen/maps,
partial/silent/muxed video, logs, samples and reports count. The decision file is
copied and hashed at freeze. The implementation does not claim that every planned
movie will fit; resource-blocked coverage must be reported and referred to parent.

Evaluator preflight reports free space, total study bytes, remaining budget and
estimated additional output. Freeze allows source file bytes plus32MiB manifest
planning space; pending complete numeric maps allow8192bytes/nativeframe. These are
advisory planning estimates. Every evaluator file write inside the shared CLI lock
then checks the exact serialized next-write length against actual space and total
study growth. A4MiB operating margin remains available for failure/report bookkeeping;
bookkeeping still respects the1GiB free reserve and1.75GiB total cap. Storage is
rechecked on each file write, not just at preflight. Concurrent external disk use is
not controlled; a secondary failure to write a receipt never masks the original error.

Renderer-specific streamed checks, inflight allowances and estimator are documented
in `outputs/pose_stage_renderer_implementation.md`. The approved2MiB/s combined
retained-video planning allowance is about16.78Mbit/s; it is not an encoder bitrate
or measured bound. This renderer retains no separate silent movie (silent estimate0).

Metadata-only full-Shpongle estimate:1,536,058,266bytes additional output
(mux581,540,250; samples716,610,560; sheets204,353,024; metadata33,554,432).
With the operating margin, this needs filesystemfree>=2,613,994,394bytes and
existingstudy<=338,795,622bytes. A600-frame diagnostic estimates178,553,856bytes
and needs free>=1,256,489,984bytes, existingstudy<=1,696,300,032bytes.
Both full original+dim entries in one preflight batch estimate3,072,116,532bytes,
which exceeds the1.75GiB cap even with an empty study: that route is resource-blocked
and was reported to the parent. The parent now authorizes **`--budget 1` only**.
Each entry is checked separately but retains/counts earlier outputs and can still
block. A combined two-full-movie batch is not authorized. No storage threshold is raised
to force coverage. No raw RGB cache, codec change,
quality reduction, coverage reduction, output relocation or evidence deletion is
allowed as a space workaround.

After reviewing these files, use the registered directory as OUT. Do not update
registered parameters to make a later failure pass.

```text
bash run_torch.sh outputs/eval_pose_stage_study.py freeze --output OUT --plan outputs/pose_stage_study_plan.json --registration OUT/registration.json
bash run_torch.sh OUT/frozen/eval_pose_stage_study.py verify --output OUT
bash run_torch.sh OUT/frozen/eval_pose_stage_study.py native --output OUT
# Only if native passes:
bash run_torch.sh OUT/frozen/eval_pose_stage_study.py references --output OUT
bash run_torch.sh OUT/frozen/eval_pose_stage_study.py schedule --output OUT
# Optional terminal numeric outcome, including failures/incomplete coverage:
bash run_torch.sh OUT/frozen/eval_pose_stage_study.py report --output OUT
bash run_torch.sh OUT/frozen/render_pose_stage_study.py --output OUT --budget 1
# Repeat renderer invocation to consume approved pending entries only.
# Then independent observations, followed by final report:
bash run_torch.sh OUT/frozen/render_pose_stage_study.py --output OUT --stage review --observation OBSERVATION.json
bash run_torch.sh OUT/frozen/render_pose_stage_study.py --output OUT --stage report
```

Use nonblocking jobs and persistent log files for real full-record mapping/rendering.
The tooling contains no vault writes or default-promotion action.
