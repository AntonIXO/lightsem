# Static four-model demo review — V1

**Decision: hold V1. Do not bind it for inference.** Correct the issues below, review the new version, then run hash-bound native synthetic tests. No incorrect real output is alleged: V1 has not been executed.

## Review scope

- Read source and JSON only. No project/driver imports, tests, checkpoint loads, model calls, array loads, rendering, vault calls or source edits.
- Reviewed `run_comparison_demo_v1.py`, its 11-test source, `COMPARISON_PLAN_V1.json`, `IMPLEMENTATION_HANDOFF_V1.json`, readiness, original `infer_overlay.py` / `render_demo.py`, and frozen training/model/overlay APIs.
- Checked source hashes and four real training `complete.json` receipts. Did not deserialize checkpoint metadata. Full review bindings are in the companion JSON.
- Parent reports V2 is already requested for sibling placement and a finite-checking palette proxy. This review binds only V1.

## Static checks that agree

- Four exact selected seed0 recipes and checkpoint references match the saved registration. Anchor `S_VR_P0` is excluded. No fit, promotion or anchor rerun is added.
- Canonical loader uses `reference_model_config` width128, then narrows independent once to76. Real receipts agree. Others use128; visual auxiliary dimensions are0/6/96/96, pose0.
- Forward/mapping calls match the frozen APIs: full11092-frame audio94 input,2773 context frames, eval/FP32, no chunking or visual/pose inputs.
- Exact original mapping is preserved: first actual color cue starts RGB replacement, HS72 circular palette summary, global fixture color, accent peaks at threshold `>=.5`, gain.35, full cosine width.25s, .2s accent NMS, maximum overlaps, protected projector dimmers/strobes/blinders/angles.
- Byte reconstruction independently covers43 writable and469 protected slots. This part is useful, but it does not establish decision provenance (B2).
- Existing/incomplete arm directories block repeat forwards. Complete arms are checked/skipped. Exclusive partial publication retains interruption evidence. Shared GPU lock performs an external compute-process check.

## Blocking issues and test gaps

### B1 — Derived output changes finalized original campaign tree

**Gate:** `blocker_before_preflight`.

Evidence: guard explicitly requires derived_root inside campaign_root; preflight creates that directory. No existing file must be overwritten for the finalized tree inventory to change.

Required: Use a new sibling/versioned delivery workspace. Bind an explicit aggregate list including original campaign plus the whole end-to-end workspace and every previous/partial delivery output. Retain original CAP=1020054732 and shared 32MiB reservation; do not reset the budget. Snapshot original tree hashes/inventory before and after. Parent reports this correction is already requested for V2.

Locations: `run_comparison_demo_v1.py:95-100`; `run_comparison_demo_v1.py:466-467`; `COMPARISON_PLAN_V1.json:campaign_root,derived_root`.

### B2 — Readback checks byte arithmetic but not the origin/semantics of saved decisions

**Gate:** `blocker_before_infer`.

Evidence: verify_mapping trusts saved accent_times_s, accent_confidences, palette_rgb and color_cues. It never recomputes accent peak/NMS/threshold from accent_score; never derives RGB/HSV/z from bound prototypes; never derives color_cues from color logits/scores; never compares generated ids with saved palette_logits argmax at initialization/cues; never checks final_ids. Current test synthetic(cue=True) uses all-zero palette_logits but id1 after the cue, which the verifier is designed to accept. This is a source-level counterexample, not an executed test result.

Required: Make zero-model-call verification derive the exact frozen cue/accent rules from saved raw scores and validate sigmoid score/logit agreement, frozen palette summary, recurrence, initialization argmax, cue argmax and final state. Require exact saved schema/shapes/dtypes first. Then retain independent RGB/dimmer/protected-byte reconstruction. Add mutations that each violate these contracts.

Locations: `run_comparison_demo_v1.py:319-364`; `run_comparison_demo_v1.py:389-399`; `test_comparison_demo_v1.py:152-188`.

### B3 — Generated-history all-state palette table can hide nonfinite outputs before argmax

**Gate:** `blocker_before_infer`.

Evidence: generated_history calls palette_logits(...).argmax before returning only integer choices. full_forward checks saved context, logits on realized previous states and integer rollout only. A nonfinite value in an unvisited all-state table entry is not necessarily seen by these checks.

Required: A transparent inference proxy must check finiteness of every canonical palette_logits result before argmax without an extra model call or arithmetic change. Add NaN/Inf synthetic cases. Parent independently identified and requested this for V2; frozen training source must stay unchanged.

Locations: `lightsem/experiments/light_layers/training.py:760-777`; `run_comparison_demo_v1.py:376-385`.

### B4 — Native-test receipt guard does not enforce the handoff receipt schema

**Gate:** `blocker_before_infer`.

Evidence: authorize checks receipt hash, status, source/test maps and zero read/forward counters. It ignores exit_code, command, log identity/hash and test count. A receipt with status=passed and exit_code=1 or missing command/log fields passes these conditions.

Required: Validate native wrapper/command/environment, successful exit, nonempty expected test execution and bound log before authorizing inference. Bind the receipt to exact plan/test/source versions. Parent-produced receipt must not infer success only from status strings.

Locations: `run_comparison_demo_v1.py:153-169`; `IMPLEMENTATION_HANDOFF_V1.json:required_native_receipt_schema`.

### B5 — Standalone verification does not authenticate saved provenance against current authorization

**Gate:** `blocker_before_infer`.

Evidence: verify_saved validates recipe/checkpoint and self-referenced NPZ hashes only. It does not compare saved plan, authorization, inputs or source_files against current bound values. verify phase also ignores complete.json and its authenticated inference/proof links. A different-plan inference sidecar with the same recipe/checkpoint can be accepted if its reconstructed baseline bytes match.

Required: Validate exact plan, permitted phase authorization, inputs, source bindings, within-arm artifact paths and completion/readback chain before treating saved output as this delivery. Preserve the no-model-call verification path; do not recover by repeating the forward.

Locations: `run_comparison_demo_v1.py:389-399`; `run_comparison_demo_v1.py:475-481`.

### B6 — Eager transitive project imports are absent from source binding

**Gate:** `binding_gap_before_native_tests`.

Evidence: Importing overlay -> dmx.fixtures executes dmx/__init__.py, which imports scenes.py and universe.py. These import audio anticipation/features/structure and execute audio/__init__.py. These six project files are absent from source_files although executed by Mapping tests and inference imports.

Required: Bind lightsem/dmx/{scenes,universe}.py and lightsem/audio/{__init__,anticipation,features,structure}.py in the new version. Do not edit frozen packages. Native tests must run with librosa/scipy/sklearn available in the dedicated torch environment.

Locations: `COMPARISON_PLAN_V1.json:source_files`; `lightsem/dmx/__init__.py:3-5`; `lightsem/dmx/scenes.py:30-32`; `lightsem/dmx/universe.py:14-16`; `lightsem/audio/__init__.py`.

### T1 — Current tests do not exercise full_forward or semantic readback

**Gate:** `test_gap_before_infer`.

Evidence: The factory test calls a fake loader that returns the expected fake config and directly calls that fake model; it never calls full_forward, the canonical loader/factory, or the generated-history path. Storage tests mock guard during actual writes; no partial-file accounting or multi-arm interruption/skip test is present. No authorization/receipt acceptance test is present. full_forward only validates output keyset and context shape, not raw head/rollout shape/dtype contracts.

Required: Add synthetic schema/finite/history/mapping/provenance/receipt mutations and filesystem partial/interruption tests. Test orchestration with fake hooks, not a real checkpoint/model forward. Keep real checkpoint loading and all four full-song forwards behind reviewed native-test and parent authorization gates.

Locations: `test_comparison_demo_v1.py:69-88`; `test_comparison_demo_v1.py:125-141`; `test_comparison_demo_v1.py:152-206`; `run_comparison_demo_v1.py:367-386`.

### D1 — Aggregate media enforcement remains a parent-side prerequisite; inference completion is not video completion

**Gate:** `blocker_before_render_and_delivery`.

Evidence: New driver never renders. It writes all_complete.json with a delivery-videos scope before any render/audit. Frozen renderer caps each arm at512MiB rather than all four at1.5GiB; audit sheet writes lack an aggregate guard. Handoff explicitly delegates this enforcement to parent, so the listed render commands alone are not a complete guarded delivery runner.

Required: Name inference/mapping completion narrowly. Before frozen render calls, parent must enforce aggregate root media reservation/cap/free floor including partial videos, logs, sheets and audits across all arms, plus exclusive dirs/no rerender of anchor. Final delivery receipt requires all four native1080p AV1/audio renders and audits.

Locations: `run_comparison_demo_v1.py:496-499`; `IMPLEMENTATION_HANDOFF_V1.json:constraints,render_commands_not_executed`; `outputs/shpongle_layers_demo_20260922T161917Z/render_demo.py:35-36,62-64,81-91`; `run_rhythm_companion.py:247-291`.

## Exact raw-output schema to check

At the saved single-song boundary:

| Arrays | Shape | Dtype |
|---|---|---|
| `accent_logit`, `accent_score` | `(11092,)` | float32 |
| `chromatic_change_logit`, `color_score` | `(2773,)` | float32 |
| `context10` | `(2773,128)` shared; `(2773,76)` independent | float32 |
| `palette_logits` | `(2773,32)` | float32 |
| `previous_ids`, `generated_palette_ids` | `(2773,)` | int64 |
| `color_cues` | `(2773,)` | bool |
| `final_ids` | scalar | int64 |
| `time40_s`, `time10_s` | `(11092,)`, `(2773,)` | float64 |

Require exact key membership and finiteness; ids0..31, first previous id32, later previous id equal prior generated id. Validate own-logit cue/NMS rules, initialization/cue argmax, holds and final state. These checks need no repeat forward.

## Storage and placement

- V1 arithmetic reserves one global32MiB across all four arms, not32MiB per arm. `size_tree` includes `.partial` files, and each serialized write is charged before publication.
- With the supplied campaign size952,976,108 bytes, full reservation gives986,530,540 bytes, below the original1,020,054,732-byte cap. No current free-space measurement was made in this static review.
- Declared total33,248,724 bytes leaves305,708 bytes inside32MiB. Independent fixed-schema arithmetic gives31,362,964 raw array bytes across all four arms, below declared raw32,069,076. Bounded metadata and ZIP overhead remain reserved separately. Omission of redundant baseline and probabilities is correct.
- **Placement still must change.** Putting a new folder inside the original finalized campaign alters that tree. Use a sibling delivery workspace and explicitly count original + whole end-to-end workspace, including partials and any old-version outputs, under the same original cap. This is not a budget reset. Verify original tree inventory and hashes before/after.
- Media stays separately accounted on root. Four per-arm512MiB guards do not enforce a1.5GiB aggregate cap; parent must provide that guard including sheets/logs/partials before rendering.

## Native environment and completion gate

- Proposed synthetic command is correct in form: `TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -m unittest discover -s /root/lightsem/outputs/shpongle_model_comparisons_20260928_v1 -p test_comparison_demo_v1.py -v`.
- Wrapper sets project working directory and `PYTHONPATH`. Static path/config inspection confirms a3.12 torch-access venv path exists. This does not prove dependencies import. Eager dmx/audio imports require librosa/scipy/sklearn as well as torch/numpy.
- Do not run the listed V1 command just to gain a passing receipt. Review and bind the corrected version and its tests first. Native synthetic tests must have zero real checkpoint/model/TEST reads.
- Then deliver all four full Shpongle outputs even if they look worse. Frozen render helper supports native1920×1080 AV1 with audio; audit checks full decoding and timestamps. Do not call inference completion video delivery, and do not treat the media audit as human audiovisual acceptance.
- Ashuf recipient-audio delivery remains explicitly outside primary donor-route evaluation. No new scientific result, generalization claim or default promotion follows from these demos.
