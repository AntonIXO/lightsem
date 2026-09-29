# V2 comparison delivery — implementation only

**No V1 or V2 code was imported, compiled, tested or executed. No checkpoint load/model call/render/vault call occurred. V1 files remain unchanged.**

## Bound candidates

- `run_comparison_demo_v2.py`: `7c90394dabf46b2fb23f48f98211435a4d17f93b5d027f43ca2f911d31be25b5`
- `test_comparison_demo_v2.py`: `810f4cd726f32ffd0d41172853b120df841a14230a069420f57be68583d4b919`
- `COMPARISON_PLAN_V2.json`: `0e9286f7428346a2ff01d2f5cd45b75fd39ac1b564d663bb209a2e48f17cffd4`

There are 22 unrun synthetic test methods. Handoff JSON contains all dependencies, native commands, expected authorization and successful test-receipt schemas, and unchanged-renderer commands for all four arms.

## Minimal V2 changes requested by parent/reviewer

1. Original campaign is read-only, including its directory tree. Derived outputs move to `/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/comparison_demos_v1`. Original campaign + the **whole** `/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1` share the unchanged1,020,054,732-byte cap. The guard reserves32MiB minus existing comparison bytes for the full four-arm set; other workspace artifacts count. It rejects overlapping/nested/aliased/symlink/foreign-volume paths. No original campaign directory is created.
2. The finite-checking proxy sees every palette logit before frozen `generated_history` calls argmax. Context/accent/cue logits are checked before rollout. Generation math, state choice and thresholds stay unchanged. The exact all-state action table is captured once; selected-state actual logits and compact33-state argmax choices are saved with cue/init indices and prior-state IDs. Dense rebatched full-T palette logits are deliberately omitted. There is no extra model pass and no tolerance widening. Actual action logits are the authority.
3. Readback recomputes accent peaks/confidences and chromatic cues from saved scores, rebuilds RGB/HSV/z from **bound prototypes**, validates every model/overlay array's shape/dtype/range, and independently replays previous/generated/final states from the actual action evidence. It binds plan, authorization, checkpoint, sources, inputs and the recipient-audio route; Ashuf remains explicitly non-primary-donor-route delivery.
4. Native test authorization checks successful exit0, exact source/test maps, plan hash, exact command and hash-checked log, zero real-model forwards and zero sealed TEST reads. Synthetic tests include wrong authorizations/logs, unvisited-state NaN/Inf, unchanged finite-case choices/call counts, false cue/event/RGB/ID/final-state evidence, wrong dtype/shape and accounting traps.
5. Added eager dependency closure: DMX scenes/universe/__init__ and audio __init__/anticipation/features/structure. Existing dependencies are retained.
6. `all_complete.json` explicitly means **INFERENCE/MAPPING ONLY**, with videos/media-audits false. Parent must write a separate final delivery receipt after all four full AV audits.
7. Root media reservation is now prospective **2112MiB**: four unchanged512MiB renderer-arm caps plus64MiB audit/source margin. Parent preflights the whole reserve and each remaining arm/audit reserve, enforces per-arm<=512MiB and total<=2112MiB, and retains the1GiB free floor. This does not change scientific storage/fit limits. Unexecuted V1 keeps its old1.5GiB proposal.

## Budget and execution notes

The conservative32MiB scientific bound remains valid: replacing each355,072-byte dense palette-logit array with at most98,924 bytes of actual-action arrays **reduces** the bound. The all-state table is temporary RAM, not a persisted additional tensor. Parent must recheck current original/workspace bytes before launch.

Use the exact native test command in JSON; it includes the declared log redirection. Create a successful receipt only after the real command exits0. The receipt and authorization must reference the same log path/hash object and plan hash. No test pass is asserted here.

The driver stages remain `preflight`, `infer`, `verify`. Inference uses the native shared GPU lock and external-process check. Four selectedseed0 recipes/order and cached audio94 are unchanged; referencewidth128/independentwidth76 checks remain. No anchor rerun, new fit, extraction, manual retuning, scientific verdict or default promotion. Test and driver files are new/versioned; all original sources and all V1 files are preserved.
