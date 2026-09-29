# V3 preparation-only implementation handoff

**Unimported, uncompiled, untested and unrun.** Root must bind/test/register before any payload read. No model/score/GPU stage exists.

Files and hashes, full authorization schema, exact command strings and remaining blocks are in `IMPLEMENTATION_HANDOFF_V3.json`. `AUTHORIZATION_SCHEMA_TEMPLATE_V3.json` is deliberately NOT authorization.

## Native synthetic tests (root only, after binding)

Working directory: `/root/lightsem`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u -m unittest discover -s /root/lightsem/outputs/legacy_struct_intensity_20260928_v1 -p test_prepare_struct_intensity_v3.py -v > /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/native_synthetic_tests_v3.log 2>&1
```

Exactly95 tests are authored. No pass is claimed. Root must require native exit0,95 actual tests, no errors/failures/skips/expected failures/unexpected successes, and the hash-bound verbose log. The preparer independently checks the exact pass count/log shape, not only an exit code.

## Preparation (separate root registration/authorization required)

Bind externally enforced900s wall and4GiB host process-tree limits first. The project CLI does not itself install an OS/cgroup limiter. CUDA is hidden.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/prepare_struct_intensity_v3.py prepare --plan /root/lightsem/docs/research/2026-09-28-end-to-end/PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V3.json --authorization /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/PARENT_BOUND_PREPARATION_AUTHORIZATION_V3.json
```

Do not redirect this command to another file. Authenticated Python stdout/stderr goes to the exclusive64KiB internal `preparation_v3.log`, with packet-cap checks on every append. Native C/C++ fd output may reach the parent harness; this is a capture limitation, not permission to create uncounted logs.

The command requires the NEW shared-workspace packet path not to exist. It only reads exact approved hashes and writes the readiness receipt. No automatic overwrite, cleanup, resume, model construction, forward, ratio or GPU fallback is implemented.

The readiness receipt binds the final internal log and requires parent-verified native exit0. A receipt alone is not execution success. All40 input/support outcomes must be audited, including failures; no replacement clips. Receipt success grants no subsequent scientific stage.

## Remaining blocks

- Native95-test result and root-verified preparation registration receipt do not yet exist.
- Root must bind all62 exact currentStage11TRAIN inputs/donors, their hashes, knownTEST-alias exclusions and unknown cross-corpus limitations. No TEST/rawmanifest access.
- Root must reconcile the12MiB TOTAL packet,32MiB comparison reservation, all other active allocations, combined1,020,054,732-byte cap and1GiB free floors. Source/tests/V3 docs/historical source/logs/receipts/temp all count.
- Restricted CPU checkpoint deserialization and real schemas have not been tested. Unsupported objects stop; no unsafe pickle/model fallback.
- Root must enforce CPU resource limits and unchanged original inventory. Registration semantic success remains root-attested; the driver authenticates bytes, not vault semantics.
- Any future full-shape GPU engineering/model/scoring stage needs additional source review, native registration, scientific preregistration and runtime admission. V3 documents only a plan. No cheap endpoint replaces the whole-rig question.

V1/V2, existing sources and the original campaign tree were not modified. The sole shell operation was read-only historical `git show`; its Git blob was verified before writing a new copy. No data arrays, checkpoints, TEST/raw manifests or vault were opened by this implementation task.
