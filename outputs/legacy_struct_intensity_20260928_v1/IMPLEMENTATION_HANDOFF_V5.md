# V5 preparation-only handoff

**Frozen source only. No V5 imports, compilation, test execution, preparation, data/checkpoint/model calls.** V4 failure evidence and all V3/V4 files are preserved. Root envelope files are untouched.

V4 native result:104 tests,102 plain passes,2 failures, exit1. Both failed in lazy `numpy.testing` initialization reading installed NumPy `direct_url.json` under an active test filesystem guard. This is not a corpus/schema result. No failed case or guard is waived.

V5 explicitly initializes `numpy.testing as npt` before per-test guards. All104 test names/guards remain. One new regression verifies prohibited `Path.read_text`/`np.load` still raise while preloaded/direct NumPy array assertions must work. **105 authored tests; none run.** Installed-package metadata initialization is not scientific payload access.

Preparation logic and numeric/role tolerances are unchanged. Only versioned paths and accounting of preserved V3/V4 plus current V5 pairs advance. Restricted CPU loader remains untested with no unsafe fallback. Synthetic tests do not prove actual pickle compatibility or OS enforcement.

Hashes, preserved failure refs, exact commands, names and blocks: `IMPLEMENTATION_HANDOFF_V5.json`. `AUTHORIZATION_SCHEMA_TEMPLATE_V5.json` is NOT authorization.

## Future105-test command (root binds/registers first; CWD/root/lightsem)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u -m unittest discover -s /root/lightsem/outputs/legacy_struct_intensity_20260928_v1 -p test_prepare_struct_intensity_v5.py -v > /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/native_synthetic_tests_v5.log 2>&1
```

## Future preparation component (NOT external-guard authority)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/prepare_struct_intensity_v5.py prepare --plan /root/lightsem/docs/research/2026-09-28-end-to-end/PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V5.json --authorization /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/PARENT_BOUND_PREPARATION_AUTHORIZATION_V5.json
```

The existing root actual-exec guard targetsV4. Root must separately bind a V5 target and actually enforce900s/4GiB/noGPU process-tree limits before preparation. Do not reuse the old envelope. No envelope file was changed here.

Names: `PARENT_BOUND_PREPARATION_AUTHORIZATION_V5.json`, `native_synthetic_tests_v5.log`, internal64KiB `preparation_v5.log`, and newworkspace `legacy_struct_intensity_v5/preparation_readiness_v5.json`. No uncounted extra preparation log. All sharedsource-dir artifacts, preservedfailurelogs/bindings, and SIX V3/V4/V5 protocol files count toward12MiB. Globalcap/comparisonreservation/freefloors unchanged.

Remaining: full105 native pass; semantic root preparation registration;62TRAIN hash/identity approvals/no knownTEST alias with exposure limits; original/resource reconciliation; V5 external limit enforcement; actual restricted-reader/cache qualification. Nativeexit0 and parentreceipt audit required. No model/scientific/smoke code or original-gate completion claim.
