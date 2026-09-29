# Structure-intensity protocol candidate V5

**Source implementation only. V5 has not been imported, compiled, tested or run.** Preparation remains blocked pending new root bindings. Every V3/V4 source, protocol, binding and log is preserved. No root external-envelope file is modified.

## Why V5 exists

The root-run V4 native suite exited1:104 tests,102 plain passes,2 failures. Both failures were dependency initialization:

- `test_producer_near_one_hop_final_tail_is_valid_without_extension`
- `test_structure_valid_segments`

Their `np.testing` attribute access lazily imported `numpy.testing`. NumPy then read its installed distribution `direct_url.json` through `Path.read_text`, which the already-active synthetic filesystem guard correctly rejected. This is not a producer-boundary/data failure or measured cache ineligibility. No failure is waived, and V4 is not rerun.

Preserved evidence: `NATIVE_SYNTHETIC_TEST_FAILURE_V4.json`, `NATIVE_SYNTHETIC_TEST_BINDING_V4.json`, and `native_synthetic_tests_v4.log`, with exact hashes in the V5 JSON/handoff.

## Only test initialization changes

The V5 test module explicitly imports `numpy.testing as npt` at module initialization, **before** any per-test filesystem or operational guard. This permits normal installed-package metadata initialization, not scientific corpus access.

All104 V4 test names and every guard remain. V5 adds one regression, for **105 authored, unrun methods**. The regression:

1. Confirms synthetic `Path.read_text` still raises the active filesystem-guard error.
2. Confirms synthetic `np.load` still raises the active numerical-file guard.
3. Requires both preloaded `npt.assert_array_equal` and `np.testing.assert_array_equal` to succeed on generated arrays inside those same guards.

No native pass is claimed. These tests do not establish actual restricted-pickle compatibility, real cache eligibility or external OS/cgroup enforcement.

## Preparation logic is unchanged from V4

Only versioned source/command/packet/log/receipt names and preserved-document accounting change in the preparer. The authoritative V5 JSON carries the unchanged source/clock/role/support/scoring contracts. V4's preserved protocol remains historical source provenance, not permission to execute its failed version.

- The only CLI stage is `prepare`. There is no model construction, forward, scientific score, ratio, bootstrap, smoke or automatic next-stage code.
- Future CPU checkpoint reads use restricted `torch.load(weights_only=True, map_location='cpu')` with the same narrow NumPy reconstruction allowlist. Unsupported formats stop; there is no unsafe fallback or model constructor. Actual compatibility remains untested.
- Exact62 explicitly hash-bound currentStage11TRAIN audio/structure files cover the fixed40 recipients and original donors. Known currentTEST aliases are excluded. Unknown cross-corpus aliases, old preprocessing/evaluation and donor exposure remain explicit; no freshTEST/disjointness/independence claim or TEST/rawmanifest grant.
- Exact recipient `i0:i0+T`, donor index0,40Hz absolute/adjacent-clock tolerances1e-4s, finite/float32 feature checks,95% valid target coverage and80-frame retained drop/breakdown/intersection rules are unchanged. All40 decisions remain visible; no replacement/repair.
- Structure cache duration L remains the last centered STFT frame; N=round(L*40), not waveform end D. Only the final segment end may reach `L+512/22050+1e-9`. Starts/intermediate bounds, nonoverlap, recognized labels and role-array agreement stay unchanged. No cache extension or gap/tail filling. The check remains **metadata consistency only**, not independent waveform-end verification.
- Native tests must match the exact published V5 command and all105 plain passes, with no failures/errors/skips/expected failures/unexpected successes, plus source/plan/test/log hashes. No V4 native result is reused as a V5 result.

## Versioned authorization and storage

New source files: `prepare_struct_intensity_v5.py`, `test_prepare_struct_intensity_v5.py`.

New authorization: `PARENT_BOUND_PREPARATION_AUTHORIZATION_V5.json`. `AUTHORIZATION_SCHEMA_TEMPLATE_V5.json` is a deliberately non-authorizing template.

New logs: `native_synthetic_tests_v5.log` and exclusive internal64KiB `preparation_v5.log`. No uncounted redirected preparation log.

New scientific-workspace packet:

`/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/legacy_struct_intensity_v5/preparation_readiness_v5.json`

A readiness receipt requires native exit0 and parent validation of all40/source/input/resource semantics. It grants no model, GPU or scientific admission.

The12MiB total packet counts the shared source directory recursively, including all preserved V3/V4 and new V5 source/tests/templates/handoffs/reviews/bindings/logs. It also counts all **six V3/V4/V5 JSON/MD protocol files**, authorizations/native receipts, readiness output, future traces and temporary files. The original+entire-workspace+external-packet combined cap remains1,020,054,732bytes; comparison reservation32MiB and both1GiB free floors remain. No storage-location bypass.

## Root admission remains separate

Before any actual preparation, root must register/bind the V5 hashes and105 native tests, validate native registration semantics, bind all62 approved input hashes/identities, reconcile the original inventory/resources, and actually enforce900s/4GiB/noGPU process-tree limits.

**The existing root external actual-exec guard still targets V4. It is not usable as V5 authority.** Root must separately bind a V5 target. This delivery does not edit root envelope files. The project CLI is not an OS/cgroup limiter, and the synthetic tests do not prove that external limit enforcement works.

The whole-rig emitted-proxy question and original rendered/independent-label gate boundary are unchanged. No blinder/C-null substitute. Any future full3439-shape/480-AR engineering plan, model outputs, scoring and scientific predictions need separate root registration. None is implemented or activated here.
