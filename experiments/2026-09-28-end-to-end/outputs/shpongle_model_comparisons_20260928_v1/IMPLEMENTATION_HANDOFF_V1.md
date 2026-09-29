# Implementation-only handoff

**No new code was imported, compiled, tested or executed. No checkpoint load, model forward, rendering or vault call was made.** Parent review/native tests are required.

- Driver: `run_comparison_demo_v1.py`
  SHA256 `1bebb4dcac2df882a07984056ec7a69e7d3379bd743699e9ea5119c9cf0c766c`
- Synthetic tests: `test_comparison_demo_v1.py` (11 test methods; multiple recipe/error cases)
  SHA256 `a5830102c3886aa20535301e208c82c9f4cd03c9daa3181684ecec0a61be7326`
- Draft fixed plan: `COMPARISON_PLAN_V1.json`
  SHA256 `6969fa75c3207a9a4dba344ce2d6611683b900241195e9413f920db6153ea60f`

`IMPLEMENTATION_HANDOFF_V1.json` lists all dependency hashes, exact native commands and authorization/receipt schemas. Parent must bind exact source/test/input/resource policy **before** running tests/preflight. For inference, the driver additionally requires an authenticated native test receipt with statuspassed, exact source/test maps, real_model_forward_calls0 and sealed_TEST_reads0. Keep authorizations/receipts immutable and separately versioned.

The driver only implements preflight, four sequential inferences, and no-model-call readback. It does not render. Parent can call the unchanged Sep22 renderer/auditor with the exact per-arm commands in the JSON, after enforcing the root aggregate1.5GiB media cap/free floor. Per-arm media subdirectories retain the old512MiB check without turning it into four unbounded reservations.

Source is new/versioned; original demo/model/overlay/renderer files are unchanged. Reuse saved audio94 and common transforms by reference. No audio/pose extraction, no donor selection, no rich-anchor rerun. The independent loader uses referencewidth128 and validates realizedwidth76. Fixed selectedseed0 order: S_none_P0, S_V6_P0, I_VR_P0, S_VR_P0_Ashuf. Ashuf is explicitly recipient-audio delivery, not its registered donor-route scientific evaluation.

Data outputs are under the original campaign `/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z/comparison_demos_20260928_v1`, charged to its exact1,020,054,732-byte cap. The guard reserves the whole32MiB before any arm, not32MiB per arm. No duplicate audio/baseline/probability arrays. A completed forward is saved once; incomplete directories block automatic rerun. Completion is published exclusively/atomically only after independent readback validates all469 protected slots and exact palette/dimmer reconstruction. A failed/interrupted mapping may need a separately bound saved-output-only recovery; it must not rerun the model.

Known unverified integration points: native NTFS hard-link publication, actual dedicated-environment imports, all synthetic tests, native checkpoint loader metadata, current live disk counts and GPU availability. These are tests/preflight tasks, not asserted passes. The shared native GPU lock also checks unexpected compute processes. The exact numerical/render policy remains the existing Sep22 policy; a poor/flat/no-cue result is still delivered, never repaired by new rules.
