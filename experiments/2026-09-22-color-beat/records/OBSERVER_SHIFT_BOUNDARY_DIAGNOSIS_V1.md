# Observer circular-shift boundary diagnosis

## Scope and status

Read-only diagnosis of the failed selected and final scoring attempts. Native command used `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh`.
No model inference, GPU work, vault operation, AP calculation, or aggregate result inspection occurred. `score_record`, `summarize_campaign`, and `visual_event_ap` were replaced with fail-fast sentinels **inside the diagnostic process only**. Original implementation file hashes were identical before and after both successful diagnostic runs. Frozen checkpoint, registration, raw prediction, and source artifacts were read and hash-checked, not modified.

## First failure (both roles)

- Recipe `S_none_P0`, seed `0`.
- Selected checkpoint: `observer_fits_v1/01_seed0_S_none_P0/best.pt`.
- Final checkpoint: `observer_fits_v1/01_seed0_S_none_P0/final.pt`.
- First prediction receipt / first record in each score index: `E3Af6_rVeSQ`.
- Head: `color` (second visited record/head, after accent).
- Draw: zero-based `4` (fifth); segment: zero-based `0`.
- Segment: `[126.6, 134.6)`; duration `8.0`; offset `1.9`.
- Candidate: `132.7`, zero-based segment index `22` selected / `23` final.

| Value | Binary64 hexadecimal |
|---|---|
| start `126.6` | `0x1.fa66666666666p+6` |
| stop `134.6` | `0x1.0d33333333333p+7` |
| candidate `132.7` | `0x1.0966666666666p+7` |
| offset `1.9` | `0x1.e666666666666p+0` |
| local and modulo residue `7.999999999999995` | `0x1.ffffffffffffap+2` |
| emitted time `134.6` | `0x1.0d33333333333p+7` |
| largest valid representable time | `0x1.0d33333333332p+7` |

The residue is strictly below duration. Adding the nonzero source-time origin rounds it to the excluded right endpoint. Thus the excess beyond the endpoint is **0 seconds**, but the value is **1 ULP = 2.842170943040401e-14 seconds** above the greatest representable valid time. It fails half-open membership, not by a material positive overshoot.

Exact rational arithmetic over the binary64 candidate, start, stop, and offset gives a real result **5.773159728050814e-15 seconds = 0.203125 endpoint ULP** below the stop. Its nearest binary64 value is the excluded endpoint. Wrapping that rounded value to the start would put this near-right-boundary event on the wrong side of the segment.

## Minimal proposal (not applied to production)

Keep the existing assignment and then correct only equality to the excluded endpoint:

```python
shifted[take] = a + np.mod(local, duration)
shifted[take] = np.where(shifted[take] == b, np.nextafter(b, a), shifted[take])
```

Place this before displacement, jitter, and close-pair diagnostics. Keep the original final membership assertion. This chooses the nearest representable in-domain time when rounding lands exactly on `b`. It does not broaden eligibility, apply a time tolerance, rerun NMS, change offset selection, change confidence/count/order, or alter other shifted values. Values below `a` or above `b` still fail. Do not use broad clipping, tolerance-based eligibility, or a second wrap to `a`.

This is a finite-representation boundary correction only. It does not claim to replace all floating modulo arithmetic with exact rational arithmetic.

## Regression/property checks

`test_observer_shift_boundary_proposal_v2.py` passed in the native CPU environment. It clones only the frozen function's text into an isolated namespace and tests the two-line proposal; the source file stays untouched.

1. Both saved frozen fixtures fail under the original function and pass all20 draws with the proposal. Exactly one time is corrected in each fixture. Confidence/count/order remain unchanged.
2. The failing value moves inward exactly one ULP; it does not wrap to start.
3. Disjoint actual-source segments retain each event's original segment at native10/40Hz. Ordinary valid outputs are bitwise unchanged.
4. Short unsupported segments remain undefined; empty candidates remain empty.
5. A deliberately invalid duration still produces `ContractError`; true escapes are not broadly clipped away.
6. Seed20260922 generated1,000,000 deterministic numeric geometry cases with nonnegative source starts up to100000s and durations0.001–8s, including deliberately near-wrap sums. Every result stayed in its segment; the130 corrected outputs were exactly upper-endpoint equalities moved one ULP. All other outputs were bitwise unchanged.

The first test driver version passed its assertions but failed **receipt serialization** because a diagnostic local variable shadowed the replacement string. Its script, partial receipt, and failure log remain preserved. Version2 only fixes that diagnostic naming/receipt issue. Use the complete v2 test receipt, not the partial v1 receipt.

## Artifacts

- Reproducer: `diagnose_observer_shift_boundary_v1.py`.
- Selected compact fixture: `observer_shift_boundary_fixture_selected_v1.json`.
- Final compact fixture: `observer_shift_boundary_fixture_final_v1.json`.
- Native frozen-input receipt: `observer_shift_boundary_diagnostic_receipt_v1.json`.
- Geometry tests: `test_observer_shift_boundary_proposal_v2.py`.
- Complete test receipt: `observer_shift_boundary_proposal_test_receipt_v2.json`.
- Preserved test-driver serialization failure: `observer_shift_boundary_proposal_test_v1_failure.txt`.

All files are in `/root/lightsem/docs/research/2026-09-22-color-beat`. Fixture provenance binds the full input paths/hashes, target content hash, checkpoint, score registration, and score-index/record positions. This diagnosis does not authorize modified scoring. Any scoring correction needs a separately registered, hash-bound amendment that preserves original raw prediction implementation bindings.
