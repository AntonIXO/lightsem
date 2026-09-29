# One-cell donor replay diagnosis

**Scope:** numerical replay only. No palette rollout, score, target metric, or resumed scientific output was produced. Frozen sources/protocol and the 101 partial predictions were not changed.

## Exact fixture

- Ordered cell 101 (zero-based): `S_VR_P0`, final checkpoint, seed 0.
- Recipient `jkGH7JIHDSk`, canonical `jd-e1da76a76ac38851fb95`; donor `hnVwwzCVoqg`.
- Missing output: `final_s0_e1da76a76ac38851fb95.npz`.
- Checkpoint: original `observer_fits_v1/03_seed0_S_VR_P0/final.pt`.

## Measurements

| Comparison | Maximum absolute color-logit error | Violations of frozen allclose |
|---|---:|---:|
| CPU recipient vs saved GPU | 7.8678131104e-6 | 0/640 |
| CPU donor vs saved GPU | 1.2025237083e-5 | 1/640 |
| Same-input GPU recipient vs saved GPU | 0, bitwise equal | 0/640 |
| Same-input GPU donor vs saved GPU | 0, bitwise equal | 0/640 |
| Repeated CPU donor vs first CPU donor | 0, bitwise equal | 0/640 |

Only the donor value at index167 (relative16.7s) crosses `atol=rtol=1e-5`: CPU `0.02396406978368759`, GPU/saved `0.023976095020771027`. Absolute error `1.2025237083435059e-5`; permitted componentwise tolerance `1.0239760950207712e-5`, ratio1.174366974.

Donor difference L2 `5.2921811211e-5` against saved-vector L2 `299.1098837628` gives relative L2 `1.7693100123e-7`. Maximum-error / saved infinity norm is `7.1199398349e-7`. Donor RMS difference is `2.0919182666e-6`.

Both recipient and donor cue inventories match saved cues exactly on CPU and GPU. Both clocks match. Recipient/donor source hashes, normalization/checkpoint bindings, donor map and exact retained/replaced columns pass. All 101 prior NPZ hashes are unchanged.

**Conclusion limited to this fixture:** backend-dependent FP32 computation explains the failed componentwise guard. The exact same source inputs and model on the original GPU reproduce the saved outputs bitwise. This is not evidence of a changed input window or donor intervention. It does not establish a global tolerance bound or palette-output CPU/GPU equivalence.

## Minimal proposal — not applied

Preserve the failed run and original guard. Before resumption, register a separate replay-proof exception for this exact checkpoint/recipient/donor/input hash. Require the complete regenerated CPU vector to equal the saved diagnostic CPU vector, exact frozen cues, and the recorded GPU bitwise reconstruction proof. Keep the original guard for every other cell. Do not broaden the numeric tolerance for unrelated cases from this one measurement.

A new remainder driver/receipt can reuse the hash-bound101 completed artifacts and generate only the missing85. It must emit a complete186-cell receipt, preserve the original CPU generation method, and fail on any new unproved mismatch. Such a driver/amendment needs prospective parent approval and new synthetic integrity tests before science resumes. None was implemented or executed by this diagnosis.

## Evidence

- Full receipt: `/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/diagnosis_first_donor_replay_v1/receipt.json`.
- Saved exact inputs, CPU/GPU/saved head arrays and clocks: adjacent `arrays.npz`.
- Diagnostic source: `diagnose_first_donor_replay_v1.py`; log: `diagnosis_first_donor_replay_v1.log`.
- One GPU diagnostic replay (recipient and donor, original batch size1), with TF32/autocast disabled. PyTorch2.14.0+cu130, RTX5070, CPU2 threads.

Original source coverage31/40 and event paired-null29/30 remain unchanged. No qualification/promotion claim follows from this numerical diagnosis.
