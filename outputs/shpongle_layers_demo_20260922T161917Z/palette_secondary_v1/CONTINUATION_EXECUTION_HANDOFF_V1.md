# Palette continuation V1: tested, not executed

43 synthetic CPU tests passed. No real resume or scoring ran.

Use `REGISTRATION_CONTINUATION_FINAL_UNAUTHORIZED_DRAFT_V1.json` as the schema.
Create a NEW authorized registration after successful native source registration.
Do not change original V1 artifacts. Do not change the tested wrapper/tests.

## Native source binding

Save a successful `update_problem(id=..., add_update=...)` RESULT directly as JSON.
It must be an object with `id=dancing-light-generative-20260903` and `updates`.
One update's `text` must contain all of:

- `palette-history-continuation-20260922-v1`
- `exp-layers-20260922-palette-secondary`
- `RETROSPECTIVE V1 EXECUTION DEVIATION`
- deviation artifact hash `df755a9bc9746e74f4270721f36df197dad1142793af70bf68346be241c56205`

And these exact SHA256 values:

- protocol: `622cbfe13690bbd3a678eb0f9d14e58db98c71377c6c0a8617bc6743576d4b35`
- wrapper: `b008682e47aeb3471f0570b898fad20068f7f246e9c4396ebaed1560801aa6a7`
- tests: `f4a66b09e0638b910d00eeeeb33f62e9198e621ae64d590987ae1068f381f2d4`
- base_registration: `42f3a1faa773603ee5e55fb34c461264cbdc834c58c33d9deb7611f97ad8c153`
- calibration: `2675fc76b4e71d951b26d2e7eea9d98bd85870f744e12fb81b00fe4ac76216d7`
- partial_receipt: `67e49eafa0331d06d553090d7e4559ebecba07e315ada2bb77825192ba6ccca3`
- exception_map: `f761df59931d83ef432359dc340cc093abf72f22ba09a0162429a804bd65df03`
- amendment: `9e329c257e894df4e0bdbbe8506e1a4566a16468f10a4c5d0f5c016dfb0a840b`

`CONTINUATION_VAULT_WRAPPER_SCHEMA_V1.json` gives the required wrapper fields.
Create a new file with status `registered` only AFTER the native result passes.
Point the NEW authorized registration's `vault_receipt` to that wrapper+hash.
Use authorized phases `resume-predict` and `score`. Native validation-error
strings, even with process exit0, are rejected. The loader verifies native
problem identity and one complete source-binding update, not just file hashes.

The registration's `execution_deviation` is copied unchanged into combined and
score reports. It explicitly declares `fully_prospective_vault_compliance=false`.
The original definition start succeeded. V1 source/test and calibration updates
were not successfully logged before101 outputs. Late disclosure is not preregistration.
This does not change the separate completed observer-study receipts or scope.

## Commands (parent authorization required)

```
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/run_palette_history_continuation_v1.py resume-predict --registration <NEW_AUTHORIZED_REGISTRATION> --output /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/predictions_continuation_v1

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/run_palette_history_continuation_v1.py score --registration <SAME_NEW_AUTHORIZED_REGISTRATION> --predictions /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/predictions_continuation_v1/combined_complete.json --output /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/palette_secondary_report_v1.json
```

Run nonblocking, redirect each phase to a new log. Inspect native status,
receipt, inventory, and output hashes before the next phase.

## Exact continuation contract

- Recheck and reuse all101 partial files in place, no copies/forwards.
- Calculate only missing85 with original CPU FP32 model+sampling,170 head
  forwards (recipient+donor per cell), same2 CPU/BLAS threads.
- Preserve original1e-5 allclose for every nonexception head.
- Six exact failed-head exceptions only, not all heads of six cells.
- Exception requires exact checkpoint, input hashes, saved prediction,
  CPU vector, GPU bitwise proof, and both cue inventories unchanged.
- No new tolerance, epsilon, cues, reference fitting, model math, or metric changes.
- New combined receipt contains both execution producers and per-entry lineage.
- Original interrupted directory never receives a `complete.json`.
- Original calibration is validated against its V1 registration antecedent.
- New scorer validates lineage and uses original frozen pm and V1 score functions.
- Storage remains64MiB across the study and >=1GiB free on /mnt.

Scientific limitations and original failed/missing gates stay unchanged.
No vault writes were performed by this worker.
