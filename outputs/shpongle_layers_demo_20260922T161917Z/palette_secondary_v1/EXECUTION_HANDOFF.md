# Palette secondary implementation handoff

Release: `IMPLEMENTATION_RELEASE_V1.json`. Frozen protocol + clarification are authoritative; the earlier review is historical planning, not the final definition.

- New driver and sidecar tests only; frozen project source files unchanged.
- 29 new synthetic CPU tests passed, 0 failed. See `PALETTE_SECONDARY_CPU_RECEIPT_V1.json`.
- No real calibration, inference or scoring was run by this implementation task.
- Read-only independent code review blockers were corrected. The release includes strict provenance, numeric, placeholder, support, timing, component and storage guards.

## Parent steps

1. Review the release hashes. Prospectively log the tested implementation/source/input bindings in `exp-layers-20260922-palette-secondary`.
2. Copy `REGISTRATION_FINAL_UNAUTHORIZED_DRAFT.json` to a **new** `REGISTRATION_AUTHORIZED_V1.json`. Set `authorized=true`, `authorized_phases=["calibrate","predict","score"]`. Do not edit it afterward. It includes the tested source/CPU receipt, six exact old prediction receipts and all frozen input bindings. The wrapper links the actual native vault receipt; it is not scientific execution permission itself.
3. Run only calibration first. Inspect its receipt. If `status=blocked`, stop. Missing or zero TRAIN calibration components must not be repaired by dropping records or changing definitions.
4. Log the successful calibration file hash before prediction. Run prediction separately.
5. Log the complete prediction receipt hash before scoring. Run scoring separately. The same immutable registration hash binds all phase receipts.
6. Preserve old failures/results. No automatic hypothesis closure, default promotion, repeated fits or R1 claim.

## Exact commands

Commands assume the parent creates the authorized registration at the path above. They are not run by this handoff. Use a nonblocking `bash()` handle for each command; inspect after its completion event. Logs are unbuffered and saved separately.

```text
CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/run_palette_history_secondary_v1.py calibrate --registration /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/REGISTRATION_AUTHORIZED_V1.json --output /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/calibration_v1.json > /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/calibration_v1.log 2>&1

CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/run_palette_history_secondary_v1.py predict --device cpu --registration /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/REGISTRATION_AUTHORIZED_V1.json --calibration /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/calibration_v1.json --output /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/predictions_v1 > /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/prediction_v1.log 2>&1

CUDA_VISIBLE_DEVICES='' TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/run_palette_history_secondary_v1.py score --registration /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/REGISTRATION_AUTHORIZED_V1.json --calibration /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/calibration_v1.json --predictions /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/predictions_v1/complete.json --output /mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/scores_v1.json > /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/score_v1.log 2>&1
```

## Measurement and scope

Registered diagnostic `layers-palette-temporal-error`: before = selected-role `reports.selected.condition_means.null_state.temporal_mean_scaled`; after = `reports.selected.condition_means.own_state.temporal_mean_scaled`. Displayed benefit remains null-minus-own. Final is the separate companion, never an alternative selected after results.

Report raw/scaled four temporal components, marginal error, generated/reference occupancy and effective diversity, their ratios, common support, censored segments, unsupported interval count, self/quantization/persistence controls, actual differing-token emitted timing, and both benefit CIs/Holm2. Missing calibration/support means undefined, not a fabricated zero/refutation.

Original source coverage **31/40<80%** and original event paired-null **29/30** remain unchanged. The event-null gate is not imported into the new palette instrument. Original shuffled-palette control remains missing; tempo-donor is a separately named limited control. Even a passing narrow secondary result cannot qualify the original LIGHT-quality hypothesis or establish human acceptance.
