# Completed overnight study, September 10, 2026

All twelve main trainings completed 15,200 scheduled updates, and all twelve pose probes completed 12,000 updates. Every independent input, checkpoint, target, score, DMX and video audit passed. Recorded valid GPU-stage elapsed time was **9.609 hours**, including **7.724 hours of main optimizer work**. Stage time includes required loading and output work; it is not CUDA kernel-only time. The GPU was used serially, with no additional main training after the frozen matrix.

**No new configuration qualified for acceptance in both seeds.** The full Shpongle demonstration follows the declared fallback: the existing `b03ea2b` checkpoint with the new adapter. This is a development result on the existing 40-song VAL partition; the previously opened TEST was not accessed.

| Study | Before | After | Computed verdict | All gates |
| --- | ---: | ---: | --- | --- |
| composition | 0.000000 | 0.375000 | directional_only | FAIL |
| conditioning | 0.000000 | 0.000000 | inconclusive | FAIL |
| fourier | 0.634184 | 0.625685 | directional_only | FAIL |
| muq | 0.634184 | 0.639349 | refutes | FAIL |
| representation | 0.392949 | 0.040783 | supports | PASS |
| rhythm | 0.582186 | 0.488415 | refutes | FAIL |

The primary units are expressive-clip fraction (composition), sufficiently conditioned-song fraction (conditioning), native trajectory energy score (Fourier and MuQ; lower is better), torso-normalized 2D joint error (representation), and annotated beat F1 (rhythm).

- **Adapter:** combined expressive coverage improved from 0/8 to 3/8 fixed clips, below the required 6/8. Group independence, emitted mirror geometry, visible-motion coverage and ending glare passed. Brightness coverage failed. Full-track Shpongle ending border luminance fell **73.57%**, exceeding the fixed 30% threshold; border luminance also decreased in every 30-second interval. This does not establish learned musical control or human preference.
- **Conditioning:** every new cell had 0/40 songs meeting the frozen response criterion. Repeated FiLM did not improve that criterion, and native/local alignment gates failed. Varying trajectories do not establish correspondence to the music.
- **Fourier:** aggregate energy score improved **1.34%**, with gains in all four matched architecture/seed comparisons, but none reached the required 5%. It failed the other acceptance gates shown in the report.
- **MuQ:** aggregate energy score worsened **0.81%**. Centered embeddings beat the mean-only control, but centered correct-versus-donor intervals included zero; local-use evidence remained unresolved under those distribution-shifting controls.
- **Reconstruction:** the pose-conditioned 24-dimensional code reduced mean joint error from **0.39295 to 0.04078**. Both tested descriptor-decoder widths and seeds support a representation bottleneck within this model budget. Direct audio-to-pose remains poor; an oracle pose-code reconstruction result is not an audio-only recovery result.
- **Rhythm:** the fixed PLP recipe scored **0.4884** beat F1 versus **0.5822** for the constant grid and also lost continuity. The separate pretrained Beat This! reference scored **0.7894**; GuitarSet pretraining overlap was not verified, so this is not an independent held-out state-of-the-art claim.

## Deliverables

- [15-page research PDF](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/research/LightSem_Overnight_Study.pdf)
- [Research report in Markdown](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/research/LightSem_Overnight_Study.md)
- [Full Shpongle before/after comparison](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/shpongle/before_after.mp4)
- [New full-stage Shpongle](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/shpongle/expression_v4.mp4)
- [Head-only Shpongle](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/shpongle/heads_only.mp4)
- [Pose comparison example](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/pose/wnw4_ZfVmrA_c0.mp4)
- [Chart data workbook](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/research/chart_data.xlsx)
- [Machine-readable outcomes](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/scientific_outcomes.json)
- [Computed experiment verdicts](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/vibescience_close.json)
- [Compute ledger](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/compute_ledger.json)

The report includes 21 primary-source links, all architecture/seed effects, confidence intervals, natural-event and alternate-coach controls, diagnostic limitations, and preserved technical incidents. The Fourier column-index error and additional MERT weight-mapping error were corrected before main training; affected unscored work was archived, and unaffected probes were retained only after exact input/code checks. All six experiment records, their magnitude gates, interpretations, calibration and causal-map records were updated in `/root/lightsem-vault`. Negative results remain conditional on the tested recipe.

Main source snapshot: `bc9eab5e730df1c8794ca55e9aa6e66799f1e9ca7f6ddb066bda6b17f8145c77`; core implementation commit: `40130f5`. All 4,320 independently regenerated DMX outputs matched exactly; maximum independent energy-score discrepancy was below 4e-15. Model weights, frozen source, datasets and generated media remain in the experiment directory outside Git.

Mechanical acceleration limits and additional fixture types remain deferred. The A4 outputs are still trained/evaluated, but the new adapter does not directly consume those four continuous outputs. The study does not resolve the product's musical-control requirement.

[Final delivery verification](/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/delivery_verification.json) passed: 15 PDF pages, 21 primary citation links, 17 data tables with matching CSV/workbook row counts, completed stage manifests and final artifact hashes.
