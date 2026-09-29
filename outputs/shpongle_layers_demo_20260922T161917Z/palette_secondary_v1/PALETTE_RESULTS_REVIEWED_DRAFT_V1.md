# Palette-history secondary results — 2026-09-22

## Outcome

**Negative in this fixed observer-proxy setup. No promotion.** The primary selected checkpoints have *higher* (worse) temporal error with generated palette history than with the trained-null, independently sampled cue-held baseline. Both checkpoint roles fail the preregistered benefit conjunction. This is not a broad refutation of palette history or proof of LIGHT quality.

| Checkpoint role | Own history | Independent null | Limited tempo-donor history | Hold first palette |
|---|---:|---:|---:|---:|
| Selected (primary) | 0.859022 | 0.830589 | 0.850720 | 1.119751 |
| Final (companion) | 0.750180 | 0.755757 | 0.754158 | 0.903743 |

These are TRAIN-IQR-scaled temporal errors; lower is better. TRAIN component IQRs are 1.598265 (log residence), 0.102808 (mark change), 0.095620 (drift/s), and 0.127058 (8-second change). Marginal IQR is 0.069469. Do not compare absolute errors across checkpoint roles as a checkpoint-selection experiment: each role supplies its own cue inventory and adapted reference.

- **Selected:** null-minus-own = **−0.028433**, or **3.42% worse** than null. Family-adjusted 97.5% paired interval **[−0.039850, −0.016604]**, Holm-2 p=0.00019998. Donor-minus-own = **−0.008302**, interval **[−0.016023, −0.000323]**, Holm p=0.019298. Both directions oppose the predicted benefit.
- **Final:** null-minus-own = **+0.005577**, or **0.738% better** than null, interval **[−0.008024, +0.022413]**, Holm p=0.795520. Donor-minus-own = **+0.003978**, interval **[−0.032216, +0.041072]**, Holm p=0.811519. This does not establish a benefit and is below both required thresholds (0.02 absolute and 15% relative).
- Persistence is worse than own-history in each role, but beating a constant palette does not rescue failure against the null and donor controls.

## Support, floors and component checks

All186 checkpoint/record cells and744 paired seed/draw rows completed. Each role has30/31 common canonical groups and30 timing-eligible groups,17 with reference events. `8UTvmHbgc5w` has92.656% valid full-window time but **zero uninterrupted8-second pairs**. Its lag component stays undefined; no component is dropped to manufacture a score. This allowed one-group loss was prespecified. The original31/40 source gate remains failed; the event study's29/30 null failure is unchanged, not imported as a palette prerequisite.

Self comparison is0 in both roles. Quantization-floor temporal errors are0.465517 selected and0.538344 final; marginal floors are0.725234 and0.928224. These are reported, **not subtracted**.

| Scaled component | Selected own | Selected null | Final own | Final null |
|---|---:|---:|---:|---:|
| Log residence | 0.448990 | 0.446238 | 0.186868 | 0.181640 |
| Mark change | 0.795741 | 0.730053 | 0.622291 | 0.650804 |
| Inter-cue drift/s | 1.409450 | 1.409450 | 1.419126 | 1.419126 |
| 8-second change | 0.781909 | 0.736617 | 0.772436 | 0.771459 |

Generated arms are held between cues, so their inter-cue drift is zero by construction. The shared nonzero **error against raw observer drift** is not a history benefit. The summary JSON retains all four arms' component errors.

## Passing guards are not a quality result

Selected own-history mean occupancy is18.864 tokens versus16.931 for null and4.833 in the adapted reference. Effective diversity is13.475 versus11.546 for null and3.331 reference. Mean per-record occupancy/diversity ratios are4.583/4.521 (null4.161/3.944), not ratios of the raw macro means. The one-sided >=0.70 anti-collapse checks pass, but **excess diversity is not success at matching the reference**.

Final own occupancy is2.836 versus2.869 null and2.156 reference; effective diversity2.214 versus2.242 null and1.803 reference. Own/null marginal errors are1.740577/1.744798 selected and2.138222/2.144781 final: the <=5% degradation guard passes, not a demonstrated meaningful marginal gain.

Actual differing-token emitted-cue AP is0.161290 own versus0.155868 null (selected); the difference0.005423 has95% interval[−0.003243,+0.013799]. Final own and null both0.044118. False emitted cues/min are28.878 own versus28.203 null (selected), and1.649 versus1.732 (final). Persistence hasAP0 and0false cues/min. AP is averaged over17 event-bearing groups. False emitted cues/min is averaged over30 timing-supported groups, including13 with no reference events. These are observer metrics, not beat accuracy or the original all-candidate color-head AP. The fixed descriptive timing noninferiority check passes; no timing-improvement claim follows.

## Provenance and limits

- Original experiment definition was registered before computation. However, source/test and calibration update calls used `problem_id` instead of `id`; native validation-error strings were mistakenly treated as success via process exit0. They were frozen locally before the first101 outputs, but not successfully logged in the required vault order. Failed receipts remain intact. **The later repair is retrospective; full prospective vault compliance is false.**
- A separately tested43-check continuation had a verified native source registration before the remaining85 outputs. The combined186-output hash was successfully logged before scoring. The101 originals were independently authenticated and reused without rerun or overwrite. Both producers are explicit.
- Exactly six final-role CPU-head replay exceptions require exact CPU/input/checkpoint hashes, stable CPU repeats, bitwise GPU reconstruction of saved vectors, and exact cues. No blanket tolerance change, GPU palette generation, changed metrics, fitting, or threshold tuning occurred. All93 selected-primary outputs preceded this numerical amendment and passed the original guard.
- Original31/40 source coverage, missing original shuffled-palette control, unknown observer/pretraining nuisances and historical DEV exposure remain. Full64-second scoring includes less-supervised edges. The delivered demo uses argmax; this secondary uses four preregistered stochastic draws. No equivalence or improvement claim for the demo follows.
- Original hypothesis qualification remains undefined. Do not call the sign-only vault auto-closure, which could turn this limited negative into an overbroad refutation.

## Artifacts

- Complete raw report: `/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/palette_secondary_report_v1.json`
- Combined186 outputs: `predictions_continuation_v1/combined_complete.json` in the same root.
- Compact report: `PALETTE_SECONDARY_RESULTS_SUMMARY_V1.json` next to this file.
- Independent read-only audit: pending at this draft; separate audit artifact will retain its own result.
- Demo remains `/root/shpongle_light_demo.mp4`, unchanged.
