# Native dance dynamics learned from recordings

The native learning experiment is implemented, trained and delivered. It changes speed inside the generated motion; there is no playback retiming, rate floor or fragment recomposition. **Neither candidate earned an overall improvement or a new baseline.**

[Lit 3D human-review gallery](http://localhost:8780/) · [Full-track comparisons](http://localhost:8779/) · [Architecture](ARCHITECTURE.md)

The 3D gallery contains A/B/C for both Shpongle and AURORA — Cure For Me, covering source 90–150 seconds. The interval deliberately includes C’s problematic Shpongle burst near 135 seconds. A is an experimental control; B/C are failure diagnostics. The four full-track videos retain both body views, actual DMX stages and synchronized traces.

## What was trained

Six matched runs completed 30,000 updates each: A/B/C with seeds 0 and 1. Displayed models always use seed 0, draw 7001.

- **A:** D7 warm start with matched continued training.
- **B:** A plus motion-derived regional speed, acceleration, low-motion duration and recurrence losses on generated trajectories.
- **C:** B plus a jointly trained whole-song planner. Its 48 dynamics and 32 continuous motif channels enter adapters in all eight flow blocks.

Just Dance contributes 72 training song groups (4.74 hours) and 24 development groups (1.50 hours) of confidence-masked **2D observations**. FineDance supplies genuine paired 3D/root/contact supervision, using the verified 109/31 group split. Both corpora and showcases are historically exposed; this is not an unseen-test claim. MuQ remains frozen. The failed source-activity ridge threshold is not used.

## Numerical result

These speed-profile values are descriptive means over available observed cells and both seeds; lower is better.

| Model | JD speed-profile error | Change versus A |
|---|---:|---:|
| A — continued training | 0.3026 | control |
| B — direct dynamics losses | **0.2035** | **−32.7%** |
| C — joint planner | 0.2147 | −29.0% |

B improves the available-cell score by 33.1% and 32.4% in the two seeds. C is 5.4% and 5.6% worse than B. B also increases 3D sliding and jerk. C has a better physical tradeoff on FineDance and Cure, but its displayed Shpongle model develops sharp bursts and worse light transfer.

For displayed Shpongle, requested-to-emitted velocity correlation falls from A’s **0.821** to **0.719** for B and **0.631** for C. On Cure the values are **0.829 / 0.767 / 0.799**. Both candidates fail physical nonregression against A on the full songs. Neither has a whole-body hold below the exploratory 0.05 m/s threshold; removing imposed zero-rate stops did not establish learned sustained rests.

The planner is active: shifting it while keeping audio fixed changes actual body and DMX. However, correct audio/plan timing does not consistently beat shifted or donor controls in emitted-light attack alignment. The target-informed constant-speed oracle also beats the models on the selected metric-control windows; it is a diagnostic, not a deployable model or rendered fallback.

**Qualification limitation:** 38 of 72 JD windows lack some region/scale observations. Missing cells were not filled with zero and windows were not changed. The complete registered primary was left unmeasured in Vibescience; the available-cell result was logged separately. Both computed primary verdicts are **inconclusive**, while the measured physical and transfer failures remain real. Artistic acceptance is the user’s review.

## Graphs and evidence

![Tradeoffs](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/early_validation_v1/tradeoffs.png)

- [Shpongle regional movement](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/plots/shpongle_regional_speed.png)
- [Cure regional movement](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/plots/cure_for_me_regional_speed.png)
- [Recorded versus generated speed example](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/plots/recorded_vs_generated.png)
- [Complete machine-readable report](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/FINAL_REPORT.json)
- [Matched validation and confidence intervals](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/VALIDATION.json)
- [Independent score recomputation](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/VALIDATION_INDEPENDENT.json)
- [Metric controls](/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/reports/METRIC_CONTROLS.json)

## Transfer and delivery checks

Native bodies remain at 30 Hz and are explicitly resampled geometrically to the 40 Hz lighting clock. Fixed V3 mapper `01`, fixture limits, 8 beams, 5 simulated lasers and 50 bulbs are unchanged. P2 color and the restored historical brightness controller are fixed. Shpongle’s first-30-second beam dimmer remains about **11.77%**.

Frozen-body stability, all six limb probes and all 54 added-output responses passed, with common outputs preserved at counts 1/63/117. Variation and source sensitivity do not by themselves establish choreography.

All ten videos — four full comparisons and six 3D clips — passed full software decoding, uniform frame timing and exact audio-reference PCM checks: **39,540 frames**. Both galleries passed browser playback, seeking, synchronization and audio-handoff checks. The 3D view removes root XZ translation for presentation; pose, height, facing and timing remain intact. Close-only neutral fill improves body readability and is explicitly separate from actual emitted lighting; wide-stage pixels remain unchanged.

The native campaign used **4.47 GPU-stage hours**, bringing the cumulative dynamic-dance ledger to **9.96 hours**, below 24. At least 20 GiB remained free. Temporary compiler caches were inventoried and removed; all scientific outputs, checkpoints, optimizer states and failures were retained.

Engineering recoveries are preserved in `records/`: an inference noise-device mismatch stopped before scores; guarded output reuse was rejected; postprocessing serialization/path issues were fixed without changing measurements. No training run, model, seed or acceptance threshold was replaced.

The next warranted experiment is stronger physical/contact and transfer supervision on generated Just Dance rollouts, with an observable primary metric. It should remain native training rather than another post-generation speed correction.
