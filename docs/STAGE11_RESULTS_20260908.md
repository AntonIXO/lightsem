# Stage11 results, 2026-09-08

All 25 planned models were trained. The temporal scene model passed its preregistered statistical gates on validation and the once-locked 40-song test. The expanded-data motion model did not pass its validation improvement gate. These results do **not** establish that lights learned to dance to music: the scene task predicts six colour/brightness statistics, and its fixture engine adds programmed effects.

The user rejected those effects as outside the intended design. The [binding dance-trained product requirement](DANCE_DRIVEN_CONTRACT.md) takes precedence for subsequent implementation. The Shpongle demo's repeated flashing is a confirmed limitation of the implementation, not a successful product result.

## Statistical evidence

Every arm used five training seeds. Generations and coaches were averaged within recordings, recordings within canonical songs; confidence intervals resampled paired songs and training seeds. Scene evaluation used four generation draws. These are repetitions within one controlled campaign, not five independent experiments.

| Question | Validation | Once-locked test | Conclusion |
| --- | --- | --- | --- |
| Expanded dance data improves motion prediction | R² −0.33825 → −0.32727; gain +0.01098, 95% CI [−0.06882, +0.09996] | Not opened for this family | Required +0.02 and positive interval failed; `directional_only` |
| Temporal flow improves scene dynamics | Temporal W1 1.73830 → 0.72116, 58.51% reduction | 1.84312 → 0.59815, 67.55% reduction; absolute reduction CI [1.12283, 1.36924] | Original statistical gates passed; `supports` |
| Scene generator uses audio beyond a beat grid | Minimum paired AP advantage +0.03806 | +0.04938; donor-control CI [0.03653, 0.06520]; all four controls passed | Original statistical gates passed; `supports` |

On test, temporal flow retained 94.36% of reference variation, had 0.639× reference high-frequency energy and marginal W1 0.45712 versus 0.46922 for pointwise. Novelty timing AP was 0.10120, versus pointwise 0.10407: this experiment did not establish better event timing than the pointwise model. A model trained with shuffled audio nearly matched the temporal distributional error (0.62280), but its AP fell to 0.04332. Matching dynamics alone is not evidence of using the music.

The descriptive validation cue rate was only 2.39 major changes/minute versus 5.96 in the reference. That is a possible source of monotony and is separate from the forced renderer pulse.

[Validation aggregate](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/aggregate_val.json) · [test aggregate](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/aggregate_test.json) · [validation audit](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/independent_validation_audit.json) · [test audit](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/independent_scene_test_audit.json).

The independent audits verified 4,699 validation inputs and 3,549 test inputs, reconstructed the aggregates and found no differences. Test was opened only for the reviewed scene family. The motion test was not substituted or assigned a fabricated result.

## Shpongle and the fixture limitation

[Full Shpongle GPU video, 4:37](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/previews/shpongle_20260908/shpongle_full_gpu.mp4), rendered on RTX 5070 at 854×480, 30 fps, with stereo audio. The GPU render took 24.68 seconds and reused the saved full-track model outputs. The earlier CPU render and its provenance are preserved.

The scene mapping fixes `effect=pulse`, `rate=1 beat`, and zero group phase spread. The actual Shpongle bus contains a uniformly spaced 90 BPM grid: 0.666667 seconds between beats. GroupEngine multiplies predicted intensity by a beat-phase pulse, adds a programmed slow wave and hue drift, and originally supplied the pan/tilt oscillators. These behaviours are not decisions of the scene model. All six saved scene output channels vary over the track; that excludes a literally constant output but does not establish expressive or appropriate choreography.

The separately requested aiming prototype replaced pan/tilt with three predicted dance descriptors and held the background projector still. It left brightness/colour bytes unchanged, so it did not remove the pulse. Its motion checkpoint failed the expanded-data improvement gate and the fixture adaptation is unvalidated. The prototype is not a trained fixture trajectory head.

[Shpongle inference provenance](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/previews/shpongle_20260908/inference.json) · [GPU render provenance](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/previews/shpongle_20260908/shpongle_full_gpu.json) · [aiming comparison](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/previews/model_motion_20260908/heart_of_glass_model_motion_comparison.mp4).

## Delivered materials and numerical qualifications

[Validation curves and CSV tables](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/overview/overview.html) include all 25 models and 75 checkpoint validation measurements, with selected checkpoints marked. The plots show checkpoint validation, not invented per-update training-loss curves. SVG, PDF and PNG exports are available alongside the HTML.

[Blind comparison manifest](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/renders/pairs.json): 40 primary pairs, four repeated pairs and four sham pairs. [Full example 1](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/full_songs/song0_S_temporal.mp4) and [full example 2](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/full_songs/song1_S_temporal.mp4) use the same historical scene renderer. Human preference remains unmeasured. These materials preserve the old experiment and are not accepted as a solution of the corrected product requirement.

The first presentation attempt failed the fixed 3% encoded-video mean-luma criterion. Its files were archived. Recovery calibrated with the same final encoded-video observer used for acceptance, preserving model outputs, comparison assignments and the original tolerance. Technical presentation verification does not assess aesthetic quality.

The [final delivery audit](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/final_delivery_audit.json) passed with zero issues: 533 artifacts, all 48 pair identities and 96 published video paths, repeat/sham byte equality, durations, audio and unchanged old ratings. It decoded all 74 distinct video payloads; the largest measured pair luma gap was 1.89306%. The supervisor subsequently authenticated and reused all 17 completed test/delivery jobs, launched no child work and recorded `completed`.

The fixed FP32 full-versus-chunk stress check remains **failed** for two of fifteen checkpoints: maximum absolute difference 1.144409e−5 against 1e−5, with three of 90,090 scalar values over tolerance. Identical-input CPU FP64 evaluation of all fifteen agreed within 1.73195e−14; independent code review found matching receptive fields and update semantics. Saved-array metric changes were at most 3.06e−7 and AP was unchanged. The primary documented this rounding qualification before unchanged locked testing. This is evidence on the tested inputs, not a universal error bound or a relabelled FP32 pass.

[Numerical review](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/primary_numerical_review.json) · [original FP32 failure](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/learned_chunk_equivalence.json) · [FP64 diagnostic](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/learned_chunk_fp64_diagnostic.json) · [4/8/16-step sensitivity](/mnt/lightsem-data/experiments/lightsem/stage11_20260908/results/numerical_sensitivity_summary.json).

## Cost and retained state

Training-loop elapsed time totalled **16.646 hours**; successful trainer invocation wall times totalled **16.775 hours**, including checkpoint validation and I/O. Full validation took 846.70 seconds; the four ancillary integration-step evaluations took 32.90 seconds. The recorded experimental task-wall allocation was 1.79350 h for D, 9.15255 h for T and 6.27419 h for A, totalling 17.22024 h. The shared temporal model is charged once. These host/process times are not pure GPU kernel time and exclude preparation, engineering and presentation work. Presentation recovery took 998.86 seconds, separately from inference and training.

No OOM, nonfinite training result or seed substitution was reported. The optional extension seed was unnecessary because the training-duration minimum was already exceeded. Checkpoints, frozen inputs, old ratings and failed diagnostic evidence remain preserved. Product defaults were not promoted. Computed experiment verdicts and interpretations are stored in the LightSem experiment record; the product problem remains open.
