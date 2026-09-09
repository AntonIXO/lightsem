# H-L manual ratings — 2026-09-08

The 39 uploaded ratings provide a useful first preference pilot. They do not establish a validated conductor improvement: the fitted model's held-out accuracy is 56.4% for music fit and 59.0% for designed appearance, and both uncertainty intervals include chance. Product defaults remain unchanged.

**Inputs and reproducibility**

- [Original ratings](/root/lightsem/outputs/stage9/hl_pairs/ratings.jsonl): 39 complete, distinct pair/rater records, all by Anton; no ties or invalid answers; all 39 generated pairs rated. Original SHA-256 remains `def1baf86ef8d0d924d88d532352994e5902a0f1204d4a90d3ee37e16894501e`.
- [Pair metadata](/root/lightsem/outputs/stage9/hl_pairs/pairs.json): nine ConductorHybrid controls, 29 recordings and 29 registered song groups. All 78 videos exist. No brightness foils were actually generated in this batch.
- [Analysis protocol](/root/lightsem/logs/hl_analysis_protocol_2026-09-08.json): saved before fitting. Original log-knob features, ridge lambda 1, five song-grouped folds, seed 0. Foils excluded from fitting; ties counted separately. No model selection on these results.
- [Fitted results](/root/lightsem/outputs/stage9/hl_pairs/bradley_terry.json): fold memberships, individual out-of-fold predictions, input/code hashes, candidate settings and explicit gate states.
- [Material audit](/root/lightsem/logs/hl_pair_materials_audit_2026-09-08.json): per-video brightness measurements, pair summaries, identity caveats, methods and source hashes.

**What the ratings say**

The questions agree on 33/39 pairs (84.6%). Music-fit choices are A 17 / B 22; designed-appearance choices A 19 / B 20. Of 18 pairs comparing defaults with perturbed controls, defaults win 8 and perturbations win 10 on each question. These are descriptive counts from one rater, not a fresh fitted-versus-default comparison.

| Question | Held-out accuracy | Conditional song-bootstrap interval | Training-fold majority-side baseline | Held-out log loss |
| --- | ---: | ---: | ---: | ---: |
| Fits the music | 22/39 = 56.4% | 42.9–70.6% | 56.4% | 0.673 |
| Looks like a lighting engineer designed it | 23/39 = 59.0% | 44.7–73.8% | 41.0% | 0.739 |

Chance accuracy is 50%; an uninformative probability of 0.5 has log loss 0.693. The music-fit classifier does not improve accuracy over the simple majority-side baseline in these folds. The appearance classifier's probability log loss is worse than the uninformative baseline despite its higher hard-choice accuracy. The intervals resample songs over fixed out-of-fold predictions; they do not include model-refitting uncertainty or establish generalisation to other raters. These accuracy figures describe the preference model, not the lighting system beating a baseline.

The preregistered H-L accuracy gate requires >60% at at least 200 decided ratings. This pilot has 39 and misses 60% on both questions. The BAS and fresh fitted-versus-default preference gates have not been measured. H-L remains open; no hypothesis is confirmed or refuted from this pilot.

**Candidate controls for a follow-up**

Both full-data fits point toward lower base brightness, deeper brightness bounce, lower contrast gain, a narrower spatial mask and slower hue/dimmer transitions. These are exploratory main-effect directions, not isolated causal effects. The small, partly brightness-confounded batch cannot validate a knob individually.

| Control | Default | Music-fit candidate | Appearance candidate |
| --- | ---: | ---: | ---: |
| base_level | 0.120 | 0.090 | 0.083 |
| bounce_depth | 0.680 | 0.974 | 0.974 |
| contrast_gain | 0.850 | 0.698 | 0.642 |
| blinder_decay_s | 0.220 | 0.258 | 0.183 |
| strobe_decay_s | 0.120 | 0.107 | 0.124 |
| spatial_sigma | 0.420 | 0.349 | 0.329 |
| slew_hue | 0.450 | 0.368 | 0.379 |
| slew_wash_dimmer | 4.000 | 2.951 | 3.338 |
| onset_percentile | 88.000 | 83.869 | 84.516 |

These retain the original fitter's half-log-unit step along its weights, bounded to the observed setting range. They have not been rendered or deployed. Hue slew describes transition speed, not palette variety: this study does not contradict or test Anton's request for more colour variation. GroupEngine's wave amplitude, wave period, hue spread and hue drift are absent from this batch.

**Material issues to fix before the next batch**

1. **DMX metric matching did not produce equal screen brightness.** Saved metadata gives a median A/B symmetric relative brightness difference of 0.67%. CPU decoding of every frame of all 78 saved videos (900 frames/video), resized to 64×36 grayscale, gives a median difference of 16.1%; 32/39 pairs exceed 5%, with a maximum of 73.4%. This measures image luma, not calibrated physical luminance or perceived brightness. The screen-brighter arm wins only 20/39 on each question, so this is not evidence that Anton simply prefers brightness.
2. **One concrete metric/renderer mismatch is verified.** [decode.py:71](/root/lightsem/lightsem/dmx/decode.py:71) sets a blinder's RGB equal to its dimmer; [emitted_light](/root/lightsem/lightsem/dmx/decode.py:102) multiplies value by dimmer again, yielding dimmer squared. The renderer instead uses fixed warm RGB and linear dimmer ([renderer.py:47](/root/lightsem/lightsem/sim/renderer.py:47), [renderer.py:175](/root/lightsem/lightsem/sim/renderer.py:175)). This discrepancy is real; it does not by itself explain every screen-luma difference. Geometry, colour and rendering also matter. The audit did not change this metric or reinterpret old scores under a new definition.
3. **The registered identity map needs an audio-level audit.** Five rated recordings have same-title variants in the incumbent motion model's training partition, covering nine pairs. The material audit lists the candidates explicitly as unverified. Titles alone do not prove identical audio; these must not be reported as confirmed leakage. Current preference CV keeps the 29 registered groups separate, but the claim that every rated song was unseen by the pretrained trunk needs stronger verification.

The next useful batch should address the DMX/render discrepancy, verify screen matching, hold audio/seed/rig constant, and directly compare the baseline with a small set of interpretable candidates. Test new wave/colour variants in a separate batch with those controls explicitly recorded. Keep these 39 ratings and videos immutable as the original pilot.

**Completed engineering and vault update**

[rate_hl_pairs.py](/root/lightsem/rate_hl_pairs.py) now resumes per rater, validates both answers and handles replay/quit without partial writes. [fit_bradley_terry.py](/root/lightsem/fit_bradley_terry.py) now validates inputs and canonical-map conflicts, excludes foils, groups folds by song, reports baselines/conditional uncertainty, and saves per-pair predictions and provenance. Sixteen CPU tests pass; the actual 39-row output has disjoint train/validation song groups in every fold. The original ratings hash is unchanged.

Vibescience now responds from `/root/lightsem-vault`; the pilot and its limitations are recorded as progress notes rather than a completed confirmation experiment. Existing Just Dance extraction remains active on the GPU. Preference fitting and the saved-video audit used CPU; no training or new GPU rendering was launched.
