# Why LightSem moves but does not yet dance

Date: 2026-09-10. Independent mechanism review. **No code changes, training, vault writes, TEST access, new model evaluation, or video acceptance occurred.** This report reads existing reports and source code. Its filter calculation is an unpreregistered analytic diagnostic, not a measured playback experiment.

## Research-service result

The generic Perplexity MCP discovery failed because `PERPLEXITY_REASON_MODEL` was unavailable. The documented `local_mcp.perplexity.list_tools()` fallback succeeded. A `perplexity_reason` request included the complete requested reports, `CLAUDE.md`, and the mapper, music model, bandwidth wrapper, and audio-dynamics code. **The request was not answered:** “Perplexity quota exhausted: Pro Search has 0 queries remaining.” No Perplexity insights are claimed. The full prompt and server error are in `outputs/dance_investigation_reason_request.json` and `outputs/dance_investigation_reason.log`. Serper also had no configured key. Three primary-source web pages were fetched directly and inspected instead; URLs and exact uses appear below.

## Main finding

**Two bottlenecks coexist. More visible movement does not establish music-conditioned choreography.**

1. The native music generator has not demonstrated the required advantage of the correct music over wrong music. This remains true after the geometric correction.
2. The mapper throws away much of the body's relational and event structure. Its floor projection, smoothing, and fixed fixture roles can make coherent dance look like slowly deforming illumination.
3. Current spatial gates detect concentration and insufficient movement. They do not measure phrase structure, anticipation, gesture timing, or whether a human sees dancing. Revision02's failed visual gate is consistent with this limitation, not a contradiction.

A replacement model sweep is not the next diagnostic. Use matched source/mapper and time-alignment controls to decide which seam loses what.

## What the existing evidence supports

### Generator: inadequate qualification, not proven complete audio blindness

The native flow's final energy score is .8796, versus 1.1934 for matched regression. But its correct-audio improvement over donor audio is only 2.34% (95% CI [-.15%, 4.64%]), and over a contiguous within-track shift only .50% ([-.60%, 1.44%]); both fail the frozen 5% criterion. Regression also fails. Thus lower marginal/distribution score cannot establish timing-specific music conditioning. These particular summary statistics and controls did not establish the required effect; they do **not** prove zero dependence in every joint, frequency, or song. [L1]

Generated motion also fails independently: flow energy is 1.85 times reference and excess foot sliding is .882 m/s. Contact labels remain populated, which rules out treating contact coverage as physical support. Native139 is an identity representation accepted for engineering fidelity, not a qualified generator. More integration steps already worsened the incumbent's jitter and velocity energy. [L1]

Code provides plausible mechanisms, not causal identification. `MusicModel.vector_field` concatenates Base92 audio with noisy motion only at its input. Residual blocks receive flow time, not a fresh music representation. A finite dilated stack can use audio, so this is not proof that concatenation must fail. Its loss is normalized native MSE plus derivative-error MSE; it has no FK/contact loss. Correct derivatives in channel coordinates need not yield stationary feet after rotation projection and FK. The energy-score result is compatible with learning a plausible motion prior without sufficiently precise audio dependence. [L4]

### Mapper: geometry was a real independent failure

The rejected mapper put targets in mid-air and compressed fixture anchors, causing ray continuation to focus on the floor. Requested and emitted hits agreed closely, and six independent bodies also collapsed. This localizes that old collapse to geometry, not pan/tilt encoding or a shared latent draw. [L2]

With source motion and every non-angle DMX byte fixed, surface revision02 changes median first-surface span from 1.817 to 6.490 m and active translation-removed 4 s extent from .454 to 1.130 m. This is direct evidence that the adapter can strongly change the visible result independently of music generation. It is not proof of better music fit. Its visual gate still fails: reconvergence around 194–201 s and one floor-facing fan family persist. [L2]

## Specific ways the mapper can erase dancing

### 1. World yaw is mixed with articulation and can cancel the lane anchors

In `surface_targets`, the regional vector is world-frame `p_region - p_root`. The code subtracts an **unrotated** neutral vector and adds the result to a fixed floor lane. This preserves world turns, but a turn also becomes an inward displacement relative to each fixed lane.

For root rotation Q, local articulation a, and neutral regional vector r0:

`world_delta = Q*a + (Q - I)*r0`.

The second term is not new limb articulation. It is the rigidly rotated neutral skeleton. A 180-degree yaw gives an approximately `-2*r0.x` horizontal term. The neutral right wrist x is about -.719 m, so at activity=1 and revision02 gain=1.8 its horizontal contribution is approximately `2.8 * 1.8 * 1.438 = +7.25 m`, against the negative outer lane anchor `-.95*5 = -4.75 m`. This is enough to cross the center even with a rigid neutral body. The opposite wrist crosses in the other direction. Clamps then flatten some trajectories.

This is an algebraic counterexample to guaranteed lane preservation, **not evidence that the actual 194–201 s interval is caused by a 180-degree rigid turn**. That requires the saved root rotation and target traces. Fixed anatomical identities are correct; swapping left/right on every turn is not the fix. [L3, L5]

### 2. The projection has a large null space

Each fixture follows one wrist or one weighted regional centroid. A 3-D regional delta becomes a 2-D floor displacement:

`x <- 2.8*dx`, `z <- 2.6*dz + 2.2*dy`.

A regional movement with `dx=0` and `2.6*dz=-2.2*dy` is invisible to this projection before the small shared-root term. Different joint configurations can also share the same centroid. All target y values are zero. Vertical body rise, forward reach, a limb fold, and changes in support do not necessarily remain distinct in the visible light trajectory. Three fixed pair roles and fixed depth offsets then favor variations of one floor composition. This is an expressivity limitation, not proof that floor-only lighting can never dance. [L3]

The four generated contact channels are not used as explicit mapping inputs. Accent uses the mean speed of four distal joints, shared across fixtures. Dimmer is dominated by common loudness, with only small regional modulation. Palette is also mainly common, smoothed over 10 seconds. The system preserves some spatial correlation but not an explicit transfer of support changes, attack/hold/release, or regional rhythmic roles. [L3]

### 3. Tracking can turn rhythmic gestures into late, low-amplitude drift

At 40 Hz, linearizing small pan/tilt excursions with no clamp or speed/acceleration saturation gives two discrete low-pass stages:

`H(z) = alpha/[1-(1-alpha)z^-1] * beta/[1-(1-beta)z^-1]`,

where `alpha=1-exp(-1/(40*tau_target))` and `beta=1/(40*tau_tracking)`.

| Filters | Frequency | Amplitude retained | Phase delay |
|---|---:|---:|---:|
| Original .55/.35 s | 1 Hz | .119 | .360 s |
| Original .55/.35 s | 2 Hz | .033 | .195 s |
| Revision02 .20/.18 s | 1 Hz | .430 | .248 s |
| Revision02 .20/.18 s | 2 Hz | .160 | .160 s |

These are **analytic small-signal values**, not fitted delays of the full nonlinear mapper. Target filtering occurs before the nonlinear angle transform, and quantization, acceleration limiting, changing activity and saturation can change the result. Even so, “active speed passes” cannot show that a rhythmic direction reversal survives with its timing intact. Gain cannot recover the original phase and may drive clamps harder. The old measured raw→EMA→tracking extent loss independently makes this mechanism worth testing. [L2, L3]

### 4. Loudness gives dynamics, not full rhythm or musical form

The direct audio path blends 400 ms momentary and 3 s short-term LUFS, then smooths activity over .30 s. It alters brightness, geometric gain and desired motor speed, not an explicit beat or phrase state. Two passages with similar loudness but different syncopation, instrumentation or harmonic change can receive similar direct controls. The 100 ms metadata cadence does not make a 3-second loudness measure a fast onset detector. [L3, W1]

The existing brightness ablation proves useful dynamic contrast. It cannot rescue the upstream music claim. It is also important to distinguish `compose(..., dynamics=False)`, which changes activity-dependent geometry, from a **dimmer-byte-only** ablation that holds geometry fixed. [L1, L3]

## Falsifiable controls before changing training

Use existing TRAIN/VAL diagnostic material only. Freeze intervals and analysis windows in advance. Keep source, RNG, gain, photometry and rendering fixed where each test requires them. Source mocap is an **offline diagnostic oracle**, never an input to music-only playback.

| Question | Matched control | Result that would change the diagnosis |
|---|---|---|
| Can the mapper transmit real dance timing? | Map paired clean source mocap and native generated motion for the same available VAL clips. Show synchronized skeleton and lights; also show identical light geometry with audio shifted. | Good skeleton/audio fit but poor oracle-light fit implicates mapping. Oracle-light fit with poor generated-light fit implicates generation. Neither alone qualifies arbitrary-music playback. |
| Is weak apparent music fit hidden by loudness? | Hold a generated-motion draw fixed and compare true vs fixed activity; use separate true vs shifted loudness experiments. Also regenerate from true vs donor/shifted music with **the same full initial noise field**, with direct audio disabled. | Geometry that barely changes across generated-music substitutions supports weak measured conditioning. Loudness-only gains are attributed only to direct controls. |
| Do turns cause contraction? | Native neutral pose under controlled yaw; articulation-only with root orientation held fixed; yaw-only with local joints fixed. Apply the same mapper. Then decompose the real rejected interval into `Q*a` and `(Q-I)*r0`. | Contraction in yaw-only is a mapper property. Dominance of the yaw term in the real interval supports that mechanism there. Failure to explain the interval redirects attention to articulation or clamps. |
| Where is timing lost? | Track the same regional signal at raw target, EMA target and emitted first-surface hit. Use a small-amplitude diagnostic frequency input, plus actual motion events. | A frequency-dependent phase/amplitude deficit downstream of intact source events identifies bandwidth loss. Matched synthetic inputs are calibration only, never production chases. |
| Is spread hiding impoverished composition? | Static fan, rigid translated fan, jitter, flashing, and beat-triggered/metronomic moving-beam controls; brightness-match views. | A proposed dance score must not call these successful choreography simply because they are wide, active, or on beat. |
| Is timing merely tempo matching? | Within-track half-beat shifts, beat-preserving whole-bar shifts, and tempo/loudness-matched donor passages. Keep non-shifted audio presentation fixed. | If an apparent advantage survives phase-destroying shifts it is not timing-specific; if whole-bar shifts preserve it, it may reflect pulse rather than phrase alignment. Specify wrap-boundary exclusions. |

Measure event lag and event precision/recall, not nearest-beat distance alone. Freeze motion-event definitions before examining music; stillness must not generate unlimited hits. Examine per-region direction reversals, velocity minima, support transitions in source motion, and persistent held poses. Report each against matched shifts and simple beat-trigger controls. Pair these with geometric movement gates so jitter and flashing cannot win. For long-form claims, test phrase/section changes separately with beat-preserving bar shifts and local boundary controls. Automatic event scores remain proxies; blinded audiovisual preference/ranking with shifted-audio negatives is needed before calling the lights dancing.

## Practical minimal revision

**First run a turn/articulation-separated surface mapper as one opt-in, fixed-source geometry intervention.** Do not change the generator, photometry, optical profile, or all smoothing gains at once.

- Decompose regional motion into body-local articulation and root orientation. Keep anatomical fixture identities fixed.
- Map body-local articulation to the existing floor lanes. Remove the rigid-neutral yaw displacement from limb offsets.
- Preserve root turns through one separately bounded, explicitly documented group-space cue, such as a small rotation of the target constellation about its center. Do not silently erase turns and call it preservation. Cap that cue by the same stage bounds and actuator limits.
- Retain held poses; do not use a rolling neutral subtraction. Compare yaw-only, articulation-only, static, and real travel/turn controls. Report both requested and emitted geometry and whether distinct body gestures remain distinguishable.

Prediction: this can reduce turn-induced cross-lane collapse with identical input motion. **It does not predict improved music conditioning or sufficient choreography.** If no meaningful turn component explains the real collapse, reject this revision as the explanation and do not keep tuning it to those timestamps.

After that seam test, a single temporal-transfer revision is more defensible than a model sweep: reduce unnecessary target lag or use bounded velocity feed-forward derived from the same motion, with noise/jitter and hardware limits as independent gates. Do not simply raise gain. A direct musical-event or structural-novelty layer is a separate candidate, not evidence that the native generator learned music. It should consume measured audio events, not a free-running oscillator, hand-picked timestamps or an ending script. [W2]

Only if the oracle mapper preserves coherent motion-to-light events while the generated input fails should the next training intervention proceed: local/multiscale audio conditioning and physical FK/contact supervision as separate matched ablations, consistent with `NEXT_REVISION.md`. EDGE's contact-consistency work supports treating contact semantics and foot velocity jointly; it does not validate LightSem's model or justify importing an entire replacement pipeline. [L6, W3]

## Sources and exact citation scope

### Local sources, inspected in full for requested reports and mapper/model

- **[L1]** `docs/research/2026-09-10-motion/RESULTS.md`: published local measurements and failed qualification. Not a new vault-state query or re-evaluation.
- **[L2]** `docs/research/2026-09-10-motion/SPATIAL_CORRECTION.md`: geometric controls, measured transfer loss, revision02 numeric improvements, failed visual acceptance.
- **[L3]** `lightsem/experiments/motion_study/lighting_map.py`: `surface_targets`, `compose`, `ema`, `limited_angles`. Code establishes available signals and transformations, not perceptual outcomes.
- **[L4]** `lightsem/experiments/motion_study/music_model.py`: `vector_field`, `losses_from_draws`, `generate`.
- **[L5]** `lightsem/experiments/motion_study/geometry.py`: canonical rest skeleton and world/root transforms. The 180-degree example is derived from code, not an observed song segment.
- **[L6]** `docs/research/2026-09-10-motion/NEXT_REVISION.md`: proposed, not executed, next training controls.
- Also read: `CLAUDE.md`, `RESEARCH.md`, `audio_dynamics.py`, `surface_bandwidth.py`.

### Primary-source web material directly retrieved and inspected

- **[W1] FFmpeg `ebur128` documentation:** https://ffmpeg.org/ffmpeg-filters.html#ebur128 — confirms 400 ms momentary, 3 s short-term, and 100 ms metadata frames. It does not claim a dance-control mapping.
- **[W2] AudioLabs FMP, novelty-based segmentation, explaining Foote (ICME 2000):** https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html — musical section boundaries can reflect instrumentation, dynamics, harmony and tempo changes; checkerboard self-similarity novelty is one measurable structural cue. No claim that a novelty peak should always trigger a lighting change or that this method passed on Shpongle.
- **[W3] EDGE author project page (CVPR 2023):** https://edge-dance.github.io/ — describes contact consistency and intentionally sliding versus planted feet, plus audio-conditioned synthesis. Supports a mechanism-specific physical diagnostic, not a claim that its checkpoint or architecture will solve this project.

Fetched HTML copies are `outputs/dance_source_{ebur128,foote,edge}.html`. No literature result supersedes the local failures. No human audiovisual viewing or fresh visual rating was done for this report.
