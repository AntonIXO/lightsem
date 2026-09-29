# Handoff: fix the collapsed lighting result

You are taking over an unfinished correction in `/root/lightsem`. Read this before
running more experiments. The user stopped the previous agent specifically to
transfer this work. The latest work should address the visible failure below,
not continue an unrelated model sweep.

## What the user wants, and what went wrong

The product is lighting that visibly dances to music. Playback input is music only.
An internally generated articulated motion sequence is now explicitly allowed.
Direct audio loudness/transitions may control brightness and movement intensity.
Quiet introductions should stay visible and restrained; active passages should
have meaningful spatial movement and choreography.

The user rejected the latest full Shpongle render: the beams converge into one
small central area throughout the song. This is the immediate issue to fix.
The previous agent checked sampled frames and numerical statistics, noticed an
overly similar fan, but failed to identify and address its severity as spatial
collapse. Do not repeat that mistake or present the brightness/speed improvements
as an adequate result.

Earlier user observations also remain relevant:

- The old pose data/reconstructions did not sufficiently describe dance.
- The incumbent `audio -> descriptors -> pose` was the only convincing path in
  the previous comparison; other alternatives looked like flashing/juggling.
- Shpongle has a quiet opening and a strong entry after a shout. Head brightness
  should visibly reflect that change.
- Heads and the overall composition were moving too quickly. The attempted fix
  overcorrected movement while concentrating the beams.

The task is to investigate, implement a correction, and verify the actual complete
render. A wider static fan alone would not establish dancing choreography.

## Current checkout and operating constraints

- Last implementation commit: `3f2f448` on `master`.
- The checkout was clean before this handoff file was added. No correction to the
  collapsed mapper has been implemented yet; the latest investigation was read-only.
- Use one GPU task at a time. Do not use subagents; the user's prior constraint
  was to work without them.
- Use `bash run_torch.sh ...` for torch/native geometry and combined rendering.
  Use `bash run.sh ...` for NumPy, OpenGL, plotting and other non-torch work.
  Do not invoke a venv Python binary directly; that has failed on this machine.
- RTX 5070, 12 GB. Training already uses BF16/compile/prefetch and is fast.
  The user removed the earlier 2–3 hour ceiling, but speed is not a reason to
  skip diagnosing a concrete geometric failure.
- All large artifacts belong under `/mnt/lightsem-data`, with roughly 5 GB free
  at handoff. Preserve historical runs and immutable source snapshots.
- Scientific records belong only in `/root/lightsem-vault`. Never write to
  `/root/vibescience-vault`, which is an unrelated project. Compare its file hashes
  before/after scientific logging; old hardcoded markdown counts are stale.
- Existing authorization covers investigation and reversible implementation work.
  No confirmation seed or TEST evaluation is earned by the existing results.
- No training/render job was left running by the latest investigation. Check
  machine state before starting GPU work.

## Start with these files and artifacts

Repository files:

- `lightsem/experiments/motion_study/lighting_map.py`: immediate mapper under review.
- `lightsem/experiments/beat_conditioned_render.py`: `_angles`, prior DMX mapping.
- `lightsem/experiments/quality_observer.py`: `beam_series`, `beam_observation`.
- `lightsem/sim/renderer.py`: DMX decoding and actual beam direction conversion.
- `lightsem/sim/shaders.py`: beam cones, floor illumination and exposure.
- `lightsem/dmx/fixtures.py`: fixture geometry and channel layout.
- `lightsem/experiments/motion_study/geometry.py`: native skeleton and anatomical axes.
- `lightsem/experiments/motion_study/delivery.py`: frozen full-track comparison harness.
- `lightsem/experiments/motion_study/delivery_results.py`: separate scientific gates.
- `docs/research/2026-09-10-motion/RESULTS.md`: complete prior outcomes and links.
- `docs/research/2026-09-10-motion/REPRODUCE.md`: launchers and artifact structure.
- `docs/research/2026-09-10-motion/NEXT_REVISION.md`: proposed model work, not executed.

Let `R3` mean:

`/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910`

Let `OUT` mean:

`/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic`

Inspect these actual artifacts first:

- `OUT/shpongle_comparison.mp4`: full incumbent-left / experimental-right comparison.
- `OUT/lighting/hybrid.mp4`: the rejected experimental result.
- `OUT/lighting/incumbent.mp4`: previous usable baseline.
- `OUT/lighting/{hybrid,independent,brightness_off,hybrid_legacy}.npz`: emitted DMX.
- `OUT/lighting/trace.npz`: requested targets, activity, dimmers and timestamps.
- `OUT/inference/shpongle.npz`: full generated native motion, all six flow draws,
  regression and incumbent 8/32-step poses. Reuse these to isolate mapper changes.
- `OUT/protocol.json`, `OUT/lighting/observations.json`, `OUT/review.json`.
- `OUT/shpongle_dynamics.png`: useful timeline, but it does not measure spatial collapse.
- `OUT/skeletons/shpongle.mp4`: native motion before rig mapping.
- `OUT/skeletons/label_corrected/*.mp4`: three historical source-video comparisons.
  The source recording is the same frontal view in both rows; native projections
  are front/oblique. Original misleading source-view headings were corrected in
  derived copies, with originals retained.

`OUT/source` is the immutable source snapshot that produced the failed result.
Changing live code does not change the existing frozen launcher. Use a distinct
revision root and freeze for the correction; do not overwrite the old evidence.

## Confirmed geometric findings from the latest investigation

The six moving heads are at x = `[-5,-3,-1,1,3,5]`, y = 6, z = -1 metres.
Floor is y = 0; stage back wall is z = -6.

The current mapper aims at points inside the stage volume, using approximately:

```python
x_target = .45 * fixture_x + shared_root_terms + regional_gesture
y_target = 1.65 + amplitude * regional_height_terms
z_target = -3.6 + shared_travel + regional_depth
```

It smooths targets for .55 seconds, then applies quantized pan/tilt speed and
acceleration limits. Requested target separation is not actual floor-footprint
separation. A ray continues past its target until it reaches the floor/wall.

For a head at height 6, the floor intersection is:

```text
x_floor = x_fixture + 6 / (6 - y_target) * (x_target - x_fixture)
```

With `x_target ≈ .45*x_fixture` and `y_target ≈ 2.5`, the fixture-spacing
coefficient at the floor becomes only about `.057`. This is a strong geometric
convergence mechanism. It must be checked against actual emitted directions,
including the rate limiter, not just requested targets.

Measured from every emitted DMX frame using `beam_series` and floor intersection:

| Variant | Median horizontal floor span | 10th–90th percentile span |
|---|---:|---:|
| Incumbent | 6.838 m | 4.480–9.448 m |
| Rejected shared hybrid | 1.817 m | 1.378–2.402 m |
| Independent six-body ablation | 2.277 m | 1.472–3.459 m |

The rejected hybrid's median depth span is only 1.184 m. At time zero all six
floor intersections lie around x=-.74..+.76 and z=-4.8..-4.38. Beam width and
projection further merge them visually. Independent draws did not remove the
central concentration, which strongly implicates the shared mapper geometry.

Another issue to investigate, not yet established as the cause: the canonical
skeleton's anatomical left offsets have positive world x. The mapper assigns
left wrist joint 20 to negative-x fixtures, and right wrist 21 to positive-x
fixtures. Check whether this sends the outer hands inward in the canonical pose.
Do not blindly flip signs: account for root orientation and explicitly define
anatomical versus stage-side mapping.

The `_angles` and renderer direction formulas appear consistent on inspection;
verify the round trip with counterexamples rather than assuming encoding is broken.

## What to research and implement

1. Reproduce the visible collapse from saved DMX. Analyze actual beam axes,
   first floor/wall hits, cone overlap and projected illuminated footprint over
   the entire song, including quiet/entry/active/ending intervals.
2. Trace where motion is lost: generated joints -> regional signals -> requested
   targets -> quantized actuator trajectory -> floor/wall hits -> rendered image.
   Separate geometric convergence, coordinate conventions, smoothing and the
   generator's own weak movement. Do not retrain before isolating this failure.
3. Research/compare mappings expressed in real illumination space: stage-surface
   targets or explicit directions with meaningful regional separation. Preserve
   shared body motion, root travel/turns and limb differences without continually
   steering every fixture toward a common invisible dancer in mid-air.
4. Add diagnostics that reject this exact result. Average angular speed, dimmer
   contrast and clipping are insufficient. Include illuminated spatial coverage,
   concurrent footprint concentration/overlap, temporal trajectory extent and
   persistence of a central focus. Distinguish genuinely changing choreography
   from a stationary wide fan, global translation, tiny jitter or flashing.
5. Validate the diagnostics on deliberate common-point, static-spread, jitter and
   meaningful separated-motion controls. Include the actual old failed DMX as a
   regression case. Freeze thresholds before evaluating the proposed correction.
6. Keep audio brightness and renderer exposure fixed initially so the geometric
   intervention can be measured. No random chases, sine waves, scripted movement
   or ending-position fades to make a broken motion path look active.
7. Render and inspect the full corrected song, with dense temporal review of
   important intervals and direct before/after comparisons. A few screenshots
   and valid video streams do not establish visual correctness. Be explicit
   about the limits of any sampled-frame inspection.
8. Record the old failure, the new hypothesis, actual before/after values and
   independent gates in Vibescience. Preserve the original computed verdicts;
   add the user's rejection and the newly identified missing spatial gate.
   Do not claim that a mapper correction qualifies the music generator.

## Prior experimental context: do not repeat or overclaim it

The representation-first plan was implemented before this rejection:

- Official FineDance public motion was imported and duplicate-group splits built.
  Clean source review passed; AIST++ is a separate source-quality reference.
- Three neural codecs and four linear compression candidates all failed dynamic
  fidelity. The accepted fallback is explicitly uncompressed native motion:
  139 channels at 30 Hz (root velocity/height, 22 local 6D rotations, four contacts).
  Its identity reconstruction is an engineering check, not a learned success.
- Matched native regression and flow ran 30,000 updates each at seed 0. Both
  failed music conditioning and foot support. Actual evaluation: 119 windows,
  33 VAL recordings / 28 canonical groups; three shorter VAL recordings were
  excluded by the fixed 32-second crop. TRAIN used 125 recordings / 107 groups.
- Flow correct-audio energy-score improvement versus donor was only 2.34%, and
  versus within-track shift only .50%; both confidence intervals include zero.
  Required improvement was 5%. Excess sliding was .882 m/s for flow, .618 for
  regression, against a .20 m/s limit. Native root drift remains a problem.
- Increasing the incumbent from 8 to 32 Euler steps increased >6 Hz pose power
  by 9.32% and velocity energy by 38.33%. This was a refutation; keep 8 steps.
- No native model was promoted. No seed-1 confirmation or TEST model evaluation.

The previous lighting result had genuine subcomponent improvements that do not
excuse spatial collapse:

- Quiet emitted head speed: 40.872 -> 2.360 degrees/s (94.23% reduction).
- Active median speed 2.657 and p95 7.929 degrees/s failed frozen minima of 3 and 8.
- Hybrid speed/acceleration maxima: 28.637 degrees/s and 79.377 degrees/s²,
  below the hard quantized limits of 30 and 80.
- Active/quiet bulb luminance ratio: 3.354 with direct audio brightness versus
  .568 with only dimmer bytes changed to the brightness-off control.
- Same hybrid DMX under legacy exposure gave ratio 1.273. Incumbent ratio: 2.152.
- Quiet mean bulb luminance .03287; active .11022. No white clipping in sampled
  5 Hz observations. Whole-scene brightness was much lower than the incumbent.
- Coherence and dynamics scientific verdicts were both `directional_only`:
  music prerequisite failed; coherence also failed active/regional/visual gates.
  The previous reporting still understated the central spatial collapse.

The opt-in optical profile independently passed white dimmer ramps:

```python
RenderConfig(
    exposure_profile='fixed',
    fixed_exposure_scale=.4,
    fixed_role_gains=(1., .3, .03, .2),  # beam, wash, flash, projector
    haze=.25, cam_orbit=0., intensity_gain=2.4,
)
```

The legacy renderer remains the default and was pixel-identical to its archived
version on 63 ramp frames. Optical calibration is not evidence of choreography.
Current mapper also applies non-head dimmer gains (.18 wash, .06 blinder, .025
strobe, .20 projector), making non-head contributions particularly subdued.

Audio loudness uses actual FFmpeg EBU R128 momentary (400 ms) and short-term (3 s)
measurements. Saved input is `R3/data/shpongle/loudness.npz`; source is
`/root/shpongle.flac`, 277.306667 seconds. The brightness controller uses a fixed
12-LU range below the track's p90 short-term level, .30-second smoothing, and
quiet/active dimmer bounds .12/.88. There is no ending-time rule.

Existing meaningful tests:

```bash
bash run_torch.sh -m unittest discover -s tests -p test_motion_study.py -q
```

All 11 passed, but they did not test spatial illumination collapse. Add the
missing counterexamples; do not cite those old passes as proof the result is good.

For scientific logging use the `vibescience-mcp-workflow` skill and existing Store
API patterns. The last full research pass used primary sources after the external
deep-research quota was exhausted; see `RESEARCH.md`. For this immediate failure,
the actual projection geometry and rendered evidence should drive the diagnosis.

## Expected delivery

A concrete explanation of the collapse, a reviewed mapper correction with
regression coverage for spatial collapse, full-track before/after movies, and an
honest account of remaining motion/model limitations. Do not declare the task
fixed merely because beams move slowly, bulbs brighten, or files render without
errors. The user's visible lighting result is the acceptance target.
