# Spatial collapse correction: analytic research and measured diagnosis

Primary evidence is this renderer, rig geometry, emitted DMX and the rejected
full Shpongle video. No model sweep or external deep-research claim is used.
The existing literature context remains in RESEARCH.md; this is a geometric
adapter failure that ray-plane tests can directly identify.

## What collapsed

Six fixtures at x=-5,-3,-1,1,3,5, y=6,z=-1 aimed inside the room. The ray continues
past the target. On the floor x_hit=x_fixture+6/(6-y_target)*(x_target-x_fixture).
With x_target=.45*x_fixture and y_target=2.5, fixture-spacing coefficient=.05714.
At y_target=6*.45=2.7, it is exactly zero. This is a focusing lens built into the
mapper, not a bad pan/tilt encoder. _angles, beam_series, and renderer round-trip
for downward, forward, backward, oblique and upward directions within DMX
quantization tolerance. Some rays hit the finite z=-6 wall before their
extrapolated floor point; first-hit measurements explicitly handle this.

Measured all11,092 emitted frames of old hybrid: floor-span median1.817376m
(p10=1.378333,p90=2.402487); first-surface span1.816600m. Four or more axes within
1.5m of the median hit on97.944% frames, longest111.4s. Central stage focus82.519%.
Half-strength soft-cone support overlap median70.507%, not just similar pans.
The renderer soft edge .018 in cosine units makes visible support wider than
nominal zoom half-angle; at half strength threshold=cos(halfangle)-.0054.

Active requested floor span1.532m vs emitted1.522m; median requested-to-emitted
hit error.117m. Thus the limiter is not the main source of spatial convergence.
Independent six-body ablation also collapses. This does not blame the generator
for this specific geometry failure, nor qualify its music conditioning.

## Coordinates

Native neutral anatomical LEFT wrist has world x+.722m; RIGHT wrist x-.719m.
The old negative-x fixtures follow anatomical left and receive positive inward
x terms even in neutral pose. New surface mode uses fixed right -> negative-x,
left -> positive-x assignments, never swaps identities during turns, and never
mirrors world x. Delta from a fixed unrotated neutral pose retains root rotation
and sustained poses; subtracting a rotating reference would erase root turns.
Inner torso uses shoulder/head/neck triangles rather than one shared centroid.
Photometry deliberately continues using the old signal to keep all non-angle
DMX bytes identical. This is documented, not a hidden whole-mapper comparison.

## Surface-space alternatives

1. Direct floor anchors: remove elevated target extrapolation and restore
   fixture-scale anchors while keeping old signals/smoothing. Mechanism control.
2. Articulated floor mapping: neutral-relative regional x -> surface x, height
   and depth -> surface z, bounded slow root and short travel shared by all.
   Floor lanes preserve concurrent composition; gestures change position and
   shape. Clamp only physical stage bounds, no scripted temporal effects.
3. Explicit directions could implement the same floor targets via normalized
   target-fixture vector. Angular spreads alone are not sufficient because
   distance, grazing incidence and the camera change visible separation.
4. Wall or combined floor/wall targets are viable future mappings, not tested
   here. No need to add them to diagnose a floor-collapse bug.

## Why a wide fan was not accepted

Revision01 fixes concentration but active translation-removed4s extent.372m
fails. Raw.676 -> EMA(.55s).432 -> actuator tracking(.35s).372m. Total active raw
extent.763m is itself below the frozen.8m bar. World/regional motion exists, but
filters attenuate it and torso signals have shorter physical lever arms.
Revision02 preregisters one gain/bandwidth bundle, not an isolated causal claim:
articulation gain1.8, extra torso gain2, EMA.20s, tracking.18s; hard speed30 and
acceleration80, quiet limit3, brightness, source, zoom and optics remain fixed.
Both revisions keep defaults opt-in; no musical or human acceptance is earned
by geometric controls. No sines/chases/randomness/time-position fades are used.

## Instrument limits

Axes: all40Hz frames. Soft-cone grid:5Hz,.25m cells, finite floor/wall stage ROI;
no truss occlusion, no radiometric or rendered-pixel interpretation of m2.
Projected area is analytic camera support, not calibrated light flux. Grid
support alone passes even common-point controls; overlap, span and persistence
are necessary. Spatial trajectories ignore brightness so flashing cannot count
as choreography. Separate lit coverage uses fixed RGB-weighted threshold. The
controls calibrate a failure detector, not human preferences or music fit.
Full videos are rendered and every raster frame measured. Chronological visual
contact review remains sampled, not continuous human audiovisual viewing.


## Measured outcome and remaining failure

Both correction experiments computed `directional_only` in **LightSem** Vibescience.
Revision01 fails active movement/trajectory/visual gates. Revision02 passes the
frozen numerical aggregate gates but **fails visual_review**. Neither is a complete
product acceptance or native-generator qualification. Original computed verdicts,
source freezes, movies and the user's rejection are all preserved.

| Diagnostic | Rejected hybrid | Surface rev01 | Surface rev02 |
|---|---:|---:|---:|
| Full median first-surface x span (m) | 1.817 | 7.344 | 6.490 |
| Rev02 first-surface span p10–p90 (m) | — | — | 5.420–8.307 |
| Full median depth span (m) | 1.184 | — | 2.580 |
| Full median half-strength cone-support overlap | 70.51% | 1.38% | 6.61% |
| Frames with4+axes within1.5m of median hit | 97.94% | 0% | 2.49% |
| Longest such focus (s) | 111.400 | 0 | 2.075 |
| Active70–90s total4s trajectory extent (m) | .824 | .438 | 1.153 |
| Active translation-removed4s extent (m) | .454 | .372 | 1.130 |
| Active median pan/tilt speed (deg/s) | 2.657 | 1.922 | 9.742 |
| Quiet0–20s median speed (deg/s) | 2.360 | 1.105 | 2.971 |

The incumbent's active/quiet medians are51.028/40.872deg/s, not the rejected
hybrid's medians. Revision02 retains maximum29.999deg/s and79.377deg/s2, with
all non-angle DMX bytes exactly unchanged. Quiet/active bulb luminance remains
.032868/.110110 (ratio3.350). No white clipping in any generated raster frame.
The same dimmer controller still determines the ending; no ending-position fade.

**Actual encoded image support**, measured in the same fixed lower-stage band
at320x180 on every frame, independently confirms broader illumination. Active
80%-energy horizontal span25->58pixels; central-quarter energy fraction
.91568->.45252. Full-track values28->57pixels and.85328->.51014. These are decoded
pixel measurements, not analytic support area. Non-head light is included and
unchanged; do not interpret them as beam-isolated radiometry.

### Complete rendered evidence and honest visual limit

Full-track movies are277.3s/8319encoded frames, with synchronized original audio;
all encoded frames decode and match ffprobe. Rendering generated8320frames, and
FFmpeg's declared source-duration cut removed the final partial-frame slot.
This is within one30Hz frame of277.306667s source duration, not a shortened clip.
The full before/after comparison also decoded end to end without errors.

The primary agent inspected **all555chronological2Hz samples**, plus144frames
on nine8Hz contact sheets covering quiet, entry, active, local contraction and
ending, plus16frames at70–72s. This is dense sampled visual review, **not continuous
human audiovisual viewing**. All-frame numerical review cannot measure human
music fit. Exact sheets, hashes and interval notes are in `review.json`.

The persistent whole-song funnel is greatly reduced and the lights have changing
regional gaps and floor sweeps. But a visible narrow central crown returns around
**194–201s**, and the composition still relies heavily on one floor-facing fan
family. The4-axis/1.5m longest-focus number2.075s understates this perceived
interval. Full-track aggregate gates therefore do not imply an acceptable
complete choreography. The preregistered visual gate stays **failed**. The
10-second blocks starting110and190also fail local spatial spread; the190block
fails concentration. No threshold was relaxed to hide these exceptions.

### Delivery paths

Root: `/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/`

- `surface_revision_02/shpongle_before_after.mp4`: full rejected-left/revised-right comparison.
- `surface_revision_02/articulated_surface.mp4`: full revised standalone movie.
- `surface_revision_01/{floor_reanchor,articulated_surface}.mp4`: both first-revision controls.
- `surface_revision_02/{measurements.json,pixels.json,review.json,comparison.json}`: numerical and actual-image evidence.
- `surface_revision_02/{spatial_timeline,surface_trajectories}.png`: all-track coverage and regional-path plots.
- Each revision's `vibescience_results.json`: computed independent gates and verdicts.

19tests pass (13in the existing motion-study command, plus6spatial tests). Tests
cover common point, static spread, jitter, rigid translation, flashing, separated
motion, the actual rejected DMX, angle/renderer round trips, finite first hits,
anatomical identity through turns, static-input stability and photometric parity.
The diagnostic controls are not injected into production movement.

No retraining, confirmation seed, TEST access, human rating, deletion or
default promotion occurred. The user subsequently authorized committing the
correction code, tests and reports; this does not change the frozen experiments.
Both vault write passes checked all unrelated-vault file hashes. Existing unrelated historical doctor findings were not silently
rewritten. Follow-up should address the remaining compositional and turn-induced
contraction limitations using these artifacts, not restart an unrelated model sweep.
