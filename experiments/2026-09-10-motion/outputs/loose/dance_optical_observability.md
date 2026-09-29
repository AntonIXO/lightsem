# Dance optical observability audit

Read-only diagnostic of `stage_geometry_02`, executed 2026-09-10. Artifacts use the parent study's `20260911` label. This is an **unpreregistered mechanism audit**, not a new gate or acceptance score.

## Answer

**The requested motion is small relative to the rendered cone profile, and the view compresses it further.** Both effects are supported. This does not show that correcting them will make the lights dance.

- Active nominal full beam angle is **13.545°**, but the shader's absolute half-strength full angle is **18.048°**. Its nonzero support extends to a **25.666° full angle**.
- Active emitted first-hit 1–2 Hz RMS is **0.74–2.59 px** per head at native 640×360 panel size. Surface half-strength equal-area radii are **6.99–15.16 px**. Torso motion is about **5% of that radius**; other regions are 17–19%.
- At half the beam length, active requested motion is only **0.33–1.26 px RMS**; emitted is **0.33–1.05 px**. A constant-distance axis proxy excludes changes in first-hit distance. The torso's request is already subpixel here; restoring actuator response alone cannot enlarge it.
- The camera gives only **2.86 px per metre of floor depth**, versus **15.26 px per metre laterally**, near `(0,0,-3)`. Camera framing and floor foreshortening are separate effects.
- These are geometry/projection quantities. Existing decoded pixels confirm a small occupied image region, **not** tracked per-head motion or visual acceptance.

## Instrument and scope

Read `CLAUDE.md`, `DANCE_DIAGNOSIS.md`, renderer, shader, fixtures, stage design/trace/measurements/render receipt, and reviewed existing images. Calculations import frozen stage02 sources through native `run.sh`; CPU FK uses native `run_torch.sh`, with `CUDA_VISIBLE_DEVICES=""`. FK logs `CUDA initialized False`. No GL context, GPU render, training, vault write, default change, or production edit occurred. The three audited live renderer/fixture files are byte-identical to their frozen copies.

`candidate.npz` has 11,092 DMX frames at 40 Hz. Its trace provides requested surface points and root-rotation-corrected regional deltas. `flow0` is a generated sequence, **not measured human motion**. It has 8,319 frames at 30 Hz. CPU FK reconstructs the canonical 22-joint skeleton in metres. All geometry ignores brightness; all six moving heads exceed the existing weighted-level `.03` threshold throughout active 70–90 s.

The existing declared stage02 gates pass except the unassessed visual gate. This is not identical to every generic observer threshold passing: the generic active absolute `trajectory` gate remains false (0.685 m vs 0.8 m). The declared dynamic gate uses relative trajectory and changing pair distances. No old gate is reinterpreted here.

## Cone calibration: softness is not degrees

For moving heads, `lightAt` uses:

```glsl
float edge = cos(nominal_half_angle);
float cone = smoothstep(edge - 0.018, edge + 0.0072, cosA);
```

`smoothstep` equals 0.5 at its midpoint, so:

`theta50 = acos(cos(nominal_half_angle) - 0.0054)`.

Nonzero support ends at `acos(cos(nominal_half_angle) - 0.018)`. These are the angular **cone factor** only, before distance falloff, haze integration, surface reflectance, tone mapping, and encoding. They are **not decoded image FWHM**.

| Setting | Nominal full angle | Absolute half-strength half-angle | Nonzero support half-angle | On-axis cone factor |
|---|---:|---:|---:|---:|
| Minimum club-rig zoom=0 | 6.000° | 6.671° | 11.296° | 0.864 |
| Actual quiet median | 11.812° | 8.392° | about 12.4° | about 0.984 |
| Actual active median | 13.545° | 9.024° | 12.833° | 0.99977 |
| Actual contraction median | 14.667° | 9.453° | about 13.1° | 1.000 |
| Maximum zoom=1 | 32.000° | 17.087° | 19.393° | 1.000 |

The quiet/contraction rows are rounded summaries; exact per-head values are in JSON. At minimum zoom, the edge's upper threshold exceeds 1, so the axis never reaches cone=1. Half of its *actual on-axis peak* occurs at **7.213°**, not 6.671°. For the active profile the difference is negligible. This distinction matters if testing narrower beams.

The current mapper sets all moving-head zoom bytes from `.22 + .12*palette_level`, independently of angular excursion. Active median zoom is **0.290196**. At fixed distance, reducing zoom to zero **without changing softness** only reduces the absolute half-strength radius to **73.6%** of current active radius. A truly 3° half-strength half-angle would reduce it to **33.0%**. These tangent ratios are geometry, not a prediction of pixel contrast or perceived motion.

## Per-head active motion versus cone size

Interval: **70–90 s**, all 800 DMX frames. Regions follow increasing fixture x: right hand, right leg, right torso, left torso, left leg, left hand.

| Head | Requested / emitted 1–2 Hz RMS px | Cone support radius px | Emitted RMS / radius | Requested / emitted 4 s extent px | Emitted extent / diameter |
|---|---:|---:|---:|---:|---:|
| mh1 | 2.38 / 2.08 | 11.55 | 0.180 | 15.45 / 15.46 | 0.67 |
| mh2 | 1.21 / 1.19 | 6.99 | 0.171 | 7.05 / 7.05 | 0.50 |
| mh3 | 0.75 / 0.75 | 15.12 | 0.050 | 6.41 / 6.40 | 0.21 |
| mh4 | 0.74 / 0.74 | 15.16 | 0.049 | 6.49 / 6.49 | 0.21 |
| mh5 | 1.24 / 1.24 | 7.05 | 0.175 | 6.43 / 6.42 | 0.46 |
| mh6 | 2.84 / 2.59 | 13.95 | 0.186 | 22.46 / 22.67 | 0.81 |

Definitions:

- Project the first valid finite floor/back-wall axis hit through the renderer's pinhole camera. Requested points are separately ray-intersected. Truss occlusion is not modeled.
- Band RMS integrates Welch density over **1≤f<2 Hz**, using 10 s Hann segments and linear detrending, then sums x/y power. This is absolute RMS, not the evaluator's requested/emitted pan-tilt power ratio. Vector-direction small-angle RMS spans **0.41–1.77° requested** and **0.41–1.56° emitted**, versus the 9.024° half-strength angle.
- Four-second extent is the norm of coordinate-wise p90–p10, median over 4 s windows advanced by 1 s. It is an extent, **not sinusoidal peak amplitude**. Nonlinear projection means it is not a universal metre-to-pixel conversion.
- Cone radius is `sqrt(projected_surface_support_area/pi)` on the frozen observer's 0.25 m finite floor/back-wall grid, sampled at 5 Hz. It uses the shader cone factor≥.5 and ignores truss/photometry. It is neither a circle fit nor decoded beam width. Independent perpendicular cross-section projected RMS radii at first-hit distance are **12.21–13.38 px**, supporting the same scale comparison without a surface footprint interpretation.
- `fixed_axis_metrics.json` holds the constant-distance axis audit. Reference distances are active per-head median emitted hit distances (**6.14–7.41 m**). At half distance, emitted RMS for mh1…mh6 is **1.047, 0.622, 0.328, 0.338, 0.640, 0.978 px**. The full-distance proxy is **2.081, 1.328, 0.586, 0.599, 1.329, 1.904 px**.

Full-track per-head emitted first-hit band RMS spans **1.24–2.21 px**; quiet spans **0.22–0.28 px**, and contraction 194–201 s spans **1.52–4.40 px**. All segment details are retained. The short contraction interval uses its full available length for Welch, not a 10 s window.

## Source physical motion and pixels

These are canonical generated body motions, not real fixture travel or measured dance. Root-relative weighted body regions differ from the mapper's `delta`: the latter also subtracts the rotated neutral region. Body turns can change both. The mapping additionally applies activity gains, bounded lateral lanes, shared facing/root cues, and a floor/wall chart.

| Head | Root-relative region 4 s extent m | Region 1–2 Hz RMS m | Mapper delta 4 s extent m | Emitted hit 4 s extent m | Emitted hit extent px |
|---|---:|---:|---:|---:|---:|
| mh1 | 0.796 | 0.118 | 0.893 | 1.479 | 15.46 |
| mh2 | 0.278 | 0.046 | 0.328 | 0.611 | 7.05 |
| mh3 | 0.184 | 0.027 | 0.104 | 0.511 | 6.40 |
| mh4 | 0.203 | 0.030 | 0.112 | 0.517 | 6.49 |
| mh5 | 0.361 | 0.060 | 0.426 | 0.677 | 6.42 |
| mh6 | 0.769 | 0.118 | 0.895 | 1.796 | 22.67 |

Fixture source positions are fixed at `(x,6,-1)`, x = −5,−3,−1,1,3,5 m. **Physical source translation is zero, and projected source translation is zero** because camera orbit is also zero. Rotating axes cannot move these top attachment points. The lamp glow is also axis-independent in the shader. This is a structural reason a persistent top row remains; sharpening edges does not remove it.

## Camera and existing decoded pixels

Receipt: fixed camera `(0,2.6,11)` aimed at `(0,3.2,-3)`, orbit=0, 640×360, 30 fps. Shader ray scale `.85` gives **80.729° vertical / 113.010° horizontal FOV** and **211.765 px focal length**. Six fixture origins occupy only **174.51 px / 27.27%** of panel width. Their row is y≈129.69 px.

Local projection derivative, magnitudes in px/m:

| Reference point | World x | World y | World z |
|---|---:|---:|---:|
| Floor `(0,0,-3)` | 15.26 | 15.40 | 2.86 |
| Wall `(0,2,-6)` | 12.49 | 12.52 | 0.44 |
| Truss `(0,6,-1)` | 17.45 | 17.26 | 4.89 |

On the floor only x/z displacements stay on the surface; on the wall only x/y do. The weak floor-z axis is a specific foreshortening problem, not evidence that every 3D direction is invisible.

Separately, CPU FFmpeg decoded all **600 existing comparison frames at 70–90 s and native 30 Hz**, with `-hwaccel none`. It retained each 640×360 panel. The title strip y<25 was excluded from support and frame differences.

| Pixel statistic | Stage01 baseline | Stage02 candidate |
|---|---:|---:|
| Median support with Rec.709-weighted encoded RGB luma >5/255 | 3.837% | 3.745% |
| Median support bbox x0,y0,x1,y1 | 227,124,412,235 | 227,124,412,235 |
| Mean absolute successive-frame luma difference, 8-bit units | 0.04768 | 0.04672 |

This threshold is an **unpreregistered descriptive mask**, not a quality gate. Encoded luma is not radiance. The full-scene support includes washes, bulbs, wall/floor response, and codec effects. These pixel numbers neither isolate moving-head motion nor establish that stage02 looks better/worse. No per-head decoded-pixel tracking was performed. The t70 image and its enlarged crops visibly retain the row/cone arrangement; static crops cannot establish temporal dance quality. The geometry overlay explicitly labels projected paths as **not pixel tracks**.

## Minimal opt-in calibration justified by this evidence

Keep all defaults unchanged. Do not retrain or remap motion based on these ratios alone.

1. **Presentation control first:** review the exact existing 30 Hz pixels with a fixed crop around the rig, alongside the full frame. This tests display scale without new rendering, new motion, or changed photometry. Enlarging a crop improves viewing size but cannot recover subpixel information. A future full-resolution render with the same FOV raises pixel sampling but preserves motion/beam-width ratios. A narrower FOV similarly scales local beam and motion together; it does not fix the broad cone ratio or floor foreshortening.
2. **One opt-in beam profile:** preserve legacy default; add an explicit angular edge half-width for moving-head beams only. For example, `cone = 1 - smoothstep(half_rad - radians(0.5), half_rad + radians(0.5), acos(clamp(cosA,-1,1)))`. This places .5 at the requested half-angle and confines the transition to ±0.5°. First hold actual zoom unchanged to isolate softness. Then a separately labeled zoom=0 diagnostic (6° nominal full angle) can test whether narrower beams expose the same trajectories. Keep axes, source positions, audio, exposure, and non-beam roles fixed. This is a proposed engineering control, not an approved visual treatment. The trig cost and 64-step volumetric sampling need GPU preflight; narrow cones may alias or lose brightness.
3. **Camera geometry only if needed:** a higher-elevation opt-in view can expose floor-depth movement better than a tighter crop. Compare with fixed world-axis markers and the same emitted sequence. No exact camera location is selected from this audit because framing, overlap, and occlusion must be checked. A close/narrow shot alone cannot remove the fixed source row or recover omitted pose dimensions.

A minimal beam test should retain a fixed-exposure comparison and report reduced total light rather than silently increasing exposure to hide it. A brightness-matched presentation may be shown separately, with its gain named. Pixel/visual acceptance must follow the opt-in test; no ratio here certifies aesthetics, audio-to-motion generalization, or physical fixtures.

## Reproduction and artifacts

Unique directory: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optical_observability_01`.

- `source_fk.py`, `source_fk.log`, `source_joints_cpu.npz`: native CPU FK.
- `audit_optics.py`, `audit.log`, `geometry_metrics.json`, `geometry_series.npz`: angle, source, emitted/requested projection, and support calculations. Input SHA256 values are retained in JSON.
- `extra_axis_proxy.py`, `fixed_axis_metrics.json`: constant-distance moving-axis check.
- `pixel_metrics.json`, `decoded_active_stats.npz`: existing-video CPU decode statistics.
- `decoded_candidate_t70.png`, `decoded_crop_quarterseconds_0.png`, `decoded_crop_quarterseconds_1.png`: exact decoded panel/crops. Crop timestamps are frame indices 0,7,15,22 and 30,37,45,52 after t70 (0, .233, .500, .733, 1.000, 1.233, 1.500, 1.733 s); names are approximate quarter-second labels.
- `proxy_path_overlay_t70_74.png`: four-second emitted geometry paths over the existing t70 frame, not optical-flow tracks.

Commands: `CUDA_VISIBLE_DEVICES="" bash run_torch.sh <dir>/source_fk.py`; `CUDA_VISIBLE_DEVICES="" bash run.sh <dir>/audit_optics.py`; `CUDA_VISIBLE_DEVICES="" bash run.sh <dir>/extra_axis_proxy.py`. These scripts write only the new diagnostic directory.
