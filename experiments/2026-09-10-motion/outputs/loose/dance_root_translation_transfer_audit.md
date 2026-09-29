# Root translation transfer audit

Unpreregistered, read-only CPU mechanism audit. Study label: `20260911`; execution: 2026-09-10. **Instrument: saved generated root trajectories and existing raw stage-target geometry, not dance quality, music causality, rendered pixels, or emitted DMX.**

## Answer

**Yes for horizontal translation, but not for every root direction.** The stage02 chart strongly compresses lateral and depth sway. It does not distinguish useful fast sway from accumulated drift. Translation is not mathematically erased: finite-input tanh retains a small derivative. Vertical bob is largely preserved and amplified near the normal root height.

The actual source matters. In these three Shpongle draws, horizontal root motion at **1–2 Hz is only about 1 cm RMS per axis**, not the ±20 cm synthetic example. The chart reduces that to **0.38–0.92 mm RMS** in its shared coordinates. This is a real transfer bottleneck, but **not evidence that restoring translation alone would produce dancing light**. There is no measured human Shpongle root reference here.

No candidate, controller, render, checkpoint, training, VAL score, vault record, or production file was changed. No music-fit inference or proposed qualified gain fix follows.

## Exact mechanism and source correction

Live `articulated_stage.py`, `lighting_map.py`, and `geometry.py` are byte-identical to the visual-review frozen copies used here. Configuration is frozen stage02 anchors `(7.7,3.2,8.0)`, lean gain **0**.

The actual vertical expression is **0.8**, not 0.5:

```
sx =  .25*tanh(root_x/3)
uy =  .8*tanh((root_y-.95)/.4)
uz = -.4*tanh(root_z/3)
```

Rigid translation cancels from `(regional_world-root) - Q@neutral_region`. It survives only in `sx` and `uy+uz`. Root yaw also has explicit facing terms. The stage branch does **not** use `root-EMA12(root)`; that exists in older mapper branches, but computing it elsewhere in `compose` does not feed the stage chart.

Local signed derivatives, before chart clipping:

- `dsx/dx = (1/12)*sech²(x/3)`; maximum **0.08333 m/m**.
- `duz/dz = -(0.4/3)*sech²(z/3)`; maximum magnitude **0.13333 m/m**.
- `duy/dy = 2*sech²((y-.95)/.4)`; maximum **2 m/m**.

These are map transfer coefficients, **not retention relative to a validated ideal light mapping**. The horizontal bounds guarantee compression at every frequency. Their dependence on absolute position makes small sway weaker after accumulated travel. Vertical and depth contributions also share one chart coordinate; the mapping is not a complete 3D pose representation.

## Isolated rigid-body tests

Canonical `NEUTRAL` skeleton, identity root orientation, activity fixed at 1, no articulation. Each axis receives ±0.2 m sinusoidal translation for 40 s at 30 Hz. Other root axes stay `(0,.95,0)`. Targets come directly from the frozen `stage_targets`, not a duplicate mapper. Independent formulas agree to <1e-12 m; translation-induced regional articulation is numerical zero.

### Exact fundamental transfer at 1 Hz

| Axis | Static offset | Shared-coordinate fundamental peak | Gain m/m |
|---|---:|---:|---:|
| x | 0 m | 16.65 mm | 0.08324 |
| x | ±3 m | 7.01 mm | 0.03503 |
| x | ±6 m | 1.18 mm | 0.00590 |
| x | ±9 m | 0.165 mm | 0.000824 |
| z | 0 m | 26.64 mm | 0.13319 |
| z | ±3 m | 11.21 mm | 0.05604 |
| z | ±6 m | 1.89 mm | 0.00944 |
| z | ±9 m | 0.264 mm | 0.001318 |
| y | .95 m | 376.91 mm | 1.88457 |
| y | .55 or 1.35 m | 175.05 mm | 0.87524 |
| y | .15 or 1.75 m | 31.49 mm | 0.15745 |

Extreme height/offset controls are mathematical stress tests, not claims about plausible dance poses.

At the central offsets, repeating at **0.1, 0.5, 1, and 2 Hz** gives the same fundamental gain to floating-point precision. Thus this nonlinear spatial bound is **not a selective drift filter**. Static input gives exactly static output. No temporal pattern is introduced by this diagnostic. The existing nonlinear chart can nevertheless create harmonics of an input motion; this is not an independent clock.

### Geometric pixel scale

640×360 stage panels. Project the exact target and the midpoint between fixture origin and target through the two frozen review cameras. These are geometric trajectories, not decoded image motion or beam widths. No brightness, optics, motor limits, DMX rounding, or temporal interpolation is applied.

Centered ±0.2 m, 1 Hz: **midshaft fundamental vector peak**, range across six heads:

| Root axis | Original camera | Closer/higher camera |
|---|---:|---:|
| x | 0.121–0.140 px | 0.227–0.257 px |
| z | 0.052–0.194 px | 0.131–0.361 px |
| y | 0.741–2.746 px | 1.852–5.109 px |

Even this sizeable horizontal source excursion yields a subpixel raw axis request in both views. Corresponding first-hit peaks are x **0.208–0.270 / 0.374–0.464 px**, z **0.127–0.336 / 0.342–0.577 px**, y **1.801–4.750 / 4.838–8.164 px** (original / closer). Offsetting x or z further compresses them. Full per-head and 1–2 Hz band results are retained in JSON.

### Turn and lean controls

- Static ±30° yaw produces a **0.350 m** signed target difference. Sinusoidal ±30° yaw retains a **0.1771 m x fundamental peak**.
- Static ±30° roll gives the same targets to **1.8e-15 m**; dynamic roll has numerical-zero target movement.
- Static ±30° pitch gives the same targets to **8.9e-16 m**. Dynamic pitch loses its signed fundamental, **but is not wholly static**: `1-cos(pitch)` produces an even response (about **0.0464 m** sampled peak-to-peak in u, principally 2 Hz for a 1 Hz pitch input).
- Adding the same x translation at static yaw/roll/pitch −30°, +30°, and 135° retains the same translation formula to <1e-12 m. Rotation does not secretly restore a high-passed root path.

The lean finding is therefore directional loss, not a claim that every pitch-derived output is identically zero.

## Saved actual Shpongle trajectories

All three have **8,319 frames at 30 Hz, 277.3 s**. OLD is saved `flow0`, integrated once by CPU FK. NEW masked and physical use evaluator `final/shpongle/000.npz` saved full world joints directly, draw seed **7001**. No crop reintegration or model loading. Evaluation envelopes, identities, complete-six-draw report status, and selected array SHA256 values were verified. **Only draw 000 is measured here; no six-draw generalization.**

### Slow trajectory scale versus faster fluctuations

| Draw | Root x range m | Root z range m | Endpoint horizontal displacement m | EMA12 slow x/z p90–p10 m |
|---|---:|---:|---:|---:|
| OLD flow0 | −0.718…2.114 | −0.192…4.001 | 2.683 | 1.444 / 2.596 |
| NEW masked | −1.062…1.085 | −0.199…7.761 | 7.540 | 0.701 / 6.159 |
| NEW physical | −0.289…2.284 | −0.122…3.165 | 3.362 | 1.548 / 2.435 |

Here **slow** means the diagnostic causal 12 s EMA, initialized at frame 0. Whole-track `root-EMA12(root)` RMS x/z is OLD **0.328/0.378 m**, masked **0.311/0.523 m**, physical **0.172/0.230 m**. Do **not** call these amplitudes dance-frequency sway: a sustained velocity leaves an EMA lag, so this residual includes slow travel. The chart itself does not use this decomposition. Slow trajectory is also not proven erroneous drift merely because it is slow.

Frequency bands separate the scales more clearly. Whole-track **0.5–2 Hz root RMS x/y/z** is OLD **33.5/23.7/22.9 mm**, masked **33.0/24.2/24.3 mm**, physical **20.7/21.2/19.3 mm**. The narrower fast band is smaller still:

| Draw | Root 1–2 Hz RMS x/y/z, mm | Shared x/height-u/depth-u RMS, mm | Output/input band RMS gain x/y/z |
|---|---:|---:|---:|
| OLD flow0 | 10.53 / 15.81 / 9.70 | .795 / 29.52 / .810 | .0755 / 1.8677 / .0835 |
| NEW masked | 10.58 / 16.44 / 9.99 | .853 / 31.13 / .383 | .0806 / 1.8928 / .0384 |
| NEW physical | 9.34 / 14.21 / 9.87 | .693 / 26.87 / .924 | .0742 / 1.8904 / .0937 |

These empirical ratios describe nonlinear full-track transfer, not a universal linear frequency response. Height/depth contributions are measured separately before their sum and chart clipping.

Masked root-z is especially far into tanh saturation: median local derivative magnitude **0.00642 m/m**, versus OLD **0.07287** and physical **0.07528**. Its depth 1–2 Hz gain is **0.02868** in 70–90 s and **0.00573** in 194–201 s. OLD is **0.12932 / 0.06843**, physical **0.12107 / 0.06847** over those same intervals. This proves that absolute accumulated offset changes subsequent sway transfer. It does not identify which source excursion was musically appropriate.

### Actual root-only pixel contribution

For each axis, keep that actual full root coordinate, hold the other axes at neutral, and use a rigid neutral body. This isolates translation from articulated gestures and avoids reporting vertical bob as horizontal sway.

Whole-track **1–2 Hz midshaft geometric RMS**, six-head range, closer camera:

| Draw | x only | z only | y only |
|---|---:|---:|---:|
| OLD | .0108–.0123 px | .0040–.0109 px | .146–.399 px |
| NEW masked | .0116–.0132 px | .0019–.0052 px | .154–.421 px |
| NEW physical | .0095–.0107 px | .0046–.0125 px | .133–.363 px |

Original-camera x/z values are smaller still, all **≤.0072 px RMS**. The real horizontal fast component is tiny in this geometric instrument. The larger combined-root pixel band is primarily the **retained vertical term**, not hidden horizontal energy.

Additional saved diagnostics compare full actual-pose raw targets to the same pose at neutral root, with activity fixed at 1. These marginal pixel differences can include seam changes and interactions with articulation. They are **not an additive root energy fraction** and must not be interpreted as one. No audio-dependent activity, rawtarget-to-DMX attenuation, or actual emitted image trajectory is measured in this audit.

## Interpretation and limits

1. A legitimate horizontal rigid sway, if present, is attenuated along with unwanted accumulation. The map has no time-scale or semantic test to distinguish them.
2. Bounding stage drift is real, but the current bounds also suppress horizontal oscillation. Translation shared across fixture lanes cannot by itself create independent regional gestures; pure shared x translation leaves inter-head target distances unchanged.
3. Vertical root movement is **not** a missing dimension in the same sense. Near actual root heights it is amplified about twofold.
4. These particular generated draws contain only centimetre-scale fast horizontal root motion. The ±0.2 m synthetic test must not be presented as the source's actual amplitude.
5. No result establishes that larger mapping gains, restored lean, alternate root filtering, or a clearer camera would satisfy the user's dancing-light goal. Aesthetic acceptance and audio causality remain outside this instrument.

## Methods and artifacts

Band RMS: Welch density, Hann windows up to 20 s, 50% overlap, linear detrending, integrate half-open bands by bin sum × frequency spacing; sum coordinate powers for vector pixels. Short segments use their full length. Reported synthetic peaks use exact sine/cosine projection over 40 s, not p90–p10. Because synthetic 1 Hz sits on the lower edge of the 1–2 Hz band, that band omits its lower Hann side-bin; **do not substitute it for total sinusoidal RMS**. All actual trajectories are measured at native 30 Hz, before the review mapper's 40 Hz interpolation.

Unique artifact directory:
`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/root_translation_audit_01/`

- `audit.py`, `audit.log`: primary CPU audit; `cuda_initialized=false`.
- `measurements.json`: exact transfers, static-offset/orientation controls, both camera projections, full/70–90/0–20/194–201 s root metrics, source/input hashes.
- `diagnostic_series.npz`: root, diagnostic EMA12, shared contributions, isolated root targets, and actual-pose root marginal target differences. These are diagnostic raw geometry, **not a DMX candidate**.
- `axis_pixels.py`, `axis_pixels.log`, `axis_pixel_metrics.json`: per-axis actual source pixels and frequency controls.
- `report.md`, `artifact_manifest.json`: this report and audit artifact hashes.

Reproduce from `/root/lightsem` with `CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 bash run_torch.sh -u <dir>/audit.py` and similarly `axis_pixels.py`, using a fresh diagnostic output location. Scripts refuse to overwrite their measurement files. No GL context or GPU work occurred.
