# Why surface rev02 still does not dance: mapper audit

Date: 2026-09-10. Read-only, unpreregistered diagnostic. No training, inference, production edits, vault writes, or acceptance claim.

## Finding

Rev02 restores large, slow spatial movement. It does **not** preserve much of the faster requested gesture motion. The current controller requests mechanically infeasible large gestures, then delays and removes their fast components. Separately, its amplified fixed-neutral subtraction still turns body rotation into inward fixture movement. Every target also belongs to one floor-facing family.

These are measured mapper failures on the saved Shpongle `flow0`. They are not proof that fixing this mapper will produce good music-conditioned dance. I did not watch the movie continuously or measure music alignment.

## Evidence and reproduction

Inputs: frozen `delivery/surface_revision_02/source`, saved `final_diagnostic/inference/shpongle.npz`, and rev02 `articulated_surface{,_trace}.npz`. Root is `/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/`.

I reconstructed native joints at 30 Hz, interpolated them to the existing 40 Hz timeline, recomputed floor targets, and decoded the actual 16-bit DMX pan/tilt into floor-axis intersections. Recomputed raw targets match the saved trace **exactly, max absolute error 0.0 m**. These are unbounded floor-plane intersections, not finite first-surface hits, cone support, or pixels.

Commands run from `/root/lightsem`:

```text
bash run_torch.sh outputs/dance_mapper_audit_20260911.py
bash run_torch.sh outputs/dance_mapper_extra_20260911.py
bash run_torch.sh -m unittest discover -s tests -p test_surface_mapping.py
```

The six existing surface contract tests pass. An initial module-style unittest command failed because `tests.test_surface_mapping` is not importable; discovery above is the successful native command. Logs preserve both outcomes.

Detailed metrics, input SHA256 hashes, and arrays: `outputs/dance_mapper_audit_20260911/{metrics.json,extra_metrics.json,audit_arrays.npz}`. Diagnostic filenames use a unique suffix, not a claimed experiment registration date.

## 1. The source moves; the output retains mostly its slower component

Selected regional signals are the exact six weighted body regions used by the mapper. RMS speed is Euclidean velocity RMS over frames and fixtures, not a human dance score.

| Signal | Quiet 0–20 s RMS speed (m/s) | Active 70–90 s RMS speed (m/s) |
|---|---:|---:|
| Source root-relative regional motion | 1.287 | 1.283 |
| Raw requested floor targets | 1.719 | 5.329 |
| After target EMA | 0.858 | 2.511 |
| Decoded emitted floor axes | 0.301 | 1.330 |

The active/quiet source-speed ratio is 0.997, compared with emitted ratio 4.424. This selected statistic does not show quieter source movement in the quiet interval. Much of the quiet/active movement contrast is introduced downstream by the audio activity gain and speed gate. It does not follow that every source motion statistic is independent of music.

Active source regional position power is split 49.1%, 30.6%, 16.5%, and 3.8% across the four bands below, normalized over 0.1–4 Hz. Thus the faster input components are present. They might include jitter; this instrument does not label them as good dance.

### Temporal fidelity against the requested-angle comparator

Welch PSD uses 40 Hz data, up to 10 s Hann windows, default 50% overlap, and per-window linear detrending. Powers are summed over all twelve pan/tilt channels. The table reports square roots of emitted/requested band-power ratios, not a linear-system transfer function.

| Frequency band | Active EMA/raw amplitude ratio | Active emitted/raw ratio | Full-track emitted/raw ratio |
|---|---:|---:|---:|
| 0.1–0.5 Hz | 0.966 | 0.982 | 0.973 |
| 0.5–1 Hz | 0.781 | 0.526 | 0.569 |
| 1–2 Hz | 0.556 | 0.243 | 0.181 |
| 2–4 Hz | 0.316 | 0.081 | 0.055 |

Large multi-second extents can pass while 1–2 Hz angular movement has only 24% of the requested RMS band amplitude in the active excerpt. This is a direct mismatch between the existing extent gate and temporal fidelity.

Active angle cross-correlation peaks when output is shifted **0.450 s behind raw targets**: zero-lag correlation 0.495 versus lag-aligned 0.811. Smoothed-target to output lag is 0.325 s. Full-track raw-to-output peak lag is 0.475 s. The comparator here is requested motion, not audio or a shuffled-music null. Correlations pool demeaned channels, and search lags 0–2 s. This is a descriptive effective lag, not constant hardware latency.

Cross-spectral analysis also shows distortion, not merely one removable delay: active 1 Hz aggregate cross-spectral gain is 0.277, phase −114 degrees, coherence 0.706. At 1.5 Hz gain is 0.098 and coherence 0.372. A global time shift alone cannot recover discarded amplitude or altered paths.

## 2. The mapper asks for movement its own limits cannot track

Rev02 uses a 0.20 s positional EMA, followed by error-only tracking with 0.18 s time constant, then quantized acceleration/speed projection. The advertised 30 degrees/s is a maximum. The actual desired-speed law is:

```text
local_limit = 3 + 27 * activity**2
```

In active 70–90 s, activity median is 0.630 and the local-limit median is **13.70 degrees/s**. Requested raw angular speed median is **24.50**, p95 **78.50 degrees/s**. After EMA those numbers are **13.38 / 41.97**. Actual emitted speed is **9.74 / 19.68**.

- 74.8% of raw fixture steps demand speed above their contemporaneous local limit; 44.4% still do after EMA.
- 95.7% of raw requested angular acceleration samples exceed 80 degrees/s²; 64.9% still do after EMA. This includes high-frequency source and interpolation components; it is not a human motion-validity statistic.
- In the active excerpt, 34.6% of emitted fixture steps are above 90% of the local desired-speed limit.
- Hard stage clipping is rare: active fractions by fixture are 0.50%, 1.375%, 0%, 0%, 0%, 0%. Clipping is not the main active bandwidth failure.

The activity-dependent limit is a **soft desired-speed limit**, not a hard instantaneous cap. The acceleration constraint and quantization can leave actual speed above it, especially while activity falls. All audited actual angular accelerations remain below the hard 80 degrees/s² bound.

Increasing articulation gain fixed an extent metric partly by asking for more travel. It did not make that travel mechanically feasible on beat-scale times. Simply removing smoothing or making the tracking time constant tiny cannot satisfy an infeasible raw trajectory under unchanged hard limits.

## 3. The return of the crown starts before the actuator

| Interval | Raw floor x-span median | Emitted floor x-span median | Source root-forward world z median |
|---|---:|---:|---:|
| 180–190 s | 6.676 m | 6.561 m | +0.923 |
| 194–201 s | 3.703 m | 4.066 m | −0.887 |

During the rejected local contraction, regional RMS speed rises to 2.091 m/s versus 1.260 immediately before it. The source is not frozen. Raw targets are already narrow. Median raw-to-emitted floor error rises from 0.620 to 1.120 m; lag rises from 0.475 to 0.550 s. Filtering can alter persistence, but did not create this local narrowing.

The source delta is `regional_world - fixed_neutral_T_pose`. Its x component is multiplied by up to 5.04 (hands/legs), or 10.08 (torso). A turn moves the fixed anatomical wrists inward relative to that reference, overcoming fixture lanes.

Independent static-pose control: hold the exact neutral articulated pose, activity=1, root stationary, and change only root yaw. Recompute raw mapper output; no DMX/controller needed.

| Root yaw | Requested floor x span |
|---|---:|
| 0 degrees | 9.500 m |
| 45 degrees | 7.377 m |
| 90 degrees | 4.534 m |
| 135 degrees | 3.726 m |
| 180 degrees | 5.026 m |

At 135 and 180 degrees outer target identities cross stage sides. Fixed anatomical identity alone does not preserve spatial composition. The real interval association plus this isolated control establish a mapper turn-contraction mechanism; they do not attribute every pixel of the reported crown to yaw alone.

## 4. Structural blind spots

Ignoring shared root terms, scalar gain and clipping, each selected regional vector uses:

```text
surface_x = 2.8 * regional_x
surface_z = 2.2 * regional_y + 2.6 * regional_z
surface_y = 0
```

The exact lost regional direction is `(0, 2.6, -2.2)`. In this recording, 28.6% of selected-region velocity energy projects onto that null direction (28.2% active). This is a geometric component-energy measurement, **not** “28% of dance information lost”: correlated components may remain observable through other axes. Depth/height cancellation ratio is 1.03 active versus 1.0 for uncorrelated terms; there is no measured pervasive destructive cancellation beyond the inevitable dimensional reduction.

All raw target y values are exactly zero. There are three fixed depth lanes, fixed region ownership, one source body, and no explicit facing, contact, or gesture-transition channel controlling composition. Terminal wrist orientation is invisible to its own joint position. This mapper has no means to distinguish poses that share its six selected regional points. It can move a floor fan; it does not encode the full articulated body or varied surface families.

## Minimal proposed next fix — not implemented or validated

Do not retrain or increase gain again on this evidence. Split two mapper changes so their effects remain identifiable:

1. **Temporal fidelity first:** replace error-only chase with target-angular-velocity feed-forward plus a small position-error correction, retaining the existing quantized hard speed/acceleration safety projection. Reduce redundant positional EMA only alongside a feasibility check. Scale infeasible fast angular excursions around the slower lane trajectory rather than let tracking smear them into long delayed sweeps. This trades fast excursion size for timing; it is not permission to exceed 30/80 or quiet pacing. Test one frozen active excerpt first against rev02, raw targets, and a mechanically feasible synthetic trajectory. Measure lag, cross-spectral gain/coherence, local saturation and all existing mechanics gates. A controller change alone cannot guarantee faithful reproduction of infeasible source paths.
2. **Turn contraction separately:** decompose local articulation from root facing. Keep regional gestures bounded around fixture lanes; map root facing to a bounded shared sweep/skew rather than an amplified subtraction that reverses those lanes. Do not silently remove root turns: retain and test the explicit facing contribution. The static yaw control above and the real 194–201 s segment must both be included. This is a proposed representation change, not a measured successful repair.

Neither patch creates surface-family diversity by itself. Keep human audiovisual review failed/unconfirmed until actually performed. Existing spatial gates and the six passing tests establish geometry contracts, not dance quality or musical timing.
