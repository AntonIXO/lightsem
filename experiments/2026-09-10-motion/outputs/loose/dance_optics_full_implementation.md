# Registered full optics runner implementation

**Implemented, not executed on study data.** No GPU context, GPU query, render, vault call, or main study measurement was made by this worker. Only CPU synthetic tests ran. Defaults and production code remain unchanged.

## Files and validation

- `outputs/run_dance_optics_full.py`: create-only stages `freeze`, `geometry`, `render --arm`, `match --arm`, `pair`, `report`.
- `tests/test_dance_optics_full.py`: 16 CPU synthetic tests. Covers DMX parity, foil channels, actual plane/truss first hits, relative half-peak inversion, camera projection, exact64-vertex circle, Welch RMS, static and invalid geometry, pixel regions, both clipping definitions, deterministic scalar rounding, frame count, receipts, and all24 single-factor contrasts.
- Test command: `CUDA_VISIBLE_DEVICES="" bash run.sh -m unittest tests.test_dance_optics_full -v`.
- Both native environments lack pytest. No dependency was installed. Tests use standard-library unittest.

## Execution order (parent owns GPU scheduling)

Use the registered output directory:

```text
/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01
```

1. `bash run.sh outputs/run_dance_optics_full.py freeze`
2. Use the **frozen script** for every later stage:
   `CUDA_VISIBLE_DEVICES="" bash run.sh <out>/frozen/run_dance_optics_full.py geometry`
3. After the parent releases GPU training, run one full arm at a time:
   `bash run.sh <out>/frozen/run_dance_optics_full.py render --arm legacy_native_legacy`
   Repeat for all13 exact keys in `design.json`. There is no arm tuning or automatic candidate selection.
4. After all13 complete:
   `bash run.sh <out>/frozen/run_dance_optics_full.py match --arm closer_native_legacy`
   `bash run.sh <out>/frozen/run_dance_optics_full.py match --arm closer_zero_angular_flux`
5. Optional CPU-only review assembly:
   `CUDA_VISIBLE_DEVICES="" bash run.sh <out>/frozen/run_dance_optics_full.py pair`
6. `CUDA_VISIBLE_DEVICES="" bash run.sh <out>/frozen/run_dance_optics_full.py report`

`--out <path>` is a global argument **before** the stage. Default already names this registration. Start each long command nonblocking through the parent harness. Capture stdout in a new log. Do not overlap render/match with training. Both rendering and hardware encoding are inside native `gpu_lock(motion_native_audio_20260910)`.

Every stage refuses an existing output directory/receipt. Failed partial directories stay as evidence; no force, overwrite, resume, or silent repair flag exists. A failed stage needs an explicit new revision, not deleting partial evidence.

## Freeze and provenance

Freeze checks design against both embedded registration plans. It requires the engineering receipt's `maximum_byte_difference == 0`, `nonhead_edge_mode_parity is True`, executed controls, and flux reference14 preview. It snapshots the exact candidate NPZ, design, registration, engineering receipt, its own script, and all `lightsem` Python/JSON source dependencies. It hashes source audio without copying the large file. Every later stage authenticates the freeze receipt, frozen files, current registered design/registration, original candidate, audio, and executable script bytes. Native imports then resolve only through the frozen source tree.

**Engineering limitation:** the existing engineering receipt lacks hashes of its tested source. Freeze retains both that receipt and current renderer source hashes, but cannot retroactively prove which source the earlier GPU process tested. Parent must establish no changes between preflight and freeze. Do not silently describe this as cryptographic tested-source binding.

Receipts use exclusive creation, fsync, read-only mode and SHA256 sidecars. Derived artifacts are hashed in the final stage receipt. Report verifies recorded artifacts. Root can override permissions; these are tamper-evident receipts, not a security boundary.

## Frozen instruments

### Geometry — not measured pixel motion

All11092 source DMX frames at40Hz. Moving-arm pan/tilt bytes remain identical. Only six head zoom bytes can change to zero. The foil fixes only six heads' four angle bytes to t80s; brightness/color/non-head channels remain source-identical. Foil intensity4.8 vs2.4 is solely in the registered renderer config.

Axis intersections include finite floor and back-wall geometry plus the shader's three truss boxes. This differs from the old observer's floor/wall-only instrument and avoids calling an infinite-floor projection an actual first hit. The metric is still analytic emitted-axis geometry, **not** camera sphere tracing, an illuminated spot, or a pixel track. Camera occlusion and shader marching discretization are not modeled.

Circle center is half the **actual per-frame first-hit distance**, not a fixed median distance. The circle is perpendicular to the emitted unit axis. Radius uses half of the actual on-axis cone peak. The legacy zoom0 axis peak is below1, so absolute cone factor0.5 would be wrong. Flux multiplication cancels out of the relative half-peak width. Exactly64 polygon vertices are projected with the frozen camera; equivalent radius is sqrt(projected area/pi).

The primary is the mean across six per-head ratios: 1–2Hz projected midshaft vector RMS divided by that head's temporal mean equivalent circle radius, active70–90s. Ratio-of-means is also reported as supplemental, not substituted. Welch uses40Hz,400-sample Hann windows,50% overlap, linear detrending and trapezoid integration over1<=f<2; x/y powers sum before square root. Short turn194–201s uses all280 available frames and is explicitly supplemental. Invalid hits/projections fail the interval instead of being dropped.

All four intervals, six heads, first-hit RMS,24 single-factor contrasts, combined contrast, and static foil contrast are retained. No brightness weighting enters geometry.

### Raw display pixels — not HDR or head tracking

Every generated640x360 RGB frame is measured before titles and video encoding. Display luma is Rec.709 RGB weighting in0..255 units. Support is descriptive luma>5; this threshold is not an additional quality gate. Both any-channel>=250 and white(all channels)>=250 are recorded per frame.

Three fixed masks:

- All pixels outside the top28 caption rows.
- The same region minus20px circles around **all14 rendered fixture origins**.
- Those bulb circles alone, also excluding the top28 rows.

These separate source-glow regions from the rest of the image, not individual heads from overlapping lighting. Raw per-frame statistics and exact per-frame SHA256 hashes are saved. Whole-stream raw RGB hashes support the brightness rerender check. Full chronological1Hz sheets contain every integer second0…277; no crop or selected interval is substituted for full coverage.

Exact decimal277.3*30 gives **8319 generated frames**. This avoids binary-float `ceil` producing8320. Receipts never call this an encoded or decoded count. Original audio is muxed, duration fixed277.3s, time uniform0, no camera orbit, AV1 NVENC and AAC192kbps. All13 original videos remain separate.

### Mean-display-brightness control

Only after all13 original receipts exist, read the fulltrack mean luma outsidebulbs for the fixed two comparators. Target=min(two means); scalar=target/each mean. Each fulltrack receives **one RGB dimming scalar**, then deterministic nearest-even uint8 rounding. No framewise scaling or increased gain.

To avoid storing ~5.75GB of raw RGB per movie, match re-renders each comparator from frozen source. The unscaled full-stream raw hash must equal the original; otherwise the matched receipt fails. Matched means and quantization residuals are recorded. This is two additional full GPU renders, not a cheap codec brightness filter. Each matched member has its own video and1Hz sheets. Optional `pair` CPU-decodes and re-encodes a1280x360 side-by-side review derivative; raw metrics refer to the separate members, not that extra transcode.

## Explicit remaining work and limitations

- Parent must freeze, run the geometry instrument and all13 original GPU arms, then both matched rerenders. This implementation has not measured or rendered registered data.
- Parent must independently decode every full original/matched video, verify actual frame counts and audio, and inspect chronological sheets/synced videos. Runner records these gates as **pending**, never passed.
- No optical-flow tracks, per-head segmented pixel motion, HDR energy calibration, photometric lumens, or subjective dancing score is implemented or claimed.
- No music-acceptance result follows from this experiment. Motion and direct audio brightness channels are held fixed.
- `report` computes numeric proxy and candidate clipping gates, all fulltrack raw-pixel control contrasts, and scalar receipts. It leaves coverage/visual/music acceptance unassessed.

## Parent validation update
Parent reran GPU parity through the actual run.sh renderer runtime in optics_flux_engineering_03. It hashes all Python/JSON source dependencies, input DMX, archived comparator sources/protocol, and preflight script before/after; exact legacy pixel and nonhead parity passed. Freeze now requires and checks those hashes. Source-binding synthetic test added;17 CPU tests pass. Original engineering01/02 evidence preserved.
