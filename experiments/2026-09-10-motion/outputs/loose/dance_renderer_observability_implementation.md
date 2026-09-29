# Renderer observability controls — implementation

## Scope

This is an opt-in rendering/presentation ablation, not a musical-quality result.
The parent supplied the audit measurements: at 640×360 the current scene support
is 3.745%; 1–2 Hz projected hit motion is about 1–3 pixels, compared with beam
support radii of 7–15 pixels. This worker did not rerun those measurements.
Those values motivate a matched observation test, not a claim that tighter optics
or a closer camera improve the generated lighting.

## Controls

`RenderConfig` adds these fields at the end to preserve positional compatibility:

- `camera_scale=0.85`: replaces the existing projection multiplier. Smaller values
  narrow the field of view; vertical FOV is `2*atan(camera_scale)`. This does not
  select or change `cam_pos`, `cam_target`, or the existing orbit.
- `beam_edge_profile="legacy"`: preserves the existing cone expressions.
  `"angular"` changes moving heads only (`u_lkind == 0`).
- `beam_edge_softness_deg=0.6`: angular transition half-width, used only in the
  angular head branch. Nominal beam width still comes from decoded DMX zoom.

For nominal half-angle `h` and softness `s`, the angular branch evaluates
`smoothstep(cos(h+s), cos(max(h-s,0)), cosA)`. It has full on-axis strength and
zero cone strength at and outside `h+s`. This is a cosine-domain interpolation
between angular bounds, not a claim of exactly half strength at `h`.
A nominal full beam of 6° and softness 0.6° gives half strength at
3.059358° off axis (full half-strength width 6.118716°).

The unchanged legacy expression uses softness 0.018 for heads/projectors and
0.16 for washes. For a nominal 6° full beam, its absolute half-strength full
width is 13.342039°. As nominal width approaches zero, the legacy
absolute half-strength width approaches 11.914070°, not zero. A narrow
legacy cone does not reach unit on-axis strength. These are analytical cone
factors, not rendered pixel widths, relative-to-peak FWHM, or luminance results.

The original shader text is unchanged outside the added uniforms, the new head-only
conditional, and replacement of the two camera multipliers. Removing these exact
additions restored the pre-edit shader source byte-for-byte during inspection.
Legacy cone expressions, wash/projector/flood response, falloff, exposure, haze,
time, orbit, role gains and default DMX zoom remain unchanged. No mapper was edited.

## Validation and failure behavior

New config values must be finite: camera scale > 0, softness in (0, 90) degrees,
and edge profile in {legacy, angular}. Angular moving-head fixture ranges must
satisfy finite `0 < beam_min_deg <= beam_max_deg <= 180`; this is checked before
GL allocation. Basic existing config geometry is also checked: positive integer
image dimensions, finite positive FPS, finite nonnegative haze/intensity, finite
camera coordinates/orbit, and a base view direction not parallel to world up.
Existing fixed-exposure validation remains in place.

All frame scalar uniforms and light arrays now fail explicitly if absent.
The added controls are not silently skipped. No GL context was created by this
worker, and no GPU render or GPU test was run.

CPU command:

```text
bash run.sh -m unittest discover -s tests -p test_renderer_observability.py -v
```

Result: 12 tests passed. Tests cover analytical center/cutoff/half-strength/floor,
unchanged non-head reference response, GLSL source equations, default values,
invalid configs and angular fixture ranges, explicit uniform uploads/missing
uniform failures, unchanged light-array bytes and unmodified input DMX. Mock GL
objects and source checks do not prove shader execution or pixel parity.

## Required parent-scheduled GPU validation (not yet run)

1. Freeze candidate source and matched emitted DMX. Record all config fields,
   renderer/device/driver, frame timestamps and source identities.
2. Require raw pixel parity between the pre-edit renderer and candidate defaults
   (`camera_scale=.85`, `beam_edge_profile="legacy"`). Use identical camera,
   time/orbit, exposure, haze and DMX, including narrow heads and non-head fixtures.
   Compare raw RGB bytes, not lossy encoded video. Do not assume source preservation
   guarantees driver/compiler pixel parity.
3. Confirm uniforms are active and shader compiles on the scheduled real GPU.
   Run isolated heads across zoom endpoints and intermediate values, plus isolated
   washes/projectors/floods. Verify non-head pixel parity when only edge mode changes.
4. Run matched camera-only, zoom-only, and edge-only controls plus any explicitly
   labelled combined arm. Hold the DMX motion candidate and all other presentation
   controls fixed. Zoom is a parent-scheduled explicit intervention, not a new
   renderer or mapper default.
5. Measure projected motion, beam support, clipping and luminance with the same
   instrument/ROI in each arm. Keep fixture bulb glow separate. Angular cones can
   change brightness as well as support; do not describe a visibility difference
   as evidence of better audio conditioning or musical quality.

No vault, training, model, mapper or original snapshot files were changed.
