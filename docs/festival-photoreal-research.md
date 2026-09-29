# Festival realism revision: optical research and local evidence

Research date: 2026-09-29. These notes support the revision of the festival scene;
only the final inspected render can establish its visual quality.

## Diagnosis from the previous delivered image

The previous poster was inspected at `runs/festival_garden/show/lightsem_festival_hq_poster.png`.
Its visible deficiencies match implementation choices: simplified people,
flat uniform magenta rings, coarse box fixtures, bright thick laser lines crossing
the whole foreground, unshadowed blue illumination, and an exceptionally clean
floor. The visible canopy is largely straight colored strands. Removing people
and darkening the world will help, but alone will not repair those material and
light-transport cues.

Recommended revision: no people, zero world illumination and no moon; physical
shadowing, fine nonemissive cloth and metal, roughness and surface variation in
real-world units, smaller bulbs, and restrained laser proxies. Keep the saved DMX
stream as the source of all time-varying light power. A material should not emit
merely to make an unlit decoration readable.

## Cycles and the installed GPU: measured

A minimal real render completed with the installed
`/opt/blender-lightsem/blender` version **4.5.14 LTS** and NVIDIA RTX 5070
(12,227 MiB, driver 610.57.04). Device discovery returned CUDA, CPU and OptiX;
only the OptiX device was enabled for the render. A 64×64 scene with one sphere,
one ground plane, one area light, black world and 8 Cycles samples rendered and
saved successfully in **0.877 seconds**, measured around `bpy.ops.render.render`.
This is a device smoke check, not a forecast of the festival animation's cost.

The proof script/log are `/tmp/festival_cycles_smoke.py` and
`/tmp/festival_cycles_smoke.log`. Runtime setup:

```python
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'OPTIX'
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
```

OptiX's documented minimum driver is 535; Blackwell CUDA binary support was
added to Blender. Local rendering evidence is more specific than either support
statement. [GPU rendering](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html),
[Blackwell support change](https://projects.blender.org/blender/blender/pulls/134170).

## Spot power is unchanged between these engines

Installed Blender RNA identifies spot `energy` as radiant power before cone
restriction. The existing beam-flux adapter remains appropriate in Cycles:

```python
cone_fraction = (1 - math.cos(full_cone_angle / 2)) / 2
light.energy = desired_in_cone_flux / cone_fraction
```

This is an approximate conversion from an authored in-cone power scale, not an
engine correction and not electrical fixture watts. At a 5.4° full cone angle,
100 W in-cone maps to roughly 180 kW in Blender's pre-restriction parameter.
A soft cone edge reduces the actual integrated flux somewhat. `normalize=True`
normalizes emitter area, not cone solid angle. Both engines read the same light
datablock; switching to Cycles does not justify another power multiplier.
[Spot API](https://docs.blender.org/api/4.5/bpy.types.SpotLight.html),
[Blender 4.5 light synchronization source](https://raw.githubusercontent.com/blender/blender/blender-v4.5-release/intern/cycles/blender/light.cpp).

## Real projected light: proved locally

The previous receiver shader approximated projection with self-emitting surfaces.
Cycles can instead evaluate a procedural shader on a spot light, illuminate a
plain fabric receiver, and cast the pattern onto intervening geometry. This is
useful for the rings, sails, canopy and stage to share actual illumination.

`/tmp/festival_projector_proof.py` produced a 256×256, 32-sample OptiX render of a
red/green rectangular checker from a real spot. The receiving plane's Principled
material has no emission. A box between the light and plane receives the pattern
and casts a visible shadow. Its rendered result is
`/tmp/festival_projector_checker.png`; the editable proof is
`/tmp/festival_projector_proof.blend`.

Verified node coordinates and power setup:

```text
Texture Coordinate.Normal → Separate XYZ
forward depth = -Z
projected x = X / depth
projected y = Y / depth
multiply x,y by DMX zoom; feed procedural image/pattern
rectangle gate = (abs(x)<half_width) * (abs(y)<half_height)
pattern intensity * rectangle gate → Emission.Strength
pattern color * DMX tint → Emission.Color
Emission → Light Output.Surface
```

The Normal output is light-local and follows light rotation. A diagnostic normal
render confirmed the central outgoing direction has negative local Z. This is
not Generated/object-position mapping. A sufficiently wide spot cone must
contain the rectangle, and actual geometry supplies shadow occlusion.
[Texture coordinate documentation](https://docs.blender.org/manual/sr/5.0/render/shader_nodes/input/texture_coordinate.html).

**Node Emission.Strength is an additional multiplier**, not a replacement for
`light.data.energy` in Blender 4.5. The default node strength was 1. Setting it to
10 left `data.energy=100`; changing `data.energy` to 20 left node strength 10.
Use physical/authored light power in `data.energy` and a dimensionless 0–1
pattern in shader strength. Keep `data.color=(1,1,1)` if the DMX color already
enters the shader, to avoid multiplying the tint twice. Light synchronization
explicitly sets strength from color × energy × 2^exposure before shader
application. [Blender 4.5 source](https://raw.githubusercontent.com/blender/blender/blender-v4.5-release/intern/cycles/blender/light.cpp).

The proof used a 1 mm source radius for sharp projected detail, a 100° spot cone,
and a rectangle with tangent-space half-extents 0.65 × 0.45. Those are diagnostic
settings, not suggested dimensions for every real fixture. For a festival
projector, a source radius around 5–10 mm is a useful initial sharpness choice.
The final script also rendered the same scene with `data.energy=0`; shader nodes
retained their checker pattern. With default output dithering, the resulting
8-bit PNG had maximum value 1 and mean 0.083 across RGB. Set
`scene.render.dither_intensity=0` for a strict zero-byte blackout check, or treat
one byte of dithering as output quantization rather than scene emission.

## Self-occlusion when enabling shadows

The old optical origins are at the centers of opaque housing cubes, with an
opaque emissive lens directly in front of the spot. Shadows disabled concealed
this construction. Once shadows are enabled, these objects can block their own
fixture output. Move the physical emitter past the housing/front lens along its
aimed local -Z, or construct a housing with an actual open aperture. If an
emissive lens is only a visual proxy, disabling shadow visibility for that lens
is a controlled approximation. Do not disable all environmental shadows to
compensate. Projector origins need the same treatment.

## Haze, exposure and geometry choices

Use a bounded volume mesh with a Volume Scatter shader and no Surface shader.
The world remains black. A density of 0.003 per scene-meter over 70 m has
optical depth 0.21; 0.008 has depth 0.56. These are meaningful starting points
for previews rather than blindly increasing density until everything glows.
Anisotropy around 0.3–0.5 is an artistic initial choice; the actual beam direction
relative to the camera changes perceived shaft brightness.
[Volume Scatter documentation](https://docs.blender.org/manual/en/4.5/render/shader_nodes/shader/volume_scatter.html).

Keep fixed AgX exposure and gamma 1; raising exposure cannot repair absent
shadowing, flat geometry or uncalibrated emission. Use physically shaded cloth
and metal for the hanging rings; keep micro-weave at millimeter scale so it does
not become a checkerboard at festival scale. Damp rough ground can reveal actual
light reflections, while fully mirror-smooth wet ground would create a different
venue. Narrow laser tubes remain a declared visualization proxy unless replaced
with a real scanner/beam integration model; reducing their width and brightness
avoids the thick luminous-cable look.

For performance, measure a representative full-scene frame before choosing
animation settings. Enable the Cycles light tree for many emitting bulbs and
lights. Denoising is useful but inspect thin cords/bulbs in consecutive frames;
low-sample temporal changes can erase them or make them flicker. A denoised still
is insufficient evidence of animation quality.

Perplexity Reason was consulted with the previous scene's lighting, geometry,
DMX constraints and GPU, and returned official Blender references. The key
implementation conclusions above were checked against the installed API,
Blender 4.5 source and actual local rendering. Recommendations that depend on the
final scene's measured cost or appearance remain recommendations.
