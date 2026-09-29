# Festival renderer research — 2026-09-29

Recommendation: keep the festival venue as a separate scene/renderer adapter inside
LightSem, with a renderer-neutral patch manifest. Prefer Blender Eevee for an
editable, reusable venue if its first-frame performance is acceptable. A true
mesh-based ModernGL renderer remains a practical delivery route because this host
already has working headless EGL. Neither choice requires changing the model.

This is an implementation recommendation, not a measured comparison. Perplexity
Reason was consulted with the accepted C baseline, DMX contract, RTX 5070 12 GB,
Arch/headless environment, existing ModernGL renderer, and all four visual
reference descriptions. Its preference was Eevee plus a deterministic `bpy`
adapter. No render-time estimates from that answer have been accepted as facts.

## Local facts

- `configs/light_layers/baseline.json` identifies accepted C: selected independent
  Rich96 seed 0, palette history, causal 300 ms RGB fades. Replay its authenticated
  saved stream; do not rerun inference or add beat-triggered lighting.
- `lightsem/dmx/fixtures.py` defines prototype personalities, explicitly not GDTF
  fixture models. The club patch uses channels 1–117: six moving heads, four
  washes, two blinders, one strobe, one projector. Moving pan/tilt is 16-bit,
  coarse byte first; the projector profile occupies channels 108–117.
- `lightsem/sim/renderer.py` reads the DMX bytes rather than internal fixture state
  and caps lighting at 16 fixtures. Preserve that byte-to-image separation.
- `lightsem/dmx/universe.py` already has an ArtDMX sender, default localhost UDP
  6454, universe 0; its docstring explicitly says physical-node use is untested.
- Initial inspection found ffmpeg but no Blender executable in PATH or usual
  `/usr/bin` and `/opt` locations. This observation predates any task installs.

Portable Blender was subsequently installed at `/opt/blender-lightsem/blender`,
version **4.5.14 LTS**, from the
[official Linux x64 tarball](https://download.blender.org/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz).
Its SHA256 matched the [official checksum file](https://download.blender.org/release/Blender4.5/blender-4.5.14.sha256):
`9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3`.
The background/factory-startup Python smoke check exited 0 and reported
`(4, 5, 14)`. No system packages were changed by this portable installation;
GPU render speed is a separate check.

## Patch and playback design

Use `accepted DMX + source fps → patch/decode → scene.apply(frame) → frame →
ffmpeg/source audio`. Use stored fps/timestamps, never derive the clock from
rounded audio duration. Keep source stream hash, patch hash, renderer settings,
audio hash and output timing in a render receipt. Index bytes at the original
time without adding another interpolation layer to the accepted fades.

Each patch entry should identify scene object, source fixture/channels, geometry,
orientation and fixed gain. Additional physical-looking fixtures can be
deterministic copies or explicit channel remaps; they are not additional learned
outputs. LED garlands can share wash RGB/dimmer; additional projector instances
can share the existing projector channels. Explain this in delivery notes.
Keep static decorative illumination distinguishable from controlled elements.

Later live input should feed the same adapter. Art-Net transports DMX universes
using UDP and assigns them Port-Addresses. [Art-Net terminology and specification](https://art-net.org.uk/art-net-introduction-and-terminology/).
BlenderDMX already supports Art-Net/sACN, GDTF fixtures and DMX property drivers;
it is a useful subsequent integration, not a prerequisite for offline byte replay.
[BlenderDMX documentation](https://blenderdmx.eu/docs/faq/).

## Scene and projection

Build the references as actual geometry: woven radial canopy, fairy-light net,
suspended concentric hoops, organic totems, geometric sails, stage/screen,
physical fixture bodies, crowd silhouettes and bounded haze.

For projection, transform receiver world positions through each projector's
view/projection transform, perform perspective division, clip behind/outside the
frustum, then sample its image/pattern. Add light-space depth comparison for
occlusion where implemented. This distinguishes projection from putting an
unrelated texture on a screen. [NVIDIA projective texture/shadow mapping](https://developer.download.nvidia.com/assets/gamedev/docs/shadow_mapping.pdf).

Eevee light nodes cannot directly supply textured projection. Receiver materials
with projector-space mapping are a useful approximation; report absent shadow
occlusion/indirect illumination honestly. A UV Project modifier alone is not a
complete projector optical model. [Eevee limitations](https://docs.blender.org/manual/en/latest/render/eevee/limitations/limitations.html).

Media/pattern content is an authored scene asset; the generic projector's DMX
channels control its brightness/color/zoom/focus/pattern selection. It is not
neural image generation. Lasers should be explicitly described as a visual
approximation driven by declared channels, not a physical scanner simulation.

## Performance and verification

Eevee uses GPU rasterization; its volume effect is single-scattering and limited
to camera rays. High scene/resource use can exhaust GPU memory. [Eevee limitations](https://docs.blender.org/manual/en/latest/render/eevee/limitations/limitations.html).
Volume resolution and step count affect quality and VRAM use. [Eevee volumes](https://docs.blender.org/manual/en/latest/render/eevee/render_settings/volumes.html).

Start with one bounded haze region and modest volumetric settings; instance or
batch fairy bulbs/crowd geometry instead of adding thousands of shadow lights.
Profile a representative bright frame, then a short clip, before full-track
rendering. Reuse scene and GPU resources across frames. Keep fixed exposure so
the visualizer does not hide DMX dimmer changes. Verify source fps/audio sync,
blackout response, changed-byte response, complete frame count, and rendered
geometry from several camera views.

Blender can execute Python and render without its UI using `-b` and `-P`; a
separate GNOME session or MCP server is unnecessary for this batch workflow.
[Blender CLI arguments](https://docs.blender.org/manual/en/4.2/advanced/command_line/arguments.html).
An editable `.blend` file is valuable when Blender is the actual backend.

## Implemented optical check

The first scene test exposed weak moving-head cones despite valid DMX and aiming.
The Blender spot API defines power before cone restriction. At a 5.39-degree
full angle, only about 0.055% of a uniform sphere's solid angle lies in the cone.
A diagnostic power increase made the beams visible; finer volume sampling alone
did not. The adapter now divides a declared beam-flux scale by
`(1 - cos(full_angle / 2)) / 2`, preserving approximate in-cone flux as zoom
changes. Its soft beam edge and uncalibrated 100 W scale remain visualization
choices, not manufacturer photometry.
[Blender 4.5 SpotLight power definition](https://docs.blender.org/api/4.5/bpy.types.SpotLight.html).

The completed first pass rendered 1,080 native 1920×1080 frames at 32 Eevee samples
in about 417 seconds on this machine. A compact 1.1 Mbps encode visibly lost fine
canopy details, so the delivery also includes a 6 Mbps rerender of the verified
native animation. These are measurements of this scene on this host, not general
renderer performance claims.
