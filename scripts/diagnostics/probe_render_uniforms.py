"""Why are all rendered frames identical? Are the light uniforms landing?

Every segment rendered to mean=30.91 std=2.41 -- bit-identical regardless of
DMX content. Either (a) the uniforms are not being set, (b) the fixture
decode returns zero intensity, or (c) the shader ignores them.
"""

from __future__ import annotations

import numpy as np

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.dmx import club_rig, render_stream
from lightsem.sim import RenderConfig, StageRenderer
from lightsem.sim.renderer import _beam_direction, _decode_fixture
from scripts.diagnostics.probe_control_synthetic import synth_edm


def main() -> None:
    y, _ = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    antic = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)
    rig = club_rig()
    stream = render_stream(rig, feats, segs, antic, fps=40.0)

    # 1. Does the DMX universe itself differ between intro and drop?
    i_intro = int(4.93 * stream.fps)
    i_drop = int(17.83 * stream.fps)
    u_i, u_d = stream.frames[i_intro], stream.frames[i_drop]
    print(f"universe intro vs drop: identical={np.array_equal(u_i, u_d)}  "
          f"n_diff_channels={(u_i != u_d).sum()}")
    print(f"  intro nonzero={np.count_nonzero(u_i)}  drop nonzero={np.count_nonzero(u_d)}")

    # 2. Does the fixture decode see intensity?
    print("\ndecoded fixtures at DROP:")
    for f in rig.fixtures[:6]:
        p = _decode_fixture(f, u_d)
        d = _beam_direction(f, p["pan"], p["tilt"])
        print(f"  {f.name:9s} dim={p['dimmer']:.3f} rgb=({p['rgb'][0]:.2f},"
              f"{p['rgb'][1]:.2f},{p['rgb'][2]:.2f}) zoom={p['zoom']:.2f} "
              f"dir=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}) kind={p['kind']}")

    # 3. Which uniforms actually exist in the linked program?
    r = StageRenderer(rig, RenderConfig(width=320, height=180))
    names = sorted(r.prog._members.keys()) if hasattr(r.prog, "_members") else []
    print(f"\nprogram members ({len(names)}):")
    print("  ", [n for n in names if not n.startswith("gl_")][:40])

    for probe in ("u_lpos", "u_lpos[0]", "u_lint[0]", "u_n_lights", "u_haze"):
        print(f"  {probe!r:16s} in prog: {probe in r.prog}")

    # 4. Render with the real universe vs a forced full-bright universe.
    a = r.render_frame(u_d, 17.83)
    forced = np.zeros(512, dtype=np.uint8)
    for f in rig.by_kind("moving_head"):
        forced[f.address - 1 + 4] = 255           # dimmer
        forced[f.address - 1 + 6] = 255           # R
        forced[f.address - 1 + 2] = 128           # tilt mid
    b = r.render_frame(forced, 0.0)
    print(f"\nreal-universe frame : mean={a.mean():.2f} std={a.std():.2f} max={a.max()}")
    print(f"forced-bright frame : mean={b.mean():.2f} std={b.std():.2f} max={b.max()}")
    print(f"identical: {np.array_equal(a, b)}")


if __name__ == "__main__":
    main()
