"""Trace ONE moving head from scene state -> DMX -> decode.

Decode reports dim=0.075 and zoom=0.00 during a drop, where the scene engine
should be writing near-full dimmer. Either write() or _decode_fixture() has
the wrong offsets. This walks the exact bytes.
"""

from __future__ import annotations

import numpy as np

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.dmx import club_rig, render_stream
from lightsem.dmx.scenes import SceneEngine
from lightsem.sim.renderer import _decode_fixture
from scripts.diagnostics.probe_control_synthetic import synth_edm


def main() -> None:
    y, _ = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    antic = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)
    rig = club_rig()

    t = 17.83  # inside a drop
    eng = SceneEngine(rig=rig, feats=feats, segments=segs, antic=antic)
    eng.apply(t)

    f = rig.by_kind("moving_head")[0]
    s = f.state
    print(f"SCENE STATE for {f.name} at t={t}:")
    print(f"  dimmer={s.dimmer:.4f} zoom={s.zoom:.4f} pan={s.pan:.4f} "
          f"tilt={s.tilt:.4f} hue={s.hue:.3f} sat={s.saturation:.3f}")
    print(f"  rgb={tuple(round(c,3) for c in s.rgb())}")

    u = rig.universe()
    a = f.address - 1
    print(f"\nRAW DMX bytes at address {f.address} (0-based {a}):")
    for off in range(12):
        print(f"  +{off:2d} (ch {a+off:3d}) = {u[a+off]:3d}")

    p = _decode_fixture(f, u)
    print(f"\nDECODED: dim={p['dimmer']:.4f} zoom={p['zoom']:.4f} "
          f"pan={p['pan']:.4f} tilt={p['tilt']:.4f} rgb={p['rgb']}")

    print("\nEXPECTED offsets per fixtures.py write():")
    print("  +0/+1 pan coarse/fine, +2/+3 tilt coarse/fine, +4 dimmer,")
    print("  +5 strobe, +6/7/8 RGB, +9 zoom, +10 focus, +11 gobo")

    # Sample dimmer over a whole drop to see the true range.
    stream = render_stream(rig, feats, segs, antic, fps=40.0)
    drop = next(s2 for s2 in segs if s2.label == "drop")
    m = (stream.times >= drop.start_s) & (stream.times < drop.end_s)
    dch = stream.frames[m][:, a + 4]
    print(f"\ndimmer channel over drop: min={dch.min()} max={dch.max()} "
          f"mean={dch.mean():.1f}")
    zch = stream.frames[m][:, a + 9]
    print(f"zoom   channel over drop: min={zch.min()} max={zch.max()} "
          f"mean={zch.mean():.1f}")


if __name__ == "__main__":
    main()
