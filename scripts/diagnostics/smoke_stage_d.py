"""Stage D smoke: render single frames and verify pixels are real.

Renders one frame per segment label and checks:
  * the GL context is the RTX 5070 via EGL
  * frames are not black and not uniform (a solid colour means the shader
    silently failed)
  * a drop frame is measurably brighter than an intro frame
Saves PNGs for visual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.dmx import club_rig, render_stream
from lightsem.sim import RenderConfig, StageRenderer
from scripts.diagnostics.probe_control_synthetic import synth_edm

OUT = Path("/root/lightsem/data/cache/frames")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    y, _ = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    antic = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)
    rig = club_rig()
    stream = render_stream(rig, feats, segs, antic, fps=40.0)

    r = StageRenderer(rig, RenderConfig(width=960, height=540))
    print("GL renderer:", r.renderer_name)
    assert "NVIDIA" in r.renderer_name.upper(), "not on the NVIDIA GPU"

    stats = {}
    seen = set()
    for s in segs:
        if s.label in seen:
            continue
        seen.add(s.label)
        t = (s.start_s + s.end_s) / 2
        idx = min(int(t * stream.fps), stream.frames.shape[0] - 1)
        img = r.render_frame(stream.frames[idx], float(t))
        p = OUT / f"seg_{s.label}.png"
        Image.fromarray(img).save(p)
        m, sd, mx = float(img.mean()), float(img.std()), int(img.max())
        stats[s.label] = m
        print(f"  {s.label:10s} t={t:6.2f}s  mean={m:6.2f} std={sd:6.2f} "
              f"max={mx:3d}  -> {p.name}")
        assert mx > 0, f"{s.label}: frame is pure black"
        assert sd > 1.0, f"{s.label}: frame is uniform - shader likely failed"

    if "drop" in stats and "intro" in stats:
        print(f"\ndrop mean {stats['drop']:.2f} vs intro mean {stats['intro']:.2f}")
        assert stats["drop"] > stats["intro"], "drop frame not brighter than intro"
        print("  -> drop brighter than intro OK")

    # Colour check: the render must not be greyscale.
    img = r.render_frame(stream.frames[int(20 * stream.fps)], 20.0)
    ch = img.reshape(-1, 3).mean(axis=0)
    print(f"\nchannel means R={ch[0]:.1f} G={ch[1]:.1f} B={ch[2]:.1f}")
    assert float(np.std(ch)) > 0.5, "render is greyscale - colour path broken"
    print("  -> colour present OK")

    print(f"\nframes in {OUT}")
    print("STAGE D SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
