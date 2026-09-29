"""Are the moving heads actually articulating, or frozen at pan=0.5?

The level probe showed PAN mean = 127.50 in every segment. That is either
(a) a frozen fixture, or (b) a symmetric oscillation whose MEAN is trivially
the centre. Mean cannot distinguish them; spread and time-series can.
"""

from __future__ import annotations

import numpy as np

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.dmx import club_rig, render_stream
from scripts.diagnostics.probe_control_synthetic import synth_edm


def main() -> None:
    y, _ = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    antic = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)
    rig = club_rig()
    stream = render_stream(rig, feats, segs, antic, fps=40.0)

    mh = rig.by_kind("moving_head")
    pan_c = [f.address - 1 for f in mh]      # coarse pan
    pan_f = [f.address for f in mh]          # fine pan
    tilt_c = [f.address - 1 + 2 for f in mh]

    print(f"{'label':11s} {'panC.std':>9s} {'panC.min':>9s} {'panC.max':>9s} "
          f"{'panF.std':>9s} {'tiltC.std':>10s}")
    for s in segs:
        m = (stream.times >= s.start_s) & (stream.times < s.end_s)
        if not m.any():
            continue
        fr = stream.frames[m].astype(float)
        print(f"{s.label:11s} {fr[:,pan_c].std():9.2f} {fr[:,pan_c].min():9.0f} "
              f"{fr[:,pan_c].max():9.0f} {fr[:,pan_f].std():9.2f} "
              f"{fr[:,tilt_c].std():10.2f}")

    # Per-fixture spread during a drop: a fan must SEPARATE the heads.
    drop = next((s for s in segs if s.label == "drop"), None)
    if drop:
        m = (stream.times >= drop.start_s) & (stream.times < drop.end_s)
        fr = stream.frames[m].astype(float)
        print(f"\nper-head coarse pan during first drop ({drop.start_s:.1f}s):")
        for f, ch in zip(mh, pan_c):
            print(f"  {f.name}: mean={fr[:,ch].mean():6.1f} "
                  f"min={fr[:,ch].min():3.0f} max={fr[:,ch].max():3.0f}")
        spread = fr[:, pan_c].mean(axis=0)
        print(f"  spread across heads (std of per-head means): {spread.std():.2f}")

    # Time series sample.
    ch = pan_c[0]
    idx = np.linspace(0, stream.frames.shape[0] - 1, 12).astype(int)
    print(f"\n{mh[0].name} coarse pan over time:")
    print("  " + "  ".join(f"{stream.times[i]:.1f}s:{stream.frames[i,ch]:3d}" for i in idx))


if __name__ == "__main__":
    main()
