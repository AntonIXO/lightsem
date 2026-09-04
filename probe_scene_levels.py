"""Why is 'drop' not brighter than 'breakdown'?

Mean over all 512 channels mixes dimmers with pan/tilt/colour, so it is a bad
brightness proxy. This probe isolates the DIMMER channels only, and also
samples the whole segment rather than a single mid-frame, to see whether the
scene mapping is actually wrong or the metric was.
"""

from __future__ import annotations

import numpy as np

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.dmx import club_rig, render_stream
from probe_control_synthetic import synth_edm


def main() -> None:
    y, drops = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    antic = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)
    rig = club_rig()
    stream = render_stream(rig, feats, segs, antic, fps=40.0)

    # Channel groups.
    mh = rig.by_kind("moving_head")
    wash = rig.by_kind("wash")
    blind = rig.by_kind("blinder")
    dim_ch = [f.address - 1 + 4 for f in mh] + [f.address - 1 for f in wash]
    blind_ch = [f.address - 1 for f in blind]
    pan_ch = [f.address - 1 for f in mh]

    print(f"dimmer channels: {dim_ch}")
    print(f"blinder channels: {blind_ch}\n")

    print(f"{'label':11s} {'dur':>6s} {'ALL512':>8s} {'DIMMER':>8s} "
          f"{'BLINDER':>8s} {'PAN':>8s}")
    for s in segs:
        m = (stream.times >= s.start_s) & (stream.times < s.end_s)
        if not m.any():
            continue
        fr = stream.frames[m].astype(float)
        print(f"{s.label:11s} {s.duration_s:6.2f} {fr.mean():8.2f} "
              f"{fr[:, dim_ch].mean():8.2f} {fr[:, blind_ch].mean():8.2f} "
              f"{fr[:, pan_ch].mean():8.2f}")

    print("\nBY LABEL (dimmer channels only, duration-weighted):")
    agg = {}
    for s in segs:
        m = (stream.times >= s.start_s) & (stream.times < s.end_s)
        if not m.any():
            continue
        v = stream.frames[m][:, dim_ch].astype(float)
        agg.setdefault(s.label, []).append(v)
    for lab, chunks in sorted(agg.items()):
        allv = np.concatenate(chunks, axis=0)
        print(f"  {lab:11s} mean={allv.mean():7.2f}  p95={np.percentile(allv,95):7.2f} "
              f"max={allv.max():6.0f}")


if __name__ == "__main__":
    main()
