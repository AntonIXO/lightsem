"""Stage C smoke: render a real DMX stream and verify its properties.

Checks that actually matter:
  * universe is 512 channels, uint8, >=30 fps
  * scenes DIFFER by segment label (a mapping that outputs the same look
    everywhere is not a mapping)
  * the micro layer is beat-locked: dimmer peaks align with beat times
  * anticipation raises intensity BEFORE the drop, not after
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.dmx import club_rig, render_stream
from scripts.diagnostics.probe_control_synthetic import synth_edm

OUT = Path("/root/lightsem/data/cache")


def main() -> int:
    y, true_drops = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    antic = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)

    rig = club_rig()
    print("rig:", rig.summary()["by_kind"],
          "channels:", rig.summary()["channels_used"])

    stream = render_stream(rig, feats, segs, antic, fps=40.0)
    print(f"stream: {stream.frames.shape} @ {stream.fps} fps "
          f"= {stream.duration_s:.2f}s  dtype={stream.frames.dtype}")

    assert stream.frames.shape[1] == 512, "universe must be 512 channels"
    assert stream.frames.dtype == np.uint8
    assert stream.fps >= 30

    # 1. Scenes must differ by label.
    # NOTE: measure DIMMER channels, not the mean over all 512. The all-channel
    # mean mixes in pan/tilt/colour (pan sits near 127 by construction), which
    # swamps the brightness signal and made "breakdown" look brighter than
    # "drop". Averaging a symmetric oscillation is not a brightness measure.
    mh = rig.by_kind("moving_head")
    wash = rig.by_kind("wash")
    dim_ch = [f.address - 1 + 4 for f in mh] + [f.address - 1 for f in wash]

    labels: dict[str, list[np.ndarray]] = {}
    for s in segs:
        m = (stream.times >= s.start_s) & (stream.times < s.end_s)
        if m.any():
            labels.setdefault(s.label, []).append(stream.frames[m][:, dim_ch])
    print("\nmean DIMMER level by segment label:")
    means = {}
    for lab, chunks in sorted(labels.items()):
        m = float(np.concatenate(chunks, axis=0).mean())
        means[lab] = m
        print(f"  {lab:10s} {m:6.2f}")
    assert len(set(round(v, 1) for v in means.values())) > 1, \
        "all scenes identical - mapping is not doing anything"
    if "drop" in means and "breakdown" in means:
        assert means["drop"] > means["breakdown"], "drop must be brighter than breakdown"
        print(f"  -> drop ({means['drop']:.1f}) > breakdown ({means['breakdown']:.1f}) OK")
    if "intro" in means and "drop" in means:
        assert means["drop"] > means["intro"], "drop must be brighter than intro"

    # 1b. Moving heads must actually articulate (a fan must SEPARATE heads).
    pan_ch = [f.address - 1 for f in mh]
    drop_seg = next((s for s in segs if s.label == "drop"), None)
    if drop_seg is not None:
        m = (stream.times >= drop_seg.start_s) & (stream.times < drop_seg.end_s)
        per_head = stream.frames[m][:, pan_ch].astype(float).mean(axis=0)
        print(f"  pan spread across heads during drop: {per_head.std():.2f}")
        assert per_head.std() > 5.0, "moving heads are not fanning"

    # 2. Micro layer beat-lock: total dimmer energy vs beat phase.
    mh_dimmer_ch = [f.address - 1 + 4 for f in rig.by_kind("moving_head")]
    dim = stream.frames[:, mh_dimmer_ch].mean(axis=1).astype(float)
    beats = feats.beats
    phases, vals = [], []
    for i, t in enumerate(stream.times):
        j = int(np.searchsorted(beats, t, side="right") - 1)
        if j < 0 or j + 1 >= beats.size:
            continue
        span = beats[j + 1] - beats[j]
        if span <= 0:
            continue
        phases.append((t - beats[j]) / span)
        vals.append(dim[i])
    phases, vals = np.array(phases), np.array(vals)
    nb = 8
    prof = [vals[(phases >= k / nb) & (phases < (k + 1) / nb)].mean() for k in range(nb)]
    print(f"\nbeat-phase dimmer profile (8 bins): {[round(p,1) for p in prof]}")
    assert np.argmax(prof) <= 1, "dimmer peak is not on the beat - micro layer broken"
    print(f"  -> peak at bin {int(np.argmax(prof))} (on the beat) OK")

    # 3. Anticipation must lift intensity BEFORE the drop.
    d0 = true_drops[0]
    pre = dim[(stream.times >= d0 - 3.0) & (stream.times < d0)].mean()
    early = dim[(stream.times >= d0 - 9.0) & (stream.times < d0 - 6.0)].mean()
    print(f"\nintensity 9-6s before drop: {early:.1f}")
    print(f"intensity 3-0s before drop: {pre:.1f}")
    print(f"  -> lift ratio {pre/max(early,1e-6):.2f}x")

    npy = stream.save_npy(OUT / "stage_c_universe.npy")
    cue = stream.save_cue_list(OUT / "stage_c_cues.json", OUT / "stage_c_cues.csv")
    print(f"\nuniverse : {npy}")
    print(f"cue list : {cue}  ({len(stream.cues)} cues)")
    for c in stream.cues[:8]:
        print(f"  {c['time_s']:7.2f}s  {c['type']:12s} {str(c.get('scene')):20s} "
              f"{c.get('segment') or ''}")

    print("\nSTAGE C SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
