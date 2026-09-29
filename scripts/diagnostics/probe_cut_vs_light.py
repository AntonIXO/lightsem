"""Are the detected 'events' LIGHTING changes or CAMERA CUTS?

This decides what a positive event-prediction result would even mean. The
target is derived from stage VIDEO histograms, so a shot change (wide -> closeup
-> crowd) makes the HSV histogram jump exactly like a lighting change does.
Concert video is edited ON the music, so audio->event coupling could be
audio->editing with the lighting rig doing nothing.

The two are separable by their SIGNATURE, without any shot-boundary labels:

  CAMERA CUT   - instantaneous (one frame at 30 fps), and the histogram often
                 RETURNS toward its previous state when the edit cuts back to
                 a similar shot. Frame-scale, partly reversible.
  LIGHT CHANGE - the rig fades or snaps to a NEW state and STAYS there.
                 Persistent, and asymmetric in time.

Three measurements on the 30 fps ORIGINAL grid (not the 10 Hz decimated one --
decimation averages 3 frames and would smear a 1-frame cut into a ramp, which
is exactly the distinction being measured):

1. RISE TIME. Frames for the novelty to go 20% -> 80%. A cut is ~1 frame.
2. REVERSION. Compare the state 2 s after the event to the state 2 s before.
   cos similarity near 1 => it came back => cut. Near 0 => it stayed => lighting.
3. VALUE-CHANNEL ASYMMETRY. Lighting changes are dominated by brightness (V)
   and saturation; camera cuts move hue/composition too. Compare which channels
   of the 6-dim vector actually move at events.

No ground truth is available for shot boundaries, so this is diagnostic, not
proof. It bounds how much of the target can be lighting at all.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from scripts.diagnostics.probe_event_target import NOV_K, event_train, novelty

FPS = 30.0
TARGETS = ["hue_cos", "hue_sin", "sat", "val", "val_p90", "hue_conc"]


def load_raw(root="data/cache/rpmc_train", limit=40):
    """Load the 30 fps light track WITHOUT decimation."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*.npz")))[:limit]:
        d = np.load(p)
        y = d["y"].astype(np.float32)
        if len(y) >= 900 and np.isfinite(y).all():
            out.append((os.path.basename(p)[:-4], y))
    return out


def rise_time(nov, i, span=45):
    """Frames for novelty to climb 20%->80% of its local peak."""
    lo = max(0, i - span)
    seg = nov[lo:i + 1]
    if len(seg) < 4:
        return np.nan
    pk, fl = seg.max(), seg.min()
    if pk - fl < 1e-9:
        return np.nan
    t20 = fl + 0.2 * (pk - fl)
    t80 = fl + 0.8 * (pk - fl)
    a = np.flatnonzero(seg >= t20)
    b = np.flatnonzero(seg >= t80)
    return float(b[0] - a[0]) if len(a) and len(b) and b[0] >= a[0] else np.nan


def main():
    songs = load_raw()
    print(f"songs analysed at native 30 fps: {len(songs)}")
    rises, revs, chan = [], [], []

    for _, y in songs:
        z = (y - y.mean(0)) / (y.std(0) + 1e-6)
        nov = novelty(y)
        _, ev = event_train(y)
        for i in ev:
            if i < 90 or i > len(y) - 90:
                continue
            rises.append(rise_time(nov, i))
            pre = z[i - 75:i - 15].mean(0)     # 2.5s..0.5s before
            post = z[i + 15:i + 75].mean(0)    # 0.5s..2.5s after
            later = z[i + 75:i + 135].mean(0)  # 2.5s..4.5s after
            d = np.linalg.norm(post - pre) + 1e-9
            back = np.linalg.norm(later - pre) / d   # <1 => returned to pre-state
            revs.append(float(back))
            chan.append(np.abs(post - pre))

    rises = np.array([r for r in rises if np.isfinite(r)])
    revs = np.array(revs)
    chan = np.stack(chan)
    out = {}

    print(f"\nevents measured: {len(revs)}")
    print("\n=== 1. rise time (a camera cut is ~1 frame @30fps = 0.033 s) ===")
    print(f"  median {np.median(rises):.1f} frames = {np.median(rises)/FPS:.3f} s")
    print(f"  p10 {np.percentile(rises,10):.1f}  p90 {np.percentile(rises,90):.1f} frames")
    frac_instant = float((rises <= 2).mean())
    print(f"  fraction rising in <=2 frames (cut-like): {frac_instant*100:.1f}%")
    out["rise_median_frames"] = float(np.median(rises))
    out["frac_cut_like_rise"] = frac_instant

    print("\n=== 2. reversion: does the state come BACK 2.5-4.5 s later? ===")
    print("  ratio <0.5 => returned (cut-like) | >1.0 => stayed/moved on (lighting-like)")
    print(f"  median {np.median(revs):.3f}  p25 {np.percentile(revs,25):.3f} "
          f"p75 {np.percentile(revs,75):.3f}")
    frac_ret = float((revs < 0.5).mean())
    frac_stay = float((revs > 1.0).mean())
    print(f"  returned (<0.5): {frac_ret*100:.1f}%   stayed (>1.0): {frac_stay*100:.1f}%")
    out["reversion_median"] = float(np.median(revs))
    out["frac_returned"] = frac_ret
    out["frac_stayed"] = frac_stay

    print("\n=== 3. which channels move at events? ===")
    m = chan.mean(0)
    for n, v in sorted(zip(TARGETS, m), key=lambda t: -t[1]):
        print(f"  {n:>10s}  {v:.4f}")
    hue = m[TARGETS.index("hue_cos")] + m[TARGETS.index("hue_sin")]
    bright = m[TARGETS.index("val")] + m[TARGETS.index("val_p90")]
    print(f"\n  hue movement {hue:.4f}  vs brightness movement {bright:.4f}"
          f"   ratio {hue/max(bright,1e-9):.2f}")
    out["channel_movement"] = dict(zip(TARGETS, m.tolist()))
    out["hue_over_brightness"] = float(hue / max(bright, 1e-9))

    print("\n=== interpretation ===")
    if frac_instant > 0.5:
        print(f"  {frac_instant*100:.0f}% of events are frame-instant -> CUT-DOMINATED.")
        print("  A predictor of these events is predicting the EDIT, not the rig.")
    elif frac_ret > 0.5:
        print(f"  {frac_ret*100:.0f}% of events revert within 4.5 s -> CUT-DOMINATED.")
    else:
        print("  Events are mostly gradual and persistent -> consistent with")
        print("  lighting state changes, though shot labels would be needed to prove it.")

    json.dump(out, open("data/cache/cut_vs_light.json", "w"), indent=2)
    print("\nwrote data/cache/cut_vs_light.json")


if __name__ == "__main__":
    main()
