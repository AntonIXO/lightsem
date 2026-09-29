"""Why does anticipation recall stay 0.000 when drops are now localised <1s?

Drop localisation is fixed (0.44/0.57 s on the synthetic control), and 7 cues
fire, yet recall is 0.000. So cues exist and drops exist but never pair. This
probe dumps the actual cue onset times against the true drop times to find
where the pairing breaks.
"""

from __future__ import annotations

import numpy as np

from lightsem.audio import CAUSAL, compute, extract, segment
from lightsem.audio.anticipation import causal_time_to_drop, tension_curve
from scripts.diagnostics.probe_control_synthetic import synth_edm


def main() -> None:
    y, true_drops = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    detected = [s.start_s for s in segs if s.label == "drop"]

    for lookahead in (2.0, 4.0):
        c = compute(feats, segs, mode=CAUSAL, lookahead_s=lookahead)
        gate = c.cue_gate.astype(int)
        onsets = feats.frame_times[
            np.flatnonzero(np.diff(np.concatenate(([0], gate))) == 1)
        ]
        print(f"\n--- lookahead={lookahead}s ---")
        print(f"cue onsets   : {[round(float(o),2) for o in onsets]}")
        print(f"true drops   : {true_drops}")
        print(f"detected drop: {[round(d,2) for d in detected]}")
        for d in true_drops:
            leads = [d - o for o in onsets]
            close = [round(float(l), 2) for l in leads if -3 <= l <= 12]
            print(f"  drop {d}: leads (positive = cue BEFORE drop) {close}")

    # Inspect the tension/ttd trace around the first build-up.
    tension, slope = tension_curve(feats, mode=CAUSAL)
    ttd = causal_time_to_drop(tension, slope, feats.frame_times, threshold=0.75)
    ft = feats.frame_times
    m = (ft >= 8.0) & (ft <= 18.0)
    print("\n  t      tension  slope    ttd")
    for t, te, sl, td in list(zip(ft[m], tension[m], slope[m], ttd[m]))[::20]:
        print(f"  {t:6.2f}  {te:6.3f}  {sl:+.4f}  {td if np.isfinite(td) else float('inf'):8.2f}")


if __name__ == "__main__":
    main()
