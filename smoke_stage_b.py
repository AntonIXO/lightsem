"""Stage B smoke: run features + structure + anticipation on REAL audio.

Reports the causal-vs-oracle gap and the rate-matched random baseline, so the
anticipation claim is priced rather than asserted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from lightsem.audio import CAUSAL, ORACLE, compute, evaluate_anticipation, extract, segment

WAV_DIR = Path("/root/lightsem/data/wav/wav")


def run_one(path: Path, duration_s: float | None = None) -> dict:
    t0 = time.time()
    feats = extract(path=str(path), duration_s=duration_s)
    t_feat = time.time() - t0

    t0 = time.time()
    segs = segment(feats)
    t_seg = time.time() - t0

    causal = compute(feats, segs, mode=CAUSAL, lookahead_s=2.0)
    oracle = compute(feats, segs, mode=ORACLE, lookahead_s=2.0)
    ev = evaluate_anticipation(feats, segs, lookahead_s=2.0)

    print(f"\n=== {path.name} ===")
    print(f"  duration {feats.duration_s:.1f}s  tempo {feats.tempo:.1f} BPM  "
          f"beats {feats.beats.size}  downbeats {feats.downbeats.size}")
    print(f"  frames {feats.n_frames}  feat {t_feat:.2f}s  seg {t_seg:.2f}s")
    print(f"  segments ({len(segs)}):")
    for s in segs:
        print(f"    {s.start_s:6.2f}-{s.end_s:6.2f}s  {s.label:10s} "
              f"e={s.energy:.3f} slope={s.energy_slope:+.3f} bright={s.brightness:.3f}")

    # Causal and oracle tension must NOT be identical - that would mean the
    # causal path is secretly reading the future.
    corr = float(np.corrcoef(causal.tension, oracle.tension)[0, 1])
    maxdiff = float(np.max(np.abs(causal.tension - oracle.tension)))
    print(f"  tension causal-vs-oracle: corr={corr:.3f} max|diff|={maxdiff:.3f}")
    assert maxdiff > 1e-6, "causal tension identical to oracle - lookahead leak!"

    print(f"  anticipation: n_drops={ev['n_drops']} n_cues={ev['n_cues']} "
          f"recall={ev['recall']:.3f} precision={ev['precision']:.3f} "
          f"mean_lead={ev['mean_lead_s']:.2f}s")
    print(f"  random baseline recall: {ev['baseline_recall_mean']:.3f} "
          f"+- {ev['baseline_recall_std']:.3f}")
    return {"track": path.name, "tempo": feats.tempo, "n_segments": len(segs),
            "labels": [s.label for s in segs], "tension_corr": corr, **ev}


def main() -> int:
    wavs = sorted(WAV_DIR.glob("*.wav"))
    if not wavs:
        print("FAIL: no wav files")
        return 1
    # A spread across genres (mBR=break, mHO=house, mLH=LA hip-hop, ...).
    picks = [wavs[0], wavs[6], wavs[12], wavs[18], wavs[24]]
    results = [run_one(p, duration_s=40.0) for p in picks]

    out = Path("/root/lightsem/data/cache/stage_b_report.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nreport: {out}")

    n_with_drops = sum(r["n_drops"] > 0 for r in results)
    print(f"tracks with >=1 detected drop: {n_with_drops}/{len(results)}")
    print("\nSTAGE B SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
