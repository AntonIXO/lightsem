"""Price the anticipation claim honestly against strong baselines.

The rate-matched random baseline hit 0.816 recall on the synthetic control
while the real detector hit 0.875 -- a margin so small the headline number is
nearly meaningless. Cause: with 7 cues over 48 s and a 12 s credit window,
random cues cover most of the timeline, so recall saturates for everyone.

Recall alone is therefore the WRONG metric. This probe scores what actually
matters for anticipatory control:

  1. TIMING PRECISION, not coverage. For each true drop, the lead of the
     nearest preceding cue. A useful system concentrates cues just before
     drops; a random one scatters them.
  2. CUE EFFICIENCY. Fraction of emitted cues that precede a drop within the
     window (precision), vs the random rate.
  3. A HARDER BASELINE than uniform-random: an "every-N-seconds metronome"
     cue process at the same rate, which is what a naive timer would do.
  4. NULL TRACK CONTROL. Steady-state audio with NO drops -- a good detector
     should emit few cues; a trigger-happy one exposes itself here.

Committed prediction: precision and lead-time concentration will beat both
baselines, while raw recall will NOT separate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from lightsem import SR
from lightsem.audio import CAUSAL, compute, extract, segment
from scripts.diagnostics.probe_control_synthetic import synth_edm

OUT = Path("/root/lightsem/data/cache")
MAX_LEAD = 12.0


def cue_onsets(feats, segs, lookahead_s=2.0):
    c = compute(feats, segs, mode=CAUSAL, lookahead_s=lookahead_s)
    gate = c.cue_gate.astype(int)
    idx = np.flatnonzero(np.diff(np.concatenate(([0], gate))) == 1)
    return feats.frame_times[idx]


def score(onsets, drops, duration, max_lead=MAX_LEAD):
    """Precision + lead concentration for a cue set."""
    if len(onsets) == 0:
        return {"n_cues": 0, "precision": float("nan"),
                "mean_abs_lead": float("nan"), "recall": 0.0}
    hits = [any(0 <= d - c <= max_lead for d in drops) for c in onsets]
    # For each drop, lead of the NEAREST preceding cue (tight timing).
    nearest = []
    for d in drops:
        pre = [d - c for c in onsets if 0 <= d - c <= max_lead]
        if pre:
            nearest.append(min(pre))
    return {
        "n_cues": int(len(onsets)),
        "precision": float(np.mean(hits)),
        "recall": float(len(nearest) / len(drops)) if drops else float("nan"),
        "mean_nearest_lead_s": float(np.mean(nearest)) if nearest else float("nan"),
        "std_nearest_lead_s": float(np.std(nearest)) if nearest else float("nan"),
    }


def metronome(rate_hz, duration, phase=0.0):
    step = 1.0 / max(rate_hz, 1e-9)
    return np.arange(phase, duration, step)


def main() -> None:
    results = {}

    # ---- 1. Synthetic EDM with known drops -------------------------------
    y, drops = synth_edm()
    feats = extract(y=y)
    segs = segment(feats)
    onsets = cue_onsets(feats, segs)
    dur = feats.duration_s
    rate = len(onsets) / dur

    real = score(onsets, drops, dur)
    print("=" * 68)
    print("SYNTHETIC EDM  (true drops at 16.0, 36.0)")
    print("=" * 68)
    print(f"  cue onsets: {[round(float(o),2) for o in onsets]}")
    print(f"  REAL      : {real}")

    rng = np.random.default_rng(0)
    rand_scores = []
    for _ in range(500):
        k = rng.poisson(rate * dur)
        fake = np.sort(rng.uniform(0, dur, size=k))
        rand_scores.append(score(fake, drops, dur))
    rand_prec = np.array([s["precision"] for s in rand_scores if np.isfinite(s["precision"])])
    rand_lead = np.array([s["mean_nearest_lead_s"] for s in rand_scores
                          if np.isfinite(s.get("mean_nearest_lead_s", np.nan))])
    print(f"  RANDOM    : precision {rand_prec.mean():.3f} +- {rand_prec.std():.3f}")
    print(f"              nearest-lead {rand_lead.mean():.2f}s +- {rand_lead.std():.2f}s")

    met_scores = [score(metronome(rate, dur, p), drops, dur)
                  for p in np.linspace(0, 1 / max(rate, 1e-9), 20)]
    met_prec = np.array([s["precision"] for s in met_scores if np.isfinite(s["precision"])])
    met_lead = np.array([s["mean_nearest_lead_s"] for s in met_scores
                         if np.isfinite(s.get("mean_nearest_lead_s", np.nan))])
    print(f"  METRONOME : precision {met_prec.mean():.3f} +- {met_prec.std():.3f}")
    print(f"              nearest-lead {met_lead.mean():.2f}s +- {met_lead.std():.2f}s")

    z_prec = (real["precision"] - rand_prec.mean()) / (rand_prec.std() + 1e-9)
    print(f"\n  precision z vs random : {z_prec:+.2f}")
    print(f"  real nearest-lead {real['mean_nearest_lead_s']:.2f}s "
          f"vs random {rand_lead.mean():.2f}s "
          f"(lower = tighter timing)")

    results["synthetic"] = {
        "real": real,
        "random_precision_mean": float(rand_prec.mean()),
        "random_precision_std": float(rand_prec.std()),
        "random_nearest_lead_mean": float(rand_lead.mean()),
        "metronome_precision_mean": float(met_prec.mean()),
        "metronome_nearest_lead_mean": float(met_lead.mean()),
        "precision_z_vs_random": float(z_prec),
    }

    # ---- 2. NULL TRACK: steady state, no drops ---------------------------
    print("\n" + "=" * 68)
    print("NULL CONTROL  (steady-state audio, NO drops)")
    print("=" * 68)
    n = int(48.0 * SR)
    t = np.arange(n) / SR
    rng2 = np.random.default_rng(1)
    null = (0.3 * np.sin(2 * np.pi * 110 * t)
            + 0.2 * np.sin(2 * np.pi * 220 * t)
            + 0.05 * rng2.standard_normal(n))
    beat = 60.0 / 128.0
    k = 0.0
    while k < 48.0:
        i = int(k * SR)
        L = int(0.15 * SR)
        idx = np.arange(L)
        env = np.exp(-idx / (0.04 * SR))
        null[i:min(i + L, n)] += (0.5 * env * np.sin(
            2 * np.pi * np.cumsum(55 * np.exp(-idx / (0.02 * SR)) + 45) / SR))[
            : min(i + L, n) - i]
        k += beat
    null = (0.9 * null / (np.max(np.abs(null)) + 1e-9)).astype(np.float32)
    sf.write(OUT / "null_steady.wav", null, SR)

    nf = extract(y=null)
    ns = segment(nf)
    n_onsets = cue_onsets(nf, ns)
    n_drops = [s.start_s for s in ns if s.label == "drop"]
    print(f"  labels        : {[s.label for s in ns]}")
    print(f"  'drops' found : {len(n_drops)}")
    print(f"  cues emitted  : {len(n_onsets)}  "
          f"(rate {len(n_onsets)/nf.duration_s:.3f} Hz)")
    print(f"  synthetic cue rate for comparison: {rate:.3f} Hz")
    results["null_track"] = {
        "n_cues": int(len(n_onsets)),
        "cue_rate_hz": float(len(n_onsets) / nf.duration_s),
        "n_labelled_drops": int(len(n_drops)),
        "synthetic_cue_rate_hz": float(rate),
    }

    (OUT / "anticipation_baselines.json").write_text(
        json.dumps(results, indent=2, default=float))
    print(f"\nsaved {OUT/'anticipation_baselines.json'}")


if __name__ == "__main__":
    main()
