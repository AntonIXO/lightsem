"""Is the weak Stage B result the DETECTOR's fault or the CORPUS's fault?

Stage B on AIST audio gave anticipation recall 0.000 on 3/5 tracks, and on
the one track that scored 0.800 the rate-matched random baseline was 0.625 --
i.e. essentially no skill. Two competing explanations:

  H1 (detector broken): the segmenter/labeller cannot find a drop even when
     one is unambiguously present.
  H2 (wrong corpus):    AIST dance tracks are short, loudness-normalised
     loops with near-constant energy and no EDM build-up/drop structure, so
     there is nothing to detect.

These are separable with a CONTROL SIGNAL: synthesise EDM-shaped audio with
drops at KNOWN times and run the identical pipeline. If the detector finds
them, H1 is refuted and H2 stands.

Committed prediction BEFORE running: the detector will localise synthetic
drops to within ~2 s, and causal anticipation will beat its rate-matched
random baseline on the synthetic track -- while AIST energy dynamic range
stays far below the synthetic one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from lightsem.audio import CAUSAL, compute, evaluate_anticipation, extract, segment
from lightsem import SR

OUT = Path("/root/lightsem/data/cache")
OUT.mkdir(parents=True, exist_ok=True)


def synth_edm(
    duration_s: float = 48.0,
    bpm: float = 128.0,
    sr: int = SR,
    drops: tuple[float, ...] = (16.0, 36.0),
    build_s: float = 8.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Synthesise EDM-shaped audio: breakdown -> build-up -> drop, twice.

    Deliberately crude but structurally honest: kick on the beat, a riser
    whose centre frequency and amplitude climb through the build, a wide
    noisy drop section, and a quiet breakdown between.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float64)
    beat_s = 60.0 / bpm

    def in_drop(x: float) -> bool:
        return any(d <= x < d + 12.0 for d in drops)

    def in_build(x: float) -> bool:
        return any(d - build_s <= x < d for d in drops)

    # Kick drum on every beat; harder inside drops.
    k = 0.0
    while k < duration_s:
        i = int(k * sr)
        env_len = int(0.18 * sr)
        idx = np.arange(env_len)
        env = np.exp(-idx / (0.045 * sr))
        freq = 55.0 * np.exp(-idx / (0.02 * sr)) + 45.0
        amp = 1.0 if in_drop(k) else (0.55 if in_build(k) else 0.35)
        seg = amp * env * np.sin(2 * np.pi * np.cumsum(freq) / sr)
        end = min(i + env_len, n)
        y[i:end] += seg[: end - i]
        k += beat_s

    # Sustained musical bed.
    bed = 0.10 * np.sin(2 * np.pi * 110.0 * t) + 0.06 * np.sin(2 * np.pi * 220.0 * t)

    # Section gains + risers.
    gain = np.full(n, 0.25)
    for d in drops:
        b0, b1 = int((d - build_s) * sr), int(d * sr)
        b1 = min(b1, n)
        if b1 > b0:
            ramp = np.linspace(0.25, 0.9, b1 - b0)
            gain[b0:b1] = ramp
            # Riser: white noise through a rising high-pass, amplitude ramp.
            noise = rng.standard_normal(b1 - b0)
            sweep = np.linspace(0.0, 1.0, b1 - b0) ** 2
            y[b0:b1] += 0.35 * sweep * noise * np.linspace(0.2, 1.0, b1 - b0)
        d0, d1 = int(d * sr), min(int((d + 12.0) * sr), n)
        gain[d0:d1] = 1.0
        # Drop: bright saw stack + noise energy.
        tt = t[d0:d1]
        y[d0:d1] += 0.28 * np.sign(np.sin(2 * np.pi * 440.0 * tt))
        y[d0:d1] += 0.16 * np.sign(np.sin(2 * np.pi * 660.0 * tt))
        y[d0:d1] += 0.10 * rng.standard_normal(d1 - d0)

    y = (y + bed) * gain
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32), list(drops)


def energy_dynamics(path_or_y, duration_s=40.0) -> dict:
    if isinstance(path_or_y, (str, Path)):
        f = extract(path=str(path_or_y), duration_s=duration_s)
    else:
        f = extract(y=path_or_y, duration_s=duration_s)
    from lightsem.audio.features import _norm

    rms = f.rms
    return {
        "rms_min": float(rms.min()),
        "rms_max": float(rms.max()),
        "rms_ratio_max_min": float(rms.max() / (rms.min() + 1e-9)),
        "rms_std_over_mean": float(rms.std() / (rms.mean() + 1e-9)),
        "rms_p90_over_p10": float(np.quantile(rms, 0.9) / (np.quantile(rms, 0.1) + 1e-9)),
    }


def main() -> None:
    print("=" * 68)
    print("CONTROL: synthetic EDM with drops at KNOWN times")
    print("=" * 68)
    y, true_drops = synth_edm()
    wav = OUT / "synth_edm.wav"
    sf.write(wav, y, SR)
    print(f"wrote {wav}  ({len(y)/SR:.1f}s)  true drops at {true_drops}")

    feats = extract(y=y)
    segs = segment(feats)
    ev = evaluate_anticipation(feats, segs, lookahead_s=2.0)
    print(f"\ntempo {feats.tempo:.1f} BPM  segments {len(segs)}")
    for s in segs:
        print(f"  {s.start_s:6.2f}-{s.end_s:6.2f}s  {s.label:10s} "
              f"e={s.energy:.3f} slope={s.energy_slope:+.3f}")

    detected = [s.start_s for s in segs if s.label == "drop"]
    print(f"\ntrue drops     : {true_drops}")
    print(f"detected drops : {[round(d,2) for d in detected]}")
    errs = []
    for d in true_drops:
        if detected:
            errs.append(min(abs(d - x) for x in detected))
    print(f"localisation error (s): {[round(e,2) for e in errs]}")

    print(f"\nanticipation: recall={ev['recall']:.3f} precision={ev['precision']:.3f} "
          f"mean_lead={ev['mean_lead_s']:.2f}s  n_cues={ev['n_cues']}")
    print(f"random baseline recall: {ev['baseline_recall_mean']:.3f} "
          f"+- {ev['baseline_recall_std']:.3f}")

    print("\n" + "=" * 68)
    print("ENERGY DYNAMIC RANGE: synthetic vs AIST")
    print("=" * 68)
    syn = energy_dynamics(y)
    print(f"synthetic : p90/p10={syn['rms_p90_over_p10']:8.2f}  "
          f"std/mean={syn['rms_std_over_mean']:.3f}")
    wavs = sorted(Path("/root/lightsem/data/wav/wav").glob("*.wav"))
    ratios = []
    for p in wavs[:10]:
        d = energy_dynamics(p)
        ratios.append(d["rms_p90_over_p10"])
        print(f"{p.name:10s}: p90/p10={d['rms_p90_over_p10']:8.2f}  "
              f"std/mean={d['rms_std_over_mean']:.3f}")
    print(f"\nAIST median p90/p10 = {np.median(ratios):.2f}   "
          f"synthetic = {syn['rms_p90_over_p10']:.2f}")

    (OUT / "control_probe.json").write_text(json.dumps({
        "true_drops": true_drops, "detected_drops": detected,
        "localisation_err_s": errs, "anticipation": ev,
        "synthetic_dynamics": syn,
        "aist_p90_p10_median": float(np.median(ratios)),
    }, indent=2, default=float))
    print(f"\nsaved {OUT/'control_probe.json'}")


if __name__ == "__main__":
    main()
