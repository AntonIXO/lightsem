"""Stage driver: reconstruct the Oslo DJ mixes on the record's clock.

    bash run.sh run_oslo_audio.py selftest   # prove the fitter, no audio needed
    bash run.sh run_oslo_audio.py resolve    # which master to obtain
    bash run.sh run_oslo_audio.py ingest     # what is in audio/sources
    bash run.sh run_oslo_audio.py align      # fit ratio + offset per track
    bash run.sh run_oslo_audio.py build      # render both mixes to wav
    bash run.sh run_oslo_audio.py verify     # score against published features

`align`, `build` and `verify` need the four recordings in
/mnt/lightsem-data/oslo/audio/sources. Obtain them yourself; the record does not
deposit them and this driver does not fetch them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from lightsem.data import oslo_audio as oa

OUT = Path("/root/lightsem/outputs/oslo_audio")


def _save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    print(f"  -> outputs/oslo_audio/{name}")


# --------------------------------------------------------------------- stages

def cmd_resolve() -> None:
    res = oa.resolve_stimulus()
    for r in res:
        print(f"\n{r['title']}  --  {r.get('artist')} ({r.get('label')}, {r.get('year')})"
              f"  want: {r.get('wanted_version') or 'any'}")
        if "error" in r:
            print(f"  metadata lookup failed: {r['error']}")
            continue
        for c in r["candidates"]:
            m, s = divmod(c["duration_s"], 60)
            print(f"   {c['title'][:44]:44s} {c['artist'][:18]:18s} "
                  f"{m:>2d}:{s:02d}  isrc={c['isrc']}")
    _save("stimulus_resolved.json", res)
    print("\nBuy the master that matches the version and duration, drop the files in")
    print(f"{oa.SOURCES}, then run: bash run.sh run_oslo_audio.py align")


def cmd_ingest() -> None:
    found = oa.ingest_dir()
    missing = [t for t in oa.STIMULUS if t not in found]
    for t in oa.STIMULUS:
        p = found.get(t)
        if p is None:
            print(f"  MISSING   {t}")
        else:
            y = oa.read_audio(p)
            print(f"  ok        {t:12s} {p.name}  {len(y)/oa.SR:7.1f}s")
    for p in getattr(oa.ingest_dir, "unmatched", []):
        print(f"  unmatched {p.name}  (rename it to contain the track title)")
    _save("ingest.json", {"found": {k: str(v) for k, v in found.items()},
                          "missing": missing})
    return missing


def cmd_align() -> None:
    found = oa.ingest_dir()
    missing = [t for t in oa.STIMULUS if t not in found]
    if missing:
        print(f"cannot align, missing: {', '.join(missing)}")
        sys.exit(1)
    rows = oa.load_stimulus_tracks()
    allp = []
    for mix in (1, 2):
        ref = oa.load_feature_series(mix)["onset_strength"]
        print(f"\nmix {mix}")
        for r in [q for q in rows if q["dj_mix"] == mix]:
            p = oa.place_track(r, found[r["title"]], ref)
            allp.append(p)
            flag = "ok " if p.margin > 0.10 else "WEAK"
            print(f"  {flag} {p.title:12s} ratio {p.ratio:.4f}  "
                  f"offset {p.offset_s:8.2f}s  r {p.corr:.3f}  "
                  f"null {p.null_corr:.3f}  margin {p.margin:+.3f}")
    _save("placements.json", oa.placement_table(allp))


def cmd_build() -> None:
    pf = OUT / "placements.json"
    if not pf.exists():
        print("run align first")
        sys.exit(1)
    rows = json.loads(pf.read_text())
    for mix in (1, 2):
        ps = [oa.Placement(**r) for r in rows if r["mix"] == mix]
        y = oa.render_mix(mix, ps)
        dst = oa.MIXES / f"{oa.MIX_NAME[mix]}.wav"
        oa.write_wav(dst, y)
        print(f"  {dst}  {len(y)/oa.SR:.1f}s  "
              f"(record says {oa.mix_duration(mix):.3f}s)")


def cmd_verify() -> None:
    report = {}
    for mix in (1, 2):
        src = oa.MIXES / f"{oa.MIX_NAME[mix]}.wav"
        if not src.exists():
            print(f"  {src.name} not built")
            continue
        y = oa.read_audio(src)
        v = oa.verify_mix(mix, y)
        v["per_track"] = oa.per_track_verify(mix, y)
        report[oa.MIX_NAME[mix]] = v
        print(f"\n{oa.MIX_NAME[mix]}  gate {'PASS' if v['gate_pass'] else 'FAIL'}")
        print(f"  {'column':26s} {'r':>7s} {'null':>7s} {'margin':>8s}")
        for c, d in v["columns"].items():
            print(f"  {c:26s} {d['r']:7.3f} {d['r_shift_null']:7.3f} {d['margin']:+8.3f}")
        print("  per track (onset r):  " + "  ".join(
            f"{t['title']}={t['onset_strength']:.2f}" for t in v["per_track"]))
    _save("verify.json", report)


# ------------------------------------------------------------------- selftest

def cmd_selftest() -> None:
    """Fit known ratios and offsets on synthetic audio.

    The alignment is the part that can fail silently -- a wrong ratio still
    renders a plausible-sounding mix, and only drifts out of the mocap clock a
    minute later. So it is checked here against ground truth built on purpose,
    with no recording involved, which also means this runs before any audio has
    been obtained.
    """
    rng = np.random.default_rng(0)
    sr = oa.SR
    ok = True

    def click_track(dur_s, bpm, seed):
        """A click train with an arrangement, because a bare one is ambiguous.

        Sections that enter and drop out, and a breakdown with no kick. Without
        that structure every bar of the signal is interchangeable and no fitter
        can be expected to find the right one -- the test would be unfair rather
        than hard.
        """
        r = np.random.default_rng(seed)
        n = int(dur_s * sr)
        y = r.normal(0, 0.02, n)
        step = int(sr * 60.0 / bpm)
        env = np.exp(-np.linspace(0, 12, int(0.09 * sr)))
        hat = np.exp(-np.linspace(0, 40, int(0.03 * sr)))
        secs = r.uniform(0.25, 1.0, 8)                  # per-eighth section gain
        drop = int(r.integers(2, 6))                    # one section loses the kick
        for k, i in enumerate(range(0, n - len(env), step)):
            g = secs[min(int(i / n * 8), 7)]
            if int(i / n * 8) != drop:
                y[i:i + len(env)] += env * g * r.uniform(0.6, 1.0)
            if k % 2:
                y[i:i + len(hat)] += hat * g * 0.3 * r.uniform(0.5, 1.0)
        for i in range(0, n - len(env), step * 4):
            if int(i / n * 8) != drop:
                y[i:i + len(env)] += env * 1.4
        return y

    print("fitting known (ratio, offset) pairs on synthetic clicks")
    for truth_ratio, truth_offset in ((1.0, 12.0), (1.0407, 40.0), (0.9603, 75.0)):
        native = click_track(90, 124, seed=int(truth_ratio * 1000))
        played = oa._resample_audio(native, truth_ratio)
        mixed = np.concatenate([
            rng.normal(0, 0.02, int(truth_offset * sr)),
            played,
            rng.normal(0, 0.02, int(20 * sr))])
        ref = oa.onset_envelope(mixed)
        cand = oa.onset_envelope(native)
        r, lag, corr = oa.fit_placement(ref, cand)
        off = lag / oa.FPS
        d_ratio = abs(r - truth_ratio)
        d_off = abs(off - truth_offset)
        good = d_ratio < 0.004 and d_off < 0.20
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} truth ratio {truth_ratio:.4f} off "
              f"{truth_offset:5.1f}s -> fitted {r:.4f} / {off:6.2f}s  "
              f"(dr {d_ratio:.4f}, dt {d_off*1000:5.1f} ms, r={corr:.3f})")

    print("\nnull separation: a different track at the same tempo must score lower")
    native = click_track(90, 124, seed=1)
    other = click_track(90, 124, seed=99)
    mixed = np.concatenate([rng.normal(0, 0.02, int(30 * sr)), native])
    ref = oa.onset_envelope(mixed)
    r_true, lag, c_true = oa.fit_placement(ref, oa.onset_envelope(native))
    _, _, c_other = oa.fit_placement(ref, oa.onset_envelope(other))
    sep = c_true - c_other
    good = sep > 0.10
    ok &= good
    print(f"  {'ok ' if good else 'FAIL'} matching {c_true:.3f} vs non-matching "
          f"{c_other:.3f}  separation {sep:+.3f}")

    print("\nfeature recompute: our columns must reproduce on identical audio")
    y = click_track(30, 126, seed=7)
    f1, f2 = oa.frame_features(y), oa.frame_features(y * 0.5)
    for c in ("spectral_centroid_hz", "bass_frac_below150hz"):
        d = float(np.nanmax(np.abs(f1[c] - f2[c])))
        good = d < 1e-6 * max(1.0, float(np.nanmax(np.abs(f1[c]))))
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} {c} gain-invariant, max delta {d:.3e}")

    print(f"\nselftest {'PASS' if ok else 'FAIL'}")
    _save("selftest.json", {"pass": bool(ok)})
    sys.exit(0 if ok else 1)


CMDS = {"resolve": cmd_resolve, "ingest": cmd_ingest, "align": cmd_align,
        "build": cmd_build, "verify": cmd_verify, "selftest": cmd_selftest}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        sys.exit(1)
    CMDS[sys.argv[1]]()
