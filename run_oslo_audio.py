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
        ref = oa.ref_fingerprint(mix)
        print(f"\nmix {mix}")
        for r in [q for q in rows if q["dj_mix"] == mix]:
            p = oa.place_track(r, found[r["title"]], ref)
            allp.append(p)
            flag = "ok " if p.margin > 0.10 else "WEAK"
            print(f"  {flag} {p.title:12s} {p.native_bpm:6.1f}->{p.played_bpm:5.1f} BPM "
                  f"ratio {p.ratio:.4f} (hint {p.ratio_hint:.4f})  "
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
    """Localise real recordings inside a mix we built, where truth is known.

    The alignment is the part that can fail silently -- a wrong ratio still
    renders a plausible-sounding mix and only drifts out of the mocap clock a
    minute later -- so it is checked against ground truth constructed on
    purpose. The construction uses the corpus's own recordings rather than
    synthesised drums, which is a correction: a bare click pattern carries
    almost nothing in the brightness, flatness and bass-fraction channels the
    matcher scores on, so it demanded a discrimination the real task never asks
    for and failed by a beat where real audio resolves to milliseconds.

    What this does not test is the thing the corpus actually asks. The
    reference here is built by our own feature code from the same audio, where
    the record's reference is a DJ mix that was EQ'd and cross-faded. That gap
    is what `align`'s margin column reports, and it is why this passing does
    not by itself mean a placement against the record is right.
    """
    rng = np.random.default_rng(0)
    sr = oa.SR
    ok = True
    found = oa.ingest_dir()
    if not found:
        print(f"no recordings in {oa.SOURCES}; localisation checks skipped")
    else:
        print("localising real recordings in a mix with known ratio and offset")
        TR, TO = 1.0135, 137.0
        prints = {}
        for title, path in sorted(found.items()):
            y = oa.read_audio(path)
            mixed = np.concatenate([rng.normal(0, 0.005, int(TO * sr)),
                                    oa._resample_audio(y, TR),
                                    rng.normal(0, 0.005, int(60 * sr))])
            ref, cand = oa.fingerprint(mixed), oa.fingerprint(y)
            prints[title] = (ref, cand)
            r, lag, corr = oa.fit_placement(ref, cand)
            off, d_r, d_t = lag / oa.FPS, abs(r - TR), abs(lag / oa.FPS - TO)
            good = d_r < 0.004 and d_t < 0.20
            ok &= good
            print(f"  {'ok ' if good else 'FAIL'} {title:12s} ratio {r:.4f} "
                  f"offset {off:7.2f}s  (dr {d_r:.4f}, dt {d_t*1000:6.1f} ms, "
                  f"r={corr:.3f})")

        print("")
        print("null separation: another recording in the same mix must score lower")
        names = sorted(prints)
        a_ref = prints[names[0]][0]
        _, _, c_true = oa.fit_placement(a_ref, prints[names[0]][1])
        worst = -9.0
        for other in names[1:]:
            _, _, c_o = oa.fit_placement(a_ref, prints[other][1])
            worst = max(worst, c_o)
            print(f"  {names[0]} vs {other:12s} {c_true:.3f} / {c_o:.3f}")
        good = c_true - worst > 0.10
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} separation {c_true - worst:+.3f}")

    print("")
    print("feature recompute: our columns must reproduce on identical audio")
    t = np.linspace(0, 30, 30 * sr)
    y = (np.sin(2 * np.pi * 110 * t) * 0.4
         + rng.normal(0, 0.05, len(t)) * (np.sin(2 * np.pi * 2.1 * t) > 0))
    f1, f2 = oa.frame_features(y), oa.frame_features(y * 0.5)
    for c in ("spectral_centroid_hz", "bass_frac_below150hz"):
        d = float(np.nanmax(np.abs(f1[c] - f2[c])))
        good = d < 1e-6 * max(1.0, float(np.nanmax(np.abs(f1[c]))))
        ok &= good
        print(f"  {'ok ' if good else 'FAIL'} {c} gain-invariant, max delta {d:.3e}")

    print("")
    print(f"selftest {'PASS' if ok else 'FAIL'}")
    _save("selftest.json", {"pass": bool(ok)})
    sys.exit(0 if ok else 1)


CMDS = {"resolve": cmd_resolve, "ingest": cmd_ingest, "align": cmd_align,
        "build": cmd_build, "verify": cmd_verify, "selftest": cmd_selftest}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__)
        sys.exit(1)
    CMDS[sys.argv[1]]()
