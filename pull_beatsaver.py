#!/usr/bin/env python
"""Pull top-rated Beat Saber lightshow maps from BeatSaver (PLAN.md rev. 8, section 6.9.1; H-H).

Research use only: per-map licences vary, BeatSaver runs a DMCA process, the audio is the mapper's
upload. Nothing here is a product training set.

    bash run.sh pull_beatsaver.py --n 400            # top-rated maps tagged 'lightshow'
    bash run.sh pull_beatsaver.py --n 400 --resume   # skip zips already on disk (default)

Writes /mnt/lightsem-data/beatsaver/maps/<id>.zip and manifest.json with, per map: id, name,
uploader, score, upvotes, bpm, duration_s, environment(s), schema version, and per difficulty the
counts of v3 lightColorEventBoxGroups / lightRotationEventBoxGroups / basicBeatmapEvents and the
number of distinct lighting group ids, so the 9.2 audit can select maps with hand-made lighting.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.beatsaver.com"
ROOT = Path("/mnt/lightsem-data/beatsaver")
UA = {"User-Agent": "lightsem-research/0.1 (stage-9 lighting corpus audit)"}


def get(url: str, retries: int = 4) -> bytes:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(url)


def inspect_zip(data: bytes) -> dict:
    """Counts of lighting objects per difficulty, from the zip in memory."""
    out = {"diffs": {}, "audio": None, "schema": None}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        info_name = next((n for n in names if n.lower() == "info.dat"), None)
        info = json.loads(z.read(info_name)) if info_name else {}
        out["schema"] = info.get("version") or info.get("_version")
        out["bpm"] = info.get("_beatsPerMinute") or (info.get("audio") or {}).get("bpm")
        out["environment"] = info.get("_environmentName") or (info.get("environmentNames") or [None])[0]
        out["environments"] = info.get("_environmentNames") or info.get("environmentNames")
        out["audio"] = next((n for n in names if n.lower().endswith((".egg", ".ogg", ".wav", ".mp3"))), None)
        for n in names:
            if not n.lower().endswith(".dat") or n.lower() in ("info.dat", "bpminfo.dat"):
                continue
            try:
                d = json.loads(z.read(n))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict):
                continue
            cg = d.get("lightColorEventBoxGroups") or []
            rg = d.get("lightRotationEventBoxGroups") or []
            basic = d.get("basicBeatmapEvents") or d.get("_events") or []
            cd = d.get("_customData") or d.get("customData") or {}
            n_events = sum(len(e.get("e", [])) for g in cg for e in g.get("e", []))
            out["diffs"][n] = {
                "color_boxes": len(cg), "color_events": n_events,
                "rotation_boxes": len(rg), "basic_events": len(basic),
                "basic_types": sorted({t for t in (e.get("_type", e.get("et")) for e in basic if isinstance(e, dict)) if isinstance(t, (int, float))})[:16],
                "chroma_events": sum(1 for e in basic if isinstance(e, dict) and (e.get("_customData") or e.get("customData"))),
                "custom_events": len(cd.get("_customEvents") or cd.get("customEvents") or []),
                "custom_environment": bool(cd.get("_environment") or cd.get("environment")),
                "groups": len({g.get("g") for g in cg}),
                "notes": len(d.get("colorNotes") or d.get("_notes") or []),
                "last_beat": max([b for b in ([g.get("b") for g in cg] + [e.get("b", e.get("_time")) for e in basic if isinstance(e, dict)]) if isinstance(b, (int, float))] + [0]),
            }
        icd = info.get("_customData") or info.get("customData") or {}
        out["vivify_asset_bundle"] = bool(icd.get("_assetBundle") or icd.get("assetBundle"))
    # Usable for the 9.2 audit = the lighting lives in parseable events (v2 _events, with or without
    # Chroma colour data, or v3 event box groups), not only in a Vivify asset bundle / custom events.
    best = max(out["diffs"].values(), key=lambda x: x["basic_events"] + x["color_events"], default={})
    out["lighting_events"] = (best.get("basic_events", 0) + best.get("color_events", 0)) if best else 0
    out["usable"] = out["lighting_events"] >= 200
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tag", default="lightshow")
    ap.add_argument("--sort", default="Rating")
    ap.add_argument("--min-score", type=float, default=0.90)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between downloads (be polite)")
    args = ap.parse_args()

    maps_dir = ROOT / "maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    man_path = ROOT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    page, n_new, t0 = 0, 0, time.time()

    def n_usable():
        return sum(1 for v in manifest.values() if v.get("usable"))

    while n_usable() < args.n:
        docs = json.loads(get(f"{API}/search/text/{page}?tags={args.tag}&sortOrder={args.sort}")).get("docs", [])
        if not docs:
            print("no more pages", flush=True); break
        for m in docs:
            if n_usable() >= args.n:
                break
            mid = m["id"]
            if mid in manifest:
                continue
            if m["stats"]["score"] < args.min_score:
                continue
            v = m["versions"][0]
            zp = maps_dir / f"{mid}.zip"
            try:
                data = zp.read_bytes() if zp.exists() else get(v["downloadURL"])
                info = inspect_zip(data)
                if info["usable"] and not zp.exists():
                    zp.write_bytes(data)          # keep only maps whose lighting is parseable
                if not zp.exists():
                    time.sleep(args.sleep)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {mid} {m['name'][:40]!r}: {e}", flush=True); continue
            manifest[mid] = {
                "name": m["name"], "uploader": m["uploader"]["name"], "uploaded": m["uploaded"],
                "score": m["stats"]["score"], "upvotes": m["stats"]["upvotes"], "downvotes": m["stats"]["downvotes"],
                "duration_s": m["metadata"].get("duration"), "bpm_meta": m["metadata"].get("bpm"),
                "song": m["metadata"].get("songName"), "artist": m["metadata"].get("songAuthorName"),
                "tags": m.get("tags"), "hash": v["hash"], "bytes": len(data), "on_disk": info["usable"], **info}
            n_new += 1
            best = max(info["diffs"].values(), key=lambda x: x["basic_events"] + x["color_events"], default={})
            print(f"{n_usable():4d}/{len(manifest):4d} {mid:6s} {m['stats']['score']:.3f} {m['name'][:44]!r} "
                  f"{'USABLE' if info['usable'] else 'skip  '} ev={info['lighting_events']} "
                  f"v2={best.get('basic_events', 0)} chroma={best.get('chroma_events', 0)} v3={best.get('color_events', 0)} "
                  f"groups={best.get('groups', 0)} vivify={int(info.get('vivify_asset_bundle', False))} "
                  f"env={info.get('environment')} {len(data)/1e6:.1f}MB", flush=True)
            if n_new % 10 == 0:
                man_path.write_text(json.dumps(manifest, indent=1))
        page += 1
    man_path.write_text(json.dumps(manifest, indent=1))
    kept = [v for v in manifest.values() if v.get("usable")]
    tot = sum(v["bytes"] for v in kept) / 1e9
    hrs = sum((v.get("duration_s") or 0) for v in kept) / 3600
    print(f"DONE {len(kept)} usable of {len(manifest)} inspected, {tot:.2f} GB, {hrs:.1f} h of audio, "
          f"{time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
