#!/usr/bin/env python
"""H-L rating CLI: plays each pair (A then B, mpv) and records two answers per pair.

    python3 rate_hl_pairs.py [--rater anton] [--pairs outputs/stage9/hl_pairs/pairs.json]

Questions (PLAN.md 1.5 / 9.6, kept separate because they dissociated): 1 "fits the music" and
2 "looks like a lighting engineer did it"; answer A / B / = (tie) / r (replay) / q (quit).
Appends to outputs/stage9/hl_pairs/ratings.jsonl; already-rated pairs for this rater are skipped.
"""
import argparse, json, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--rater", default="anton")
ap.add_argument("--pairs", default="/root/lightsem/outputs/stage9/hl_pairs/pairs.json")
a = ap.parse_args()
root = Path(a.pairs).parent; pairs = json.loads(Path(a.pairs).read_text())["pairs"]
out = root / "ratings.jsonl"
done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
done = {d for d in done}
for p in pairs:
    if p["id"] in done:
        continue
    while True:
        for tag in ("A", "B"):
            print(f"\n{p['id']}  playing {tag} ...", flush=True)
            subprocess.run(["mpv", "--really-quiet", "--fs", str(root / f"{p['id']}__{tag}.mp4")])
        q1 = input("1) fits the music better: A / B / = (tie) / r (replay) / q: ").strip().lower()
        if q1 == "r":
            continue
        if q1 == "q":
            raise SystemExit
        q2 = input("2) looks like a lighting engineer did it: A / B / = : ").strip().lower()
        break
    with out.open("a") as f:
        f.write(json.dumps({"id": p["id"], "rater": a.rater, "fits_music": q1, "engineer": q2, "t": time.time()}) + "\n")
print("done")
