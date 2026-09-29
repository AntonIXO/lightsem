#!/usr/bin/env python
"""H-L rating CLI: plays each pair (A then B, mpv) and records two answers per pair.

    python3 scripts/training/rate_hl_pairs.py [--rater anton] [--pairs experiments/legacy/stages/outputs/stage9/hl_pairs/pairs.json]

Questions (PLAN.md 1.5 / 9.6, kept separate because they dissociated): 1 "fits the music" and
2 "looks like a lighting engineer did it"; answer A / B / = (tie) / r (replay) / q (quit).
Appends to experiments/legacy/stages/outputs/stage9/hl_pairs/ratings.jsonl; already-rated pairs for this rater are skipped.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path


def read_answer(question: str) -> str:
    """Read a preference or session command; incomplete input quits safely."""
    while True:
        try:
            answer = input(f"{question}: A / B / = (tie) / r (replay) / q: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "q"
        if answer in {"a", "b", "=", "r", "q"}:
            return answer
        print("Please enter A, B, =, r, or q.")


def rate_pairs(pairs_path: Path, rater: str) -> bool:
    """Append complete answers for this rater; return False when they quit."""
    root = pairs_path.parent
    pairs = json.loads(pairs_path.read_text())["pairs"]
    out = root / "ratings.jsonl"
    ratings = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    done = {rating["id"] for rating in ratings if rating.get("rater") == rater}
    for pair in pairs:
        if pair["id"] in done:
            continue
        while True:
            for tag in ("A", "B"):
                print(f"\n{pair['id']}  playing {tag} ...", flush=True)
                subprocess.run(["mpv", "--really-quiet", "--fs", str(root / f"{pair['id']}__{tag}.mp4")])
            fits_music = read_answer("1) fits the music better")
            if fits_music == "q":
                return False
            if fits_music == "r":
                continue
            engineer = read_answer("2) looks like a lighting engineer did it")
            if engineer == "q":
                return False
            if engineer == "r":
                continue
            break
        with out.open("a") as stream:
            stream.write(json.dumps({"id": pair["id"], "rater": rater, "fits_music": fits_music,
                                     "engineer": engineer, "t": time.time()}) + "\n")
        done.add(pair["id"])
    print("done")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater", default="anton")
    parser.add_argument("--pairs", default="/root/lightsem/experiments/legacy/stages/outputs/stage9/hl_pairs/pairs.json")
    args = parser.parse_args(argv)
    rate_pairs(Path(args.pairs), args.rater)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
