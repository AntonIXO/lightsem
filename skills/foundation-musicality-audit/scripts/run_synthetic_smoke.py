#!/usr/bin/env python3
"""Run the repository's deterministic metric sanity benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/lightsem"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if not (repo / "run.sh").is_file():
        raise SystemExit(f"not a LightSem repository: {repo}")
    command = [
        "bash", str(repo / "run.sh"),
        "scripts/evaluation/benchmark_music_motion_metrics.py",
        "--out", str(args.out.resolve()),
    ]
    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
