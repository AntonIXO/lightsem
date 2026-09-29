#!/usr/bin/env python3
"""Sequentially render and audit prepared baseline06 comparison variants."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.light_layers.prepare_baseline_review import sha

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["output_root"])
    prep = json.loads((root / "PREPARATION.json").read_text())
    started = time.monotonic()
    completed = []
    manifest = dict(plan, status="rendering", variants=[])
    expected_pcm = plan.get("pcm_sha256")
    env = dict(os.environ, TORCH_ENV="/root/lightsem/.venv/torch-access",
               PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")

    def write(name, obj):
        temp = root / (name + ".tmp")
        temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        temp.replace(root / name)

    def progress(status, **extra):
        write("PROGRESS.json", dict(status=status, completed=completed,
              elapsed_seconds=time.monotonic()-started, **extra))

    try:
        for path, expected in plan["source_files"].items():
            assert sha(path) == expected, "Bound source changed: " + path
        for row in prep["variants"]:
            assert sha(row["streams"]) == row["streams_sha256"]
            dest = Path(row["path"])
            if row["id"] == "01" and row["status"] == "verified_existing":
                assert sha(dest) == row["sha256"]
                audit = json.loads(Path(row["audit"]).read_text())
                assert audit["decoded_frames"] == 8319 and audit["pts_uniform"] and audit["full_software_AV_decode"]
                row["status"] = "verified_existing"
            else:
                dest.parent.mkdir(parents=True, exist_ok=False)
                for mode in ("render", "audit"):
                    progress("running", variant=row["id"], phase=mode)
                    cmd = ["bash", str(REPO / "run_torch.sh"), "-u",
                           str(REPO / "scripts/rendering/render_light_layers_review.py"), mode,
                           "--streams", row["streams"], "--output", str(dest),
                           "--label", row["caption"], "--audio", plan["inputs"]["audio"]["path"],
                           "--max-bytes", str(plan.get("max_media_bytes", 50_000_000))]
                    with (dest.parent / (mode + ".log")).open("xb") as log:
                        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=720, check=True)
                audit_path = dest.with_name(dest.stem + "_audit.json")
                audit = json.loads(audit_path.read_text())
                render = json.loads(dest.with_name(dest.stem + "_render.json").read_text())
                assert audit["decoded_frames"] == 8319 and audit["pts_uniform"] and audit["full_software_AV_decode"]
                assert render["source_audio_sha256"] == plan["inputs"]["audio"]["sha256"]
                assert render["video_sha256"] == sha(dest)
                row.update(status="rendered_and_verified", sha256=render["video_sha256"],
                           audit=str(audit_path), bytes=dest.stat().st_size)
            if expected_pcm is None:
                expected_pcm = audit["pcm_sha256"]
            else:
                assert audit["pcm_sha256"] == expected_pcm
            assert dest.stat().st_size <= plan.get("max_media_bytes", 50_000_000)
            manifest["pcm_sha256"] = expected_pcm
            completed.append(row["id"])
            manifest["variants"].append(row)
            write("MANIFEST.json", manifest)
        for entry in plan["inputs"].values():
            assert sha(entry["path"]) == entry["sha256"]
        manifest["status"] = "all_four_verified"
        write("MANIFEST.json", manifest)
        progress("complete")
    except BaseException as exc:
        progress("failed", error=type(exc).__name__ + ": " + str(exc))
        raise


if __name__ == "__main__":
    main()
