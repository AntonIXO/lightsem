#!/usr/bin/env python3
"""Build a small, deterministic RTMPose TRAIN manifest for stereo adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha(path)}) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b"))
    p.add_argument("--out", type=Path, default=Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b_stereo_v1"))
    p.add_argument("--groups", type=int, default=32)
    args = p.parse_args(argv)
    source_manifest = args.source_root / "data/MANIFEST.json"
    rows = json.loads(source_manifest.read_text())["entries"]
    selected = []
    seen = set()
    for row in sorted(rows, key=lambda x: (x.get("group", ""), x["key"])):
        if row.get("partition") != "train" or row.get("domain") != "jd":
            continue
        group = row.get("group")
        if group in seen:
            continue
        if not Path(row["wav_path"]).is_file():
            continue
        seen.add(group)
        selected.append(row)
        if len(seen) >= args.groups:
            break
    if len(selected) != args.groups:
        raise ValueError(f"Only found {len(selected)} eligible groups, requested {args.groups}")
    payload = {
        "schema_version": 1,
        "status": "stereo_native_training_pilot",
        "test_access": False,
        "source_root": str(args.source_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha(source_manifest),
        "selection": "first deterministic duplicate-resolved RTMPose JD TRAIN group per group id",
        "groups": len(selected),
        "entries": [{k: row[k] for k in (
            "key", "feature_key", "domain", "partition", "group", "wav_path", "wav_sha256",
            "pose_cache", "pose_cache_sha256", "frames", "timeline_path") if k in row} for row in selected],
    }
    write(args.out / "STEREO_NATIVE_MANIFEST.json", payload)
    print(json.dumps({"status": "complete", "groups": len(selected), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
