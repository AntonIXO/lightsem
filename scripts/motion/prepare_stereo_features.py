#!/usr/bin/env python3
"""Prepare authenticated StereoAudioV1 caches for TRAIN/DEV pilot records.

The loader keeps the original L/R waveform for descriptor extraction and never
opens a TEST record.  MuQ/Base92 extraction remains a separate, already pinned
input; ``--base92`` may bind an existing cache path to the output manifest.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from lightsem.audio.stereo_v1 import SCHEMA_VERSION, StereoAudioV1, extract_stereo_v1, metamorphic_check, sha256


PILOT_ROOT = Path("/mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1")


def write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserving existing pilot artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    sidecar = path.with_name(path.name + ".sha256.json")
    sidecar.write_text(json.dumps({"sha256": sha256(path)}) + "\n")


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "record"


def _records(manifest: dict) -> list[dict]:
    records = manifest.get("records", manifest.get("entries"))
    if not isinstance(records, list) or not records:
        raise ValueError("Input manifest needs a nonempty records or entries list")
    return records


def run(input_manifest: Path, out: Path, *, limit: int | None = None,
        base92_manifest: Path | None = None) -> dict:
    source = json.loads(input_manifest.read_text())
    rows = _records(source)
    base92 = {}
    if base92_manifest is not None:
        base = json.loads(base92_manifest.read_text())
        for item in _records(base):
            key = str(item.get("id", item.get("key", item.get("record_id", ""))))
            if key:
                base92[key] = item
    if limit is not None:
        rows = rows[:limit]
    feature_dir = out / "features"
    result_rows = []
    for row in rows:
        split = str(row.get("split", row.get("partition", ""))).lower()
        if split == "test":
            raise ValueError("Stereo pilot refuses TEST records")
        if split not in ("train", "dev", "val", "showcase"):
            raise ValueError(f"Unsupported split for stereo pilot: {split!r}")
        record_id = str(row.get("id", row.get("key", row.get("record_id", ""))))
        if not record_id:
            raise ValueError("Every stereo record needs id/key/record_id")
        audio_path = Path(row.get("wav_path", row.get("audio_path", ""))).resolve(strict=True)
        info = sf.info(str(audio_path))
        if info.channels not in (1, 2):
            raise ValueError(f"Only mono/stereo audio is supported: {audio_path}")
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = waveform.T
        descriptor = extract_stereo_v1(waveform, int(sample_rate))
        destination = feature_dir / f"{_safe(record_id)}.npz"
        descriptor.metadata.update({
            "record_id": record_id, "split": split,
            "audio_path": str(audio_path), "audio_sha256": sha256(audio_path),
        })
        descriptor.save(destination)
        metamorphic = metamorphic_check(waveform, int(sample_rate))
        row_result = {
            "id": record_id, "split": split, "audio_path": str(audio_path),
            "audio_sha256": sha256(audio_path), "feature_path": str(destination),
            "feature_sha256": sha256(destination), "metadata_path": str(destination.with_suffix(".json")),
            "feature_dim": descriptor.feature_dim, "event_dim": descriptor.event_dim,
            "spatial_dim": descriptor.spatial_10.shape[-1], "relation_dim": descriptor.relation_2.shape[-1],
            "frames_40": len(descriptor.time_40), "frames_10": len(descriptor.time_10),
            "frames_2": len(descriptor.time_2), "metamorphic": metamorphic,
            "test_audio_opened": False,
        }
        linked = base92.get(record_id)
        if linked:
            path = Path(linked.get("base92_path", linked.get("path", "")))
            if path.exists():
                row_result["base92_path"] = str(path)
                row_result["base92_sha256"] = sha256(path)
            else:
                raise FileNotFoundError(f"Bound Base92 cache is missing: {path}")
        result_rows.append(row_result)
    protocol = {
        "schema_version": 1, "stereo_schema_version": SCHEMA_VERSION,
        "artifact": "audio_stereo_spatial_20261007_pilot_v1",
        "source_manifest": str(input_manifest), "source_manifest_sha256": sha256(input_manifest),
        "test_access": False, "audio_contract": "original L/R retained; no independent channel normalization",
        "spatial_semantics": "stereo image descriptors; not physical 3D localization",
        "muq_contract": "pinned MuQ/Base92 remains an external mono input; this cache does not replace it",
        "entries": result_rows,
    }
    write_exclusive(out / "STEREO_FEATURE_MANIFEST.json", protocol)
    return protocol


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=PILOT_ROOT)
    parser.add_argument("--base92-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = run(args.manifest, args.out, limit=args.limit, base92_manifest=args.base92_manifest)
    print(json.dumps({"status": "complete", "entries": len(result["entries"]),
                      "out": str(args.out / "STEREO_FEATURE_MANIFEST.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
