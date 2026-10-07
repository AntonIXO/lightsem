"""Build the expanded, duplicate-safe native TRAIN admission manifest.

This is an admission artifact only. It does not train or overwrite the frozen
96-group development manifest. Each Just Dance song group contributes its best
audited coach track; protected development/quarantine groups have already been
removed by ``audit_native_full_data.py``.
"""
from pathlib import Path
import argparse, json, hashlib


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--fine-dance-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit = json.loads(a.audit.read_text())
    rows = [r for r in audit["candidates"] if r["wav_present"] and r["known_group"]]
    best = {}
    for row in rows:
        key = row["known_group"]
        rank = (-row["quality"]["coverage"], row["key"], row["coach_index"])
        if key not in best or rank < best[key][0]: best[key] = (rank, row)
    jd = []
    for _, row in sorted(best.values(), key=lambda x: x[1]["known_group"]):
        jd.append({"domain": "jd", "partition": "train", "group": row["known_group"],
                   "key": row["key"], "coach_index": row["coach_index"],
                   "pose_path": row["pose_path"], "wav_path": row["wav_path"],
                   "pose_sha256": digest(row["pose_path"]), "wav_sha256": digest(row["wav_path"]),
                   "coverage": row["quality"]["coverage"], "eligible_windows": row["quality"]["eligible_windows"]})
    fd = json.loads(a.fine_dance_manifest.read_text())
    fd_rows = [{"domain": "fd", **{k: r[k] for k in ("partition", "canonical_id", "key", "path", "audio_path", "cache_path", "sha256", "audio_sha256", "cache_sha256")}} for r in fd["entries"] if r["partition"] == "train"]
    result = {"schema_version": 1, "status": "admission_manifest_not_trained", "selection": "all audited known Just Dance groups with paired waveform; best coach by coverage; existing development/quarantines retained", "inputs": {"audit": digest(a.audit), "fine_dance_manifest": digest(a.fine_dance_manifest)}, "counts": {"jd_groups": len(jd), "jd_tracks": len(jd), "fd_groups": len({r["canonical_id"] for r in fd_rows}), "fd_records": len(fd_rows)}, "justdance": jd, "finedance": fd_rows, "missing_audio_not_admitted": sorted({r["key"] for r in audit["candidates"] if not r["wav_present"]}), "warnings": ["No training was run by this command", "Just Dance remains observed 2D; no metric 3D labels are fabricated", "Audio extraction for missing-waveform video sources is a separate, storage-gated step"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__": main()
