#!/usr/bin/env python3
"""Independent saved-artifact checks after the baseline06 media queue finishes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from lightsem.conductor.rig import canonical_rig


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    manifest = json.loads((root / "MANIFEST.json").read_text())
    progress = json.loads((root / "PROGRESS.json").read_text())
    assert manifest["status"] == "all_four_verified" and progress["status"] == "complete"
    assert [v["id"] for v in manifest["variants"]] == ["01", "02", "03", "04"]
    for item in manifest["inputs"].values():
        assert digest(item["path"]) == item["sha256"]
    for path, expected in manifest["source_files"].items():
        assert digest(path) == expected
    with np.load(manifest["inputs"]["overlay"]["path"], allow_pickle=False) as z:
        baseline = z["overlay"].copy()
    rgb = np.zeros(512, bool)
    for fixture in canonical_rig().fixtures:
        offsets = {"moving_head": [6, 7, 8], "wash": [2, 3, 4], "projector": [2, 3, 4]}.get(fixture.kind, [])
        for offset in offsets:
            rgb[fixture.address - 1 + offset] = True
    assert int((~rgb).sum()) == 479
    pcm_hashes, rows = set(), []
    for item in manifest["variants"]:
        path = Path(item["path"])
        assert item["status"] == "rendered_and_verified"
        assert path.stat().st_size <= manifest["max_media_bytes"]
        assert digest(path) == item["sha256"]
        assert digest(item["streams"]) == item["streams_sha256"]
        with np.load(item["streams"], allow_pickle=False) as z:
            stream = z["candidate"]
            assert np.array_equal(z["time"], np.arange(len(baseline)) / 40)
        assert np.array_equal(stream[:, ~rgb], baseline[:, ~rgb])
        if item["id"] == "01":
            assert np.array_equal(stream, baseline)
        audit = json.loads(Path(item["audit"]).read_text())
        render = json.loads(path.with_name(path.stem + "_render.json").read_text())
        assert audit["decoded_frames"] == 8319 and audit["full_software_AV_decode"] and audit["pts_uniform"]
        assert render["video_sha256"] == item["sha256"]
        assert render["source_audio_sha256"] == manifest["inputs"]["audio"]["sha256"]
        pcm_hashes.add(audit["pcm_sha256"])
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], text=True))
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        assert (v["codec_name"], v["width"], v["height"], v["r_frame_rate"]) == ("av1", 1920, 1080, "30/1")
        assert abs(float(v["duration"]) - 277.3) < .001
        assert a["codec_name"] == "aac" and abs(float(a["duration"]) - 277.3) < .06
        rows.append({"arm": item["letter"], "path": str(path), "bytes": path.stat().st_size,
                     "sha256": item["sha256"], "frames": audit["decoded_frames"],
                     "all_non_RGB_slots_unchanged": True, "max_RGB_channel_step_byte": item["max_RGB_channel_step_byte"]})
    assert pcm_hashes == {manifest["pcm_sha256"]}
    assert shutil.disk_usage(root).free > 2**30
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert total < manifest["resources"]["output_cap_bytes"]
    result = {"status": "PASS_ALL_FOUR_READY_FOR_HUMAN_REVIEW", "variants": rows,
              "original_model_audio_and_video_unchanged": True,
              "bound_sources_unchanged": True, "identical_decoded_audio_sha256": next(iter(pcm_hashes)),
              "delivery_package_bytes": total, "root_free_bytes": shutil.disk_usage(root).free,
              "media_decode_evidence": "Per-file complete software decode/timestamp audits authenticated; ffprobe and hashes independently checked here",
              "human_preference_for_new_arms": "pending", "baseline_selection": "User previously selected original06"}
    with (root / "INDEPENDENT_DELIVERY_AUDIT.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
