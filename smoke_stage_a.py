"""Stage A smoke test: prove the data layer loads real sequences end to end."""

from __future__ import annotations

import sys
import time

import numpy as np

from lightsem.data import AistDataset, write_manifest


def main() -> int:
    ds = AistDataset()
    files = ds.keypoint_files()
    print(f"keypoint files on disk : {len(files)}")
    print(f"wav files on disk      : {len(list(ds.wav_dir.glob('*.wav')))}")
    if not files:
        print("FAIL: no keypoint files found")
        return 1

    # 1. Filename parsing must accept cAll.
    from lightsem.data import parse_name

    n_parsed = sum(parse_name(p.stem) is not None for p in files)
    print(f"parseable filenames    : {n_parsed}/{len(files)}")
    assert n_parsed == len(files), "filename regex rejects real files"

    # 2. Load one real sequence with the NaN policy applied.
    kp, meta = ds.load(files[0])
    print(f"\nloaded {meta.name}")
    print(f"  shape      : {kp.shape}")
    print(f"  dancer     : {meta.dancer}  music: {meta.music_id}  choreo: {meta.choreo_id}")
    print(f"  frames     : {meta.n_frames}  duration: {meta.duration_s:.2f}s")
    print(f"  nan_frac   : {meta.nan_frac:.4f}  usable: {meta.usable} ({meta.reason})")
    print(f"  wav        : {meta.wav_path}")
    assert np.isfinite(kp).all(), "NaN survived the interpolation policy"
    print("  all finite after NaN policy: OK")

    # 3. Full corpus scan -> manifest.
    t0 = time.time()
    metas = ds.build_manifest()
    dt = time.time() - t0
    print(f"\nscanned {len(metas)} sequences in {dt:.1f}s")

    paths = write_manifest(metas)
    import json

    summary = json.loads(paths["summary"].read_text())
    print(json.dumps(summary, indent=2))
    print(f"\nmanifest : {paths['parquet']}")
    print(f"summary  : {paths['summary']}")

    if summary["n_usable"] == 0:
        print("FAIL: zero usable sequences")
        return 1
    print("\nSTAGE A SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
