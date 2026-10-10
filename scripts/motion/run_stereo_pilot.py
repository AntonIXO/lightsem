#!/usr/bin/env python3
"""Guarded, sequential launcher for the StereoAudioV1 representation pilot."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import shutil
import time

from lightsem.audio.stereo_v1 import sha256


DEFAULT_OUT = Path("/mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1")
DEFAULT_LOCK = Path("/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock")
FREE_FLOOR = 20 * 1024 ** 3


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing pilot receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


@contextmanager
def gpu_lock(path: Path, *, wait: bool = False, timeout_s: float | None = None,
             poll_s: float = 30.0):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    started = time.monotonic()
    last_notice = started
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError as exc:
            if not wait or (timeout_s is not None and time.monotonic() - started >= timeout_s):
                handle.close()
                raise RuntimeError(f"GPU curator lock is busy: {path}") from exc
            now = time.monotonic()
            if now - last_notice >= 60.0:
                print(json.dumps({"status": "waiting_for_gpu_lock", "waited_s": now - started}), flush=True)
                last_notice = now
            time.sleep(poll_s)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def preflight(manifest: Path, out: Path, device: str) -> dict:
    if shutil.disk_usage(out if out.exists() else out.parent).free < FREE_FLOOR:
        raise RuntimeError("20 GiB free-space floor would be violated")
    payload = json.loads(manifest.read_text())
    if payload.get("test_access", True):
        raise ValueError("Stereo pilot manifest must prove test_access=false")
    entries = payload.get("entries", [])
    if not entries or any(row.get("split") == "test" for row in entries):
        raise ValueError("Pilot manifest must contain nonempty TRAIN/DEV rows and no TEST")
    splits = {split: sum(row.get("split") == split for row in entries) for split in ("train", "dev", "val", "showcase")}
    result = {
        "schema_version": 1, "status": "ready", "artifact": "audio_stereo_spatial_20261007_pilot_v1",
        "manifest": str(manifest), "manifest_sha256": sha256(manifest), "device": device,
        "entries": len(entries), "split_counts": splits, "test_access": False,
        "arms": ["R1_fused", "R2_factorized"], "seeds": [0],
        "max_updates_per_arm": 3000, "minimum_free_bytes": FREE_FLOOR,
        "created_at": time.time(), "note": "Quality gates are applied only by the locked evaluation report; no automatic promotion.",
    }
    _write(out / "PREFLIGHT.json", result)
    return result


def train_arms(manifest: Path, out: Path, device: str, steps: int, batch_size: int, split: str,
               wait_for_gpu: bool, max_wait_hours: float) -> dict:
    from scripts.motion.train_stereo_conditioner import train
    preflight_path = out / "PREFLIGHT.json"
    if not preflight_path.exists():
        raise FileNotFoundError("Run --stage preflight before --stage train")
    pre = json.loads(preflight_path.read_text())
    if pre.get("manifest_sha256") != sha256(manifest) or pre.get("status") != "ready":
        raise ValueError("Pilot preflight does not bind the requested manifest")
    if steps > 3000:
        raise ValueError("Pilot cap is 3000 equal updates per representation arm")
    rows = []
    lock = gpu_lock(DEFAULT_LOCK, wait=wait_for_gpu, timeout_s=max_wait_hours * 3600.) if device.startswith("cuda") else nullcontext()
    with lock:
        for arm, directory in (("fused", "R1"), ("factorized", "R2")):
            rows.append(train(manifest, out / "learning" / directory, arm=arm, steps=steps,
                              batch_size=batch_size, frames=320, seed=0, device=device, split=split))
    result = {"schema_version": 1, "status": "complete", "manifest": str(manifest),
              "manifest_sha256": sha256(manifest), "test_access": False, "arms": rows}
    _write(out / "TRAINING_SUMMARY.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=("train", "dev", "val"), default="train")
    parser.add_argument("--wait-for-gpu", action="store_true",
                        help="wait behind the shared GPU curator lock instead of failing when busy")
    parser.add_argument("--max-wait-hours", type=float, default=24.)
    args = parser.parse_args(argv)
    result = preflight(args.manifest, args.out, args.device) if args.stage == "preflight" else train_arms(
        args.manifest, args.out, args.device, args.steps, args.batch_size, args.split,
        args.wait_for_gpu, args.max_wait_hours)
    print(json.dumps({"status": result["status"], "stage": args.stage}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
