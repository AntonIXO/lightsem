#!/usr/bin/env python
"""H-B arm (b): fit one trailing one-pole per channel to the corpus spectrum
(calibrate_output_smoothing) on the validation split and write the constants
into a COPY of the checkpoint, so the unfiltered arm keeps its own file.

    bash run_torch.sh calibrate_smoothing.py --ckpt .../light_trunk_v3causal.pt
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from lightsem.model.infer import LightModel, calibrate_output_smoothing
from lightsem.model.light_data import load_tensor, make_splits
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus

OUT = Path("/root/lightsem/outputs/stage1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/lightsem-data/ckpt/light_trunk_v3causal.pt")
    ap.add_argument("--split", default="val_track")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tag", default="v3causal")
    ap.add_argument("--audio-dir", default="", help="bus directory; cache/audio_offline for an offline checkpoint")
    args = ap.parse_args()
    z, meta = load_tensor()
    sp = make_splits(meta)
    idx = getattr(sp, args.split)
    rng = np.random.default_rng(0)
    idx = idx[rng.permutation(len(idx))[:args.n]]
    lm = LightModel(args.ckpt)
    clips = []
    for i in idx:
        nm = meta[i]["name"]
        d = load_descriptors(CACHE_ROOT / "motion" / f"{nm}.npz")
        bus = slice_bus(load_bus((Path(args.audio_dir) if args.audio_dir else CACHE_ROOT / "audio") / f"{nm.split('_')[4]}.npz"), d.duration_s)
        clips.append((bus, d))
    print(f"calibrating smoothing on {len(clips)} {args.split} clips")
    taus = calibrate_output_smoothing(lm, clips)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    blob["smooth_tau"] = taus
    out = Path(args.ckpt).with_name(Path(args.ckpt).stem + "_smooth.pt")
    torch.save(blob, out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"smooth_tau_{args.tag}.json").write_text(json.dumps(
        {"ckpt_in": args.ckpt, "ckpt_out": str(out), "split": args.split, "n_clips": len(clips),
         "smooth_tau": taus}, indent=1))
    print(f"wrote {out} with {taus}")


if __name__ == "__main__":
    main()
