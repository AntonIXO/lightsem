#!/usr/bin/env python
"""Fit the block-B hazard thresholds to the corpus event rate, in place.

    bash run_torch.sh calibrate_events.py --ckpt /mnt/.../light_trunk_v3causal.pt

Stage 1 measured the model emitting 88.9 light events per clip where light
mapped from the real dance emits 46.5 and an audio beat trigger 59.8. The cause
was not the model: `descriptors()` floored the event threshold at the clip's own
97th percentile, so the emitted rate was a constant of the code and the model's
confidence only decided WHICH of a fixed number of frames fired. A train denser
than the beat grid sits at its own circular-shift null however well each flash
is placed, which is what the +0.0103 Light-BAS was measuring.

This fits one level per channel on the validation split -- never on training
clips, and never on the test split the gates are read from -- so the rate the
model emits is the rate the corpus has, and the threshold is a property of the
checkpoint rather than of whichever clip it happens to be rendering.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lightsem.model.infer import LightModel, calibrate_event_thresholds
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
    ap.add_argument("--corpus", default="aist", choices=["aist", "jd"])
    ap.add_argument("--structure", default="none", choices=["none", "real", "shuffled"],
                    help="jd, H-C: feed the checkpoint the structure channels it was trained with")
    ap.add_argument("--tensor", default="", help="tensor path for the split definition")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    z, meta = load_tensor(Path(args.tensor) if args.tensor else None)
    if args.corpus == "jd":
        from lightsem.model.jd_data import make_jd_splits
        sp = make_jd_splits(meta)
    else:
        sp = make_splits(meta)
    idx = getattr(sp, args.split)
    rng = np.random.default_rng(0)
    idx = idx[rng.permutation(len(idx))[:args.n]]

    lm = LightModel(args.ckpt)
    if lm.event_thr:
        print(f"checkpoint already carries {lm.event_thr}; refitting")
        lm.event_thr = {}

    clips = []
    for i in idx:
        nm = meta[i]["name"]
        if args.corpus == "jd":
            d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{nm}.npz")
            off = float(d.meta.get("start_s", 0.0))
            bus = slice_bus(load_bus((Path(args.audio_dir) if args.audio_dir else CACHE_ROOT / "justdance_audio") / f"{meta[i]['music_id']}.npz"),
                            d.duration_s, offset_s=off)
            if args.structure != "none":
                from lightsem.model.jd_data import structure_channels
                mids = sorted({m["music_id"] for m in meta})
                src = meta[i]["music_id"] if args.structure == "real" else mids[(mids.index(meta[i]["music_id"]) + 1) % len(mids)]
                extra = structure_channels(src, off if args.structure == "real" else 0.0, len(bus.times))
                clips.append((bus, d, extra)); continue
        else:
            d = load_descriptors(CACHE_ROOT / "motion" / f"{nm}.npz")
            bus = slice_bus(load_bus((Path(args.audio_dir) if args.audio_dir else CACHE_ROOT / "audio") / f"{nm.split('_')[4]}.npz"),
                            d.duration_s)
        clips.append((bus, d))
    print(f"calibrating on {len(clips)} {args.split} clips")

    thr = calibrate_event_thresholds(lm, clips)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"event_thresholds_{args.tag}.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "split": args.split, "n_clips": len(clips),
         "thresholds": thr}, indent=1))
    if args.dry_run:
        print("dry run, checkpoint not written")
        return
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    blob["event_thr"] = thr
    torch.save(blob, args.ckpt)
    print(f"wrote thresholds into {args.ckpt}")


if __name__ == "__main__":
    main()
