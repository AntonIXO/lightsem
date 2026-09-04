#!/usr/bin/env python
"""Diagnose probe3 small-n: pose clip duration vs audio track duration."""
import glob
import os
import pickle
import re

import librosa
import numpy as np

DATA = "/root/lightsem/data"
KP_FPS = 60.0
SR = 22050
HOP = 512


def parse_name(fn):
    m = re.match(r"(g\w+?)_(s\w+?)_(c[\w\d]+?)_(d\d+)_(m\w+?)_(ch\d+)",
                 os.path.basename(fn))
    if not m:
        return None
    g, s, c, d, mus, ch = m.groups()
    return dict(genre=g, sit=s, cam=c, dancer=d, music=mus, chore=ch)


kp_files = sorted(glob.glob(os.path.join(DATA, "kp3d", "**", "*.pkl"), recursive=True))
wavs = {os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(DATA, "wav", "**", "*.wav"), recursive=True)}

durs = []
for p in kp_files[:400]:
    m = parse_name(p)
    if not m or m["music"] not in wavs:
        continue
    with open(p, "rb") as f:
        d = pickle.load(f)
    kp = d.get("keypoints3d", d.get("keypoints3d_optim"))
    if kp is None:
        continue
    durs.append((len(kp) / KP_FPS, m["music"]))

pd_ = np.array([x[0] for x in durs])
print(f"POSE clip duration (n={len(pd_)}): "
      f"min={pd_.min():.1f}s  median={np.median(pd_):.1f}s  max={pd_.max():.1f}s")

print("\nAUDIO track duration:")
for mus in ["mWA5", "mWA4", "mPO1", "mBR4"]:
    y, _ = librosa.load(wavs[mus], sr=SR, mono=True)
    ad = len(y) / SR
    _, bt = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="time")
    sub = pd_[[i for i, x in enumerate(durs) if x[1] == mus]]
    print(f"  {mus}: audio={ad:.1f}s  beats={len(bt)}  "
          f"pose_clips_median={np.median(sub):.1f}s" if len(sub) else
          f"  {mus}: audio={ad:.1f}s  beats={len(bt)}")
    if len(sub):
        n_usable = int((bt < np.median(sub)).sum())
        print(f"      -> beats inside pose clip: {n_usable} / {len(bt)}")
