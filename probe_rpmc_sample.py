"""Pull ONE real RPMC-L2 sample: what does the light target actually look like?

Answers questions the README does not: what are the 8 numbered light subgroups,
what is in the (F,3,256) tensor, and what is the music/light frame-rate ratio.
"""
import json
import time

import numpy as np

from lightsem.data.rpmc_remote import open_h5

t0 = time.time()
f, raw = open_h5()
keys = list(f.keys())
print(f"songs in dataset: {len(keys)}  (opened {time.time()-t0:.0f}s)")

k = keys[0]
g = f[k]
light = g["light"]
sub = sorted(light.keys(), key=lambda s: int(s))
print(f"light subgroups: {sub}")

# Read the un-rotated light track and one rotated one.
L0 = light["0"][:]
L30 = light["30"][:]
print(f"\nlight['0']  shape={L0.shape} dtype={L0.dtype}")
print(f"  sum over last axis (first frame): {L0[0].sum(axis=-1)}")
print(f"  -> rows are HISTOGRAMS: H(180 bins used of 256), S(256), V(256)")

# Is subgroup N a hue rotation of subgroup 0?
h0 = L0[:, 0, :180]
h30 = L30[:, 0, :180]
best, bestshift = -1, None
for s in range(0, 180, 5):
    r = np.corrcoef(np.roll(h0, s, axis=1).ravel(), h30.ravel())[0, 1]
    if r > best:
        best, bestshift = r, s
print(f"\nlight['30'] vs roll(light['0'], {bestshift}) -> r={best:.4f}")
print("  => numbered subgroups are HUE-ROTATION AUGMENTATIONS, not fixtures")

mus = g["music"]
mel = mus["mel_spectrogram_db"]
print(f"\nmusic frames {mel.shape[1]}  light frames {L0.shape[0]}  "
      f"ratio {mel.shape[1]/L0.shape[0]:.3f}")

# Dominant hue/val per frame = a usable low-dim target.
hue_bins = L0[:, 0, :180].astype(np.float64)
val_bins = L0[:, 2, :].astype(np.float64)
dom_hue = hue_bins.argmax(axis=1)
mean_val = (val_bins * np.arange(256)).sum(1) / np.maximum(val_bins.sum(1), 1)
print(f"\ndominant hue: min={dom_hue.min()} max={dom_hue.max()} "
      f"unique={len(np.unique(dom_hue))}")
print(f"mean value  : min={mean_val.min():.1f} max={mean_val.max():.1f} "
      f"std={mean_val.std():.1f}")
print(f"  brightness DYNAMIC RANGE p90/p10 = "
      f"{np.percentile(mean_val,90)/max(np.percentile(mean_val,10),1e-6):.2f}")

out = {
    "n_songs": len(keys),
    "light_subgroups": sub,
    "light_shape": list(L0.shape),
    "music_frames": int(mel.shape[1]),
    "music_light_ratio": float(mel.shape[1] / L0.shape[0]),
    "rotation_confirmed_r": float(best),
    "rotation_shift": int(bestshift),
    "mean_val_std": float(mean_val.std()),
    "mb_fetched": raw.bytes_fetched / 1e6,
}
with open("data/cache/rpmc_probe.json", "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nfetched {raw.bytes_fetched/1e6:.0f} MB total in {time.time()-t0:.0f}s")
f.close()
