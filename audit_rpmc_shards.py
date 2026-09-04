"""Audit the extracted RPMC-L2 shards BEFORE training on them.

Checks that matter:
  - shapes/dtypes align, no NaN/Inf
  - x and y are frame-aligned per song
  - does the LIGHT target actually vary, or is it near-constant?
  - is there any audio->light signal at all, priced against a shuffled control?

That last one decides whether training is worth starting.
"""
import glob
import json

import numpy as np

files = sorted(glob.glob("data/cache/rpmc_train/*.npz"))
print(f"shards: {len(files)}")

frames, bad, xs, ys = [], [], [], []
for p in files:
    d = np.load(p)
    x, y = d["x"].astype(np.float32), d["y"].astype(np.float32)
    if x.shape[0] != y.shape[0] or x.shape[1] != 512 or y.shape[1] != 6:
        bad.append((p, x.shape, y.shape))
        continue
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        bad.append((p, "NONFINITE", ""))
        continue
    frames.append(x.shape[0])
    xs.append(x)
    ys.append(y)

frames = np.array(frames)
print(f"valid {len(xs)}  bad {len(bad)}")
for b in bad[:5]:
    print("   BAD:", b)
print(f"frames/song: min={frames.min()} med={int(np.median(frames))} "
      f"max={frames.max()} total={frames.sum():,}")

X = np.concatenate(xs)
Y = np.concatenate(ys)
print(f"\nX {X.shape}  Y {Y.shape}  ({X.nbytes/1e6:.0f} MB in RAM as f32)")

names = ["hue_cos", "hue_sin", "sat", "val", "val_p90", "hue_conc"]
print("\n--- LIGHT TARGET VARIANCE (is there anything to predict?) ---")
for i, n in enumerate(names):
    c = Y[:, i]
    print(f"  {n:9s} mean={c.mean():+.3f} std={c.std():.3f} "
          f"p10={np.percentile(c,10):+.3f} p90={np.percentile(c,90):+.3f}")

# Between-song vs within-song variance: if all variance is BETWEEN songs, a
# model can only learn "which venue", not "what the music is doing".
print("\n--- variance decomposition (within-song is the useful part) ---")
for i, n in enumerate(names):
    per = np.array([y[:, i].mean() for y in ys])
    within = np.mean([y[:, i].var() for y in ys])
    between = per.var()
    frac = within / max(within + between, 1e-9)
    print(f"  {n:9s} within={within:.4f} between={between:.4f} "
          f"within_frac={frac:.2f}")

# Cheap signal check: correlation of audio embedding with each light dim,
# vs the same with y shuffled ACROSS songs (destroys pairing, keeps marginals).
print("\n--- audio->light signal vs shuffled control ---")
rng = np.random.default_rng(0)
sub = rng.choice(len(X), size=min(200_000, len(X)), replace=False)
Xs, Ys = X[sub], Y[sub]
Xc = Xs - Xs.mean(0)
Xc /= (Xs.std(0) + 1e-8)
perm = rng.permutation(len(Ys))
for i, n in enumerate(names):
    y = Ys[:, i]
    yc = (y - y.mean()) / (y.std() + 1e-8)
    r = np.abs(Xc.T @ yc / len(yc))
    ysh = Ys[perm, i]
    ysh = (ysh - ysh.mean()) / (ysh.std() + 1e-8)
    rsh = np.abs(Xc.T @ ysh / len(ysh))
    print(f"  {n:9s} max|r|={r.max():.3f} mean|r|={r.mean():.3f}   "
          f"shuffled max|r|={rsh.max():.3f}  ratio={r.mean()/max(rsh.mean(),1e-9):.2f}x")

json.dump({"n_shards": len(xs), "total_frames": int(frames.sum()),
           "bad": len(bad)}, open("data/cache/rpmc_audit.json", "w"), indent=2)
print("\nwrote data/cache/rpmc_audit.json")
