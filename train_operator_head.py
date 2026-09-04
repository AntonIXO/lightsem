#!/usr/bin/env python
"""HEAD B — the operator head: audio -> when a lighting operator changes the scene.

Stage 2's probe established the signal with a ridge on the corpus's openl3
embeddings (+0.31 lift over the rhythm baseline, z 3.66). This trains the
deployable version: the same causal Conformer used for the dance head, on the
product's own causal features, predicting the same validated target.

The competitor is not "no model". It is the rhythm of the operator's own past
changes, which needs no audio and is a real deployable system, plus the ridge
that already passed. A neural head that does not beat both is not worth its
latency.

    bash run_torch.sh train_operator_head.py --steps 6000
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lightsem.model.light_trunk import LightTrunk, TrunkConfig, causality_probe
from probe_event_target import (average_precision, event_train, hazard_feats, target)

ROOT = "/mnt/lightsem-data/cache/rpmc_causal"
OUT = Path("/root/lightsem/outputs/stage2")
CKPT = Path("/mnt/lightsem-data/ckpt")
WIN = 300          # 30 s at 10 Hz


def load_all():
    songs = []
    for p in sorted(glob.glob(f"{ROOT}/*.npz")):
        d = np.load(p, allow_pickle=False)
        x, y = d["x"].astype(np.float32), d["y"].astype(np.float32)
        if len(x) < WIN + 20 or not np.isfinite(x).all():
            continue
        tr, _ = event_train(y)
        songs.append({"key": Path(p).stem, "x": x, "y": y,
                      "haz": hazard_feats(tr).astype(np.float32),
                      "t": target(tr).astype(np.float32)})
    return songs


def split(songs, val=0.15, test=0.15):
    def b(k):
        return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8], "big") / 2 ** 64
    tr, va, te = [], [], []
    for s in songs:
        u = b(s["key"])
        (te if u < test else va if u < test + val else tr).append(s)
    return tr, va, te


def evaluate(model, songs, device, mu, sd, use_hazard):
    model.eval()
    lifts = []
    with torch.no_grad():
        for s in songs:
            if s["t"].sum() < 3:
                continue
            x = torch.from_numpy(s["x"]).to(device)[None]
            if use_hazard:
                h = torch.from_numpy(s["haz"]).to(device)[None]
                x = torch.cat([x, h], -1)
            x = (x - mu) / sd
            p = torch.sigmoid(model(x)["b_logit"][0, :, 0]).float().cpu().numpy()
            lifts.append(average_precision(p, s["t"]) / s["t"].mean())
    return np.array(lifts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--with-hazard", action="store_true",
                    help="give the model the rhythm features too (the fair upper form)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    songs = load_all()
    tr, va, te = split(songs)
    print(f"{len(songs)} songs: train {len(tr)} val {len(va)} test {len(te)}")

    in_dim = songs[0]["x"].shape[1] + (songs[0]["haz"].shape[1] if args.with_hazard else 0)
    cfg = TrunkConfig(in_dim=in_dim, d_model=192, n_blocks=4, attn_window=120,
                      n_a=1, n_b=1, n_c=1)
    model = LightTrunk(cfg).to(device)
    print(f"params {model.n_params()/1e6:.2f} M, in_dim {in_dim}")

    def batch(n, rng):
        xs, ys = [], []
        for _ in range(n):
            s = tr[rng.integers(0, len(tr))]
            i = rng.integers(0, len(s["x"]) - WIN)
            x = s["x"][i:i + WIN]
            if args.with_hazard:
                x = np.concatenate([x, s["haz"][i:i + WIN]], 1)
            xs.append(x); ys.append(s["t"][i:i + WIN])
        return (torch.from_numpy(np.stack(xs)).to(device),
                torch.from_numpy(np.stack(ys)).to(device))

    rng = np.random.default_rng(0)
    sample = np.concatenate([s["x"][:600] for s in tr[:200]])
    if args.with_hazard:
        sample = np.concatenate([np.concatenate([s["x"][:600], s["haz"][:600]], 1)
                                 for s in tr[:200]])
    mu = torch.from_numpy(sample.mean(0)).to(device)
    sd = torch.from_numpy(sample.std(0) + 1e-6).to(device)

    pos = float(np.concatenate([s["t"] for s in tr]).mean())
    pw = torch.tensor([(1 - pos) / pos], device=device)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                              pct_start=0.05)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        x, y = batch(args.batch, rng)
        x = (x - mu) / sd
        loss = F.binary_cross_entropy_with_logits(model(x)["b_logit"][:, :, 0], y,
                                                  pos_weight=pw)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:5d} loss {loss.item():.4f}  {time.time()-t0:.0f}s",
                  flush=True)

    lifts = evaluate(model, te, device, mu, sd, args.with_hazard)

    # Baselines fitted on TRAINING songs and applied to held-out ones. Fitting
    # the rhythm weights on the test song itself turns the baseline into an
    # oracle: it scored 2.79 that way against 1.24 when fitted honestly, which
    # would have made the model look far worse than it is.
    def fit_on(sgs, key):
        XtX = Xty = None
        for s in sgs:
            f = s[key].astype(np.float64)
            if XtX is None:
                XtX = np.zeros((f.shape[1],) * 2); Xty = np.zeros(f.shape[1])
            XtX += f.T @ f
            Xty += f.T @ s["t"]
        return np.linalg.solve(XtX + 30 * np.eye(XtX.shape[0]), Xty)

    w_h = fit_on(tr, "haz")
    w_x = fit_on(tr, "x")
    rhythm, ridge_audio = [], []
    for s in te:
        if s["t"].sum() < 3:
            continue
        rhythm.append(average_precision(s["haz"] @ w_h, s["t"]) / s["t"].mean())
        ridge_audio.append(average_precision(s["x"] @ w_x, s["t"]) / s["t"].mean())
    rhythm = np.array(rhythm)
    ridge_audio = np.array(ridge_audio)
    d = lifts - rhythm[: len(lifts)]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(6000)])
    res = {"n_songs": len(songs), "n_test": int(len(lifts)),
           "with_hazard": args.with_hazard, "params_m": model.n_params() / 1e6,
           "model_lift": float(lifts.mean()), "rhythm_lift": float(rhythm.mean()),
           "ridge_on_same_causal_features_lift": float(ridge_audio.mean()),
           "delta": float(d.mean()),
           "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           "ridge_reference_delta": 0.3145,
           "causality_leak": causality_probe(
               model, torch.randn(2, 400, in_dim, device=device), 150),
           "train_s": time.time() - t0}
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                "with_hazard": args.with_hazard},
               CKPT / f"operator_head{'_haz' if args.with_hazard else ''}.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "with_hazard" if args.with_hazard else "audio_only"
    (OUT / f"operator_head_model_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
