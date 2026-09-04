#!/usr/bin/env python
"""Stage 1 — train the three-block audio -> descriptor model (runbook rb-ls2-01).

    bash run_torch.sh train_light.py --steps 8000
    bash run_torch.sh train_light.py --eval-only --ckpt /mnt/.../light_trunk.pt

Blocks and objectives, per PLAN.md section 3:
  A  deterministic skeleton   L1 on corpus-z-scored channels
  B  events                   BCE hazard with positive weighting
  C  style residual           conditional rectified flow (sampled, never averaged)

Two baselines run in the same harness because a number without them means
nothing here: an MSE model over A and C together (the collapse control the plan
demands) and the per-channel training mean (R2 = 0 by construction).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lightsem.model.light_data import (BLOCK_A, BLOCK_A_V2, BLOCK_B, BLOCK_C,
                                       BLOCK_C_V2, load_tensor, make_splits)
from lightsem.model.light_trunk import LightTrunk, TrunkConfig, causality_probe

CKPT_DIR = Path("/mnt/lightsem-data/ckpt")
OUT = Path("/root/lightsem/outputs/stage1")
WIN = 160


class Windows:
    def __init__(self, z, meta, idx, win=WIN, a_cols=None, c_cols=None):
        self.X, self.B = z["X"], z["B"]
        if a_cols is None:
            self.A, self.C = z["A"], z["C"]
        else:
            AC = np.concatenate([z["A"], z["C"]], axis=1)
            self.A, self.C = AC[:, a_cols], AC[:, c_cols]
        cid = z["clip_id"]
        starts = np.zeros(len(meta), dtype=np.int64)
        off = 0
        for i, m in enumerate(meta):
            starts[i] = off
            off += m["T"]
        self.spans = [(starts[i], starts[i] + meta[i]["T"]) for i in idx
                      if meta[i]["T"] >= win]
        self.win = win

    def batch(self, n, rng, device):
        xs, as_, bs, cs = [], [], [], []
        for _ in range(n):
            s, e = self.spans[rng.integers(0, len(self.spans))]
            i = rng.integers(s, e - self.win + 1)
            sl = slice(i, i + self.win)
            xs.append(self.X[sl]); as_.append(self.A[sl])
            bs.append(self.B[sl]); cs.append(self.C[sl])
        t = lambda a: torch.from_numpy(np.stack(a)).to(device)   # noqa: E731
        return t(xs), t(as_), t(bs), t(cs)

    def all_windows(self, stride=WIN):
        for s, e in self.spans:
            for i in range(s, e - self.win + 1, stride):
                yield i


def evaluate(model, z, meta, idx, device, mu, sd, n_max=400, rng=None,
             flow_samples=4, a_names=BLOCK_A, a_cols=None, c_cols=None):
    """R2 per block-A channel, event AP, and block-C sample diversity."""
    rng = rng or np.random.default_rng(0)
    w = Windows(z, meta, idx, a_cols=a_cols, c_cols=c_cols)
    if not w.spans:
        return None
    starts = list(w.all_windows(stride=WIN))
    if len(starts) > n_max:
        starts = [starts[i] for i in rng.permutation(len(starts))[:n_max]]
    model.eval()
    A_true, A_pred, B_true, B_prob, C_samps, C_true = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 32):
            chunk = starts[i:i + 32]
            X = torch.from_numpy(np.stack([z["X"][s:s + WIN] for s in chunk])).to(device)
            X = (X - mu) / sd
            out = model(X)
            A_pred.append(out["a"].float().cpu().numpy())
            A_true.append(np.stack([w.A[s:s + WIN] for s in chunk]))
            B_prob.append(torch.sigmoid(out["b_logit"]).float().cpu().numpy())
            B_true.append(np.stack([z["B"][s:s + WIN] for s in chunk]))
            C_true.append(np.stack([w.C[s:s + WIN] for s in chunk]))
            if hasattr(model, "head_c_mse"):
                # The collapse control: an MSE head is deterministic, so every
                # "sample" is the same look. Sampling the untrained flow head
                # here instead would report random noise as diversity.
                det = model.head_c_mse(out["h"]).float().cpu().numpy()
                C_samps.append(np.stack([det for _ in range(flow_samples)]))
            else:
                C_samps.append(np.stack([model.head_c.sample(out["h"]).float().cpu().numpy()
                                         for _ in range(flow_samples)]))
    A_true = np.concatenate(A_true); A_pred = np.concatenate(A_pred)
    B_true = np.concatenate(B_true); B_prob = np.concatenate(B_prob)
    C_true = np.concatenate(C_true)
    C_samps = np.concatenate(C_samps, axis=1)          # (S, N, T, C)

    r2 = {}
    for ci, name in enumerate(a_names):
        yt = A_true[:, :, ci].ravel(); yp = A_pred[:, :, ci].ravel()
        ss = ((yt - yp) ** 2).sum(); st = ((yt - yt.mean()) ** 2).sum()
        r2[name] = float(1 - ss / max(st, 1e-9))

    ap = {}
    for bi, name in enumerate(BLOCK_B):
        yt = B_true[:, :, bi].ravel().astype(int); yp = B_prob[:, :, bi].ravel()
        tol = 8                                          # +-0.2 s at 40 fps
        yt_dil = np.zeros_like(yt)
        pos = np.flatnonzero(yt)
        for p in pos:
            yt_dil[max(p - tol, 0):p + tol + 1] = 1
        order = np.argsort(-yp)
        tp = np.cumsum(yt_dil[order]); fp = np.cumsum(1 - yt_dil[order])
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(tp[-1], 1)
        ap[name] = float(np.sum(np.diff(np.r_[0, rec]) * prec))
        ap[name + "_base"] = float(yt_dil.mean())
    div = float(np.mean([np.linalg.norm(C_samps[a] - C_samps[b], axis=-1).mean()
                         for a in range(len(C_samps)) for b in range(a + 1, len(C_samps))]))
    div_real = float(np.linalg.norm(C_true[1:] - C_true[:-1], axis=-1).mean())
    return {"r2_block_a": r2, "r2_a_mean": float(np.mean(list(r2.values()))),
            "event_ap": ap, "div_flow_samples": div, "div_real_neighbour": div_real,
            "n_windows": len(starts)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--steps", type=int, default=8000)
    ap_.add_argument("--batch", type=int, default=48)
    ap_.add_argument("--lr", type=float, default=3e-4)
    ap_.add_argument("--baseline-mse", action="store_true",
                     help="train A and C with one MSE head: the collapse control")
    ap_.add_argument("--tag", default="v1")
    ap_.add_argument("--eval-only", default="")
    ap_.add_argument("--v2", action="store_true",
                     help="measurement-driven block split + autoregressive flow head")
    ap_.add_argument("--offline", action="store_true",
                     help="H-A: symmetric attention window and conv padding; the whole-track model")
    ap_.add_argument("--tensor", default="",
                     help="training tensor path (default cache/train_tensor.npz; H-A uses the offline one)")
    args = ap_.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z, meta = load_tensor(Path(args.tensor) if args.tensor else None)
    if meta and meta[0].get("corpus") == "jd":
        from lightsem.model.jd_data import make_jd_splits
        sp = make_jd_splits(meta)                 # by song key; test_dancer holds the held-out songs
    else:
        sp = make_splits(meta)
    print({k: len(getattr(sp, k)) for k in ("train", "val_track", "test_genre", "test_dancer")})
    all_names = list(BLOCK_A) + list(BLOCK_C)
    if args.v2:
        a_names, c_names = list(BLOCK_A_V2), list(BLOCK_C_V2)
    else:
        a_names, c_names = list(BLOCK_A), list(BLOCK_C)
    a_cols = np.array([all_names.index(n) for n in a_names])
    c_cols = np.array([all_names.index(n) for n in c_names])
    tr = Windows(z, meta, sp.train, a_cols=a_cols, c_cols=c_cols)
    rng = np.random.default_rng(0)

    sub = rng.integers(0, len(z["X"]), 200000)
    mu = torch.from_numpy(z["X"][sub].mean(0)).to(device)
    sd = torch.from_numpy(z["X"][sub].std(0) + 1e-6).to(device)

    cfg = TrunkConfig(in_dim=z["X"].shape[1], n_a=len(a_names), n_b=len(BLOCK_B),
                      n_c=len(c_names), ar_flow=args.v2, offline=args.offline)
    model = LightTrunk(cfg).to(device)
    if args.baseline_mse:
        model.head_c_mse = torch.nn.Linear(cfg.d_model, cfg.n_c).to(device)
    print(f"params {model.n_params()/1e6:.1f} M on {device}")

    pos = np.clip(z["B"].mean(0), 1e-4, 1)
    pw = torch.tensor((1 - pos) / pos, dtype=torch.float32, device=device)
    if args.eval_only:
        blob = torch.load(args.eval_only, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        mu = torch.as_tensor(blob["mu"], device=device)
        sd = torch.as_tensor(blob["sd"], device=device)
        args.steps = 0

    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                                total_steps=max(args.steps, 1),
                                                pct_start=0.05)
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        X, A, B, C = tr.batch(args.batch, rng, device)
        X = (X - mu) / sd
        out = model(X)
        la = F.l1_loss(out["a"], A)
        lb = F.binary_cross_entropy_with_logits(out["b_logit"], B, pos_weight=pw)
        lc = (F.mse_loss(model.head_c_mse(out["h"]), C) if args.baseline_mse
              else model.head_c.loss(out["h"], C))
        loss = la + 0.3 * lb + lc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}  A {la.item():.4f} "
                  f"B {lb.item():.4f} C {lc.item():.4f}  {time.time()-t0:.0f}s", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ck = CKPT_DIR / f"light_trunk_{args.tag}.pt"
    if not args.eval_only:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                    "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                    "block_names": {"a": a_names, "c": c_names}}, ck)

    x_probe = torch.from_numpy(np.stack([z["X"][s:s + WIN] for s in
                                         list(tr.all_windows())[:4]])).to(device)
    leak = causality_probe(model, (x_probe - mu) / sd, WIN // 2)

    res = {"tag": args.tag, "steps": args.steps, "baseline_mse": args.baseline_mse,
           "v2": args.v2, "offline": args.offline, "tensor": args.tensor or "default",
           "block_a": a_names, "block_c": c_names,
           "params_m": model.n_params() / 1e6, "causality_leak": leak,
           "train_s": time.time() - t0,
           "splits": {k: int(len(getattr(sp, k)))
                      for k in ("train", "val_track", "test_genre", "test_dancer")}}
    for name in ("train", "val_track", "test_genre", "test_dancer"):
        idx = getattr(sp, name)
        res[name] = evaluate(model, z, meta, idx, device, mu, sd, rng=rng,
                             a_names=a_names, a_cols=a_cols, c_cols=c_cols)
        if res[name]:
            print(f"{name:12s} R2(A) {res[name]['r2_a_mean']:+.4f}  "
                  f"AP {[round(res[name]['event_ap'][b],3) for b in BLOCK_B]}  "
                  f"div {res[name]['div_flow_samples']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"train_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"causality leak {leak}  -> {OUT}/train_{args.tag}.json")


if __name__ == "__main__":
    main()
