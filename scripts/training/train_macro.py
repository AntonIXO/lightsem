"""Train the LightSem macro layer.

Objective: predict the per-song-CENTRED scene vector at multiple horizons from
a causal 6.4 s window of openl3. Centring is applied to the TARGET during
training so the model spends no capacity on venue palette identity -- that is
given by the show, not predicted from audio.

Metric: R2_within on held-out SONGS. Baselines (data/cache/baselines.json):
    per-song-mean oracle   0.0000   <- zero point
    best fair linear      -0.1091   <- linear cannot beat holding the mean
So any R2_within > 0 is real skill. We report test only once, at the end.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from lightsem.model.data import TARGET_NAMES, load_songs, split_songs
from lightsem.model.encoder import MacroEncoder

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tensors(songs, window, horizons, device):
    """Pack songs into padded GPU tensors + a valid (song, end) index."""
    xs = [torch.from_numpy(s.x) for s in songs]
    ys = [torch.from_numpy(s.y - s.y.mean(0, keepdims=True)) for s in songs]  # CENTRED
    lens = [len(a) for a in xs]
    T = max(lens)
    X = torch.zeros(len(xs), T, xs[0].shape[1])
    Y = torch.zeros(len(ys), T, ys[0].shape[1])
    for i, (a, b) in enumerate(zip(xs, ys)):
        X[i, : len(a)] = a
        Y[i, : len(b)] = b
    hmax = max(horizons)
    idx = [(i, e) for i, L in enumerate(lens)
           for e in range(window, L - hmax + 1)]
    return X.to(device), Y.to(device), torch.tensor(idx, dtype=torch.long, device=device)


def gather_batch(X, Y, idx, sel, window, horizons):
    si, ei = idx[sel, 0], idx[sel, 1]
    ar = torch.arange(window, device=X.device)
    xw = X[si[:, None], (ei[:, None] - window + ar[None, :])]        # [B,W,512]
    tg = torch.stack([Y[si, ei - 1 + h] for h in horizons], 1)       # [B,H,6]
    return xw, tg


@torch.no_grad()
def eval_r2(model, songs, window, horizons, device, bs=4096):
    """R2_within per horizon per target, evaluated on whole songs."""
    model.eval()
    hmax = max(horizons)
    num = torch.zeros(len(horizons), len(TARGET_NAMES), device=device)
    den = torch.zeros(len(TARGET_NAMES), device=device)
    for s in songs:
        L = len(s.x)
        if L < window + hmax:
            continue
        x = torch.from_numpy(s.x).to(device)
        yc = torch.from_numpy(s.y - s.y.mean(0, keepdims=True)).to(device)
        ends = torch.arange(window, L - hmax + 1, device=device)
        preds = []
        for i in range(0, len(ends), bs):
            e = ends[i : i + bs]
            ar = torch.arange(window, device=device)
            xw = x[(e[:, None] - window + ar[None, :])]
            p, _ = model(xw)
            preds.append(p)
        P = torch.cat(preds)                                   # [N,H,6]
        # de-mean predictions per song, mirroring the baseline treatment
        P = P - P.mean(0, keepdim=True)
        for hi, h in enumerate(horizons):
            tgt = yc[ends - 1 + h]
            tgt = tgt - tgt.mean(0, keepdim=True)
            num[hi] += ((tgt - P[:, hi]) ** 2).sum(0)
            if hi == 0:
                den += (tgt ** 2).sum(0)
    r2 = 1.0 - num / den.clamp_min(1e-9)
    model.train()
    return r2.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=64)      # 6.4 s @10 Hz
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--horizons", type=str, default="0,10,20,40")  # frames @10Hz
    ap.add_argument("--out", type=str, default="data/cache/macro_model.pt")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"device={DEV} horizons={horizons} (= {[h/10 for h in horizons]} s)")

    songs = load_songs()
    tr, va, te = split_songs(songs)
    print(f"songs train={len(tr)} val={len(va)} test={len(te)}")

    Xtr, Ytr, Itr = build_tensors(tr, args.window, horizons, DEV)
    print(f"train windows: {len(Itr):,}  gpu tensors {Xtr.nbytes/1e9:.2f} GB")

    model = MacroEncoder(d_model=args.d_model, n_layers=args.layers,
                         horizons=tuple(horizons)).to(DEV)
    print(f"params: {model.n_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.15)
    lossf = nn.SmoothL1Loss(beta=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")

    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project="lightsem-macro", config=vars(args))
        except Exception as e:
            print("wandb off:", e)

    best, best_ep, hist = -9e9, -1, []
    t0 = time.time()
    for ep in range(args.epochs):
        tot = 0.0
        for _ in range(args.steps_per_epoch):
            sel = torch.randint(0, len(Itr), (args.batch,), device=DEV)
            xw, tg = gather_batch(Xtr, Ytr, Itr, sel, args.window, horizons)
            with torch.amp.autocast("cuda", enabled=DEV.type == "cuda"):
                pred, _ = model(xw)
                loss = lossf(pred, tg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()

        r2 = eval_r2(model, va, args.window, horizons, DEV)
        m0 = float(r2[0].mean())          # horizon 0 = "now"
        mall = float(r2.mean())
        hist.append({"epoch": ep, "loss": tot / args.steps_per_epoch,
                     "val_r2_h0": m0, "val_r2_all": mall})
        flag = ""
        if m0 > best:
            best, best_ep = m0, ep
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "horizons": horizons}, args.out)
            flag = " *"
        print(f"ep {ep:3d} loss {tot/args.steps_per_epoch:.4f} "
              f"val_R2_now {m0:+.4f} val_R2_allh {mall:+.4f}{flag}", flush=True)
        if run:
            run.log({"loss": tot / args.steps_per_epoch, "val_r2_h0": m0,
                     "val_r2_all": mall, "epoch": ep})

    print(f"\nbest val R2_now {best:+.4f} @ epoch {best_ep}  ({time.time()-t0:.0f}s)")

    # ---- single final test evaluation with the best checkpoint ----
    ck = torch.load(args.out, weights_only=False)
    model.load_state_dict(ck["model"])
    r2te = eval_r2(model, te, args.window, horizons, DEV)
    print(f"\n{'horizon':>9s} " + " ".join(f"{n:>9s}" for n in TARGET_NAMES) + "     MEAN")
    print("-" * 82)
    for hi, h in enumerate(horizons):
        print(f"{h/10:6.1f} s  " + " ".join(f"{x:9.4f}" for x in r2te[hi])
              + f"  {r2te[hi].mean():7.4f}")
    base = json.load(open("data/cache/baselines.json"))
    print(f"\nTEST R2_within (now)   = {r2te[0].mean():+.4f}")
    print(f"per-song-mean oracle   = {np.mean(base['per_song_mean_oracle']):+.4f}  <- zero point")
    print(f"best fair linear       = {np.mean(base['ridge_centred_1frame']):+.4f}")

    json.dump({"val_hist": hist, "best_val_r2_h0": best,
               "test_r2": r2te.tolist(), "horizons": horizons,
               "targets": TARGET_NAMES},
              open("data/cache/macro_results.json", "w"), indent=2)
    print("\nwrote data/cache/macro_results.json")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
