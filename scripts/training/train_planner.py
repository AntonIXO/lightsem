#!/usr/bin/env python
"""H-J: the section planner -- audio-only, bar-rate, offline -- writing knobs for the engine (PLAN.md 9.4).

One token per bar (four beats of the 10 Hz features' own beat phase), a small Transformer
over the whole song (may look ahead: the show is pre-rendered), trained on the AUTHORED corpora
to predict per bar what a designer decides before a section starts and holds:

    knobs = hue cos, hue sin, saturation, level class (bar-mean value, z-scored within the song),
            peak class (p90, z-scored), hue concentration, activity (value std in the bar,
            z-scored), look changes in the bar (log1p count)

Nothing per-frame: H-C refuted section channels INTO the per-frame model; this writes knobs the
engine applies on top of the decoder (scripts/legacy_stages/run_stage9_planner_eval.py). ``--causal`` trains the
control arm the gate needs next to the offline planner.

    bash run_torch.sh scripts/training/train_planner.py --steps 4000 --run-tag v1 [--causal]
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.diagnostics.probe_event_target import event_train
from scripts.training.train_group_decoder import CORPORA, load_corpus
from scripts.training.train_operator_head import split

CKPT = Path("/mnt/lightsem-data/ckpt"); OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9")
KNOBS = ("hue_cos", "hue_sin", "sat", "level_z", "peak_z", "hue_conc", "activity_z", "look_log1p")
MAX_BARS = 256


def bars_of(phase: np.ndarray, beats_per_bar: int = 4) -> np.ndarray:
    """Beat phase (T,) in [0,1) -> bar index per frame (wraps of the phase count beats)."""
    wraps = np.concatenate([[0], (np.diff(phase) < -0.5).astype(int)])
    beat_idx = np.cumsum(wraps)
    return beat_idx // beats_per_bar


def bar_pool(x: np.ndarray, bar: np.ndarray) -> np.ndarray:
    """(T, d) -> (n_bars, 3d) mean / std / max per bar."""
    n = int(bar.max()) + 1
    out = np.zeros((n, 3 * x.shape[1]), np.float32)
    for b in range(n):
        m = bar == b
        if m.any():
            v = x[m]; out[b] = np.concatenate([v.mean(0), v.std(0), v.max(0)])
    return out


def knobs_of(y: np.ndarray, ev: np.ndarray, bar: np.ndarray) -> np.ndarray:
    n = int(bar.max()) + 1
    k = np.zeros((n, len(KNOBS)), np.float32)
    lv = np.array([y[bar == b, 3].mean() if (bar == b).any() else 0 for b in range(n)])
    pk = np.array([y[bar == b, 4].mean() if (bar == b).any() else 0 for b in range(n)])
    act = np.array([y[bar == b, 3].std() if (bar == b).any() else 0 for b in range(n)])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-3)   # noqa: E731
    for b in range(n):
        m = bar == b
        if not m.any():
            continue
        k[b, 0:3] = y[m, 0:3].mean(0); k[b, 5] = y[m, 5].mean()
        k[b, 7] = np.log1p(ev[m].sum())
    k[:, 3] = z(lv); k[:, 4] = z(pk); k[:, 6] = z(act)
    return k


class Planner(nn.Module):
    def __init__(self, in_dim: int, d: int = 256, n_layers: int = 4, n_heads: int = 4, causal: bool = False):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.SiLU())
        self.pos = nn.Embedding(MAX_BARS, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d, len(KNOBS))
        self.causal = causal

    def forward(self, x, pad_mask=None):
        B, N, _ = x.shape
        h = self.inp(x) + self.pos(torch.arange(N, device=x.device))[None]
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), 1) if self.causal else None
        h = self.enc(h, mask=mask, src_key_padding_mask=pad_mask)
        return self.out(h)


def prepare(songs):
    for s in songs:
        bar = bars_of(s["phase"])
        s["bar"] = bar; s["bx"] = bar_pool(s["x"], bar)
        ev = event_train(s["y"])[0]
        s["knobs"] = knobs_of(s["y"], ev, bar)
    return songs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--corpora", default="rpmc,jd")
    ap.add_argument("--run-tag", default="v1")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-only", default="", help="checkpoint: skip training, report held-out metrics")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    songs = []
    for c in a.corpora.split(","):
        songs += load_corpus(c, a.limit)
    songs = [s for s in prepare(songs) if 8 <= len(s["bx"]) <= MAX_BARS]
    tr, va, te = split(songs)
    print(f"{len(songs)} songs, train {len(tr)} test {len(te)}, {time.time()-t0:.0f}s; causal={a.causal}", flush=True)
    sample = np.concatenate([s["bx"] for s in tr[:300]])
    mu = torch.from_numpy(sample.mean(0)).to(device); sd = torch.from_numpy(sample.std(0) + 1e-6).to(device)
    kv = np.concatenate([s["knobs"] for s in tr]).var(0) + 1e-3
    wk = torch.from_numpy((1 / kv) / (1 / kv).mean()).float().to(device)
    model = Planner(sample.shape[1], causal=a.causal).to(device)
    print(f"planner {sum(p.numel() for p in model.parameters())/1e6:.2f} M params", flush=True)
    if a.eval_only:
        b = torch.load(a.eval_only, map_location=device, weights_only=False); model.load_state_dict(b["model"])
        mu = torch.as_tensor(b["mu"], device=device); sd = torch.as_tensor(b["sd"], device=device); a.steps = 0
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05) if a.steps > 0 else None
    rng = np.random.default_rng(0)

    def batch(sgs):
        N = max(len(s["bx"]) for s in sgs)
        X = np.zeros((len(sgs), N, sample.shape[1]), np.float32); Y = np.zeros((len(sgs), N, len(KNOBS)), np.float32)
        M = np.ones((len(sgs), N), bool)
        for i, s in enumerate(sgs):
            n = len(s["bx"]); X[i, :n] = s["bx"]; Y[i, :n] = s["knobs"]; M[i, :n] = False
        return (torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device), torch.from_numpy(M).to(device))

    model.train()
    for step in range(a.steps):
        X, Y, M = batch([tr[i] for i in rng.integers(0, len(tr), a.batch)])
        P = model((X - mu) / sd, M)
        loss = ((((P - Y) ** 2) * wk).mean(-1) * (~M)).sum() / (~M).sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == a.steps - 1:
            print(f"  step {step:5d} loss {loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
    model.eval(); res = {"causal": a.causal, "corpora": a.corpora, "n_songs": len(songs), "per_corpus": {}}
    with torch.no_grad():
        for cname in a.corpora.split(","):
            tes = [s for s in te if s["corpus"] == cname]
            r2 = {k: [] for k in KNOBS}; r2_shuf = {k: [] for k in KNOBS}; Pall, Yall, Sall = [], [], []
            for s in tes:
                X, Y, M = batch([s]); P = model((X - mu) / sd, M)[0].cpu().numpy(); Yn = s["knobs"]
                # shuffled-section control: the prediction permuted in 8-bar blocks
                nb = len(Yn); blocks = np.array_split(np.arange(nb), max(nb // 8, 1)); perm = rng.permutation(len(blocks))
                Ps = np.concatenate([P[blocks[j]] for j in perm]); Pall.append(P); Yall.append(Yn); Sall.append(Ps)
                for j, k in enumerate(KNOBS):
                    ss = ((Yn[:, j] - Yn[:, j].mean()) ** 2).sum() + 1e-6
                    r2[k].append(1 - ((Yn[:, j] - P[:, j]) ** 2).sum() / ss)
                    r2_shuf[k].append(1 - ((Yn[:, j] - Ps[:, j]) ** 2).sum() / ss)
            Pa, Ya, Sa = np.concatenate(Pall), np.concatenate(Yall), np.concatenate(Sall)
            pooled = {k: float(1 - ((Ya[:, j] - Pa[:, j]) ** 2).sum() / (((Ya[:, j] - Ya[:, j].mean()) ** 2).sum() + 1e-6)) for j, k in enumerate(KNOBS)}
            pooled_shuf = {k: float(1 - ((Ya[:, j] - Sa[:, j]) ** 2).sum() / (((Ya[:, j] - Ya[:, j].mean()) ** 2).sum() + 1e-6)) for j, k in enumerate(KNOBS)}
            res["per_corpus"][cname] = {"n_test": len(tes), "r2": {k: float(np.mean(v)) for k, v in r2.items()},
                                        "r2_shuffled_sections": {k: float(np.mean(v)) for k, v in r2_shuf.items()},
                                        "r2_pooled": pooled, "r2_pooled_shuffled_sections": pooled_shuf}
            print(f"  [{cname}] held-out bar-knob R2 pooled over bars (shuffled-section control): " + " ".join(f"{k} {pooled[k]:+.2f}/{pooled_shuf[k]:+.2f}" for k in KNOBS), flush=True)
    tag = a.run_tag + ("_causal" if a.causal else "")
    if a.eval_only:
        (OUT / f"planner_{tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["per_corpus"], indent=1)); return
    torch.save({"model": model.state_dict(), "in_dim": sample.shape[1], "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                "causal": a.causal, "knobs": KNOBS}, CKPT / f"planner_{tag}.pt")
    (OUT / f"planner_{tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
