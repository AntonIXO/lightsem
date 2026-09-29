#!/usr/bin/env python
"""H-O, the deployable form of the with-hazard operator head: does its 3.17x cue lift on Just Dance
visual survive when the hazard features come from its OWN emitted cues instead of the corpus's?

Held-out songs (train_operator_head's hash split, the exp-016 test set), three scores per song:
  true hazard   the checkpoint as evaluated in exp-016 (teacher-forced rhythm features; the ceiling)
  own cues      autoregressive in 1 s blocks: cues fire when the head's probability crosses a
                threshold calibrated on TRAINING songs to the corpus cue rate, refractory 2 s, and
                the rhythm features are recomputed from those cues
  audio only    the audio-only checkpoint on the same songs (the floor)

    bash run_torch.sh scripts/diagnostics/eval_head_own_cues.py --root /mnt/lightsem-data/cache/justdance_visual --tag jdvis
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from lightsem.model.light_trunk import LightTrunk, TrunkConfig
from scripts.diagnostics.probe_event_target import average_precision, hazard_feats
from scripts.training.train_operator_head import load_all, split

CKPT = Path("/mnt/lightsem-data/ckpt"); OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9")


def load(ckpt, device):
    b = torch.load(ckpt, map_location=device, weights_only=False)
    m = LightTrunk(TrunkConfig(**b["cfg"])).to(device).eval(); m.load_state_dict(b["model"])
    return m, torch.as_tensor(b["mu"], device=device), torch.as_tensor(b["sd"], device=device), bool(b.get("with_hazard"))


@torch.no_grad()
def score(m, mu, sd, x, haz, device):
    xt = torch.from_numpy(np.concatenate([x, haz], 1) if haz is not None else x).to(device)[None]
    return torch.sigmoid(m((xt - mu) / sd)["b_logit"][0, :, 0]).cpu().numpy()


@torch.no_grad()
def own_cues(m, mu, sd, x, device, thr, block=10, refractory=20):
    T = len(x); cues = np.zeros(T, np.float32); out = np.zeros(T, np.float32); last = -10 ** 6
    for b0 in range(0, T, block):
        hz = hazard_feats(cues).astype(np.float32)
        p = score(m, mu, sd, x[:b0 + block], hz[:b0 + block], device)[b0:b0 + block]
        out[b0:b0 + block] = p
        for j, v in enumerate(p):
            if v > thr and (b0 + j) - last >= refractory:
                cues[b0 + j] = 1.0; last = b0 + j
    return out, cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/lightsem-data/cache/justdance_visual")
    ap.add_argument("--tag", default="jdvis")
    a = ap.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"
    songs = load_all(a.root); tr, va, te = split(songs)
    mh, muh, sdh, wh = load(CKPT / f"operator_head_haz_{a.tag}.pt", device); assert wh
    ma, mua, sda, _ = load(CKPT / f"operator_head{'_' + a.tag if a.tag != 'rpmc' else ''}.pt", device) if (CKPT / f"operator_head_{a.tag}.pt").exists() else load(CKPT / "operator_head.pt", device)
    # calibrate the own-cue threshold on training songs to the corpus cue rate (cues per minute of the event train)
    rate = np.mean([(np.diff(np.concatenate([[0], s["t"]])) > 0).sum() / (len(s["t"]) / 600) for s in tr])
    thrs = np.linspace(0.1, 0.9, 9); rates = []
    for t in thrs:
        rates.append(np.mean([own_cues(mh, muh, sdh, s["x"], device, t)[1].sum() / (len(s["x"]) / 600) for s in tr[:10]]))
    thr = float(thrs[int(np.argmin(np.abs(np.array(rates) - rate)))])
    print(f"corpus cue rate {rate:.2f}/min -> threshold {thr:.2f}", flush=True)
    rows = {"true_hazard": [], "own_cues": [], "audio_only": [], "own_cue_rate": []}
    for s in te:
        if s["t"].sum() < 3:
            continue
        base = s["t"].mean()
        rows["true_hazard"].append(average_precision(score(mh, muh, sdh, s["x"], s["haz"], device), s["t"]) / base)
        p, c = own_cues(mh, muh, sdh, s["x"], device, thr)
        rows["own_cues"].append(average_precision(p, s["t"]) / base); rows["own_cue_rate"].append(c.sum() / (len(c) / 600))
        rows["audio_only"].append(average_precision(score(ma, mua, sda, s["x"], None, device), s["t"]) / base)
    rng = np.random.default_rng(0)
    d = np.array(rows["own_cues"]) - np.array(rows["audio_only"]); bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    res = {"tag": a.tag, "n_test": len(rows["own_cues"]), "threshold": thr, "corpus_cue_rate": float(rate),
           "lift": {k: float(np.mean(v)) for k, v in rows.items()}, "own_minus_audio": {"delta": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}}
    (OUT / f"head_own_cues_{a.tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
