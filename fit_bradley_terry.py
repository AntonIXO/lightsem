#!/usr/bin/env python
"""H-L: Bradley-Terry over the conductor knobs from the pairwise ratings (PLAN.md 9.6, gate G9.6).

P(A > B) = sigmoid(w . (phi_A - phi_B)) on log-knob differences, ridge toward the defaults (w -> 0),
fitted separately for the two questions; 5-fold held-out accuracy on decided pairs (ties dropped),
the over-bright foils reported separately (a rater who picks the brighter clip fails the foil check),
and the fitted knob vector written out so its Light-BAS can be priced.

    bash run.sh fit_bradley_terry.py [--ratings outputs/stage9/hl_pairs/ratings.jsonl]
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ratings", default="/root/lightsem/outputs/stage9/hl_pairs/ratings.jsonl")
ap.add_argument("--lam", type=float, default=1.0)
a = ap.parse_args()
root = Path(a.ratings).parent; meta = json.loads((root / "pairs.json").read_text()); knobs = meta["knobs"]; pairs = {p["id"]: p for p in meta["pairs"]}
rows = [json.loads(l) for l in Path(a.ratings).read_text().splitlines() if l.strip()]
res = {"n_ratings": len(rows), "questions": {}}
for q in ("fits_music", "engineer"):
    X, y, foil_ok = [], [], []
    for r in rows:
        p = pairs[r["id"]]; ans = r[q]
        if ans not in ("a", "b"):
            continue
        d = np.array([np.log(p["phi"]["A"][k]) - np.log(p["phi"]["B"][k]) for k in knobs])
        X.append(d); y.append(1.0 if ans == "a" else 0.0)
        if p.get("foil_overbright"):
            foil_ok.append(ans.upper() != p["foil_overbright"])
    X, y = np.array(X), np.array(y)
    n = len(y)
    if n < 10:
        res["questions"][q] = {"n_decided": int(n), "note": "fewer than 10 decided pairs"}; continue

    def fit(Xt, yt):
        w = np.zeros(Xt.shape[1])
        for _ in range(500):                                   # gradient ascent with ridge
            p = 1 / (1 + np.exp(-(Xt @ w)))
            w += 0.1 * (Xt.T @ (yt - p) / len(yt) - a.lam * w / len(yt))
        return w
    rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(n), 5); acc = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f); w = fit(X[tr], y[tr])
        acc.append(float(((X[f] @ w > 0) == (y[f] > 0.5)).mean()))
    w = fit(X, y)
    fitted = {k: float(meta["defaults"][k] * np.exp(0.5 * w[i])) for i, k in enumerate(knobs)}   # move half a log-unit along w
    res["questions"][q] = {"n_decided": int(n), "heldout_accuracy": float(np.mean(acc)), "acc_ci_lo": float(np.mean(acc) - 1.96 * np.sqrt(np.mean(acc) * (1 - np.mean(acc)) / n)),
                           "w": dict(zip(knobs, np.round(w, 3).tolist())), "foil_pass_rate": float(np.mean(foil_ok)) if foil_ok else None, "fitted_knobs": fitted,
                           "gate_G9.6-acc": bool(np.mean(acc) > 0.60 and n >= 200)}
(root / "bradley_terry.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
