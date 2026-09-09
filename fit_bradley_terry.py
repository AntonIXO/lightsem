#!/usr/bin/env python
"""H-L preferences on CPU, validated on held-out canonical song groups.

Brightness foils and ties are reported separately from the binary fit. Original
log-knob features and ridge strength are retained. Fitted settings are candidates;
fresh-render preference and BAS gates must still be measured (PLAN.md 9.6).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

QUESTIONS = ("fits_music", "engineer")
DEFAULT_RATINGS = Path("/root/lightsem/outputs/stage9/hl_pairs/ratings.jsonl")
DEFAULT_SONG_MAP = Path("/mnt/lightsem-data/cache/justdance_songs.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(ratings_path, song_map_path=None):
    meta = json.loads((Path(ratings_path).parent / "pairs.json").read_text())
    defaults = np.array([meta["defaults"][k] for k in meta["knobs"]], float)
    if not np.isfinite(defaults).all() or (defaults <= 0).any():
        raise ValueError("default log-knob values must be finite and positive")
    pairs = {p["id"]: p for p in meta["pairs"]}
    if len(pairs) != len(meta["pairs"]):
        raise ValueError("duplicate pair IDs in pairs.json")
    song_map = {}
    if song_map_path and Path(song_map_path).exists():
        for row in json.loads(Path(song_map_path).read_text())["rows"]:
            key = row["song_key"]
            if row["yt_id"] in song_map and song_map[row["yt_id"]] != key:
                raise ValueError("conflicting canonical song IDs")
            song_map[row["yt_id"]] = key
    groups, unresolved, recording_keys = {}, set(), {}
    for pid, pair in pairs.items():
        foil = pair.get("foil_overbright")
        if foil is not None:
            foil = str(foil).strip().upper()
            if foil not in ("A", "B"):
                raise ValueError(f"{pid}: foil_overbright must be A, B or null")
            pair["foil_overbright"] = foil
        explicit = pair.get("song_key")
        canonical = song_map.get(pair.get("music_id"))
        if explicit and canonical and explicit != canonical:
            raise ValueError(f"{pid}: explicit song_key conflicts with canonical song map")
        key = explicit or canonical
        mid = pair.get("music_id")
        if mid and key:
            if mid in recording_keys and recording_keys[mid] != key:
                raise ValueError(f"{pid}: conflicting song keys for one recording")
            recording_keys[mid] = key
        if key:
            groups[pid] = "song:" + key
        else:
            if not pair.get("music_id"):
                raise ValueError(f"{pid}: missing music_id / song_key")
            groups[pid] = "recording:" + pair["music_id"]
            unresolved.add(pid)
        for side in ("A", "B"):
            values = np.array([pair["phi"][side][k] for k in meta["knobs"]], float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{pid}: log-knob values must be finite and positive")
    rows, seen = [], set()
    for line, text in enumerate(Path(ratings_path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if row["id"] not in pairs or not row.get("rater"):
            raise ValueError(f"rating line {line}: unknown pair or missing rater")
        identity = (row["rater"], row["id"])
        if identity in seen:
            raise ValueError(f"rating line {line}: duplicate rater/pair {identity}")
        seen.add(identity)
        for question in QUESTIONS:
            answer = str(row.get(question, "")).strip().lower()
            if answer not in ("a", "b", "="):
                raise ValueError(f"rating line {line}: invalid {question} answer")
            row[question] = answer
        rows.append(row)
    return meta, pairs, rows, groups, sorted(unresolved & {r["id"] for r in rows})


def fit_weights(X, y, lam=1.0):
    """Summed logistic loss + lambda/2 ||w||², matching the original objective."""
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("lambda must be finite and positive")

    def objective(w):
        score = X @ w
        loss = np.sum(np.logaddexp(0.0, score) - y * score) + lam * (w @ w) / 2
        grad = X.T @ (expit(score) - y) + lam * w
        return loss, grad

    fit = minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-9, "ftol": 1e-12, "maxiter": 500})
    if not fit.success:
        raise RuntimeError(f"preference fit did not converge: {fit.message}")
    return fit.x


def grouped_folds(groups, seed=0):
    unique = np.array(sorted(set(groups)))
    if len(unique) < 2:
        return []
    shuffled = np.random.default_rng(seed).permutation(unique)
    groups = np.asarray(groups)
    return [np.flatnonzero(np.isin(groups, held))
            for held in np.array_split(shuffled, min(5, len(unique)))]


def correctness(probability, labels):
    # Exact indifference receives chance credit instead of an A/B side bias.
    return np.where(np.isclose(probability, 0.5, rtol=0, atol=1e-12), 0.5,
                    ((probability > 0.5) == (labels > 0.5)).astype(float))


def cluster_interval(values, groups, seed=0, draws=2000):
    """Song-bootstrap interval conditional on fixed OOF predictions, no refit."""
    unique = sorted(set(groups))
    groups = np.asarray(groups)
    sums = np.array([values[groups == g].sum() for g in unique])
    counts = np.array([sum(groups == g) for g in unique])
    sampled = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    means = sums[sampled].sum(1) / counts[sampled].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def analyse_question(meta, pairs, rows, groups, question, lam=1.0, seed=0):
    decided, foils = [], []
    for row in rows:
        if pairs[row["id"]].get("foil_overbright"):
            foils.append(row)
        elif row[question] in ("a", "b"):
            decided.append(row)
    foil_decided = [r for r in foils if r[question] in ("a", "b")]
    result = {
        "answer_counts": dict(Counter(r[question] for r in rows)),
        "n_decided": len(decided), "n_unique_pairs": len({r["id"] for r in decided}),
        "n_song_groups": len({groups[r["id"]] for r in decided}),
        "n_foil_rows": len(foils), "n_foil_decided": len(foil_decided),
        "foil_pass_rate": float(np.mean([r[question].upper() != pairs[r["id"]]["foil_overbright"]
                                        for r in foil_decided])) if foil_decided else None,
        "gate_G9.6-acc": False,
    }
    if len(decided) < 10 or result["n_song_groups"] < 2:
        result["note"] = "Need 10 non-foil decided ratings and two song groups to fit."
        return result
    knobs = meta["knobs"]
    X = np.array([[np.log(pairs[r["id"]]["phi"]["A"][k]) -
                   np.log(pairs[r["id"]]["phi"]["B"][k]) for k in knobs] for r in decided])
    y = np.array([r[question] == "a" for r in decided], float)
    g = np.array([groups[r["id"]] for r in decided])
    scores, baseline = np.empty(len(y)), np.empty(len(y))
    fold_rows = []
    for fold_id, test in enumerate(grouped_folds(g, seed)):
        train = np.setdiff1d(np.arange(len(y)), test)
        w = fit_weights(X[train], y[train], lam)
        scores[test] = X[test] @ w
        baseline[test] = float(y[train].mean())
        fold_rows.append({"fold": fold_id, "train_songs": sorted(set(g[train])),
                          "test_songs": sorted(set(g[test])),
                          "test_pair_ids": [decided[i]["id"] for i in test]})
    predictions = expit(scores)
    correct = correctness(predictions, y)
    w = fit_weights(X, y, lam)
    candidate = {}
    for j, k in enumerate(knobs):
        observed = [pairs[r["id"]]["phi"][side][k] for r in decided for side in ("A", "B")]
        # Original half-log-unit proposal, bounded to settings actually shown.
        log_value = np.log(meta["defaults"][k]) + 0.5 * w[j]
        candidate[k] = float(np.exp(np.clip(log_value, np.log(min(observed)), np.log(max(observed)))))
    acc = float(correct.mean())
    result.update({
        "heldout_accuracy": acc,
        "song_macro_accuracy": float(np.mean([correct[g == key].mean() for key in sorted(set(g))])),
        "chance_accuracy": 0.5,
        "train_fold_majority_accuracy": float(correctness(baseline, y).mean()),
        "conditional_oof_song_bootstrap_ci95": cluster_interval(correct, g, seed),
        "oof_log_loss": float(np.mean(np.logaddexp(0, scores) - y * scores)),
        "w": dict(zip(knobs, w.tolist())), "fitted_knobs": candidate, "candidate_only": True,
        "gate_G9.6-acc": bool(acc > 0.60 and result["n_decided"] >= 200 and
                              all(key.startswith("song:") for key in g)),
        "folds": fold_rows,
        "out_of_fold": [{"id": row["id"], "rater": row["rater"], "song_group": g[i],
                         "answer": row[question], "probability_a": float(predictions[i]),
                         "correct": float(correct[i])} for i, row in enumerate(decided)],
    })
    return result


def analyse(ratings_path, song_map_path=DEFAULT_SONG_MAP, lam=1.0, seed=0):
    meta, pairs, rows, groups, unresolved = load_inputs(ratings_path, song_map_path)
    question_results = {q: analyse_question(meta, pairs, rows, groups, q, lam, seed) for q in QUESTIONS}
    return {
        "schema_version": 2, "n_ratings": len(rows), "raters": dict(Counter(r["rater"] for r in rows)),
        "n_unique_pairs": len({r["id"] for r in rows}),
        "unrated_pair_ids": sorted(set(pairs) - {r["id"] for r in rows}),
        "canonical_song_unresolved_pair_ids": unresolved,
        "question_agreement_count": sum(r[QUESTIONS[0]] == r[QUESTIONS[1]] for r in rows),
        "questions": question_results,
        "protocol": {"validation": "five folds grouped by canonical song; seed fixed before fitting",
                     "lambda": lam, "seed": seed, "foil_policy": "excluded from fit and validation",
                     "ties": "reported, excluded from binary fit",
                     "interval": "2000 song bootstrap draws conditional on fixed OOF predictions; no model refit",
                     "generalisation": "new songs for observed raters; not new-rater validation"},
        "gates": {"G9.6-acc": any(q["gate_G9.6-acc"] for q in question_results.values()),
                  "G9.6-bas": None, "G9.6-fresh": None},
        "deployment_eligible": False,
        "notes": ["Exploratory conductor-knob analysis; does not evaluate GroupEngine wave/colour controls.",
                  "Fresh fitted-versus-default render preference and BAS gates have not been measured.",
                  "The original accuracy gate counts 200 non-foil decided ratings; unique pairs and songs are reported separately because repeat raters do not increase song diversity."],
        "provenance": {"ratings_path": str(Path(ratings_path).resolve()), "ratings_sha256": sha256(ratings_path),
                       "pairs_sha256": sha256(Path(ratings_path).parent / "pairs.json"),
                       "song_map_sha256": sha256(song_map_path) if song_map_path and Path(song_map_path).exists() else None,
                       "fitter_sha256": sha256(__file__)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    ap.add_argument("--song-map", type=Path, default=DEFAULT_SONG_MAP)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = analyse(args.ratings, args.song_map, args.lam, args.seed)
    output = args.out or args.ratings.parent / "bradley_terry.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['n_ratings']} ratings -> {output}")
    for q, r in result["questions"].items():
        print(q, json.dumps({k: v for k, v in r.items() if k not in ("folds", "out_of_fold")}))


if __name__ == "__main__":
    main()
