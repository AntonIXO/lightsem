#!/usr/bin/env python
"""Score the Stage 0c category test: dance-derived hybrid vs the beat trigger.

Reads the rater's sheet and the sealed key, decodes A/B or L/R answers into
systems, and reports the win rate per question with an exact binomial CI and a
two-sided sign test. `equal` answers are reported both ways: dropped, and
counted as half a win for each side.
"""
import csv, json, sys
from math import comb
from pathlib import Path

D = Path("outputs/stage0c/category")
key = {t["trial_id"]: t for t in json.load(open(D / "KEY.json"))}
rows = list(csv.DictReader(open(D / "score_sheet.csv")))

Q = {"Which one is ON THE MUSIC?": "on_music",
     "Which one looks like a LIGHTING ENGINEER did it?": "engineer"}


def system_for(t, ans):
    """A/B name the files; L/R name the screen side. Both map through the key."""
    a = ans.strip().upper()
    if a == "EQUAL":
        return "equal"
    if a in ("A", "B"):
        for side in ("left", "right"):
            if t[f"{side}_file"].endswith(f"__{a}.mp4"):
                return t[f"{side}_system"]
    if a == "L":
        return t["left_system"]
    if a == "R":
        return t["right_system"]
    raise ValueError(ans)


def exact_ci(k, n, alpha=0.05):
    """Clopper-Pearson via bisection on the binomial CDF."""
    def cdf(p, k):
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k + 1))
    def solve(f, lo, hi):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda p: 1 - cdf(p, k - 1) < alpha / 2, 0, 1)
    hi = 1.0 if k == n else solve(lambda p: cdf(p, k) > alpha / 2, 0, 1)
    return lo, hi


def sign_test(k, n):
    """Two-sided exact P(X >= k) under p = 0.5, doubled and capped at 1."""
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * p)


out = {"n_trials_rated": 0, "n_foil_rated": 0, "questions": {}, "trials": []}
per_q = {q: [] for q in Q.values()}
for r in rows:
    t = key[r["trial_id"]]
    ans = r["answer(L/R/equal)"].strip()
    if not ans:
        continue
    if "FOIL" in t["clip"]:
        out["n_foil_rated"] += 1
        continue
    q = Q[r["question"]]
    winner = system_for(t, ans)
    conf = r["confidence(1-5)"].strip()
    per_q[q].append(winner)
    out["trials"].append({"trial": r["trial_id"], "clip": t["clip"], "question": q,
                          "answer": ans, "winner": winner,
                          "confidence": int(conf) if conf else None,
                          "notes": r["notes"].strip(),
                          "brightness_mean_ratio": round(t["match"]["mean_ratio"], 3),
                          "brightness_p95_ratio": round(t["match"]["p95_ratio"], 3)})

out["n_trials_rated"] = len({x["trial"] for x in out["trials"]})
for q, ws in per_q.items():
    n_all = len(ws)
    hyb = ws.count("dance_derived_hybrid")
    beat = ws.count("beat_trigger")
    eq = ws.count("equal")
    n_dec = hyb + beat
    lo, hi = exact_ci(hyb, n_dec) if n_dec else (0, 1)
    # 'equal' as half a win each, Wilson interval on the fractional count
    k_half = hyb + 0.5 * eq
    p_half = k_half / n_all
    z = 1.959964
    den = 1 + z * z / n_all
    cen = (p_half + z * z / (2 * n_all)) / den
    hw = z * ((p_half * (1 - p_half) / n_all + z * z / (4 * n_all * n_all)) ** 0.5) / den
    out["questions"][q] = {
        "hybrid": hyb, "beat_trigger": beat, "equal": eq,
        "winrate_decided": round(hyb / n_dec, 3) if n_dec else None,
        "ci95_exact_decided": [round(lo, 3), round(hi, 3)],
        "sign_test_p_two_sided": round(sign_test(hyb, n_dec), 4) if n_dec else None,
        "winrate_equal_as_half": round(p_half, 3),
        "ci95_wilson_equal_as_half": [round(cen - hw, 3), round(cen + hw, 3)],
        "gate_G2_ci_excludes_0.5": bool(lo > 0.5),
    }

out["protocol"] = {
    "raters": 1, "preregistered_raters": 5,
    "tracks": out["n_trials_rated"], "preregistered_tracks": 20,
    "edm_tracks": 0, "preregistered_edm_tracks": 10,
    "foil_trials_rated": out["n_foil_rated"], "foil_trials_offered": 10,
    "brightness_matched": all(key[r["trial_id"]]["match"].get("matched", False)
                              for r in rows if "FOIL" not in key[r["trial_id"]]["clip"]),
    "blinding_caveat": ("answers were given by file letter; __A is the hybrid and __B the "
                        "beat trigger in every trial, so the letter was a constant label "
                        "across trials even though left/right was randomised in the key"),
}
json.dump(out, open(D / "score_result.json", "w"), indent=1)
print(json.dumps(out["questions"], indent=1))
print(json.dumps(out["protocol"], indent=1))
for x in out["trials"]:
    print(x["trial"], x["question"][:8], x["winner"][:12].ljust(12), x["confidence"], x["notes"])
