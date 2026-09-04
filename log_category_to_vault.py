#!/usr/bin/env python
"""Record the Stage 0c category test (rb-ls2-00c, part ii) in the vault.

The gate was preregistered in the runbook on 2026-09-03, before any rating.
The hypothesis record is created here, after the sheet was filled, from that
gate unchanged; the record says so. Rater: Anton, 2026-09-04, one sitting.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
R = json.loads(Path("/root/lightsem/outputs/stage0c/category/score_result.json").read_text())
qm, qe, pr = R["questions"]["on_music"], R["questions"]["engineer"], R["protocol"]

c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))

hyp = c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "Dance is not a category error for light. Light derived from real dance motion "
        "through the hybrid conductor (dance envelopes, audio-phased transients), "
        "brightness-matched against a 30-line audio beat trigger on the same music, is "
        "preferred by a blind rater on at least one of two questions: 'which is on the "
        "music?' and 'which looks like a lighting engineer did it?'. If the beat trigger "
        "wins both, block C (the sampled style residual) leaves the product."),
    "predicted_effects": [
        {"diagnostic_id": "human-preference-winrate", "direction": "up",
         "magnitude_note": "win rate above 0.5 with a CI excluding 0.5 on at least one question"}],
    "rationale": (
        "exp-001 showed the hybrid beats the beat trigger on Light-BAS (+0.124 vs +0.094) "
        "while staying close to the dance-derived look (FID 43 vs 794). Both arms take "
        "transient phase from the same audio beat grid, so 'on the music' is expected to be "
        "close; the envelopes and amplitudes the dancer supplies are what the second "
        "question can see."),
    "plan": (
        "run_stage0c.py category: 10 AIST++ clips, 8 s, canonical rig, headless render, "
        "left/right randomised in a sealed KEY.json, brightness matched on mean and 95th "
        "percentile (mean ratio 1.03-1.17, p95 ratio 0.84-0.96, all matched=True). Ten FOIL "
        "trials (beat trigger vs a 1.6x brighter copy of itself) offered as a rater control. "
        "Gate preregistered in rb-ls2-00c on 2026-09-03; this record written after rating."),
    "gates": [
        {"id": "G2-engineer", "description": "'lighting engineer' win rate > 0.5, CI excluding 0.5"},
        {"id": "G2-music", "description": "'on the music' win rate > 0.5, CI excluding 0.5"},
        {"id": "G-protocol", "description": (
            "preregistered protocol met: >= 5 raters, 20 tracks (10 dance, 10 club/EDM), "
            "FOIL control rated; blocking, so a pass on the reduced protocol reads as "
            "directional only")}],
    "topic_tags": ["audio-to-light", "motion-to-light", "music-induced-movement",
                   "evaluation-metric"],
    "problem_tags": ["matched-baseline"],
})
assert isinstance(hyp, dict) and hyp.get("ok") is not False, hyp
hid = hyp.get("id") or hyp.get("hypothesis_id")
print("hypothesis:", hid)

trials = [{"t": x["trial"], "clip": x["clip"], "q": x["question"], "winner": x["winner"],
           "conf": x["confidence"], "notes": x["notes"]} for x in R["trials"]]
exp = c.call("start_experiment", {"hypothesis_id": hid, "git_ref": "25e1f4a",
    "config_note": json.dumps({
        "stage": "0c-ii", "runbook": "rb-ls2-00c", "rater": "Anton", "date": "2026-09-04",
        "sheet": "outputs/stage0c/category/score_sheet.csv",
        "systems": {"A": "dance_derived_hybrid", "B": "beat_trigger"},
        "protocol": pr, "trials": trials})})
assert isinstance(exp, dict) and exp.get("ok") is not False, exp
eid = exp.get("id") or exp.get("experiment_id")
print("experiment:", eid)

c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
    {"diagnostic_id": "human-preference-winrate", "before": 0.5,
     "after": qe["winrate_equal_as_half"]}]})

c.call("record_gate_results", {"experiment_id": eid, "results": [
    {"gate_id": "G2-engineer", "passed": True, "evidence": (
        f"hybrid {qe['hybrid']}, beat trigger {qe['beat_trigger']}, equal {qe['equal']} of "
        f"10 trials. Decided-only win rate {qe['winrate_decided']}, exact binomial CI "
        f"{qe['ci95_exact_decided']}, two-sided sign test p = {qe['sign_test_p_two_sided']}. "
        f"'equal' counted as half a win each: {qe['winrate_equal_as_half']}, Wilson CI "
        f"{qe['ci95_wilson_equal_as_half']}. Both intervals exclude 0.5, the lower one by "
        f"0.018 and 0.041. One rater, so this is one person's threshold, not a population's.")},
    {"gate_id": "G2-music", "passed": False, "evidence": (
        f"hybrid {qm['hybrid']}, beat trigger {qm['beat_trigger']}, equal {qm['equal']}. "
        f"Win rate {qm['winrate_decided']}, CI {qm['ci95_exact_decided']}, p = "
        f"{qm['sign_test_p_two_sided']}. Chance, and expected by construction: after "
        f"exp-001 the hybrid takes its transient phase from the same audio beat grid the "
        f"beat trigger uses, so the two arms are phased identically and this question "
        f"cannot separate them. The rater's note on T05 says it directly: the beat trigger "
        f"is 'on the music actually' but has 'too fast changes for light'.")},
    {"gate_id": "G-protocol", "passed": False, "evidence": (
        f"1 rater of 5; {pr['tracks']} tracks of 20; 0 club/EDM tracks of 10; 0 of 10 FOIL "
        f"trials rated (Anton: not needed); brightness matched by construction on every "
        f"pair (mean ratio 1.03-1.17, p95 ratio 0.84-0.96). Blinding caveat: {pr['blinding_caveat']}.")},
]})

c.call("close_experiment", {"experiment_id": eid, "notes": (
    "The category error did not happen. On 'which looks like a lighting engineer did it' "
    f"the dance-derived hybrid won {qe['hybrid']} of {qe['hybrid'] + qe['beat_trigger']} "
    "decided trials against the beat trigger on brightness-matched pairs; on 'which is on "
    f"the music' it was {qm['hybrid']}-{qm['beat_trigger']}, a tie, which is what the design "
    "predicts because both arms carry the same audio-phased transients.\n"
    "What the rater's notes add, and this is the useful part: the discriminating cue was "
    "dynamics, not timing. T05 (beat trigger): 'too fast changes for light, but on the music "
    "actually'. T07 (both bad): 'A does not moving, B is moving boring' -- the hybrid was "
    "static on gJS_sBM ch01 and lost both questions. T03 (both bad): equal on both. So the "
    "hybrid loses when its envelopes are flat, and the beat trigger loses when it flashes on "
    "every beat. That is the same failure exp-009 measured in the model's light (85.8 events "
    "per clip against 30.1 real-mapped) seen through a human: a light that changes on every "
    "beat is 'on the music' and is not 'a lighting engineer'.\n"
    "Status: block C stays in the product, provisionally. The protocol was reduced (one rater, "
    "ten AIST++ tracks, no EDM half, no FOIL), so this is a directional result and the "
    "preregistered bar is still open for the Stage 3 A/B with >= 20 raters.")})

c.call("interpret_experiment", {"experiment_id": eid, "lesson": (
    "Two lessons. (1) The two questions measure different things and the design should keep "
    "both: 'on the music' is a timing question that any beat-phased system saturates; "
    "'lighting engineer' is a dynamics question -- does the light vary in amount, width and "
    "rate the way a designed show does. Dance-derived envelopes carry the second and the "
    "beat trigger does not. (2) Event density is a human-visible failure, not only a metric "
    "one: the rater independently named 'too fast changes' as the tell for a non-engineer, "
    "which is the model's own largest measured defect (light-event-rate-ratio 2.85). "
    "The offline path's H-B (zero-phase spectral matching) is therefore aimed at the thing a "
    "human actually sees, and its diversity gate is the one that keeps it honest.")})

c.call("update_runbook", {"runbook_id": "rb-ls2-00c-late-lights-and-category-test", "note": (
    "2026-09-04 part (ii) rated by Anton, one sitting, 10 AIST++ pairs, FOIL skipped. "
    "'lighting engineer': hybrid 8, beat trigger 1, equal 1 (CI excludes 0.5). 'on the "
    "music': 5-4-1, chance, as the shared audio phasing predicts. G2 passes on one question "
    "at a reduced protocol; block C stays. Part (i) late lights (outputs/stage0c/late_lights, "
    "18 pairs) is still unrated.")})

c.call("update_problem", {"id": P, "add_update": (
    "2026-09-04 Stage 0c category test: dance-derived hybrid light beats the beat trigger "
    "8-1 (1 equal) on 'looks like a lighting engineer', ties 5-4 on 'on the music'; one rater, "
    "10 brightness-matched AIST++ pairs. Category error rejected directionally; block C stays. "
    "Rater's tell for 'not an engineer' was change rate, the same defect the model carries.")})

lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov, "optiHealth vault must not change"
c.close()
