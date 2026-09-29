#!/usr/bin/env python
"""Record Stage 2B — the operator head on all 699 RPMC-L2 shows."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client
P = "dancing-light-generative-20260903"
R = json.loads(Path("/root/lightsem/experiments/legacy/stages/outputs/stage2/operator_head.json").read_text())
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
am = R["audio_marginal"]
hyp = c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": ("The August audio->lighting-event result holds on the full RPMC-L2 "
                  "corpus: across 699 professional live shows, audio predicts when a "
                  "lighting operator changes the scene, over and above the rhythm of "
                  "that operator's own past changes."),
    "predicted_effects": [{"diagnostic_id": "event-ap-lift-audio-marginal", "direction": "up",
        "magnitude_note": "marginal lift over the rhythm baseline >= +0.09 with z > 3"}],
    "rationale": ("The August probe measured +0.09 lift at z +2.83 on the 140 songs that "
                  "fitted on the root disk. Five times the data should sharpen it or kill "
                  "it; a result that only exists at n = 140 is not a result."),
    "plan": ("All 699 songs extracted from the local HDF5 into the same cache format the "
             "August probe used, so the event definition, the features and the honest "
             "competitor are unchanged. Ridge fitted by streaming the normal equations "
             "one song at a time (the 3584-dimensional audio features for 478 training "
             "songs are 34 GB and the machine has 31). Split by song; average precision "
             "per held-out song as a lift over that song's own base rate; lag-shift "
             "control on the audio score."),
    "gates": [{"id": "G-op", "description": "audio marginal lift >= +0.09 over the rhythm "
                                            "baseline, z > 3, shift control peaking at 0 s"}],
    "topic_tags": ["audio-to-light", "evaluation-metric"],
    "problem_tags": ["matched-baseline"]})
hid = hyp.get("id") or hyp.get("hypothesis_id")
exp = c.call("start_experiment", {"hypothesis_id": hid, "config_note": json.dumps({
    "stage": "2B", "runbook": "rb-ls2-02", "n_songs": R["n_songs"],
    "split": "by song, 478 train / 122 val / 99 test",
    "features": "512-d openl3 x 4 causal pools + 3 novelty differences = 3584",
    "baseline": "rhythm of past changes (time since last event, recent rate, phase)",
    "horizon_s": 1.0, "corpus": "RPMC-L2 full, CC-BY-4.0, 40 GB local"})})
eid = exp.get("id") or exp.get("experiment_id")
print("experiment:", eid)
c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
    {"diagnostic_id": "event-ap-lift-audio-marginal", "before": 0.09,
     "after": round(am["lift"], 4)}]})
c.call("record_gate_results", {"experiment_id": eid, "results": [{
    "gate_id": "G-op", "passed": bool(R["gates"]["G_event_ap_lift"]["passed"]),
    "evidence": (f"Audio's marginal lift over the rhythm baseline is {am['lift']:+.4f}x "
                 f"(CI [{am['ci95'][0]:+.4f}, {am['ci95'][1]:+.4f}], z = {am['z']:.2f}) "
                 f"across {am['n_songs']} held-out shows, against a gate of +0.09 and "
                 f"z > 3. Absolute lifts: audio alone {R['audio_only']['lift_mean']:.3f}x, "
                 f"rhythm alone {R['rhythm_only']['lift_mean']:.3f}x, together "
                 f"{R['rhythm_plus_audio']['lift_mean']:.3f}x. The lag-shift control peaks "
                 f"at {R['shift_curve']['peak_lag_s']:+d} s.")}]})
c.call("close_experiment", {"experiment_id": eid, "notes": (
    "Confirmed and strengthened at five times the sample size. The August number "
    "(+0.09 lift, z +2.83, 140 songs) becomes +0.31 lift (CI [+0.15, +0.48], z 3.66) "
    "across 699 shows and 99 held-out songs, with the lag-shift control peaking at zero.\n"
    "Read it with the same care as before. The shift curve is peaked at 0 but not sharply "
    "— neighbouring lags reach 1.46 against 1.48 at zero — which matches the August "
    "characterisation of a reactive, timbre-driven coupling over one to two seconds rather "
    "than an anticipatory one. Nothing here supports an anticipation claim, and the plan's "
    "separate anticipation gate (lift at +4 s above +0.05 with a CI excluding zero) is not "
    "tested by this run.\n"
    "Practical weight: this is the strongest audio-to-light coupling measured anywhere in "
    "the project. It comes from professional operators, not from dancers, and it is about "
    "WHEN to act rather than what level to hold — the same shape as every other positive "
    "result here.")})
c.call("interpret_experiment", {"experiment_id": eid, "lesson": (
    "The operator head is the load-bearing part of the product, and it should be trained "
    "first and weighted highest. Dance data explains what a light should DO between cues; "
    "operator data explains WHEN to cut, and does so with an effect three times the size. "
    "For Stage 3 the conductor should take scene-change timing from the operator head, "
    "transient phase from the audio beat grid, and envelopes from the dance-trained head — "
    "three sources, each used for what it measurably carries.")})
c.call("update_runbook", {"runbook_id": "rb-ls2-02-operator-head-699",
    "note": "2026-09-04 part B RUN on all 699 songs. Gate PASSED: audio marginal lift "
            "+0.3145x, z 3.66, CI [+0.149,+0.479], shift control peaks at 0 s. Part A "
            "(video->descriptor scaling) not run."})
lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov
c.close()
