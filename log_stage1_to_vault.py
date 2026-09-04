#!/usr/bin/env python
"""Record Stage 1 (rb-ls2-01) — the trained three-block model — in the vault."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
S = Path("/root/lightsem/outputs/stage1")
V = json.loads((S / "train_v1.json").read_text())
M = json.loads((S / "train_mse.json").read_text())
L = json.loads((S / "stage1_light_eval.json").read_text())["summary"]

c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))

hyp = c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "A causal Conformer with three heads — deterministic block A, event hazard "
        "block B, conditional-flow block C — trained on AIST++ produces light that "
        "lands on the music like dance-derived light does, with a feature distribution "
        "close to it and sampled variety a regression head cannot give."),
    "predicted_effects": [
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "up",
         "magnitude_note": "model >= 0.9 x the real-dance-mapped value"},
        {"diagnostic_id": "fid-light-kinetic", "direction": "down",
         "magnitude_note": "FID(model, real) < half the structure-destroying null"},
        {"diagnostic_id": "light-diversity-kinetic", "direction": "up",
         "magnitude_note": "model diversity >= 0.7 x real; MSE control must collapse"}],
    "rationale": (
        "MotionStreamer reports FID 59 for an MSE head against 10.7 for a diffusion head "
        "on the same autoregressive trunk. Stage 0a showed transient phase must come from "
        "audio, so the model feeds the hybrid conductor rather than the motion-only one."),
    "plan": (
        "9.3 M parameter causal Conformer, 6 blocks, d_model 256, 4 s attention window, "
        "40 fps; 749k frames from 1408 clips; L1 on 13 z-scored block A channels, BCE "
        "hazard on 3 event types, rectified flow on 10 block C channels; 12k steps. "
        "Splits by track, by genre and by dancer, scored separately. Collapse control: "
        "the same trunk with an MSE head over block C."),
    "gates": [
        {"id": "G-causal", "description": "future-perturbation leak exactly 0"},
        {"id": "G-bas", "description": "model Light-BAS minus null >= 0.9 x real-mapped"},
        {"id": "G-fid", "description": "FID(model, real) < 0.5 x structure-destroying null"},
        {"id": "G-div", "description": "model diversity >= 0.7 x real; MSE control collapses"}],
    "topic_tags": ["audio-to-light", "generative-modelling", "real-time"],
    "problem_tags": ["conditional-mean-collapse", "matched-baseline"],
})
hid = hyp.get("id") or hyp.get("hypothesis_id")
exp = c.call("start_experiment", {"hypothesis_id": hid, "config_note": json.dumps({
    "stage": "1", "runbook": "rb-ls2-01", "params_m": V["params_m"],
    "steps": V["steps"], "train_s": round(V["train_s"]), "splits": V["splits"],
    "device": "RTX 5070", "ckpt": "/mnt/lightsem-data/ckpt/light_trunk_v1.pt",
    "conductor": "hybrid (dance envelopes + audio-phased transients)"})})
eid = exp.get("id") or exp.get("experiment_id")
print("experiment:", eid)

b = L["bas_minus_null"]; f = L["fid_light"]; dv = L["diversity"]
c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
    {"diagnostic_id": "light-bas-minus-shift-null",
     "before": round(b["real_dance"]["delta"], 4), "after": round(b["model"]["delta"], 4)},
    {"diagnostic_id": "fid-light-kinetic",
     "before": round(f["beat_trigger_vs_real"], 1), "after": round(f["model_vs_real"], 1)},
    {"diagnostic_id": "light-diversity-kinetic",
     "before": round(dv["across_clips_real"], 3), "after": round(dv["across_clips_model"], 3)}]})

c.call("record_gate_results", {"experiment_id": eid, "results": [
    {"gate_id": "G-causal", "passed": True,
     "evidence": f"future-perturbation leak {V['causality_leak']} at cuts of 50/100/200/300 "
                 f"frames; attention is masked to the past and windowed, the depthwise "
                 f"convolution is left-padded"},
    {"gate_id": "G-bas", "passed": False,
     "evidence": f"model {b['model']['delta']:+.4f}, CI "
                 f"{[round(x,4) for x in b['model']['ci95']]} (includes zero) against "
                 f"real-mapped {b['real_dance']['delta']:+.4f} and beat trigger "
                 f"{b['beat_trigger']['delta']:+.4f}. Mechanism found: the model's block A "
                 f"outputs are over-smoothed toward the corpus median, so its light flashes "
                 f"6.5 times a second instead of the real 4.3 and does so uniformly. A dense "
                 f"uniform event train has a HIGH circular-shift null by construction, so "
                 f"even correctly beat-phased transients cannot show above it."},
    {"gate_id": "G-fid", "passed": False,
     "evidence": f"FID(model, real) {f['model_vs_real']:.1f} against half the null "
                 f"{0.5*f['real_vs_block_shuffled']:.1f}; the beat trigger scores "
                 f"{f['beat_trigger_vs_real']:.0f}, so the model is 6x closer to "
                 f"dance-derived light than the baseline is, and still not close enough"},
    {"gate_id": "G-div", "passed": False,
     "evidence": f"model {dv['across_clips_model']:.2f} against real "
                 f"{dv['across_clips_real']:.2f}, a ratio of "
                 f"{dv['across_clips_model']/dv['across_clips_real']:.2f} versus the 0.7 "
                 f"required. The collapse control behaves exactly as predicted: the MSE "
                 f"head's diversity across style samples is 0.000 against the flow head's "
                 f"2.03, at equal block A accuracy (val-track R2 0.057 vs 0.035), so the "
                 f"generative head buys variety for free — there is simply not enough of it"}]})

c.call("close_experiment", {"experiment_id": eid, "notes": (
    "The machinery works and the corpus is the ceiling, as Stage 0b predicted.\n"
    "What passed: causality is exact (leak 0.0), the model trains in 9 minutes on one "
    "RTX 5070, and the generative head does what the literature says it should — the MSE "
    "control collapses to 0.000 diversity while the flow head gives 2.03 at the same block "
    "A accuracy.\n"
    "Block A on unseen tracks: mean R2 +0.035, best channels footfall rate 0.220, metrical "
    "level 0.082, vertical bounce amplitude 0.038. On unseen DANCERS +0.069. On unseen "
    "GENRES -0.183 — negative, meaning worse than predicting the mean, which is exactly "
    "what Stage 0b's finding predicts: what the model can learn from this corpus is "
    "genre-conditioned style, so removing the genre removes the skill.\n"
    "Block B events: footfall AP 0.392 against a 0.347 base rate on unseen tracks, 0.655 "
    "against 0.572 on unseen dancers; hits show no lift at all. Consistent with Stage 0a — "
    "event phase belongs to the choreography.\n"
    "The light-level gates fail for one identifiable reason rather than three. Predicting "
    "near the conditional median makes every channel smooth; a smooth descriptor stream "
    "drives a light that flashes constantly and evenly; and an even event train sits at its "
    "own circular-shift null however well each flash is phased. The metric is not being "
    "unfair — it is naming the failure the plan was built to catch.")})
c.call("interpret_experiment", {"experiment_id": eid, "failure_mode": "recipe_conditional",
    "lesson": (
        "Two fixes follow from the mechanism, and neither is 'train longer'. First, the "
        "block A / block C boundary was drawn from the literature and should be redrawn "
        "from the Stage 0b audit: on this corpus almost nothing clears the identity "
        "baseline, so almost everything belongs in the SAMPLED block, not the regressed "
        "one. Regressing an unpredictable channel does not produce a cautious estimate, it "
        "produces a flat line, and a flat line makes bad light. Second, the flow head "
        "samples each frame independently given the trunk state, so its variety is "
        "high-frequency jitter rather than the burst-and-rest structure real dance has; it "
        "needs to condition on its own previous output. Both are v2 work. Meanwhile the "
        "diagnostic to keep is the one that caught this: report the event RATE and the "
        "circular-shift null next to every Light-BAS, because a model can raise alignment "
        "and lose the comparison by flashing more.")})
c.call("update_runbook", {"runbook_id": "rb-ls2-01-latent-dancer-offline",
    "note": ("2026-09-04 RUN as the three-block model (not a latent dancer). Causality gate "
             "passed exactly; BAS/FID/diversity gates failed with a single identified "
             "mechanism (over-smoothed block A -> uniformly dense light -> high shift null). "
             "MSE collapse control behaved as predicted (diversity 0.000 vs 2.03). v2: move "
             "unpredictable channels from the regressed block to the sampled one and make "
             "the flow head autoregressive.")})
lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov
c.close()
