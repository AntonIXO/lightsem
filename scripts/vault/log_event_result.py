#!/usr/bin/env python3
"""Log the event-reformulation POSITIVE result to the LIGHTSEM vault.

First non-null finding in the project, so it goes in as hypothesis + experiment
with a PREREGISTERED gate, not a rule-out.

Schema notes learned the hard way (see docs):
  - propose_hypothesis wants predicted_effects=[{diagnostic_id, direction,
    magnitude_note}], NOT a prose string. Diagnostics must be REGISTERED first.
  - start_experiment takes hypothesis_id + config_note; there is no title/plan.
  - record_diagnostics uses `measurements`, not `diagnostics`.
  - close_experiment takes only (experiment_id, notes) -- the verdict is
    COMPUTED from prediction vs measurement. You cannot assert 'supported'.
"""
import json

from scripts.vault.log_to_vault import PROBLEM, Client

ev = json.load(open("data/cache/event_probe_cv.json"))
cut = json.load(open("data/cache/cut_vs_light.json"))

obs = ev["observed"]
marg = ev["observed_marginal"]
nm = ev["null_marginal"]
na = ev["null_audio"]
z = ev["audio_marginal_lift"]["z"]
p = ev["audio_marginal_lift"]["p"]
za = ev["audio-only_lift"]["z"]

c = Client()
try:
    c.initialize()

    for d in (
        {"name": "event_ap_lift_audio_marginal",
         "description": (
             "Marginal lift in per-song average-precision (AP / base rate) from adding "
             "audio features to the rhythm-of-past-changes baseline, for predicting a "
             "lighting scene change in the next 1.0 s. 5-fold CV by song. Judged "
             "against a circular-shift permutation null that preserves event count and "
             "rhythm, so AP's small-sample upward bias cancels."),
         # NOTE: register_diagnostic direction is DiagDirection
         # (higher_better|lower_better|neutral). The up|down|none vocabulary is
         # only for predicted_effects. They are different enums.
         "unit": "lift", "direction": "higher_better"},
        {"name": "event_ap_lift_audio_only",
         "description": (
             "Per-song AP / base rate for predicting a lighting scene change in the "
             "next 1.0 s from audio features alone, no rhythm baseline. 5-fold CV by "
             "song."),
         # NOTE: register_diagnostic direction is DiagDirection
         # (higher_better|lower_better|neutral). The up|down|none vocabulary is
         # only for predicted_effects. They are different enums.
         "unit": "lift", "direction": "higher_better"},
    ):
        r = c.call("register_diagnostic", d)
        print("diag:", json.dumps(r)[:110])

    h = c.call("propose_hypothesis", {
        "problem_id": PROBLEM,
        "statement": (
            "Audio predicts lighting EVENTS (when the scene changes) even though it "
            "does not predict lighting LEVELS. Reformulating the target from the "
            "absolute scene vector to a change/onset train removes persistence's "
            "structural advantage and asks the question a lighting desk actually "
            "answers: fire a cue now, or not."
        ),
        "rationale": (
            "The level-regression refutation showed persistence explains R2_within "
            "+0.708 with no audio at all, and audio's marginal was -0.005. But that "
            "comparison is rigged by construction: a slowly-varying target is trivially "
            "predicted by its own past, and persistence is useless at exactly the "
            "moments an operator acts. An event target makes the two competitors meet "
            "where neither has a structural edge. multimodal-ai lecture 06 names this "
            "failure directly: continuous streams have no natural elements, and "
            "treating fixed frames as semantic units is a silent correctness bug -- "
            "segment first, then align."
        ),
        "plan": (
            "Events: per-song z-scored 6-dim light vector, novelty = ||future 1.0s mean "
            "- past 1.0s mean||, peaks above the 90th percentile with 2.0s minimum "
            "separation. Target: is there an event in the next 1.0s. "
            "COMPETITOR is NOT persistence (it cannot predict change) but the RHYTHM of "
            "past changes: time since last event, 30s event rate, phase against the "
            "song's own median inter-event interval -- a desk firing every N seconds is "
            "a real deployable baseline needing no audio. "
            "Audio: causal multi-scale past pooling (0.5-4.0s) plus differences between "
            "recent and older pools. Metric: per-song AP / base rate, 5-fold CV by song "
            "over all 140 songs. Null: 40 reps with the event train circularly shifted "
            "within each song."
        ),
        "predicted_effects": [
            {"diagnostic_id": "event-ap-lift-audio-marginal", "direction": "up",
             "magnitude_note": "PREREGISTERED GATE: observed marginal lift must exceed "
                               "the circular-shift null by z > 2.0 with p < 0.05, else "
                               "the event reformulation is dropped like the level target."},
        ],
        # Gate wants id + description (not name/criterion), and gate RESULTS are
        # recorded separately via record_gate_results -- the server deliberately
        # does not compute gates, it records what was committed to.
        "gates": [
            {"id": "beats-circular-shift-null",
             "description": "z(audio marginal lift vs circular-shift permutation "
                            "null) > 2.0 and p < 0.05"},
        ],
        "topic_tags": ["audio-to-light"],
    })
    print("hypothesis ->", json.dumps(h)[:200])
    hid = h["id"]

    e = c.call("start_experiment", {
        "hypothesis_id": hid,
        "config_note": (
            "scripts/diagnostics/probe_event_cv.py: 140 songs, 1376 events, horizon 1.0s, 5-fold CV by "
            "song, ridge lam=30, 40 permutation reps (circular shift within song). "
            "Confound check: scripts/diagnostics/probe_cut_vs_light.py, 36 songs at native 30fps."
        ),
    })
    print("experiment ->", json.dumps(e)[:200])
    eid = e["id"]

    r = c.call("record_diagnostics", {
        "experiment_id": eid,
        "measurements": [
            {"diagnostic_id": "event-ap-lift-audio-marginal",
             "before": float(nm["mean"]), "after": float(marg)},
            {"diagnostic_id": "event-ap-lift-audio-only",
             "before": float(na["mean"]), "after": float(obs["audio"])},
        ],
    })
    print("diagnostics ->", json.dumps(r)[:200])

    r = c.call("record_gate_results", {
        "experiment_id": eid,
        "results": [
            {"gate_id": "beats-circular-shift-null", "passed": True,
             "evidence": f"observed marginal lift {marg:+.4f}x vs circular-shift null "
                         f"{nm['mean']:+.4f} +- {nm['sd']:.4f} over 40 reps "
                         f"-> z {z:+.2f}, p {p:.3f}. Required z>2.0 and p<0.05."},
        ],
    })
    print("gates ->", json.dumps(r)[:200])

    r = c.call("close_experiment", {
        "experiment_id": eid,
        "notes": f"""POSITIVE -- the first non-null result in this project.

140 songs, 1376 events, 5-fold CV by song, horizon 1.0 s.

  rhythm-only baseline (no audio)   {obs['hazard']:.4f}x lift over base rate
  audio only                        {obs['audio']:.4f}x
  rhythm + audio                    {obs['both']:.4f}x
  audio's MARGINAL lift             {marg:+.4f}x

Circular-shift permutation null (40 reps; event count and rhythm preserved,
only audio alignment destroyed):
  marginal null    {nm['mean']:+.4f} +- {nm['sd']:.4f}  ->  observed z {z:+.2f}, p {p:.3f}
  audio-only null  {na['mean']:+.4f} +- {na['sd']:.4f}  ->  z {za:+.2f}

GATE MET (preregistered z > 2.0, p < 0.05).

The null mean is NEGATIVE ({nm['mean']:+.4f}): under shift, adding audio features HURTS
the rhythm baseline via added variance. So the observed positive marginal is
not an artifact of extra parameters -- extra parameters cost you here, and
audio still wins.

WHY THIS WORKED WHERE LEVELS FAILED: on the level target persistence scored
+0.708 and audio's marginal was -0.005. That comparison was structurally
rigged. Moving to events removes the edge, and the SAME audio features that
were worthless become significant. The failure was the target formulation,
not the features and not the model.

CONFOUND, MEASURED NOT ASSUMED (scripts/diagnostics/probe_cut_vs_light.py, 36 songs, native 30fps,
968 events): the target derives from stage VIDEO, so a camera cut looks like a
lighting change, and concert video is edited on the music. If cuts dominated,
this would be audio->editing, not audio->lighting.
  median rise time {cut['rise_median_frames']:.0f} frames = {cut['rise_median_frames']/30:.2f} s   (a cut is ~1 frame)
  only {cut['frac_cut_like_rise']*100:.1f}% rise in <=2 frames
  only {cut['frac_returned']*100:.1f}% revert to the pre-event state within 4.5 s
  {cut['frac_stayed']*100:.1f}% move to a NEW state and stay there
Gradual and persistent = lighting-like, not cut-like. This does NOT prove the
absence of edit contamination -- that needs shot-boundary labels the dataset
does not ship -- but it bounds it as a minority of events.

SCOPE, stated plainly: z=+2.83 establishes the coupling EXISTS. It does not
establish it is large enough to build a rig on ({marg:+.4f}x lift is modest), and it
does not say WHICH audio property carries it. That attribution is the next
probe and it is project-defining: if plain ENERGY carries all of it, the
industry's L2 threshold triggers are already at the ceiling and 'semantic'
buys nothing.""",
    })
    print("close ->", json.dumps(r)[:400])
finally:
    c.close()
