#!/usr/bin/env python
"""Stage 10 pre-registration (PLAN.md rev. 10 section 6.10): the cue-based renderer that Stage 9's
results point at. Idempotent for the runbook and the diagnostic, NOT for the hypotheses.

    bash run.sh scripts/vault/queue_stage10_to_vault.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client

P = "dancing-light-generative-20260903"; VAULT = Path("/root/lightsem-vault")
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


if not (VAULT / "diagnostics" / "cue-transition-distance.md").exists():
    ok(c.call("register_diagnostic", {"id": "cue-transition-distance", "name": "cue_transition_frechet_distance", "direction": "lower_better",
                                      "unit": "distance", "topic_tags": ["evaluation-metric", "audio-to-light"],
                                      "description": "Frechet distance between the look-TRANSITION statistics of a generated cue stream and the authored corpora's "
                                                     "(inter-cue interval median and CV, hue jump at cues, value-class change at cues, fraction of cues within 100 ms "
                                                     "of a beat / downbeat, cues per minute), reported with a split-half floor of the authored set and a shuffled-donor null."}))
if not (VAULT / "runbooks" / "rb-ls2-10-cue-renderer.md").exists():
    ok(c.call("register_runbook", {"id": "rb-ls2-10-cue-renderer", "problem_id": P, "priority": 1, "topic_tags": ["audio-to-light", "generative-modelling", "learned-conductor"],
        "prompt": ("STAGE 10 -- THE CUE-BASED RENDERER (PLAN.md rev. 10 section 6.10). Stage 9 measured that every authored corpus on disk carries "
                   "cue TIMING (look-change AP lifts 1.4x RPMC, 2.4x JD, 1.65x Beat Saber) and the corpus's look statistics, but not per-frame or per-bar "
                   "light LEVELS (frame value r <= 0.18, bar knobs R2 ~ 0, a sampled look no better than a donor). A decoder that owns the look must "
                   "therefore be a sampler over looks with learned cue timing. Two hypotheses. H-N: hold the sampled look between cues fired by the "
                   "calibrated look-change hazard (refractory 2 s, threshold matched to the corpus cue rate) -- scored on cue-timing AP lift, on the new "
                   "cue-transition-distance against the authored corpora with a split-half floor and a shuffled-donor null, and on the JD light protocol "
                   "(Light-BAS minus null not below the conductor at no higher onset density). H-O: the with-hazard operator head (3.17x) beats the "
                   "audio-only decoder (2.35x) because the rhythm of the operator's OWN past cues carries information; feeding the decoder the hazard "
                   "features of its own emitted cues (teacher-forced in training, autoregressive at inference in 1 s blocks) must recover part of that "
                   "margin on held-out songs, priced against the same decoder with the TRUE hazard (the ceiling) and without it (the floor); a size sweep "
                   "(d 192/256/384, 3/6 blocks) reports whether capacity moves the cue lift at all. Nothing in this stage regresses a level; nothing "
                   "enters a loss from Beat Saber; every number carries its shuffled-donor and split-half counterparts."),
        "gates": ["G10.1-cue: cue-held sampler's cue-transition distance below the per-frame sampler's and below the shuffled-donor null, above the split-half floor",
                  "G10.1-bas: Light-BAS minus null not below the conductor without a higher onset density",
                  "G10.2-self: self-hazard decoder's held-out cue lift above the audio-only decoder's with CI excluding 0 (JD and RPMC)",
                  "G10.2-size: size sweep reported, lift vs parameters, causal only"]}))
H = []
def hyp(tag, **kw):
    r = ok(c.call("propose_hypothesis", dict(problem_id=P, **kw))); hid = r.get("id") or r.get("hypothesis_id"); print(f"{tag}: {hid}"); H.append((tag, hid))
hyp("H-N", statement=("A look sampler that HOLDS its sampled look between cues fired by the calibrated look-change hazard (refractory 2 s, rate matched "
                      "to the corpus) reproduces the authored corpora's look-transition statistics better than a per-frame sampler and than a shuffled "
                      "donor, and lands on the beat no worse than the hand-written conductor at no higher onset density."),
    rationale=("Stage 9 (exp-020..022): per-frame regression collapses to constants, per-frame sampling jitters (1039 events per clip) and correlates with "
               "the target at r 0.04, while the look-change hazard carries lifts of 2.4x (JD) and 1.4x (RPMC). What the data supports is WHEN, plus the "
               "corpus's look distribution; a cue-held sampler is the object that uses exactly that and nothing else."),
    plan=("rb-ls2-10 step 1: scripts/legacy_stages/run_stage10_cue_eval.py -- calibrate the hazard threshold on training songs to the corpus cue rate; at each cue sample a look "
          "from the v2/v3 flow head and hold it; cue-timing AP lift, cue-transition distance (split-half floor, shuffled donor), JD light protocol, demos."),
    predicted_effects=[{"diagnostic_id": "cue-transition-distance", "direction": "down", "magnitude_note": "below the per-frame sampler and the shuffled donor; the floor is the split-half of the authored set"},
                       {"diagnostic_id": "light-bas-minus-shift-null", "direction": "up", "magnitude_note": "not below the conductor's +0.12 at events per clip <= the conductor's"}],
    gates=[{"id": "G10.1-cue", "description": "cue-transition distance below the per-frame sampler and the shuffled donor, above the floor"},
           {"id": "G10.1-bas", "description": "Light-BAS minus null not below the conductor without a higher onset density"}],
    topic_tags=["audio-to-light", "generative-modelling", "learned-conductor", "evaluation-metric"], problem_tags=["conditional-mean-collapse", "matched-baseline", "metric-gaming"])
hyp("H-O", statement=("Feeding the group decoder the hazard features of its OWN emitted cues (teacher-forced in training, autoregressive in 1 s blocks at "
                      "inference) raises its held-out cue-timing lift above the audio-only decoder on Just Dance visual and RPMC-L2, recovering part of the "
                      "margin the with-hazard operator head holds (3.17x vs 2.35x on JD); decoder size alone does not."),
    rationale=("The with-hazard head's margin over audio-only (exp-016: 3.17x vs 2.2x) is information about the operator's own rhythm of changes; it is "
               "available at inference only from the model's own past cues. The ceiling is the same decoder with the true hazard, the floor the audio-only one."),
    plan=("rb-ls2-10 step 2: scripts/training/train_group_decoder.py --self-hazard over a size sweep (d 192/256/384, 3/6 blocks); held-out cue lift with own cues "
          "(autoregressive), with the true hazard (ceiling) and without (floor); CI over songs; RPMC and JD separately."),
    predicted_effects=[{"diagnostic_id": "event-ap-lift-audio-marginal", "direction": "up", "magnitude_note": "own-cue hazard lift above audio-only with CI excluding 0; the true-hazard ceiling reported next to it"}],
    gates=[{"id": "G10.2-self", "description": "own-cue lift above audio-only with CI excluding 0 on JD and RPMC"},
           {"id": "G10.2-size", "description": "size sweep reported, causal only"}],
    topic_tags=["audio-to-light", "generative-modelling", "real-time"], problem_tags=["leaked-future", "matched-baseline"])
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md"))); assert ov2 == ov
print("lightsem", len(list(VAULT.rglob("*.md"))), "md; optiHealth", ov, "->", ov2); c.close()
