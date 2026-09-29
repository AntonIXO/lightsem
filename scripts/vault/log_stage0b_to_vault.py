#!/usr/bin/env python
"""Record Stage 0b (rb-ls2-00b) — the predictability audit — in the vault."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client

PROBLEM = "dancing-light-generative-20260903"
SH = json.loads(Path("/root/lightsem/experiments/legacy/stages/outputs/stage0b/audit_shared.json").read_text())
BR = json.loads(Path("/root/lightsem/experiments/legacy/stages/outputs/stage0b/burger_replication.json").read_text())


def main():
    lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
    ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    c = Client()
    assert c.initialize()["name"] == "vibescience-lightsem"

    hyp = c.call("propose_hypothesis", {
        "problem_id": PROBLEM,
        "statement": (
            "The music-induced-movement literature's shared skeleton is present in "
            "AIST++: causal audio predicts the across-dancer mean of quantity of "
            "motion, vertical bounce amplitude, head speed and the embodied metrical "
            "level on held-out tracks, beating a dancer-identity baseline."),
        "predicted_effects": [
            {"diagnostic_id": "descriptor-audio-r2-over-identity", "direction": "up",
             "magnitude_note": "audio R2 > 0.3 at 2-8 s on the gate channels, "
                               "beating dancer identity by > 0.1"},
        ],
        "rationale": (
            "Burger et al. 2013 measured stimulus-level correlations of 0.55-0.73 "
            "between pulse clarity / low-band flux / high-band flux and quantity of "
            "motion, head speed and hand speed; Toiviainen 2010 reports the vertical "
            "bounce locking to the tactus. If those hold here, block A is trainable."),
        "plan": (
            "29 descriptor channels x 4 causal timescales (0.1/0.5/2/8 s trailing EMA), "
            "749k frames from 1408 clips. Two audits: (a) per-clip targets against a "
            "dancer-identity and a genre-mean baseline; (b) the across-dancer MEAN per "
            "track, which is the only thing an audio-only model can possibly predict, "
            "with its split-half reliability as the ceiling. Multi-output ridge, alpha "
            "chosen on held-out inner tracks, GroupKFold by music id, bootstrap CIs. "
            "Plus a direct replication of Burger's stimulus-level correlations."),
        "gates": [
            {"id": "G-audit", "description": "QoM, vertical bounce amplitude, head speed "
                                             "and metrical level beat dancer identity "
                                             "with audio R2 > 0.3 at 2-8 s"},
        ],
        "topic_tags": ["music-induced-movement", "audio-to-light", "cross-modal"],
        "problem_tags": ["ill-posed-ground-truth", "matched-baseline"],
    })
    hid = hyp.get("id") or hyp.get("hypothesis_id")
    exp = c.call("start_experiment", {
        "hypothesis_id": hid,
        "config_note": json.dumps({
            "stage": "0b", "runbook": "rb-ls2-00b",
            "n_frames": 749173, "n_tracks": 60, "n_clips": 1408,
            "timescales_s": [0.1, 0.5, 2.0, 8.0],
            "model": "multi-output ridge, alphas 1e1..1e7 chosen on held-out inner tracks",
            "splits": "GroupKFold by music id (5 folds per-clip, 6 folds shared)",
            "baselines": ["dancer identity (per-dancer train mean)",
                          "genre mean", "split-half reliability ceiling"],
            "causality": "trailing EMA features and targets; no future audio",
        }),
    })
    eid = exp.get("id") or exp.get("experiment_id")
    print("experiment:", eid)

    best = SH["per_channel"]["laterality"]["2.0"]
    c.call("record_diagnostics", {
        "experiment_id": eid,
        "measurements": [
            {"diagnostic_id": "descriptor-audio-r2-over-identity",
             "before": 0.30, "after": round(float(best["r2_within_track"]), 3)},
        ],
    })

    c.call("record_gate_results", {
        "experiment_id": eid,
        "results": [{
            "gate_id": "G-audit", "passed": False,
            "evidence": (
                "No channel comes close. Per-clip audit: audio's held-out R2 is <= 0.15 "
                "everywhere and dancer identity wins on 23 of 29 channels (footfall rate "
                "identity R2 0.403 vs audio 0.035; vertical bounce amplitude 0.241 vs "
                "-0.018). Shared-component audit (across-dancer mean, the only target an "
                "audio-only model can reach): best between-track R2 is hand_spread 0.211, "
                "CI [-0.075, +0.464], and metric_level 0.150, CI [-0.026, +0.361] — every "
                "CI includes zero. Within-track dynamics are small but real: laterality "
                "0.192, CI [+0.064, +0.325]; hand_speed 0.183, CI [-0.004, +0.337]; "
                "com_speed 0.128; hip_yaw_speed 0.117; vertical bounce 0.092. The gate "
                "asked for > 0.3 on QoM, vertical bounce, head speed and metrical level; "
                "all four are below 0.05 within-track."),
        }],
    })

    c.call("rule_out", {
        "problem_id": PROBLEM,
        "claim": (
            "AIST++ can supply block A, the shared audio->movement skeleton. ELIMINATED "
            "for this corpus. What varies between AIST++ tracks is the dance GENRE, not "
            "the acoustics, and the movement follows the genre."),
        "method": (
            "Direct replication of Burger et al. 2013's stimulus-level design on all 60 "
            "AIST++ tracks: one row per track, movement averaged over every dancer and "
            "frame, audio averaged over the track, bootstrap CIs over tracks; then the "
            "same correlations with genre and tempo partialled out; then the share of "
            "between-track movement variance explained by the genre label alone; then a "
            "check of whether audio can recover genre at all."),
        "evidence": (
            "Not one published correlation replicates, and several invert. Pulse clarity "
            "-> quantity of motion: Burger +0.62, here -0.078 (CI [-0.341, +0.169]). "
            "Low-band flux -> head speed: +0.73 vs -0.102. High-band flux -> hand speed: "
            "+0.65 vs -0.321. High-band flux -> hand spread: +0.57 vs -0.396 (CI excludes "
            "zero, wrong sign). The audio features do vary across the corpus (CV 0.16 to "
            "0.68), so this is not a range restriction. The explanation is the corpus "
            "design: the genre label alone explains 0.835 of between-track variance in "
            "quantity of motion, 0.916 in centre-of-mass speed, 0.879 in vertical bounce "
            "amplitude and 0.955 in footfall rate. With genre and tempo partialled out, "
            "every audio-movement correlation collapses to between -0.18 and +0.21 with "
            "CIs spanning zero. AIST++ is choreographed street dance whose music was "
            "chosen to match the style, so the arrow runs genre -> both music and "
            "movement; Burger's participants moved freely to varied music, where the "
            "arrow runs music -> movement. The shared component is real and highly "
            "reliable (split-half up to 0.94 for footfall rate) — it is simply not "
            "acoustic. Block A therefore needs free-dancing data; the Oslo club-dancing "
            "EDM mocap (Zenodo, CC-BY-4.0) is the corpus for it."),
        "topic_tags": ["music-induced-movement", "cross-modal", "motion-to-light"],
    })

    c.call("close_experiment", {
        "experiment_id": eid,
        "notes": (
            "Gate fails and the reason is the corpus, not the method. Three things are "
            "now measured rather than assumed:\n"
            "(1) There IS a large, highly reliable shared movement component in AIST++ "
            "(split-half reliability of the across-dancer mean reaches 0.94), so the "
            "averaging design works.\n"
            "(2) That component is genre, not audio: the genre label explains 0.65-0.96 "
            "of between-track movement variance, and audio recovers genre only weakly "
            "(28 % accuracy over 10 genres from 10 track-mean features, chance 10 %; a "
            "proper audio classifier would do better, so this is a floor).\n"
            "(3) Within a track, causal audio does predict a little of the shared "
            "dynamics — laterality R2 0.192 (CI [+0.064, +0.325]), hand speed 0.183, "
            "centre-of-mass speed 0.128, hip yaw 0.117, vertical bounce 0.092 — which is "
            "the honest ceiling for a dance-supervised block A trained on this corpus.\n"
            "Two method bugs were found and fixed during the audit, both of which "
            "produced spectacular nonsense first: selecting the ridge penalty on data the "
            "model had been fitted to (held-out R2 of -19), and scoring audio against a "
            "single dancer's trace rather than the across-dancer mean, which charges "
            "audio for variance no audio model could explain."),
    })
    c.call("interpret_experiment", {
        "experiment_id": eid,
        "failure_mode": "recipe_conditional",
        "lesson": (
            "The recipe is conditional on the corpus. Dance-derived supervision only "
            "carries audio information when the music is the only thing that varies; in "
            "a choreographed corpus the choreography carries it instead. Practically: "
            "(a) keep AIST++ for what it is good for — style priors per genre, motion "
            "realism as an FID reference, and validating the video-to-pose path against "
            "mocap; (b) get free-dancing data for block A (Oslo EDM club mocap, CC-BY-4.0, "
            "16 and 29 dancers to DJ-mixed EDM with drop/break annotations), which is also "
            "the concert domain the product actually ships into; (c) in the meantime train "
            "Stage 1 with the honest ceiling in mind — within-track shared dynamics at "
            "R2 0.1-0.2, timing from audio, style from a genre token — and never report a "
            "block A number without the reliability ceiling next to it."),
    })
    c.call("update_runbook", {
        "runbook_id": "rb-ls2-00b-predictability-audit",
        "note": ("2026-09-04 RUN. Gate FAILED on AIST++. Burger 2013 does not replicate "
                 "here (every predicted correlation zero or inverted); genre explains "
                 "0.65-0.96 of between-track movement variance. Within-track shared "
                 "dynamics R2 0.09-0.19. Re-run required on free-dancing data (Oslo EDM "
                 "mocap) before block A can be called dead or alive in general."),
    })
    lv2 = len(list(Path("/root/lightsem-vault").rglob("*.md")))
    ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    print(f"lightsem {lv}->{lv2} md, optiHealth {ov}->{ov2} md")
    assert ov2 == ov
    c.close()


if __name__ == "__main__":
    main()
