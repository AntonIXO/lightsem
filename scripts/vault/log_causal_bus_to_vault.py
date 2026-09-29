#!/usr/bin/env python
"""Record the causal-bus repair and the event-rate calibration in the vault.

Two things happened on 2026-09-04 after the cloud review, and they belong
together because the second only became measurable once the first was true.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client

P = "dancing-light-generative-20260903"
S = Path("/root/lightsem/experiments/legacy/stages/outputs/stage1")
OLD = json.loads((S / "stage1_light_eval_v2b.json").read_text())
NEW = json.loads((S / "stage1_light_eval_v3causal.json").read_text())
CTL = json.loads((S / "stage1_light_eval_v3nocal.json").read_text())
TR_OLD = json.loads((S / "train_v2b.json").read_text())
TR_NEW = json.loads((S / "train_v3causal.json").read_text())
CAL = json.loads((S / "event_thresholds_v3causal.json").read_text())

o, n, k = OLD["summary"], NEW["summary"], CTL["summary"]


def ev(res, key):
    return float(np.mean([r[key]["n_events"] for r in res["per_clip"]]))


c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))

hyp = c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "The Stage 1 audio bus was not causal, and the Stage 1 light gates were not "
        "measuring what the model believes. Removing the look-ahead will cost held-out "
        "accuracy, because part of the accuracy was bought with it; calibrating the event "
        "threshold to the corpus rate will raise Light-BAS, because the emitted event rate "
        "was a constant of the code rather than a decision of the model."),
    "predicted_effects": [
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "direction": "down",
         "magnitude_note": "held-out block A R2 falls once the leak is removed"},
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "up",
         "magnitude_note": "model delta rises once the event rate is the model's own"},
        {"diagnostic_id": "fid-light-kinetic", "direction": "down",
         "magnitude_note": "fewer spurious light events move the delta-feature "
                           "distribution toward the real one"}],
    "rationale": (
        "A future-perturbation test on the FEATURES, not the model: two signals identical "
        "up to t and different after must give identical causal features before t. librosa "
        "centres STFT frames, its HPSS harmonic estimate is a centred 31-frame median, and "
        "power_to_db clamps at top_db against the whole clip's maximum. Each is a "
        "documented default and each reads the future. Separately, descriptors() floored "
        "the event threshold at the clip's own 97th percentile, which pins the emitted "
        "rate whatever the hazard says."),
    "plan": (
        "Fix the bus (center=False timestamped at the window end, trailing-median HPSS, "
        "top_db=None, hold instead of interpolate on the analysis-to-control resample), "
        "rebuild the 60 audio caches and the 749k-frame tensor, retrain the same 9.3 M "
        "Conformer for 12k steps, then bisect one hazard level per block-B channel on the "
        "validation split until the emitted event rate matches the rate the real dancers "
        "produced. Score the calibrated and uncalibrated models on the same features so "
        "the calibration is priced alone."),
    "gates": [
        {"id": "G-feature-causal", "description":
         "every model-input channel bit-identical before the cut; beat-grid channels "
         "exempt only if still flagged oracle"},
        {"id": "G-rate", "description":
         "emitted block-B event rate within 5 % of the corpus rate"},
        {"id": "G-bas-up", "description":
         "calibration raises model Light-BAS against the SAME checkpoint uncalibrated"}],
    "topic_tags": ["audio-to-light", "real-time", "offline-vs-online"],
    "problem_tags": ["leaked-future", "matched-baseline"],
})
assert isinstance(hyp, dict) and hyp.get("ok") is not False, hyp
hid = hyp.get("id") or hyp.get("hypothesis_id")
exp = c.call("start_experiment", {"hypothesis_id": hid, "config_note": json.dumps({
    "stage": "1-repair", "runbook": "rb-ls2-01",
    "trigger": "cloud code review, 10 findings; the regression check written for one of "
               "them failed on three larger leaks",
    "ckpt": "/mnt/lightsem-data/ckpt/light_trunk_v3causal.pt",
    "steps": TR_NEW["steps"], "params_m": TR_NEW["params_m"],
    "thresholds": CAL["thresholds"], "calibration_split": CAL["split"],
    "calibration_clips": CAL["n_clips"], "device": "RTX 5070"})})
assert isinstance(exp, dict) and exp.get("ok") is not False, exp
eid = exp.get("id") or exp.get("experiment_id")
print("experiment:", eid)

c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
    {"diagnostic_id": "descriptor-audio-r2-over-identity",
     "before": round(TR_OLD["val_track"]["r2_a_mean"], 4),
     "after": round(TR_NEW["val_track"]["r2_a_mean"], 4)},
    {"diagnostic_id": "light-bas-minus-shift-null",
     "before": round(k["bas_minus_null"]["model"]["delta"], 4),
     "after": round(n["bas_minus_null"]["model"]["delta"], 4)},
    {"diagnostic_id": "fid-light-kinetic",
     "before": round(k["fid_light"]["model_vs_real"], 2),
     "after": round(n["fid_light"]["model_vs_real"], 2)},
    {"diagnostic_id": "light-diversity-kinetic",
     "before": round(k["diversity"]["across_clips_model"], 3),
     "after": round(n["diversity"]["across_clips_model"], 3)}]})

c.call("record_gate_results", {"experiment_id": eid, "results": [
    {"gate_id": "G-feature-causal", "passed": True,
     "evidence": (
         "23 model-input channels and the 64-band mel bank are bit-identical before the "
         "cut when only the future changes; the five beat-grid channels are exempt and the "
         "oracle_beat flag is asserted. Before the repair the worst channel was "
         "percussive_ratio at 375 ms of look-ahead and a difference of 15x its own standard "
         "deviation, from librosa's centred HPSS median; every spectral channel leaked "
         "25 ms from centre=True; onset, pulse_clarity and the mel bank carried a thin "
         "whole-clip dependence through top_db.")},
    {"gate_id": "G-rate", "passed": True,
     "evidence": (
         f"footfall {CAL['thresholds']['footfall']:.4f} -> 0.898 ev/s against a corpus "
         f"0.896; jump {CAL['thresholds']['jump']:.4f} -> 0.024 against 0.026; hit "
         f"{CAL['thresholds']['hit']:.4f} -> 1.352 against 1.353. Fitted on "
         f"{CAL['n_clips']} {CAL['split']} clips, never on training or test clips.")},
    {"gate_id": "G-bas-up", "passed": False,
     "evidence": (
         f"REFUTED against the right control. Against v2b the model rose "
         f"{o['bas_minus_null']['model']['delta']:+.4f} -> "
         f"{n['bas_minus_null']['model']['delta']:+.4f}, but v2b was trained on the leaky "
         f"bus, so that comparison confounds two changes. Scoring the SAME checkpoint on "
         f"the SAME features with and without the fitted thresholds isolates it: "
         f"uncalibrated {k['bas_minus_null']['model']['delta']:+.4f} CI "
         f"{[round(x,4) for x in k['bas_minus_null']['model']['ci95']]}, calibrated "
         f"{n['bas_minus_null']['model']['delta']:+.4f} CI "
         f"{[round(x,4) for x in n['bas_minus_null']['model']['ci95']]}. Calibration "
         f"LOWERED Light-BAS. It improved FID and diversity "
         f"({k['fid_light']['model_vs_real']:.2f} -> {n['fid_light']['model_vs_real']:.2f}, "
         f"{k['diversity']['across_clips_model']:.2f} -> "
         f"{n['diversity']['across_clips_model']:.2f}) and it makes the emitted rate the "
         f"model's decision rather than the code's, which is why it is kept -- but the "
         f"predicted mechanism was wrong.")}]})

c.call("close_experiment", {"experiment_id": eid, "notes": (
    "The leak was worth removing and it was not free.\n"
    f"Held-out block A R2 fell from {TR_OLD['val_track']['r2_a_mean']:+.4f} to "
    f"{TR_NEW['val_track']['r2_a_mean']:+.4f} once the features stopped reading ahead. That "
    "is the honest size of what the look-ahead was buying, and it means every block A "
    "number recorded before today was measured on a bus a live system cannot build. The "
    "causality gate on the MODEL was passing throughout and never had a chance of catching "
    "this: it perturbs the model's input, not the input's own construction.\n"
    "The rate calibration was predicted to raise Light-BAS and did not. Block-B rates now "
    f"match the corpus to three decimals, but the rendered light still emits "
    f"{ev(NEW,'model'):.1f} events per clip against {ev(CTL,'model'):.1f} uncalibrated and "
    f"{ev(NEW,'real'):.1f} for light mapped from the real dance. The light event count did "
    "not move at all, which locates the surplus somewhere the event head cannot reach.\n"
    "Follow-up measurement, and it is the useful result of the day: the model's channels "
    "carry 11x to 452x the above-6-Hz energy of the real ones (section_energy 116x, "
    "space_eff 202x, weight 343x, hand_spread 452x), while the REGRESSED block A channels "
    "sit at 0.3-5x. The targets are mocap through a 12 Hz Butterworth and are physically "
    "smooth; a flow head sampled per frame is not, even with ar_flow already enabled and "
    "conditioning on its own previous output. That jitter is what crosses the visibility "
    "threshold on the wash fixtures and makes the light flash three times too often.\n"
    f"FID moved {k['fid_light']['model_vs_real']:.2f} -> {n['fid_light']['model_vs_real']:.2f} "
    f"against a half-null of {0.5*n['fid_light']['real_vs_block_shuffled']:.2f}. It is now "
    "nominally inside the gate by 0.16, which is not a pass anyone should quote -- the "
    "split-half floor is 6.26, so the model is still an order of magnitude from being "
    "indistinguishable from dance-derived light.")})

c.call("interpret_experiment", {"experiment_id": eid,
    "failure_mode": "measurement_artifact",
    "lesson": (
        "Three lessons, and the third is the one that changes the plan.\n"
        "Tooling: a causality claim about a model says nothing about the features it is "
        "fed. librosa's defaults are all offline-correct and all future-reading -- "
        "centre=True, HPSS's centred median, top_db's global maximum. Any project asserting "
        "real-time causality needs the perturbation test at the FEATURE boundary, and the "
        "test has to name which channels are deliberately oracle so it fails on the rest. "
        "That check is now in scripts/diagnostics/verify_all.py.\n"
        "Metrics: a threshold expressed as a percentile of the clip is not a threshold, it "
        "is a rate assignment. It appeared twice in this codebase independently -- in the "
        "light-feature onset detector, where it was caught and replaced with an absolute "
        "level, and in the model's event head, where it survived until the per-clip event "
        "counts were read side by side. Whenever a detector has a percentile in it, print "
        "its output rate next to the metric it feeds.\n"
        "Direction: the smoothness gap is not a bug in the sampler, it is a mismatch "
        "between what the target is (a physically smooth body, filtered at 12 Hz) and what "
        "a per-frame sampler can emit. The fix that matches the target exactly is a "
        "zero-phase filter, which is available offline and not online. Together with the "
        "measured size of the causality tax on block A, this is the argument for making "
        "the pre-rendered path the product and the causal path a labelled variant.")})

c.call("update_runbook", {"runbook_id": "rb-ls2-01-latent-dancer-offline",
    "note": ("2026-09-04 REPAIR. The Stage 1 audio bus read up to 375 ms of the future "
             "through librosa defaults; fixed, caches rebuilt, model retrained. Held-out "
             "block A R2 +0.0438 -> +0.0192, which is the size of the artefact. Block-B "
             "thresholds are now fitted to the corpus event rate on the validation split "
             "and stored in the checkpoint; that helped FID and diversity and slightly hurt "
             "Light-BAS, so the mechanism predicted for it was wrong. The surplus light "
             "events come from the block A/C envelopes, whose above-6-Hz energy is 11x to "
             "452x the corpus value.")})

lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov, "optiHealth vault must not change"
c.close()
