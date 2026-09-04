#!/usr/bin/env python
"""Record Stage 4 (software half), the video-pose validation, and verification."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client
P = "dancing-light-generative-20260903"
L = json.loads(Path("/root/lightsem/outputs/stage4/latency.json").read_text())
V = json.loads(Path("/root/lightsem/outputs/stage0a/pose_validation.json").read_text())
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))

# --- Stage 4 (software half) ------------------------------------------------
h = c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": ("The deployed path meets its latency budget and is causal by "
                  "construction: transients under 20 ms, the learned path under 65 ms "
                  "audio-in to DMX-out, and no dependence on future audio."),
    "predicted_effects": [{"diagnostic_id": "audio-to-dmx-latency-ms", "direction": "down",
        "magnitude_note": "learned path p99 <= 65 ms, transient path <= 20 ms"}],
    "rationale": ("ITU-R BT.1359-1 puts audible sound-before-light at +45 ms and "
                  "flash-tone JNDs at 20-70 ms, so a light that follows a kick by 80 ms "
                  "reads as a miss regardless of how good the model is."),
    "plan": ("Time each stage over many blocks on the RTX 5070 and report p50/p99: "
             "analysis hop, model forward with a fully primed streaming context, "
             "conductor, DMX packing. Test causality by replacing all audio after a cut "
             "with noise and asserting the output before the cut is bit-identical."),
    "gates": [{"id": "G-lat-transient", "description": "transient path <= 20 ms"},
              {"id": "G-lat-learned", "description": "learned path p99 <= 65 ms"},
              {"id": "G-causal-mech", "description": "future perturbation changes nothing "
                                                     "before the cut"}],
    "topic_tags": ["real-time", "audio-to-light"], "problem_tags": ["matched-baseline"]})
hid = h.get("id") or h.get("hypothesis_id")
e = c.call("start_experiment", {"hypothesis_id": hid, "config_note": json.dumps({
    "stage": "4 (software half)", "runbook": "rb-ls2-04", "device": L["device"],
    "receptive_field_frames": L["receptive_field_frames"],
    "receptive_field_s": L["receptive_field_s"],
    "not_measured": "fixture response and Art-Net on real hardware; photodiode timing"})})
eid = e.get("id") or e.get("experiment_id")
b = L["budget"]
c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
    {"diagnostic_id": "audio-to-dmx-latency-ms", "before": 65.0,
     "after": round(b["learned_path_p99_ms"], 1)}]})
c.call("record_gate_results", {"experiment_id": eid, "results": [
    {"gate_id": "G-lat-transient", "passed": True, "evidence": L["gates"]["transient_path_le_20ms"]["evidence"]},
    {"gate_id": "G-lat-learned", "passed": True, "evidence": L["gates"]["learned_path_le_65ms"]["evidence"]},
    {"gate_id": "G-causal-mech", "passed": True, "evidence": L["gates"]["causality_bit_identical"]["evidence"]}]})
c.call("close_experiment", {"experiment_id": eid, "notes": (
    "All three software gates pass, and one plan assumption is corrected on the way.\n"
    "The plan specified a 3-4 s KV cache. That is the attention WINDOW, not the "
    "receptive field: windows compose across layers, so six blocks of a 160-frame window "
    "plus a 15-tap causal convolution look back 1039 frames, 26 s of audio. The error was "
    "not theoretical — chunked inference primed with one window gave block A differences "
    "of 0.79 between chunk sizes, meaning the same audio produced different light "
    "depending on how it was sliced. With the full context the model costs 3.7 ms p50 and "
    "4.3 ms p99 per frame (against 1.7 ms when under-primed), and the learned path lands "
    "at 61.9 ms p50 / 62.5 ms p99 against its 65 ms budget.\n"
    "The transient path cannot share the learned path's 23.2 ms analysis hop: that alone "
    "costs 28.2 ms and misses a 20 ms budget. On its own 5.8 ms hop, onset detection plus "
    "network is 10.8 ms. This is an independent, purely engineering argument for the same "
    "two-path split that Stage 0a reached from the data.\n"
    "Not measured here and not claimed: fixture response, Art-Net on a real node, and a "
    "photodiode against a click track. The 30 ms fixture term in the budget is an "
    "assumption.")})
c.call("update_runbook", {"runbook_id": "rb-ls2-04-causal-student-latency",
    "note": "2026-09-04 software half RUN. Transient 10.8 ms, learned p99 62.5 ms, "
            "causality leak exactly 0.0. Receptive field is 26 s, not the 3-4 s the plan "
            "assumed. Hardware half (photodiode, Art-Net node, lookahead sweep) not run."})

# --- video pose path --------------------------------------------------------
h2 = c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": ("Descriptors extracted from ordinary video with an off-the-shelf 2D pose "
                  "model match mocap-derived descriptors well enough to use video as a "
                  "data source (r > 0.8 on speed, laterality and spread)."),
    "predicted_effects": [{"diagnostic_id": "descriptor-audio-r2-over-identity",
        "direction": "up", "magnitude_note": "proxy: channel-wise correlation to mocap > 0.8"}],
    "rationale": ("Mocap corpora are small and licence-bound; dance video is not. A pose "
                  "model returns the same COCO-17 skeleton AIST++ ships, so the same "
                  "descriptor code can run on both and be compared directly."),
    "plan": ("Download AIST camera-1 video for 12 clips whose mocap we hold, run "
             "YOLO11x-pose with tracking, embed the 2D skeleton as (lateral, up, 0), and "
             "push it through the identical prepare/extract path. Correlate channel by "
             "channel against the mocap descriptors."),
    "gates": [{"id": "G3", "description": "mean r over quantity of motion, head speed, "
                                          "hand speed, spread and laterality > 0.8"}],
    "topic_tags": ["pose", "motion-to-light"], "problem_tags": ["matched-baseline"]})
hid2 = h2.get("id") or h2.get("hypothesis_id")
e2 = c.call("start_experiment", {"hypothesis_id": hid2, "config_note": json.dumps({
    "stage": "0a G3", "model": "yolo11x-pose", "n_clips": V["n_clips"],
    "video": "AIST c01, 1920x1080 at 59.94 fps, frame count matches the mocap exactly",
    "tracking": "bytetrack, longest track kept (largest-box selection swaps subject "
                "mid-clip in a mirrored studio)"})})
eid2 = e2.get("id") or e2.get("experiment_id")
pc = V["per_channel"]
top = sorted([(v["mean_r"], k) for k, v in pc.items() if v["transferable"]], reverse=True)
c.call("record_gate_results", {"experiment_id": eid2, "results": [{
    "gate_id": "G3", "passed": False,
    "evidence": (f"{V['gate_G3']['evidence']}. The average hides a clean split. Channels "
                 f"that transfer: " + ", ".join(f"{k} {r:.2f}" for r, k in top[:6]) +
                 ". Channels that do not: twist 0.14, hip yaw 0.19, footfall rate 0.30, "
                 "symmetry 0.40 — every one of them needs depth or exact contact timing. "
                 "The vertical channel, which Stage 0a identified as the one carrying "
                 "musical phase, is among the best transferred (vertical position 0.87, "
                 "bounce 0.84, bounce amplitude 0.79).")}]})
c.call("close_experiment", {"experiment_id": eid2, "notes": (
    "Gate fails as written and the result is still usable, which is why the per-channel "
    "table matters more than the average. Eight of eighteen transferable channels correlate "
    "above 0.65 with mocap, including the whole vertical family (0.79-0.87) and the shape "
    "channels (hand spread 0.81, centre-of-mass speed 0.78). The channels that fail are "
    "exactly the ones the method cannot see: torso twist and hip yaw need depth, footfall "
    "rate needs contact timing a single camera does not give.\n"
    "So video scales the energy, bounce and shape half of block A and not the contact half. "
    "For a product whose light mapping is driven by intensity, bounce rate and which side "
    "the body is on, that is the useful half.")})
c.call("interpret_experiment", {"experiment_id": eid2, "failure_mode": "recipe_conditional",
    "lesson": ("Split the gate by channel rather than averaging it. A 2D pose path is a "
               "valid source for the vertical, energy and shape channels (r 0.65-0.87) and "
               "must not be used for twist, hip yaw or footfall timing; those come from 3D "
               "lifting or are dropped. Recorded in lightsem/motion/pose2d.py as "
               "TRANSFERABLE and NOT_TRANSFERABLE so the distinction is in the code, not "
               "only in a note.")})

c.call("update_problem", {"problem_id": P, "add_update": (
    "2026-09-04 implementation pass complete. Built and verified: descriptor extractor "
    "(mocap and video paths), constant-tempo beat grid, canonical-rig conductors "
    "(motion-only, hybrid, beat-trigger baseline), evaluation suite with nulls, causal "
    "Conformer with three heads, streaming inference, blind A/B harness. 14 of 14 "
    "verification checks pass, including two bugs they caught: a DMX round trip and a "
    "streaming context primed to one attention window instead of the 26 s receptive field. "
    "Stage gates: 0a partial (FID sensitivity passed, dance-derived timing failed), 0b "
    "failed on AIST++ and passed on Oslo free-dance EDM, 1 failed at the light level with "
    "an identified mechanism, 2B passed decisively (+0.31 lift, z 3.66, 699 shows), 4 "
    "software half passed. Human stages 0c and 3 have materials generated and await raters.")})
lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov
c.close()
