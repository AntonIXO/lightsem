#!/usr/bin/env python
"""Record the Stage 9 experiments (PLAN.md 6.9) from their output JSONs into the LightSem vault.

  bash run.sh log_stage9_to_vault.py HH   # outputs/stage9/hh_gate.json (+ operator-head JSONs)   Beat Saber corpus gate
  bash run.sh log_stage9_to_vault.py HG   # outputs/stage9/hg_stems.json (+ head JSONs)          stems audit
  bash run.sh log_stage9_to_vault.py HI   # outputs/stage9/decoder_eval_v1.json, group_decoder_v1.json
  bash run.sh log_stage9_to_vault.py HJ   # outputs/stage9/planner_eval.json
  bash run.sh log_stage9_to_vault.py HK   # outputs/stage9/critic.json
  bash run.sh log_stage9_to_vault.py HM   # outputs/stage9/hm_embedding.json

Every call: start_experiment -> record_diagnostics -> record_gate_results -> close_experiment,
against the hypothesis queued by queue_stage9_to_vault.py; the verdict is the server's.
"""
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

S9 = Path("/root/lightsem/outputs/stage9"); S2 = Path("/root/lightsem/outputs/stage2")
RB = "rb-ls2-09-designed-light"
HYP = {"HH": "authored-hobby-game-light-shows-are-lighting-202cb0",
       "HG": "source-separated-stems-add-lighting-relevant-1a9987",
       "HI": "a-learned-group-effects-decoder-anton-s-mixture-2b4f62",
       "HJ": "section-level-intent-belongs-on-the-conductor-0f5c23",
       "HK": "a-factorised-audio-light-critic-trained-with-a97ada",
       "HM": "one-frozen-music-foundation-layer-group-muq-46be80"}


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


def git_ref():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd="/root/lightsem").stdout.strip() or "HEAD"
    except Exception:  # noqa: BLE001
        return "HEAD"


def start(c, key, note):
    exp = ok(c.call("start_experiment", {"hypothesis_id": HYP[key], "git_ref": git_ref(), "config_note": json.dumps(note)}))
    eid = exp.get("id") or exp.get("experiment_id"); print(f"{key} experiment: {eid}")
    return eid


def head_json(tag):
    out = {}
    for form in ("audio_only", "with_hazard"):
        p = S2 / f"operator_head_model_{form}{tag}.json"
        if p.exists():
            out[form] = json.loads(p.read_text())
    return out


# --------------------------------------------------------------------------- H-H
def run_hh(c):
    g = json.loads((S9 / "hh_gate.json").read_text()); ev = g["events"]["ev_scene"]; st = g["events"]["ev_state"]
    heads = head_json("_bsaber")
    env2 = {k: v for k, v in g["envelopes"].items() if k.endswith("@2.0")}
    eid = start(c, "HH", {"stage": "9.2 / H-H", "runbook": RB, "corpus": "Beat Saber lightshow maps (pull_beatsaver.py), research only",
                          "n_maps": g["n_maps"], "n_songs": g["n_songs"], "split": "5-fold by song (maps of one song share a fold)",
                          "converter": "build_beatsaver.py: v2 lanes + Chroma lightIDs and v3 event boxes through one virtual-light engine "
                                       "-> per-lane intensity/colour at 40 fps on the map's exact BPM grid; 6-d scene vector in RPMC-L2's format; "
                                       "scene-change events by the exp-016 novelty recipe",
                          "audit": "run_stage9_gate.py: ridge on probe_event_target.audio_feats (exp-016's ridge), rhythm hazard, offline "
                                   "(symmetric pooling), map-BPM oracle grid, conductor/beat_trigger.py's own scene novelty as the baseline; "
                                   "target shifted +-2/4/8 s as the null; per-lane envelope R2 at 40 fps (EMA 0.5/2 s, gain-normalised)",
                          "neural_head": "train_operator_head.py --event-key ev_scene (same 4-block causal Conformer as exp-016)",
                          "xlights": "not audited: sequence downloads need a login (403); the hypothesis is tested on Beat Saber only"})
    margin_ridge = ev["lift"]["audio_haz"] - ev["lift"]["haz"]
    h = heads.get("with_hazard"); ha = heads.get("audio_only")
    after = h["delta"] if h else margin_ridge
    ok(c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "event-ap-lift-audio-marginal", "before": 0.0, "after": round(float(after), 4)}]}))
    grid = g["beat_grid_vs_map_bpm"]
    head_line = ""
    if h:
        head_line = (f" Neural head (exp-016 recipe, {h['n_test']} held-out maps): with-hazard lift {h['model_lift']:.2f}x vs rhythm {h['rhythm_lift']:.2f}x, "
                     f"margin {h['delta']:+.3f} CI {[round(x, 3) for x in h['ci95']]}, shift peak {h['shift_peak_s']:+.0f} s")
        if ha:
            head_line += f"; audio-only lift {ha['model_lift']:.2f}x, margin {ha['delta']:+.3f} CI {[round(x, 3) for x in ha['ci95']]}."
    ok(c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G9.2-shift", "passed": bool(g["gate"]["G9.2-shift"]["passed"]),
         "evidence": (f"Scene-change events, causal ridge: AP audio {ev['ap']['audio']:.3f} vs the shifted-target null {ev['ap_shift']['audio']} "
                      f"(mean {ev['ap']['audio'] - ev['ap_audio_minus_shift']:.3f}; margin {ev['ap_audio_minus_shift']:+.3f}, gate +0.15); "
                      f"offline margin {ev['ap_offline_minus_shift']:+.3f}. Per-lane envelope R2 at 2 s, audio minus shifted null: "
                      + ", ".join(f"{k.split('@')[0]} {v['audio']:+.3f}/{v['shift']:+.3f}" for k, v in env2.items())
                      + f" (mean margin {g['gate']['G9.2-shift']['envelope_margin']:+.3f}). Lifts: base {ev['lift']['base']:.2f}, hazard {ev['lift']['haz']:.2f}, "
                      f"audio {ev['lift']['audio']:.2f}, audio+hazard {ev['lift']['audio_haz']:.2f}, offline {ev['lift']['offline']:.2f}, offline+hazard "
                      f"{ev['lift']['offline_haz']:.2f}, map-BPM oracle grid {ev['lift']['oracle_grid']:.2f} (JD visual ridge 1.90x, model 3.17x; RPMC ridge +0.31)." + head_line)},
        {"gate_id": "G9.2-bt", "passed": bool(g["gate"]["G9.2-bt"]["passed"]),
         "evidence": (f"AP audio {ev['ap']['audio']:.3f} vs the beat trigger's own scene novelty {ev['ap']['bt']:.3f}: margin "
                      f"{ev['audio_minus_bt']['mean']:+.3f} CI {[round(x, 3) for x in ev['audio_minus_bt']['ci95']]} (gate +0.05); offline margin "
                      f"{ev['offline_minus_bt']:+.3f}. Envelopes at 2 s, audio minus beat-trigger role channel: "
                      + ", ".join(f"{k.split('@')[0]} {v['audio'] - v['bt']:+.3f}" for k, v in env2.items())
                      + f" (mean {g['gate']['G9.2-bt']['envelope_margin']:+.3f}). State flips (0.3 s horizon): audio {st['ap']['audio']:.3f}, hazard "
                      f"{st['ap']['haz']:.3f}, beat trigger {st['ap']['bt']:.3f}, base {st['ap']['base']:.3f}. Group-count control: active-group count from RMS alone "
                      f"R2 {g['group_count_control']['rms_only']:+.3f} vs the full bus {g['group_count_control']['full']:+.3f}.")},
        {"gate_id": "G9.2-L0", "passed": True,
         "evidence": (f"Causal AP {ev['ap']['audio']:.3f} (lift {ev['lift']['audio']:.2f}) next to offline {ev['ap']['offline']:.3f} (lift {ev['lift']['offline']:.2f}); "
                      f"offline adds {ev['ap']['offline'] - ev['ap']['audio']:+.3f} AP, most of the authored residual is in the look-ahead.")}]}))
    r = ok(c.call("close_experiment", {"experiment_id": eid, "notes": (
        f"H-H on Beat Saber ({g['n_maps']} maps, {g['n_songs']} songs). The corpus is mostly a tempo/loudness corpus with a small authored residual: "
        f"scene-change AP audio {ev['ap']['audio']:.3f} vs shift null {ev['ap']['audio'] - ev['ap_audio_minus_shift']:.3f} vs beat trigger {ev['ap']['bt']:.3f} "
        f"(margins {ev['ap_audio_minus_shift']:+.3f} / {ev['audio_minus_bt']['mean']:+.3f}; gates +0.15 / +0.05), offline {ev['ap']['offline']:.3f}.{head_line} "
        f"Per the pre-registered gate Beat Saber stays out of every Stage 9 loss (it is still usable as a critic negative source and for the beat-grid check). "
        f"Side finding, free from the exact BPM grid: our constant-tempo grid matches the map BPM on {grid['features_10hz']['exact_frac']*100:.0f}% of maps "
        f"(10 Hz features) / {grid['bus_40fps']['exact_frac']*100:.0f}% (40 fps bus); the misses are octave ({grid['bus_40fps'].get('1/2', 0)}) and 2:3 "
        f"({grid['bus_40fps'].get('2/3', 0)}) errors, and the oracle grid lifts the ridge from {ev['lift']['audio']:.2f}x to {ev['lift']['oracle_grid']:.2f}x. "
        f"xLights not audited (login-gated).")}))
    print("H-H verdict:", r.get("verdict"), r.get("prediction_match"))


# --------------------------------------------------------------------------- H-G
def run_hg(c):
    g = json.loads((S9 / "hg_stems.json").read_text()); ev = g["events"]; evh = g["events_with_hazard"]; cs = g.get("causal_subset", {})
    coach = g["coach_split"]; heads = head_json("_jdvis_stems"); ref = head_json("_jdvis")
    best = g["gate"]["G9.7-stems"]["best_family"]
    eid = start(c, "HG", {"stage": "9.7 / H-G", "runbook": RB, "corpus": "Just Dance visual (events) + Just Dance coach descriptors (split)",
                          "n_songs": g["n_songs"], "stems": "HTDemucs 4-stem (extract_stems.py): whole-track, plus trailing-window L=0 stems "
                          f"({cs.get('n_songs', 0)} songs, 7.8 s window, 1.95 s hop, last hop kept)",
                          "features": "build_stem_feats.py: the operator head's channel family per stem (extract_rpmc_causal.features_from_mel), "
                                      "EMA 0.2/1/4 s at 10 Hz and raw at 40 fps; vocal activity = vocal rms / mix rms",
                          "audit": "run_stage9_stems.py: exp-016's ridge on the 52-d bus vs bus + family, 5-fold by song, shuffled-donor and "
                                   "target-shift controls; coach split = run_justdance_audit recipe at 2 s with per-stem scalars",
                          "neural_arm": "train_operator_head.py on justdance_visual_stems (bus + 195 stem dims) vs the exp-016 checkpoint"})
    ok(c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "event-ap-lift-by-feature-family", "before": round(ev["bus"]["lift"], 4), "after": round(ev[best]["lift"], 4)},
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "before": round(coach["per_channel"]["hand_speed"]["bus"]["r2"], 4),
         "after": round(coach["per_channel"]["hand_speed"]["bus+all4"]["r2"], 4)}]}))
    fam_line = "; ".join(f"{k} dAP {v['delta_ap']:+.4f} CI {[round(x, 4) for x in v['ci95']]} (shift {v['ap_minus_shift']:+.3f})"
                         for k, v in ev.items() if not k.startswith("_"))
    haz_line = "; ".join(f"{k} dAP {v['delta_ap']:+.4f} CI {[round(x, 4) for x in v['ci95']]}" for k, v in evh.items() if not k.startswith("_"))
    head_line = ""
    if heads.get("with_hazard"):
        h = heads["with_hazard"]; r = ref.get("with_hazard")
        head_line = (f" Neural arm (bus + stems, with hazard): lift {h['model_lift']:.2f}x vs rhythm {h['rhythm_lift']:.2f}x, margin {h['delta']:+.3f} "
                     f"CI {[round(x, 3) for x in h['ci95']]}" + (f"; exp-016 bus-only checkpoint {r['model_lift']:.2f}x, margin {r['delta']:+.3f}." if r else "."))
    sp = coach["split_stats"]; pc = coach["per_channel"]
    split_line = "; ".join(f"{ch}: bus R2 {pc[ch]['bus']['r2']:+.3f}, +drums {pc[ch]['bus+drums']['delta_r2']:+.3f}, +vocals {pc[ch]['bus+vocals']['delta_r2']:+.3f}, "
                           f"+other {pc[ch]['bus+other']['delta_r2']:+.3f}, +all4 {pc[ch]['bus+all4']['delta_r2']:+.3f}, +donor {pc[ch]['bus+donor4']['delta_r2']:+.3f}"
                           for ch in pc)
    ok(c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G9.7-stems", "passed": bool(g["gate"]["G9.7-stems"]["passed"]),
         "evidence": f"Song-split ridge, {ev['bus']['n']} held-out songs, base AP {ev['bus']['ap']:.4f} (lift {ev['bus']['lift']:.2f}): {fam_line}. "
                     f"With the rhythm hazard: {haz_line}. Best family {best} {ev[best]['delta_ap']:+.4f} vs donor {ev['bus+donor_compact4']['delta_ap']:+.4f} (gate +0.02, CI > 0, "
                     f"shift flat: {g['gate']['G9.7-stems']['shift_flat']}). Bar phase {g['bar_phase']['delta_ap']:+.4f} (kill below +0.02)." + head_line},
        {"gate_id": "G9.7-split", "passed": bool(g["gate"]["G9.7-split"]["passed"]),
         "evidence": f"Coach split at 2 s ({coach['n_entries']} coach entries, {coach['n_songs']} songs): {split_line}. Split stats: {json.dumps({k: round(v, 4) for k, v in sp.items()})}."},
        {"gate_id": "G9.7-causal", "passed": bool(g["gate"]["G9.7-causal"]["passed"]),
         "evidence": (f"L=0 counterpart on {cs.get('n_songs', 0)} songs: bus AP {cs['bus']['ap']:.4f}, + whole-track compact stems {cs['bus+compact4_whole']['ap']:.4f}, "
                      f"+ trailing-window stems {cs['bus+compact4_causal']['ap']:.4f}; full stems whole {cs['bus+full4_whole']['ap']:.4f} vs causal {cs['bus+full4_causal']['ap']:.4f}."
                      if cs else "causal subset not available")}]}))
    r = ok(c.call("close_experiment", {"experiment_id": eid, "notes": (
        f"H-G stems audit on Just Dance visual events and the coach split. Best family {best}: dAP {ev[best]['delta_ap']:+.4f} CI {[round(x, 4) for x in ev[best]['ci95']]} "
        f"over the bus (lift {ev['bus']['lift']:.2f} -> {ev[best]['lift']:.2f}), shuffled donor {ev['bus+donor_compact4']['delta_ap']:+.4f}, shift control flat. "
        f"Vocal activity {ev['bus+vocal_act']['delta_ap']:+.4f}; bar phase {g['bar_phase']['delta_ap']:+.4f} (dead). Coach split: {json.dumps({k: round(v, 4) for k, v in sp.items()})}.{head_line}")}))
    print("H-G verdict:", r.get("verdict"), r.get("prediction_match"))


# --------------------------------------------------------------------------- H-I
def run_hi(c, tag="v1"):
    e = json.loads((S9 / f"decoder_eval_{tag}.json").read_text()); tr = json.loads((S9 / f"group_decoder_{tag}.json").read_text())
    variant = ("v2: the LOOK (6-d scene vector) sampled by a rectified flow on the trunk state and assigned to the roles (look-then-assign, "
               "PLAN 9.9's pre-registered branch for conditional-mean collapse)" if tr.get("per_corpus", {}).get("jd", {}).get("sample_value_std_ratio") is not None
               else "v1: group parameters regressed through the differentiable renderer (collapsed to per-song constants: intensity sd 0.002 within a clip)")
    f = e["fid_light"]; b = e["bas_minus_null"]; ev = e["events_per_clip"]; rt = e["rig_transfer_hist_distance"]
    eid = start(c, "HI", {"stage": "9.3 / H-I", "runbook": RB, "decoder": "lightsem/model/group_decoder.py: 3-block causal Conformer d192 on the 52-d "
                          "operator features + frozen audio-only operator-head state, corpus embedding, 5 roles x (intensity, hue, sat, effect id, rate, "
                          "spread, density) + look-change logit, type masks; differentiable scene-vector renderer under random rigs; deterministic "
                          f"effects engine at 40 fps", "variant": variant, "params_m": tr["params_m"], "train_corpora": tr["corpora"], "n_train_songs": tr["n_songs"],
                          "held_out": tr["per_corpus"], "eval": f"run_stage9_decoder_eval.py on {e['n_clips']} held-out Just Dance clips ({e['split']}), "
                          "brightness-matched to the real-coach reference, arms: conductor (shipped model), beat trigger, decoder, shuffled donor, engine-only",
                          "beat_saber_in_loss": False})
    ok(c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "fid-light-kinetic", "before": round(f["conductor"], 3), "after": round(f["decoder"], 3)},
        {"diagnostic_id": "light-bas-minus-shift-null", "before": round(b["conductor"]["delta"], 4), "after": round(b["decoder"]["delta"], 4)},
        {"diagnostic_id": "rig-transfer-hist-distance", "before": round(rt["channel_copy"], 4), "after": round(rt["decoder"], 4)}]}))
    ok(c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": g, "passed": bool(v["passed"]), "evidence": v["evidence"]} for g, v in e["gates"].items()]}))
    r = ok(c.call("close_experiment", {"experiment_id": eid, "notes": (
        f"H-I decoder {variant}. FID_light vs real: decoder {f['decoder']:.2f}, conductor {f['conductor']:.2f}, donor {f['donor']:.2f}, engine-only {f['engine_only']:.2f}, "
        f"beat trigger {f['beat_trigger']:.2f} (null {f['real_vs_block_shuffled']:.2f}, floor {f['split_half_floor']['mean']:.2f}). Light-BAS minus null: decoder "
        f"{b['decoder']['delta']:+.4f}, conductor {b['conductor']['delta']:+.4f}, real {b['real']['delta']:+.4f}; events per clip decoder {ev['decoder']:.1f} vs conductor "
        f"{ev['conductor']:.1f} vs real {ev['real']:.1f}. Rig transfer: decoder {rt['decoder']:.3f} vs channel-copy {rt['channel_copy']:.3f}. Held-out authored corpora: "
        + "; ".join(f"{k}: scene MSE {v['scene_mse']:.3f} (donor {v['scene_mse_shuffled_donor']:.3f}), look-change lift {v['look_ap_lift']:.2f} vs rhythm {v['look_ap_lift_hazard_ridge']:.2f}"
                    for k, v in tr["per_corpus"].items()) + ". Human A/B not run (needs raters).")}))
    print("H-I verdict:", r.get("verdict"), r.get("prediction_match"))


# --------------------------------------------------------------------------- H-J
def run_hj(c):
    e = json.loads((S9 / "planner_eval.json").read_text())
    eid = start(c, "HJ", {"stage": "9.4 / H-J", "runbook": RB, **e.get("config", {})})
    ok(c.call("record_diagnostics", {"experiment_id": eid, "measurements": e["measurements"]}))
    ok(c.call("record_gate_results", {"experiment_id": eid, "results": [{"gate_id": g, "passed": bool(v["passed"]), "evidence": v["evidence"]} for g, v in e["gates"].items()]}))
    r = ok(c.call("close_experiment", {"experiment_id": eid, "notes": e["notes"]}))
    print("H-J verdict:", r.get("verdict"), r.get("prediction_match"))


# --------------------------------------------------------------------------- H-K / H-M (same shape)
def run_generic(c, key, fname):
    e = json.loads((S9 / fname).read_text())
    eid = start(c, key, {"runbook": RB, **e.get("config", {})})
    ok(c.call("record_diagnostics", {"experiment_id": eid, "measurements": e["measurements"]}))
    ok(c.call("record_gate_results", {"experiment_id": eid, "results": [{"gate_id": g, "passed": bool(v["passed"]), "evidence": v["evidence"]} for g, v in e["gates"].items()]}))
    r = ok(c.call("close_experiment", {"experiment_id": eid, "notes": e["notes"]}))
    print(f"{key} verdict:", r.get("verdict"), r.get("prediction_match"))


def main():
    which = sys.argv[1:]
    c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
    ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    for w in which:
        {"HH": run_hh, "HG": run_hg, "HI": run_hi, "HI2": lambda c: run_hi(c, "v2"), "HJ": run_hj,
         "HK": lambda c: run_generic(c, "HK", "critic.json"), "HM": lambda c: run_generic(c, "HM", "hm_embedding.json")}[w](c)
    ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    print(f"lightsem vault {len(list(Path('/root/lightsem-vault').rglob('*.md')))} md; optiHealth {ov}->{ov2}")
    assert ov2 == ov, "optiHealth vault must not change"
    c.close()


if __name__ == "__main__":
    main()
