#!/usr/bin/env python
"""Record the Stage 7 experiments (rb-ls2-07) from their JSON outputs.

  bash run.sh log_stage7_to_vault.py HA   # after train_v4offline + stage1_light_eval_v4offline
  bash run.sh log_stage7_to_vault.py HB   # after the three H-B arms
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
S = Path("/root/lightsem/outputs/stage1")
HYP = {"HA": "most-of-the-gap-between-the-model-s-light-0-018-8b5f6c",
       "HB": "the-model-s-light-flashes-three-times-too-often-73aafb"}


def J(name):
    return json.loads((S / name).read_text())


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


def run_ha(c):
    tr_off, tr_c = J("train_v4offline.json"), J("train_v3causal.json")
    ev_off, ev_c = J("stage1_light_eval_v4offline.json")["summary"], J("stage1_light_eval_v3causal.json")["summary"]
    r2_off, r2_c = tr_off["val_track"]["r2_a_mean"], tr_c["val_track"]["r2_a_mean"]
    bas_off, bas_c = ev_off["bas_minus_null"]["model"], ev_c["bas_minus_null"]["model"]
    exp = ok(c.call("start_experiment", {"hypothesis_id": HYP["HA"], "git_ref": "HEAD",
        "config_note": json.dumps({"stage": "7 H-A", "runbook": "rb-ls2-07-offline-prerendered",
            "trunk": "same 9.3 M Conformer, offline=True (symmetric attention window, symmetric conv padding)",
            "features": "extract_bus(causal=False) -> cache/audio_offline, tensor train_tensor_offline.npz",
            "frames": 749173, "steps": tr_off["steps"], "splits": tr_off["splits"],
            "causal_counterpart": "v3causal (same frames, splits, steps)",
            "leak_offline": tr_off["causality_leak"], "leak_causal": tr_c["causality_leak"]})}))
    eid = exp.get("id") or exp.get("experiment_id"); print("H-A experiment:", eid)
    c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "before": round(r2_c, 4), "after": round(r2_off, 4)},
        {"diagnostic_id": "light-bas-minus-shift-null", "before": round(bas_c["delta"], 4), "after": round(bas_off["delta"], 4)},
        {"diagnostic_id": "fid-light-kinetic", "before": round(ev_c["fid_light"]["model_vs_real"], 2),
         "after": round(ev_off["fid_light"]["model_vs_real"], 2)},
        {"diagnostic_id": "light-diversity-kinetic", "before": round(ev_c["diversity"]["across_clips_model"], 3),
         "after": round(ev_off["diversity"]["across_clips_model"], 3)}]})
    g_r2 = r2_off >= 0.04; g_bas = bas_off["delta"] >= 0.105
    c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G-ceiling-r2", "passed": bool(g_r2), "evidence": (
            f"offline held-out (val_track) block A R2 {r2_off:+.4f} against the causal {r2_c:+.4f} on the same "
            f"749k frames, splits and 12k steps; test_genre {tr_off['test_genre']['r2_a_mean']:+.4f} vs "
            f"{tr_c['test_genre']['r2_a_mean']:+.4f}, test_dancer {tr_off['test_dancer']['r2_a_mean']:+.4f} vs "
            f"{tr_c['test_dancer']['r2_a_mean']:+.4f}. Gate: >= +0.04.")},
        {"gate_id": "G-ceiling-bas", "passed": bool(g_bas), "evidence": (
            f"offline Light-BAS minus shift null {bas_off['delta']:+.4f} CI {[round(x,4) for x in bas_off['ci95']]} "
            f"against the causal {bas_c['delta']:+.4f} CI {[round(x,4) for x in bas_c['ci95']]}; real-mapped "
            f"{ev_off['bas_minus_null']['real_dance']['delta']:+.4f}, beat trigger "
            f"{ev_off['bas_minus_null']['beat_trigger']['delta']:+.4f}. Gate: >= +0.105.")},
        {"gate_id": "G-tax-reported", "passed": True, "evidence": (
            "Both numbers above are quoted next to their L=0 counterparts on identical splits; the offline "
            f"model's own causality probe leaks {tr_off['causality_leak']:.4g} (expected non-zero) against "
            f"{tr_c['causality_leak']:.4g} for the causal one.")}]})
    verdict = "the corpus binds" if not (g_r2 and g_bas) else "the model was the constraint"
    c.call("close_experiment", {"experiment_id": eid, "notes": (
        f"H-A, the look-ahead ceiling. Same trunk, same data, whole-track features and symmetric context. "
        f"Held-out block A R2 {r2_c:+.4f} -> {r2_off:+.4f}; Light-BAS {bas_c['delta']:+.4f} -> {bas_off['delta']:+.4f}; "
        f"FID {ev_c['fid_light']['model_vs_real']:.2f} -> {ev_off['fid_light']['model_vs_real']:.2f}; diversity "
        f"{ev_c['diversity']['across_clips_model']:.2f} -> {ev_off['diversity']['across_clips_model']:.2f}; light events "
        f"per clip {ev_c.get('events_per_clip', {}).get('model', float('nan')):.1f} -> "
        f"{ev_off.get('events_per_clip', {}).get('model', float('nan')):.1f} against real "
        f"{ev_off.get('events_per_clip', {}).get('real', float('nan')):.1f}. Reading: {verdict}. "
        f"PLAN.md 6.7 said that if an offline model with the entire track cannot clear +0.105 the binding "
        f"constraint is data, and the next move is Stage 8 (Just Dance), which is running.")})
    return eid


def run_hb(c):
    raw, ema_, zp = (J("stage1_light_eval_v3hb_raw.json")["summary"],
                     J("stage1_light_eval_v3hb_ema.json")["summary"],
                     J("stage1_light_eval_v3hb_zp12.json")["summary"])
    taus = J("smooth_tau_v3causal.json")["smooth_tau"]
    exp = ok(c.call("start_experiment", {"hypothesis_id": HYP["HB"], "git_ref": "HEAD",
        "config_note": json.dumps({"stage": "7 H-B", "runbook": "rb-ls2-07-offline-prerendered",
            "checkpoint": "light_trunk_v3causal.pt, no retraining", "arms": {
                "raw": "no output filtering", "ema": f"causal one-pole fitted per channel on val_track: {taus}",
                "zp12": "zero-phase 4th-order Butterworth, 12 Hz, the targets' own filter"},
            "n_clips": raw["n_clips"], "split": raw["split"], "seeds": raw["seeds"]})}))
    eid = exp.get("id") or exp.get("experiment_id"); print("H-B experiment:", eid)
    c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "light-event-rate-ratio", "before": round(raw["light_event_rate_ratio"], 3),
         "after": round(zp["light_event_rate_ratio"], 3)},
        {"diagnostic_id": "descriptor-hf-energy-ratio", "before": round(raw["hf_ratio_max"], 2),
         "after": round(zp["hf_ratio_max"], 2)},
        {"diagnostic_id": "fid-light-kinetic", "before": round(raw["fid_light"]["model_vs_real"], 2),
         "after": round(zp["fid_light"]["model_vs_real"], 2)},
        {"diagnostic_id": "light-diversity-kinetic", "before": round(raw["diversity"]["across_clips_model"], 3),
         "after": round(zp["diversity"]["across_clips_model"], 3)},
        {"diagnostic_id": "light-bas-minus-shift-null", "before": round(raw["bas_minus_null"]["model"]["delta"], 4),
         "after": round(zp["bas_minus_null"]["model"]["delta"], 4)}]})

    def arm_line(name, a):
        worst = max(a["descriptor_hf_energy_ratio"].items(), key=lambda kv: kv[1])
        return (f"{name}: event-rate ratio {a['light_event_rate_ratio']:.2f} ({a['events_per_clip']['model']:.1f} vs real "
                f"{a['events_per_clip']['real']:.1f}); worst HF ratio {worst[0]} {worst[1]:.1f}x; FID "
                f"{a['fid_light']['model_vs_real']:.2f}; diversity {a['diversity']['across_clips_model']:.2f}; "
                f"Light-BAS {a['bas_minus_null']['model']['delta']:+.4f}")
    lines = "\n".join(arm_line(n, a) for n, a in (("raw", raw), ("ema", ema_), ("zp12", zp)))
    c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G-smooth-rate", "passed": bool(abs(zp["light_event_rate_ratio"] - 1) <= 0.3), "evidence": lines},
        {"gate_id": "G-smooth-hf", "passed": bool(zp["hf_ratio_max"] <= 2.0), "evidence": (
            f"zero-phase arm per-channel HF ratios: { {k: round(v, 1) for k, v in zp['descriptor_hf_energy_ratio'].items()} }")},
        {"gate_id": "G-smooth-fid", "passed": bool(zp["fid_light"]["model_vs_real"] < raw["fid_light"]["model_vs_real"]),
         "evidence": f"FID raw {raw['fid_light']['model_vs_real']:.2f}, ema {ema_['fid_light']['model_vs_real']:.2f}, "
                     f"zero-phase {zp['fid_light']['model_vs_real']:.2f}; half-null {0.5*raw['fid_light']['real_vs_block_shuffled']:.2f}"},
        {"gate_id": "G-smooth-div", "passed": bool(zp["diversity"]["across_clips_model"] >= raw["diversity"]["across_clips_model"] - 1e-9),
         "evidence": f"diversity raw {raw['diversity']['across_clips_model']:.3f}, ema {ema_['diversity']['across_clips_model']:.3f}, "
                     f"zero-phase {zp['diversity']['across_clips_model']:.3f}; real {raw['diversity']['across_clips_real']:.3f}"}]})
    c.call("close_experiment", {"experiment_id": eid, "notes": (
        "H-B, zero-phase spectral matching, three arms on identical descriptors from the causal checkpoint.\n" + lines)})
    return eid


if __name__ == "__main__":
    which = sys.argv[1].upper()
    c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
    ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    {"HA": run_ha, "HB": run_hb}[which](c)
    lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
    ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    print(f"lightsem {lv} md; optiHealth {ov}->{ov2}"); assert ov2 == ov
    c.close()
