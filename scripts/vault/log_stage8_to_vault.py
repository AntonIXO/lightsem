#!/usr/bin/env python
"""Record the Stage 8 experiments from their outputs.

  bash run.sh scripts/vault/log_stage8_to_vault.py HE   # audit_jd_full.json      (coach motion predictable?)
  bash run.sh scripts/vault/log_stage8_to_vault.py HC   # jd_none/struct/shuf evals (structure conditioning)
  bash run.sh scripts/vault/log_stage8_to_vault.py HF   # operator_head_model_with_hazard_jdvis.json (background as a show)
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client

P = "dancing-light-generative-20260903"
S1 = Path("/root/lightsem/experiments/legacy/stages/outputs/stage1"); S2 = Path("/root/lightsem/experiments/legacy/stages/outputs/stage2"); S8 = Path("/root/lightsem/experiments/legacy/stages/outputs/stage8")
HYP = {"HE": "just-dance-coach-motion-is-predictable-from-bcb46b",
       "HC": "conditioning-on-whole-track-structure-section-4ccfa8",
       "HF": "the-just-dance-background-with-coach-and-ui-598f33"}


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


def corpus_note():
    idx = json.loads((Path("/mnt/lightsem-data/cache/justdance_motion_index.json")).read_text())
    usable = [r for r in idx if r.get("coverage", 0) >= 0.8]
    return {"coach_entries_total": len(idx), "coach_entries_usable": len(usable),
            "songs_usable": len({r["song_key"] for r in usable}),
            "multi_coach_entries": sum(1 for r in usable if r.get("n_coaches", 1) > 1),
            "hours_usable": round(sum(r["duration_s"] for r in usable) / 3600, 2)}


def run_he(c):
    a = json.loads((S8 / "audit_jd_full.json").read_text())
    ho = a["held_out_song"]
    fam = ["qom@2.0", "vert_bounce_amp@2.0", "head_speed@2.0", "com_speed@2.0", "spread@2.0", "hand_speed@2.0"]
    best = max(fam, key=lambda k: ho[k]["r2"])
    exp = ok(c.call("start_experiment", {"hypothesis_id": HYP["HE"], "git_ref": "HEAD", "config_note": json.dumps({
        "stage": "8.2", "runbook": "rb-ls2-08-justdance", "recipe": "scripts/legacy_stages/run_justdance_audit.py: the Oslo per-track recipe per coach, "
        "both sides causally gain-normalised (30 s trailing), 13 causal bus scalars x 3 EMA taus, ridge 1e3, 5-fold by song key",
        **corpus_note(), "n_songs_audit": a["n_songs"], "n_entries_audit": a["n_entries"]})}))
    eid = exp.get("id") or exp.get("experiment_id"); print("H-E experiment:", eid)
    c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "before": 0.0, "after": round(ho[best]["r2"], 4)}]})
    lines = "; ".join(f"{k} R2 {ho[k]['r2']:+.3f} CI {[round(x,3) for x in ho[k]['ci95']]} r {ho[k]['r']:+.3f}" for k in fam)
    ic = a.get("inter_coach", {}).get("pairs", {})
    ic_line = "; ".join(f"{k} r {v['mean_r']:+.2f} ({v['n_pairs']} pairs)" for k, v in ic.items() if k.endswith("@2.0"))
    sc = a.get("section_contrast", {})
    sc_line = "; ".join(f"{k} {v['ratio_mean']:.2f}x CI {[round(x,2) for x in v['ci95']]} ({v['n_entries']})" for k, v in sc.items())
    g_r2 = ho[best]["r2"] >= 0.20
    g_con = bool(sc) and all(v["ci95"][0] > 1.0 for v in sc.values())
    c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G8.2-r2", "passed": bool(g_r2), "evidence": f"held-out-song R2 at 2 s: {lines}. Best channel {best}. Gate >= +0.20 (Oslo +0.30, AIST++ <= 0.21 collapsing under genre)."},
        {"gate_id": "G8.2-genre", "passed": False, "evidence": (
            "Not measurable as preregistered: the corpus carries no genre or Effort labels (the wiki's ratings were never "
            "scraped), so nothing could be partialled out. Recorded as not passed rather than skipped. What the corpus does "
            f"give is inter-coach agreement on multi-coach routines at 2 s: {ic_line or 'n/a'} -- coaches hearing the same "
            "music move together, which is the shared part a model can learn.")},
        {"gate_id": "G8.2-contrast", "passed": bool(g_con), "evidence": (
            f"coach movement in the loudest third of a song's sections over the quietest third (structure labels' "
            f"segment energy): {sc_line or 'n/a'}. Oslo's drop/breakdown ratio was 1.68-2.53x.")}]})
    bounce = a.get("bounce_period_over_beat", {})
    c.call("close_experiment", {"experiment_id": eid, "notes": (
        f"Stage 8.2 audit on {a['n_songs']} held-out songs, {a['n_entries']} coach channels. {lines}. Inter-coach: {ic_line}. "
        f"Section contrast: {sc_line}. Bounce period over beat: median {bounce.get('median', float('nan')):.2f}, near 1x or 2x in "
        f"{100*bounce.get('frac_near_1_or_2', 0):.0f}% of entries (the 8.3 sync check).")})
    return eid


def run_hc(c):
    ev = {t: json.loads((S1 / f"stage1_light_eval_{t}.json").read_text())["summary"]
          for t in ("jd_none_smooth", "jd_struct_smooth", "jd_shuf_smooth", "jd_none_raw", "jd_struct_raw", "jd_shuf_raw", "jd_aistonly")}
    tr = {t: json.loads((S1 / f"train_{t}.json").read_text()) for t in ("jd_none", "jd_struct", "jd_shuf")}
    exp = ok(c.call("start_experiment", {"hypothesis_id": HYP["HC"], "git_ref": "HEAD", "config_note": json.dumps({
        "stage": "8.5 / H-C", "runbook": "rb-ls2-08-justdance", "corpus": corpus_note(),
        "arms": {"none": "Just Dance tensor, no structure channels", "struct": "+8 structure channels (role one-hot, section energy, "
                 "tanh(seconds to next rise/drop / 30))", "shuf": "the same channels from another song (the control)"},
        "trunk": "9.3 M causal Conformer, v2 blocks, hip_yaw_speed and footfall_rate dropped (2D pose), 12k steps",
        "eval": "held-out songs (test split by song key), 40 clips, 3 seeds; per-arm event thresholds and H-B smoothing fitted on val songs",
        "held_out_r2": {t: tr[t]["test_dancer"]["r2_a_mean"] for t in tr}})}))
    eid = exp.get("id") or exp.get("experiment_id"); print("H-C experiment:", eid)
    b = lambda t: ev[t]["bas_minus_null"]["model"]  # noqa: E731
    c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "light-bas-minus-shift-null", "before": round(b("jd_none_smooth")["delta"], 4), "after": round(b("jd_struct_smooth")["delta"], 4)},
        {"diagnostic_id": "fid-light-kinetic", "before": round(ev["jd_none_smooth"]["fid_light"]["model_vs_real"], 2),
         "after": round(ev["jd_struct_smooth"]["fid_light"]["model_vs_real"], 2)},
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "before": round(tr["jd_none"]["test_dancer"]["r2_a_mean"], 4),
         "after": round(tr["jd_struct"]["test_dancer"]["r2_a_mean"], 4)}]})

    def line(t):
        s_ = ev[t]; bb = s_["bas_minus_null"]
        return (f"{t}: Light-BAS {bb['model']['delta']:+.4f} CI {[round(x,4) for x in bb['model']['ci95']]} (real-mapped "
                f"{bb['real_dance']['delta']:+.4f}, beat trigger {bb['beat_trigger']['delta']:+.4f}); FID {s_['fid_light']['model_vs_real']:.2f} "
                f"(half-null {0.5*s_['fid_light']['real_vs_block_shuffled']:.2f}); diversity {s_['diversity']['across_clips_model']:.2f}/"
                f"{s_['diversity']['across_clips_real']:.2f}; events {s_['events_per_clip']['model']:.1f}/{s_['events_per_clip']['real']:.1f}")
    lines = "\n".join(line(t) for t in ("jd_aistonly", "jd_none_raw", "jd_none_smooth", "jd_struct_raw", "jd_struct_smooth", "jd_shuf_raw", "jd_shuf_smooth"))
    d_struct = b("jd_struct_smooth")["delta"] - b("jd_none_smooth")["delta"]
    d_shuf = b("jd_shuf_smooth")["delta"] - b("jd_none_smooth")["delta"]
    # a paired bootstrap across the same 40 clips
    import numpy as np
    pc = {t: json.loads((S1 / f"stage1_light_eval_{t}.json").read_text())["per_clip"] for t in ("jd_none_smooth", "jd_struct_smooth", "jd_shuf_smooth")}
    names = [r["name"] for r in pc["jd_none_smooth"]]
    def vec(t):
        m = {r["name"]: r["model"]["bas"] - r["model"]["null_mean"] for r in pc[t]}
        return np.array([m.get(n, np.nan) for n in names])
    v0, v1, v2 = vec("jd_none_smooth"), vec("jd_struct_smooth"), vec("jd_shuf_smooth")
    okm = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(v2)
    rng = np.random.default_rng(0)
    def ci(dv):
        bs = np.array([dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(4000)])
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ci_s, ci_h = ci((v1 - v0)[okm]), ci((v2 - v0)[okm])
    c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G-struct-lift", "passed": bool(d_struct > 0 and ci_s[0] > 0), "evidence": (
            f"structure minus none, paired over {int(okm.sum())} held-out clips: {d_struct:+.4f} CI {[round(x,4) for x in ci_s]}.\n" + lines)},
        {"gate_id": "G-struct-contrast", "passed": False, "evidence": (
            "Not measured at the light level: the evaluator reports clip-level brightness, not per-section intensity. "
            "The movement-level contrast is in the H-E record (section_contrast).")},
        {"gate_id": "G-struct-shuffle", "passed": bool(d_struct > 0 and ci_s[0] > 0 and d_shuf < 0.5 * d_struct), "evidence": (
            f"shuffled-structure minus none: {d_shuf:+.4f} CI {[round(x,4) for x in ci_h]} against the real-structure gain "
            f"{d_struct:+.4f}. Passes only if the real gain exists and the shuffled channel delivers less than half of it.")}]})
    c.call("close_experiment", {"experiment_id": eid, "notes": (
        "H-C on Just Dance, three arms plus the AIST-only model on the same held-out songs.\n" + lines + "\n"
        f"Held-out-song block A R2: none {tr['jd_none']['test_dancer']['r2_a_mean']:+.4f}, struct "
        f"{tr['jd_struct']['test_dancer']['r2_a_mean']:+.4f}, shuffled {tr['jd_shuf']['test_dancer']['r2_a_mean']:+.4f} "
        f"(AIST++ causal model on its own held-out tracks: +0.019).")})
    return eid


def run_hf(c):
    r = json.loads((S2 / "operator_head_model_with_hazard_jdvis.json").read_text())
    exp = ok(c.call("start_experiment", {"hypothesis_id": HYP["HF"], "git_ref": "HEAD", "config_note": json.dumps({
        "stage": "8.4 / H-F", "runbook": "rb-ls2-08-justdance", "data": "justdance_visual: background HSV scene vector (coaches and UI "
        "masked) at 10 Hz in RPMC-L2's 6-d format, causal audio features as for the operators; split by song hash",
        "head": "4-block Conformer d192 with hazard features (causal), same as exp-008", "n_songs": r["n_songs"], "n_test": r["n_test"]})}))
    eid = exp.get("id") or exp.get("experiment_id"); print("H-F experiment:", eid)
    c.call("record_diagnostics", {"experiment_id": eid, "measurements": [
        {"diagnostic_id": "event-ap-lift-audio-marginal", "before": 0.0, "after": round(r["delta"], 4)}]})
    line = (f"model lift {r['model_lift']:.3f}x, rhythm baseline {r['rhythm_lift']:.3f}x, ridge on the same features "
            f"{r['ridge_on_same_causal_features_lift']:.3f}x, margin {r['delta']:+.4f} CI {[round(x,4) for x in r['ci95']]}, "
            f"{r['n_test']} held-out songs; shift-curve peak {r.get('shift_peak_s')} s")
    c.call("record_gate_results", {"experiment_id": eid, "results": [
        {"gate_id": "G8.4-lift", "passed": bool(r["delta"] >= 0.09 and r["ci95"][0] > 0), "evidence": line + ". Gate: margin >= +0.09 with CI excluding 0 (RPMC-L2 operators: +0.555 with hazard, +0.096 audio-only)."},
        {"gate_id": "G8.4-hue", "passed": False, "evidence": "Not measured: the head predicts scene-change events only; the palette (hue KL) test is not implemented."},
        {"gate_id": "G8.4-shift", "passed": bool(r.get("shift_peak_s") is not None and abs(r["shift_peak_s"]) <= 2.0),
         "evidence": f"target-shift curve { {k: round(v,2) for k, v in r.get('shift_curve_s', {}).items() if v is not None} }"}]})
    c.call("close_experiment", {"experiment_id": eid, "notes": "H-F, the Just Dance background as a designed show. " + line})
    return eid


if __name__ == "__main__":
    which = sys.argv[1].upper()
    c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
    ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    {"HE": run_he, "HC": run_hc, "HF": run_hf}[which](c)
    lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
    ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    print(f"lightsem {lv} md; optiHealth {ov}->{ov2}"); assert ov2 == ov
    c.close()
