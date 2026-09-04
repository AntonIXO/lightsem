#!/usr/bin/env python
"""Stage 9 closing note: runbook rb-ls2-09 + problem update with the verdicts of exp-017..023 and
the next lever (PLAN.md rev. 9, section 6.9.10). Numbers are read from outputs/stage9.

    bash run.sh note_stage9_results_to_vault.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"; S9 = Path("/root/lightsem/outputs/stage9")
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


hh = json.loads((S9 / "hh_gate.json").read_text())["events"]["ev_scene"]
hg = json.loads((S9 / "hg_stems.json").read_text())["events"]
hm = json.loads((S9 / "hm_embedding.json").read_text())["arms"]
d1 = json.loads((S9 / "decoder_eval_v1.json").read_text()); d2 = json.loads((S9 / "decoder_eval_v2.json").read_text())
pl = json.loads((S9 / "planner_eval.json").read_text())
cr = json.loads((S9 / "critic.json").read_text()) if (S9 / "critic.json").exists() else None
zs = json.loads((S9 / "decoder_eval_v1_zerospread.json").read_text()) if (S9 / "decoder_eval_v1_zerospread.json").exists() else None

note = (
    "2026-09-05 STAGE 9 RUN, one overnight pass (PLAN.md rev. 9 section 6.9.10; exp-017..023). "
    f"H-H Beat Saber (298 maps, song split): scene-change AP audio {hh['ap']['audio']:.3f} vs shifted null {hh['ap']['audio']-hh['ap_audio_minus_shift']:.3f} vs beat trigger {hh['ap']['bt']:.3f} "
    f"(margins {hh['ap_audio_minus_shift']:+.3f} / {hh['audio_minus_bt']['mean']:+.3f} against gates +0.15 / +0.05; offline {hh['ap']['offline']:.3f}); Conformer head 1.65x vs rhythm 1.30x -> a tempo corpus with a small authored residual, EXCLUDED from every loss. "
    f"H-G stems: four full stems {hg['bus+full4']['delta_ap']:+.4f} AP audio-only (CI {[round(x,4) for x in hg['bus+full4']['ci95']]}), donor {hg['bus+donor_compact4']['delta_ap']:+.4f}, bar phase {hg['bus+bar_phase']['delta_ap']:+.4f}; the Conformer head with stems 2.99x vs 3.17x without; coach split absent. "
    f"H-M MERT-330M layer groups: {hm['bus+rhy32']['delta_ap']:+.4f} raw, {hm['bus+rhy32_resid']['delta_ap']:+.4f} after song-mean subtraction, random projection {hm['bus+randproj32']['delta_ap']:+.4f}, song-ID probe 83%. "
    f"H-I decoder: v1 regressed parameters collapse to per-song constants; v2 sampled look (rectified flow, look-then-assign) has variance but r~0.04 with the target; G9.3 FID_light v1 {d1['fid_light']['decoder']:.0f} / v2 {d2['fid_light']['decoder']:.0f} vs conductor {d1['fid_light']['conductor']:.1f} (donors {d1['fid_light']['donor']:.0f} / {d2['fid_light']['donor']:.0f}), "
    f"BAS-null v1 {d1['bas_minus_null']['decoder']['delta']:+.3f} / v2 {d2['bas_minus_null']['decoder']['delta']:+.3f} vs conductor {d1['bas_minus_null']['conductor']['delta']:+.3f}; rig transfer v1 {d1['rig_transfer_hist_distance']['decoder']:.3f} vs copy {d1['rig_transfer_hist_distance']['channel_copy']:.3f}. REFUTED AS BUILT (two light-level gates missed twice). "
    + (f"Engine diagnostic (zero phase spread, 30 clips): BAS-null {zs['bas_minus_null']['decoder']['delta']:+.3f}, FID {zs['fid_light']['decoder']:.0f}. " if zs else "")
    + f"H-J planner: bar knobs unpredictable within songs (pooled R2 ~0); on v2 FID plan {pl['fid_light']['plan']:.0f} vs shuffled-section {pl['fid_light']['plan_shuffled']:.0f} vs no planner {pl['fid_light']['no_planner']:.0f}, BAS plan {pl['bas_minus_null']['plan']['delta']:+.3f} vs no planner {pl['bas_minus_null']['no_planner']['delta']:+.3f} (G9.4-bas failed). "
    + (f"H-K critic: real vs beat-trigger {cr['pair_accuracy']['vs_beat_trigger']['acc']:.3f} (C_R alone {cr['pair_accuracy']['R_vs_beat_trigger']['acc']:.3f}), vs other song {cr['pair_accuracy']['vs_other_song']['acc']:.3f}, vs 2+-bar shift {cr['pair_accuracy']['vs_shift']['acc']:.3f}; "
       f"additive projection R2 {cr['best_of_8']['additive_projection_r2']:.2f}, best-of-8 order changes on {cr['best_of_8']['order_change_frac']*100:.0f}% of clips; best-of-8 vs random-of-8 FID {cr['best_of_8']['fid_light']['best_of_8']:.0f} vs {cr['best_of_8']['fid_light']['random_of_8']:.0f}, BAS {cr['best_of_8']['bas_minus_null']['best']:+.3f} vs {cr['best_of_8']['bas_minus_null']['rand']:+.3f}. " if cr else "")
    + "H-L not run (needs raters). WHAT IT SETTLES: every authored corpus carries cue TIMING (look-change lifts 1.4-3.2x), not LEVELS (frame value r <= 0.18, bar knobs R2 ~0, sampled look = donor); "
    "feature engineering has a +0.02 AP ceiling here and the Conformer head converts none of it; FID_light against dance-derived light is the wrong metric for a look sampler. "
    "NEXT LEVER: a cue-based renderer (calibrated look-change hazard firing look transitions drawn from the corpus look distribution, scored on cue timing and transition statistics against authored corpora) and the H-L human study; console show files remain the only route to level design.")
ok(c.call("update_runbook", {"runbook_id": "rb-ls2-09-designed-light", "note": note}))
ok(c.call("update_problem", {"id": P, "add_update": (
    "2026-09-05: Stage 9 (rb-ls2-09) run end to end; H-H, H-G, H-M, H-J directional-only, H-I refuted as built, H-K recorded, H-L open. "
    "Reading: the authored corpora on disk (RPMC-L2, Just Dance visual, Beat Saber) support learning WHEN a look changes and the corpus's look statistics, "
    "not per-frame or per-bar light levels; the decent-not-human gap is not closed by stems, a frozen encoder, a regressed or sampled group decoder, or a bar planner. "
    "Next: cue-based rendering from the calibrated look-change hazard, scored on cue timing; human preference study; console show files for level design.")}))
lv = len(list(Path("/root/lightsem-vault").rglob("*.md"))); ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}"); assert ov2 == ov
c.close()
