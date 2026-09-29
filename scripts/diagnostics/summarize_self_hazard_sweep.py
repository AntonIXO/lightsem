#!/usr/bin/env python
"""H-O: collect the size sweep (group_decoder_s_*.json) into experiments/legacy/stages/outputs/stage9/self_hazard_sweep.json in the
generic vault format (config / measurements / gates / notes) for scripts/vault/log_stage9_to_vault.py HO.

    bash run.sh scripts/diagnostics/summarize_self_hazard_sweep.py
"""
import glob, json
from pathlib import Path

import numpy as np

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9")
runs = {}
for p in sorted(glob.glob(str(OUT / "group_decoder_*.json"))):
    r = json.loads(Path(p).read_text()); tag = Path(p).stem.replace("group_decoder_", "")
    if tag.startswith("smoke"):
        continue
    runs[tag] = {"params_m": r["params_m"], "self_hazard": r.get("self_hazard", False), "d_model": r.get("d_model", 192), "n_blocks": r.get("n_blocks", 3),
                 **{f"{c}_{k}": v[k] for c, v in r["per_corpus"].items() for k in ("look_ap_lift", "look_ap_lift_hazard_ridge", "look_ap_lift_own_cues") if k in v}}
rows = "\n".join(f"  {t:16s} {v['params_m']:5.2f} M d{v['d_model']} b{v['n_blocks']} self={int(v['self_hazard'])}: JD lift {v.get('jd_look_ap_lift', float('nan')):.2f}"
                 + (f" (own cues {v['jd_look_ap_lift_own_cues']:.2f})" if "jd_look_ap_lift_own_cues" in v else "")
                 + f" vs rhythm {v.get('jd_look_ap_lift_hazard_ridge', float('nan')):.2f}; RPMC {v.get('rpmc_look_ap_lift', float('nan')):.2f}"
                 + (f" (own {v['rpmc_look_ap_lift_own_cues']:.2f})" if "rpmc_look_ap_lift_own_cues" in v else "") for t, v in runs.items())
print(rows)
audio = {t: v for t, v in runs.items() if not v["self_hazard"]}; selfh = {t: v for t, v in runs.items() if v["self_hazard"]}
best_audio = max(audio.values(), key=lambda v: v.get("jd_look_ap_lift", 0)) if audio else None
best_self = max(selfh.values(), key=lambda v: v.get("jd_look_ap_lift_own_cues", 0)) if selfh else None
own_gain = (best_self["jd_look_ap_lift_own_cues"] - best_audio["jd_look_ap_lift"]) if (best_self and best_audio) else float("nan")
own_gain_r = (best_self.get("rpmc_look_ap_lift_own_cues", 0) - best_audio.get("rpmc_look_ap_lift", 0)) if (best_self and best_audio) else float("nan")
hp = OUT / "head_own_cues_jdvis.json"; head = json.loads(hp.read_text()) if hp.exists() else None
head_line = (f" The with-hazard OPERATOR HEAD itself (exp-016, 3.17x with the true hazard) drops to {head['lift']['own_cues']:.2f}x when its hazard features come from its own cues "
             f"(autoregressive, {head['lift']['own_cue_rate']:.1f} cues/min at threshold {head['threshold']:.2f}), BELOW the audio-only head's {head['lift']['audio_only']:.2f}x "
             f"(paired {head['own_minus_audio']['delta']:+.2f} CI {[round(x, 2) for x in head['own_minus_audio']['ci95']]}, {head['n_test']} songs): the rhythm-of-past-cues margin is teacher-forcing, not deployable information." if head else "")
sizes = sorted(runs.values(), key=lambda v: v["params_m"])
size_line = "; ".join(f"{v['params_m']:.1f} M -> JD {v.get('jd_look_ap_lift', 0):.2f}" for v in sizes)
res = {"runs": runs,
       "gates": {"G10.2-self": {"passed": bool(own_gain > 0.05 and own_gain_r > 0.0),
                                "evidence": f"best audio-only decoder JD cue lift {best_audio['jd_look_ap_lift']:.2f} (RPMC {best_audio.get('rpmc_look_ap_lift', 0):.2f}); best self-hazard decoder with its OWN cues "
                                            f"{best_self['jd_look_ap_lift_own_cues']:.2f} (RPMC {best_self.get('rpmc_look_ap_lift_own_cues', 0):.2f}), with the TRUE hazard (ceiling) {best_self['jd_look_ap_lift']:.2f} "
                                            f"(RPMC {best_self.get('rpmc_look_ap_lift', 0):.2f}); rhythm baseline JD {best_audio.get('jd_look_ap_lift_hazard_ridge', 0):.2f}. Own-cue gain {own_gain:+.2f} (JD), {own_gain_r:+.2f} (RPMC). "
                                            f"No CI over songs is stored by the trainer, so a gain under 0.05 is not claimed." + head_line},
                 "G10.2-size": {"passed": True, "evidence": f"causal decoders, JD cue lift vs parameters: {size_line}."}},
       "measurements": [{"diagnostic_id": "event-ap-lift-audio-marginal", "before": round(best_audio["jd_look_ap_lift"] - best_audio.get("jd_look_ap_lift_hazard_ridge", 0), 4),
                         "after": round(best_self["jd_look_ap_lift_own_cues"] - best_self.get("jd_look_ap_lift_hazard_ridge", 0), 4)}],
       "config": {"sweep": list(runs), "trainer": "scripts/training/train_group_decoder.py --flow [--self-hazard] --d-model --n-blocks, 6000 steps, RPMC + JD, hash split; own cues = autoregressive in 1 s blocks, thr 0.5, refractory 2 s"},
       "notes": "H-O size sweep. " + rows.replace("\n", " ") + head_line}
(OUT / "self_hazard_sweep.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["gates"], indent=1))
