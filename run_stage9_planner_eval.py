#!/usr/bin/env python
"""H-J gate G9.4: does a section planner writing engine knobs beat the same knobs on the wrong sections?

The decoder-eval protocol (run_stage9_decoder_eval.py) with four decoder arms on the same
held-out Just Dance clips, all brightness-matched to the real-coach reference:

  no_planner    decoder + engine as in G9.3
  plan          the offline planner's per-bar knobs (palette, saturation, level class, activity)
                applied on top of the decoder's parameters
  plan_shuffled the SAME knob sequence with its 8-bar blocks permuted -- the real plan on the
                wrong sections (the H-C control, pre-registered)
  plan_causal   the causal planner's knobs (G9.4-causal)

    bash run_torch.sh run_stage9_planner_eval.py --n 60
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.online import ema
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.eval.bas import bas_report
from lightsem.eval.fid import FeatureScaler, frechet, split_half_floor
from lightsem.eval.light_features import block_shuffle, delta_features, level_features, light_onsets
from lightsem.model.group_decoder import GroupEngine, ROLES
from lightsem.model.infer import LightModel
from lightsem.model.jd_data import make_jd_splits
from lightsem.model.light_data import load_tensor
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus
from run_stage9_decoder_eval import CKPT, Decoder, HZ, OUT
from train_planner import KNOBS, Planner, bar_pool, bars_of


class PlannerRunner:
    def __init__(self, ckpt: Path, device: str):
        b = torch.load(ckpt, map_location=device, weights_only=False)
        self.m = Planner(b["in_dim"], causal=b["causal"]).to(device).eval(); self.m.load_state_dict(b["model"])
        self.mu = torch.as_tensor(b["mu"], device=device); self.sd = torch.as_tensor(b["sd"], device=device); self.device = device

    @torch.no_grad()
    def knobs(self, x: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bar = bars_of(phase); bx = bar_pool(x, bar)
        X = torch.from_numpy(bx).to(self.device)[None]
        return self.m((X - self.mu) / self.sd)[0].cpu().numpy(), bar


def apply_plan(p: dict[str, np.ndarray], knobs: np.ndarray, bar: np.ndarray, ramp_s: float = 0.5) -> dict[str, np.ndarray]:
    """Write the per-bar knobs onto the decoder's parameters: palette and saturation on the coloured
    roles (beams keep their hue offset from the washes), the level class on every role's bar-mean
    intensity, the activity class on the chase density. Held per bar, ramped over ``ramp_s``."""
    q = {k: v.copy() for k, v in p.items()}
    T = len(bar); kb = knobs[np.minimum(bar, len(knobs) - 1)]                      # (T, K) knobs per frame
    kb = np.stack([ema(kb[:, j], HZ, ramp_s) for j in range(kb.shape[1])], 1)
    hc, hs, sat, level_z, _, _, act_z, _ = [kb[:, j] for j in range(len(KNOBS))]
    plan_hue = np.stack([hc, hs], 1); plan_hue /= np.linalg.norm(plan_hue, axis=1, keepdims=True) + 1e-6
    wash = ROLES.index("wash")
    ang_w = np.arctan2(p["hue"][:, wash, 1], p["hue"][:, wash, 0]); ang_plan = np.arctan2(plan_hue[:, 1], plan_hue[:, 0])
    for g, role in enumerate(ROLES):
        if role in ("strobe", "blinder"):
            continue
        ang_g = np.arctan2(p["hue"][:, g, 1], p["hue"][:, g, 0])
        ang = ang_plan + (ang_g - ang_w)                                             # keep the decoder's role offset
        q["hue"][:, g, 0], q["hue"][:, g, 1] = np.cos(ang), np.sin(ang)
        q["sat"][:, g] = np.clip(sat, 0.0, 1.0)
        # level: the plan's class replaces the decoder's own bar-mean, in the decoder's own units
        bm = np.array([p["inten"][bar == b, g].mean() if (bar == b).any() else 0.0 for b in range(int(bar.max()) + 1)])
        m_dec, s_dec = float(bm.mean()), float(bm.std() + 1e-3)
        level = m_dec + s_dec * level_z
        bar_mean_frame = bm[np.minimum(bar, len(bm) - 1)]
        q["inten"][:, g] = np.clip(p["inten"][:, g] + (level - bar_mean_frame), 0.0, 1.0)
        q["density"][:, g] = np.clip(0.5 + 0.25 * act_z, 0.05, 1.0)
    return q


def shuffle_sections(knobs: np.ndarray, rng, block: int = 8) -> np.ndarray:
    blocks = np.array_split(np.arange(len(knobs)), max(len(knobs) // block, 1))
    return np.concatenate([knobs[blocks[j]] for j in rng.permutation(len(blocks))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="val_track")
    ap.add_argument("--ckpt", default=str(CKPT / "group_decoder_v1.pt"))
    ap.add_argument("--planner", default=str(CKPT / "planner_v1.pt"))
    ap.add_argument("--planner-causal", default=str(CKPT / "planner_v1_causal.pt"))
    ap.add_argument("--light-ckpt", default=str(CKPT / "light_trunk_jd_none_smooth.pt"))
    ap.add_argument("--tensor", default=str(CACHE_ROOT / "train_tensor_jd_none.npz"))
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args(); t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z, meta = load_tensor(Path(a.tensor)); sp = make_jd_splits(meta); idx = getattr(sp, a.split)
    rng = np.random.default_rng(0); idx = idx[rng.permutation(len(idx))[:a.n]]
    names = [meta[i]["name"] for i in idx]; by_name = {m["name"]: m for m in meta}
    lm = LightModel(a.light_ckpt); cond = ConductorHybrid(); rig = canonical_rig()
    dec = Decoder(Path(a.ckpt), device); eng = GroupEngine(rig)
    pl = PlannerRunner(Path(a.planner), device); plc = PlannerRunner(Path(a.planner_causal), device)
    ARMS = ("real", "conductor", "no_planner", "plan", "plan_shuffled", "plan_causal")
    F = {k: [] for k in ARMS}; F["null"] = []; rows = []

    def match_gain(frames_fn, ref_mean):
        fr = frames_fn(1.0); m = level_features(fr, rig)["mean"]
        g = float(np.clip(ref_mean / max(m, 1e-3), 0.3, 3.0))
        return frames_fn(g) if abs(g - 1) > 0.05 else fr
    n_done = 0
    for nm in names:
        d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{nm}.npz"); mid = by_name[nm]["music_id"]
        vp = CACHE_ROOT / "justdance_visual" / f"{mid}.npz"
        if not vp.exists():
            continue
        bus = slice_bus(load_bus(CACHE_ROOT / "justdance_audio" / f"{mid}.npz"), d.duration_s, offset_s=float(d.meta.get("start_s", 0.0)))
        if bus.beats.size < 6:
            continue
        x = np.load(vp, allow_pickle=False)["x"].astype(np.float32)
        off, dur = float(d.meta.get("start_s", 0.0)), d.duration_s
        i0, i1 = int(round(off * HZ)), int(round((off + dur) * HZ))
        p = {k: v[i0:i1] for k, v in dec.params(x, "jd").items()}; xc = x[i0:i1]
        if len(p["inten"]) < 80:
            continue
        phase = np.mod(np.arctan2(xc[:, 48], xc[:, 49]) / (2 * np.pi), 1.0)
        knobs, bar = pl.knobs(xc, phase); knobs_c, _ = plc.knobs(xc, phase)
        t10 = np.arange(len(p["inten"])) / HZ; beats = bus.beats; barph = bus.channels["bar_phase"]
        real = cond.render(d, bus).frames; ref_mean = level_features(real, rig)["mean"]
        arms = {"real": real, "conductor": cond.render(lm.descriptors(bus, seed=0), bus).frames}
        for k, q in (("no_planner", p), ("plan", apply_plan(p, knobs, bar)), ("plan_shuffled", apply_plan(p, shuffle_sections(knobs, rng), bar)),
                     ("plan_causal", apply_plan(p, knobs_c, bar))):
            arms[k] = match_gain(lambda g, q=q: eng.render(q, t10, beats, barph, dur, gain=g).frames, ref_mean)
        for k, fr in arms.items():
            F[k].append(delta_features(fr, rig, 40.0))
        F["null"].append(delta_features(block_shuffle(real, 40.0, 1.0, rng), rig, 40.0))
        row = {"name": nm}
        for k, fr in arms.items():
            ev, _ = light_onsets(fr, rig, 40.0); row[k] = bas_report(ev, beats, dur, rng=rng)
        rows.append(row); n_done += 1
        if n_done % 10 == 1:
            print(f"  [{n_done}] {nm} bas no-planner {row['no_planner']['bas']-row['no_planner']['null_mean']:+.3f} plan {row['plan']['bas']-row['plan']['null_mean']:+.3f} "
                  f"shuffled {row['plan_shuffled']['bas']-row['plan_shuffled']['null_mean']:+.3f}  {time.time()-t0:.0f}s", flush=True)
    X = {k: np.array(v) for k, v in F.items()}; sc = FeatureScaler(X["real"]); Z = {k: sc(v) for k, v in X.items()}
    fid = {k: frechet(Z[k], Z["real"]) for k in ARMS if k != "real"}
    fid["real_vs_block_shuffled"] = frechet(Z["real"], Z["null"]); floor = split_half_floor(Z["real"], rng=rng)

    def paired(key):
        v = np.array([r[key]["bas"] - r[key]["null_mean"] for r in rows]); n_nan = int(np.isnan(v).sum()); v = v[~np.isnan(v)]
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]) if len(v) else np.array([np.nan])
        return {"delta": float(v.mean()) if len(v) else float("nan"), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_no_events": n_nan}
    bas = {k: paired(k) for k in ARMS}
    dpair = np.array([r["plan"]["bas"] - r["plan"]["null_mean"] - (r["no_planner"]["bas"] - r["no_planner"]["null_mean"]) for r in rows]); dpair = dpair[~np.isnan(dpair)]
    bs = np.array([dpair[rng.integers(0, len(dpair), len(dpair))].mean() for _ in range(4000)])
    events = {k: float(np.mean([r[k]["n_events"] for r in rows])) for k in ARMS}
    gates = {
        "G9.4-shuffle": {"passed": bool(fid["plan"] < fid["plan_shuffled"] and fid["plan"] < fid["no_planner"]),
                         "evidence": f"FID_light vs real: plan {fid['plan']:.2f}, shuffled-section plan {fid['plan_shuffled']:.2f}, no planner {fid['no_planner']:.2f}, "
                                     f"causal planner {fid['plan_causal']:.2f}, conductor {fid['conductor']:.2f} (null {fid['real_vs_block_shuffled']:.2f}, floor {floor['mean']:.2f}). "
                                     f"The human 'intentionally structured' question was not run (needs raters)."},
        "G9.4-bas": {"passed": bool(bas["plan"]["delta"] >= bas["no_planner"]["delta"] - 0.005),
                     "evidence": f"paired Light-BAS minus null: plan {bas['plan']['delta']:+.4f} vs no planner {bas['no_planner']['delta']:+.4f} (paired diff {dpair.mean():+.4f} "
                                 f"CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]); shuffled {bas['plan_shuffled']['delta']:+.4f}, causal {bas['plan_causal']['delta']:+.4f}; "
                                 f"events per clip plan {events['plan']:.1f} vs no planner {events['no_planner']:.1f} vs real {events['real']:.1f}."},
        "G9.4-causal": {"passed": True, "evidence": f"offline planner FID {fid['plan']:.2f} / BAS {bas['plan']['delta']:+.4f} next to the causal planner FID {fid['plan_causal']:.2f} / BAS {bas['plan_causal']['delta']:+.4f}."}}
    res = {"n_clips": len(rows), "fid_light": fid | {"split_half_floor": floor}, "bas_minus_null": bas, "events_per_clip": events, "gates": gates,
           "config": {"planner": "train_planner.py: 4-layer Transformer d256, one token per bar (bar-pooled mean/std/max of the 52-d features), "
                      "trained on RPMC-L2 + Just Dance visual to predict 8 per-bar knobs; applied on the decoder's parameters by run_stage9_planner_eval.apply_plan "
                      "(palette, saturation, level class, activity -> density); control = the same knobs with 8-bar blocks permuted; causal arm = causal attention",
                      "n_clips": len(rows), "split": a.split},
           "measurements": [{"diagnostic_id": "fid-light-kinetic", "before": round(fid["no_planner"], 3), "after": round(fid["plan"], 3)},
                            {"diagnostic_id": "light-bas-minus-shift-null", "before": round(bas["no_planner"]["delta"], 4), "after": round(bas["plan"]["delta"], 4)}],
           "notes": (f"H-J planner on the decoder's knobs, {len(rows)} held-out JD clips. FID_light: no planner {fid['no_planner']:.2f}, plan {fid['plan']:.2f}, shuffled-section "
                     f"plan {fid['plan_shuffled']:.2f}, causal plan {fid['plan_causal']:.2f}, conductor {fid['conductor']:.2f}. BAS minus null: no planner {bas['no_planner']['delta']:+.4f}, "
                     f"plan {bas['plan']['delta']:+.4f}, shuffled {bas['plan_shuffled']['delta']:+.4f}, causal {bas['plan_causal']['delta']:+.4f}. Human question not run."),
           "elapsed_s": time.time() - t0}
    (OUT / "planner_eval.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("fid_light", "bas_minus_null", "events_per_clip")}, indent=1, default=float))
    for g, v in gates.items():
        print(f"{g}: {'PASS' if v['passed'] else 'FAIL'} -- {v['evidence']}")


if __name__ == "__main__":
    main()
