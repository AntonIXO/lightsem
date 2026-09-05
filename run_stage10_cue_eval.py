#!/usr/bin/env python
"""H-N (PLAN.md 6.10): the cue-held look sampler, scored on what the corpora support -- cue timing
and look-transition statistics -- plus the JD light protocol.

Arms on held-out Just Dance songs (the visual scene vector is the authored reference):
  per_frame   decoder v2: a look sampled every frame, one-pole smoothed (Stage 9's arm)
  cue_held    the same flow head sampled once per CUE and held; cues from the decoder's look-change
              hazard, threshold calibrated on TRAINING songs to the corpus cue rate, refractory 2 s
  cue_held_bt cues from the beat trigger's scene novelty instead (timing control)
  donor       another held-out song's cue-held stream on this song (shuffled-donor null)
  authored    the corpus's own scene vector (the reference; its split-half distance is the floor)

Transition features per stream (all from the 10 Hz scene vector and its cue train): cues per
minute, inter-cue interval median and CV, mean |hue jump| at cues, mean |value-class change| at
cues, fraction of cues within 100 ms of a beat, hue jump between cues (should be ~0 when held).
cue-transition-distance = Frechet distance on those 7 features (standardised on the authored set).
Then the DMX protocol on the same clips (Light-BAS minus null, events per clip, brightness-matched).

    bash run_torch.sh run_stage10_cue_eval.py --ckpt /mnt/lightsem-data/ckpt/group_decoder_v2.pt --tag v2
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
from lightsem.eval.light_features import level_features, light_onsets
from lightsem.model.group_decoder import GroupEngine, assign_look, render_scene, window_norm
from lightsem.model.infer import LightModel
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus
from probe_event_target import average_precision, event_train, novelty, target
from run_stage9_decoder_eval import CKPT, Decoder, HZ, OUT
from train_critic import beat_trigger_scene
from train_group_decoder import load_corpus
from train_operator_head import split

TRANS = ("cues_per_min", "iei_median_s", "iei_cv", "hue_jump_at_cue", "value_change_at_cue", "frac_on_beat", "hue_drift_between_cues")


def pick_cues(score: np.ndarray, thr: float, refractory: int = 20) -> np.ndarray:
    cues = np.zeros(len(score), np.float32); last = -10 ** 6
    for i, v in enumerate(score):
        if v > thr and i - last >= refractory:
            cues[i] = 1.0; last = i
    return cues


def pick_cues_topk(score: np.ndarray, k: int, refractory: int = 20) -> np.ndarray:
    """Rate-matched cue picking for songs outside the calibration corpus: local maxima of the hazard,
    tallest first, at most `k` of them, `refractory` frames apart."""
    peaks = [i for i in range(1, len(score) - 1) if score[i] >= score[i - 1] and score[i] > score[i + 1]]
    cues = np.zeros(len(score), np.float32); taken = []
    for i in sorted(peaks, key=lambda i: -score[i]):
        if len(taken) >= k:
            break
        if all(abs(i - j) >= refractory for j in taken):
            cues[i] = 1.0; taken.append(i)
    return cues


def transition_features(y: np.ndarray, cues: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """(T, 6) scene vector at 10 Hz + (T,) cue train + beat phase -> the 7 transition features."""
    T = len(y); idx = np.flatnonzero(cues); mins = T / HZ / 60.0
    hue = np.arctan2(y[:, 1], y[:, 0]); vz = (y[:, 3] - y[:, 3].mean()) / (y[:, 3].std() + 1e-6)
    iei = np.diff(idx) / HZ if len(idx) > 2 else np.array([np.nan])
    dh = lambda a, b: np.abs(((hue[b] - hue[a] + np.pi) % (2 * np.pi)) - np.pi)  # noqa: E731
    pre = np.clip(idx - 5, 0, T - 1); post = np.clip(idx + 5, 0, T - 1)
    on_beat = np.mean((phase[idx] < 0.1) | (phase[idx] > 0.9)) if len(idx) else np.nan
    drift = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 10:
            drift.append(dh(a + 5, b - 5))
    return np.array([len(idx) / max(mins, 1e-6), np.nanmedian(iei), np.nanstd(iei) / (np.nanmean(iei) + 1e-6),
                     np.mean(dh(pre, post)) if len(idx) else np.nan, np.mean(np.abs(vz[post] - vz[pre])) if len(idx) else np.nan,
                     on_beat, np.mean(drift) if drift else 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT / "group_decoder_v2.pt"))
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--n-dmx", type=int, default=40, help="held-out JD coach clips for the light protocol")
    ap.add_argument("--light-ckpt", default=str(CKPT / "light_trunk_jd_none_smooth.pt"))
    a = ap.parse_args(); t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dec = Decoder(Path(a.ckpt), device); rng = np.random.default_rng(0)
    songs = load_corpus("jd"); tr, va, te = split(songs)
    print(f"JD visual: train {len(tr)} test {len(te)}", flush=True)

    @torch.no_grad()
    def hazard_and_h(s):
        xt = torch.from_numpy(s["x"]).to(device)[None]
        if dec.cond:
            xt = torch.cat([xt, dec.op.trunk((xt - dec.omu) / dec.osd)], -1)
        if dec.cfg.cond_dim > (dec.op.cfg.d_model if dec.cond else 0):        # a self-hazard checkpoint: use the true hazard here (ceiling)
            xt = torch.cat([xt, torch.from_numpy(s["haz"]).to(device)[None]], -1)
        p = dec.m((xt - dec.mu) / dec.sd, torch.tensor([dec.cids["jd"]], device=device))
        return torch.sigmoid(p["look_logit"][0]).cpu().numpy(), p
    # ---- calibrate the cue threshold on TRAINING songs to the corpus cue rate
    rate = np.mean([event_train(s["y"])[0].sum() / (len(s["y"]) / HZ / 60) for s in tr])
    haz_tr = [hazard_and_h(s)[0] for s in tr[:120]]
    thrs = np.linspace(0.05, 0.95, 37)
    rates = [np.mean([pick_cues(h, t).sum() / (len(h) / HZ / 60) for h in haz_tr]) for t in thrs]
    thr = float(thrs[int(np.argmin(np.abs(np.array(rates) - rate)))])
    print(f"corpus cue rate {rate:.2f}/min -> hazard threshold {thr:.3f} (rate {rates[int(np.argmin(np.abs(np.array(rates) - rate)))]:.2f}/min)", flush=True)

    # ---- per held-out song: streams and their transition features
    F = {k: [] for k in ("authored", "per_frame", "cue_held", "cue_held_bt", "donor")}; ap_lift = {"hazard": [], "beat_trigger": []}
    streams = []
    for s in te:
        h, p = hazard_and_h(s); T = len(s["x"])
        ev_true = event_train(s["y"])[0]; tgt = target(ev_true)
        if tgt.sum() >= 3:
            ap_lift["hazard"].append(average_precision(h, tgt) / tgt.mean())
        bt_nov = novelty(beat_trigger_scene(s["x"]))
        if tgt.sum() >= 3:
            ap_lift["beat_trigger"].append(average_precision(bt_nov, tgt) / tgt.mean())
        g = torch.Generator(device=device).manual_seed(0)
        look = dec.m.sample_look(p["h"], generator=g)[0].cpu().numpy()                     # per-frame samples (normalised units)
        yn = window_norm(torch.from_numpy(s["y"])[None])[0].numpy()
        # per-frame arm: one-pole smoothed samples
        L_pf = np.stack([ema(look[:, j], HZ, 0.8 if j in (0, 1, 2, 5) else 0.3) for j in range(6)], 1)
        cues = pick_cues(h, thr); cues_bt = pick_cues(bt_nov / (bt_nov.max() + 1e-9), float(np.percentile(bt_nov / (bt_nov.max() + 1e-9), 100 * (1 - rate * (T / HZ / 60) / T))))
        def hold(cue_train):
            L = np.zeros_like(look); cur = look[0]
            for t in range(T):
                if cue_train[t] > 0:
                    cur = look[t]
                L[t] = cur
            return np.stack([ema(L[:, j], HZ, 0.5) for j in range(6)], 1)              # a 0.5 s crossfade at each cue
        L_ch = hold(cues); L_bt = hold(cues_bt)
        F["authored"].append(transition_features(yn, ev_true, s["phase"]))
        F["per_frame"].append(transition_features(L_pf, event_train(L_pf)[0], s["phase"]))
        F["cue_held"].append(transition_features(L_ch, cues, s["phase"]))
        F["cue_held_bt"].append(transition_features(L_bt, cues_bt, s["phase"]))
        streams.append({"key": s["key"], "L_ch": L_ch, "cues": cues, "phase": s["phase"], "T": T})
    for j, st in enumerate(streams):                                                        # donor: another song's cue-held stream on this song's clock
        d = streams[(j + 1) % len(streams)]; n = min(st["T"], d["T"])
        F["donor"].append(transition_features(d["L_ch"][:n], d["cues"][:n], st["phase"][:n]))
    X = {k: np.nan_to_num(np.array(v)) for k, v in F.items()}; sc = FeatureScaler(X["authored"]); Z = {k: sc(v) for k, v in X.items()}
    dist = {k: frechet(Z[k], Z["authored"]) for k in F if k != "authored"}; floor = split_half_floor(Z["authored"], rng=rng)
    means = {k: dict(zip(TRANS, np.round(X[k].mean(0), 3).tolist())) for k in F}
    print("cue-transition distance:", {k: round(v, 2) for k, v in dist.items()}, "floor", round(floor["mean"], 2), flush=True)
    print("cue lift: hazard %.2f  beat-trigger novelty %.2f" % (np.mean(ap_lift["hazard"]), np.mean(ap_lift["beat_trigger"])), flush=True)
    for k, m in means.items():
        print(f"  {k:12s} " + " ".join(f"{n} {v}" for n, v in m.items()), flush=True)

    # ---- the DMX protocol on held-out coach clips: cue-held vs per-frame vs conductor
    from lightsem.model.jd_data import make_jd_splits
    from lightsem.model.light_data import load_tensor
    z, meta = load_tensor(CACHE_ROOT / "train_tensor_jd_none.npz"); sp = make_jd_splits(meta); idx = sp.val_track
    idx = idx[rng.permutation(len(idx))[:a.n_dmx]]; by_name = {m["name"]: m for m in meta}
    lm = LightModel(a.light_ckpt); cond = ConductorHybrid(); rig = canonical_rig(); eng = GroupEngine(rig)
    rows = []
    for k in idx:
        nm = meta[k]["name"]; d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{nm}.npz"); mid = by_name[nm]["music_id"]
        vp = CACHE_ROOT / "justdance_visual" / f"{mid}.npz"
        if not vp.exists():
            continue
        off, dur = float(d.meta.get("start_s", 0.0)), d.duration_s
        bus = slice_bus(load_bus(CACHE_ROOT / "justdance_audio" / f"{mid}.npz"), dur, offset_s=off)
        if bus.beats.size < 6:
            continue
        s = next((q for q in songs if q["key"] == mid), None)
        if s is None:
            continue
        h, p = hazard_and_h(s); T = len(s["x"])
        g = torch.Generator(device=device).manual_seed(0); look = dec.m.sample_look(p["h"], generator=g)[0].cpu().numpy()
        cues = pick_cues(h, thr)
        L = np.zeros_like(look); cur = look[0]
        for t in range(T):
            if cues[t] > 0:
                cur = look[t]
            L[t] = cur
        base = {kk: v[0].float().cpu().numpy() for kk, v in p.items() if kk != "h"}
        base["spread"] = np.zeros_like(base["spread"])                                       # the corrected engine (on the beat)
        i0, i1 = int(round(off * HZ)), int(round((off + dur) * HZ))
        q_ch = assign_look(L, base, tau_level=0.5, tau_colour=0.5); q_pf = assign_look(look, base)
        q_ch = {kk: v[i0:i1] for kk, v in q_ch.items()}; q_pf = {kk: v[i0:i1] for kk, v in q_pf.items()}
        if len(q_ch["inten"]) < 80:
            continue
        t10 = np.arange(len(q_ch["inten"])) / HZ; barph = bus.channels["bar_phase"]
        real = cond.render(d, bus).frames; ref = level_features(real, rig)["mean"]
        conductor = cond.render(lm.descriptors(bus, seed=0), bus).frames
        def engine(q):
            fr = eng.render(q, t10, bus.beats, barph, dur).frames; gg = float(np.clip(ref / max(level_features(fr, rig)["mean"], 1e-3), 0.3, 3.0))
            return eng.render(q, t10, bus.beats, barph, dur, gain=gg).frames if abs(gg - 1) > 0.05 else fr
        arms = {"real": real, "conductor": conductor, "cue_held": engine(q_ch), "per_frame": engine(q_pf)}
        row = {"name": nm}
        for kk, fr in arms.items():
            ev, _ = light_onsets(fr, rig, 40.0); r = bas_report(ev, bus.beats, dur, rng=rng); row[kk] = {"bas_null": r["bas"] - r["null_mean"], "events": r["n_events"]}
        rows.append(row)
    bas = {kk: float(np.nanmean([r[kk]["bas_null"] for r in rows])) for kk in ("real", "conductor", "cue_held", "per_frame")}
    evn = {kk: float(np.mean([r[kk]["events"] for r in rows])) for kk in bas}
    paired = np.array([r["cue_held"]["bas_null"] - r["conductor"]["bas_null"] for r in rows]); paired = paired[~np.isnan(paired)]
    bs = np.array([paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(4000)])
    gates = {"G10.1-cue": {"passed": bool(dist["cue_held"] < dist["per_frame"] and dist["cue_held"] < dist["donor"] and dist["cue_held"] > floor["mean"]),
                           "evidence": f"cue-transition distance to the authored JD set ({len(te)} held-out songs): cue-held {dist['cue_held']:.2f}, per-frame {dist['per_frame']:.2f}, "
                                       f"cue-held on beat-trigger cues {dist['cue_held_bt']:.2f}, shuffled donor {dist['donor']:.2f}, split-half floor {floor['mean']:.2f}. "
                                       f"Cue lift: hazard {np.mean(ap_lift['hazard']):.2f}x vs beat-trigger novelty {np.mean(ap_lift['beat_trigger']):.2f}x. Means: {json.dumps(means)}"},
             "G10.1-bas": {"passed": bool(bas["cue_held"] >= bas["conductor"] - 0.005 and evn["cue_held"] <= 1.1 * evn["conductor"]),
                           "evidence": f"Light-BAS minus null on {len(rows)} held-out coach clips (engine with zero phase spread): cue-held {bas['cue_held']:+.4f}, per-frame {bas['per_frame']:+.4f}, "
                                       f"conductor {bas['conductor']:+.4f}, real coach {bas['real']:+.4f}; paired cue-held minus conductor {paired.mean():+.4f} CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]; "
                                       f"events per clip cue-held {evn['cue_held']:.0f}, per-frame {evn['per_frame']:.0f}, conductor {evn['conductor']:.0f}, real {evn['real']:.0f}."}}
    res = {"ckpt": a.ckpt, "threshold": thr, "corpus_cue_rate_per_min": float(rate), "n_test_songs": len(te), "distance": dist | {"floor": floor}, "means": means,
           "cue_lift": {k: float(np.mean(v)) for k, v in ap_lift.items()}, "bas_minus_null": bas, "events_per_clip": evn, "gates": gates,
           "measurements": [{"diagnostic_id": "cue-transition-distance", "before": round(dist["per_frame"], 3), "after": round(dist["cue_held"], 3)},
                            {"diagnostic_id": "light-bas-minus-shift-null", "before": round(bas["conductor"], 4), "after": round(bas["cue_held"], 4)}],
           "config": {"arm": "run_stage10_cue_eval.py: v2 flow head sampled once per cue and held (0.5 s crossfade), cues from the decoder's look-change hazard with the threshold "
                      f"calibrated on training songs to the corpus rate ({rate:.2f}/min, thr {thr:.3f}), refractory 2 s; engine with zero phase spread", "ckpt": a.ckpt},
           "notes": f"H-N cue-held sampler ({a.tag}). {gates['G10.1-cue']['evidence']} {gates['G10.1-bas']['evidence']}", "elapsed_s": time.time() - t0}
    (OUT / f"cue_eval_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    for g_, v in gates.items():
        print(f"{g_}: {'PASS' if v['passed'] else 'FAIL'} -- {v['evidence'][:600]}")


if __name__ == "__main__":
    main()
