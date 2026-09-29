#!/usr/bin/env python
"""H-I gate G9.3: was the hand-written conductor the bottleneck?  (PLAN.md 9.3)

The Stage 1 light-level protocol (scripts/legacy_stages/run_stage1_eval.py, Just Dance held-out songs) with the
group-effects decoder + engine as a new arm, every arm brightness-matched to the reference:

  real        hybrid conductor on the coach's REAL motion (the FID_light reference, as before)
  conductor   the shipped model: causal trunk -> descriptors -> hybrid conductor
  beat_trigger the 30-line baseline
  decoder     audio -> group decoder -> effects engine (this stage's claim)
  donor       ANOTHER held-out song's decoder output realised on this song's beat grid: the
              shuffled-donor control the FID gain must beat by > 2x (H-C's trap, pre-registered)
  engine_only the decoder's slow knobs (2 s) driving beat-trigger effects: is the learned
              effect choice doing anything, or only the palette/intensity plan (9.9)

FID_light on delta features against real (split-half floor, block-shuffle null), Light-BAS
minus its circular-shift null, light events per clip (a BAS bought by flashing every beat is
a fail), and the RIG-TRANSFER test: the decoder re-realised on an unseen rig (8 heads, 6
washes, 4 blinders, no strobe) against channel-copy of the canonical DMX, on the emitted
HSV-histogram distance to the canonical render.

    bash run_torch.sh scripts/legacy_stages/run_stage9_decoder_eval.py --n 60 --ckpt /mnt/lightsem-data/ckpt/group_decoder_v1.pt
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from lightsem.conductor.beat_trigger import render_beat_trigger
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.decode import decode_stream, emitted_light
from lightsem.dmx.encode import pack_universe
from lightsem.dmx.fixtures import Fixture, Rig
from lightsem.eval.bas import bas_report
from lightsem.eval.fid import FeatureScaler, frechet, split_half_floor
from lightsem.eval.light_features import block_shuffle, delta_features, level_features, light_onsets
from lightsem.model.group_decoder import DecoderConfig, GroupDecoder, GroupEngine, assign_look, slow_knobs
from lightsem.model.infer import LightModel
from lightsem.model.jd_data import make_jd_splits
from lightsem.model.light_data import load_tensor
from lightsem.model.light_trunk import LightTrunk, TrunkConfig
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9")
CKPT = Path("/mnt/lightsem-data/ckpt")
HZ = 10.0


def unseen_rig() -> Rig:
    """A rig the decoder never saw: 8 heads, 6 washes, 4 blinders, projector, NO strobe."""
    fx, addr = [], 1
    for i, x in enumerate(np.linspace(-6.0, 6.0, 8)):
        fx.append(Fixture(name=f"mh{i+1}", kind="moving_head", address=addr, position=(float(x), 6.5, -1.5),
                          direction=(0.0, -1.0, 0.2), beam_min_deg=5.0, beam_max_deg=30.0)); addr += 12
    for i, x in enumerate(np.linspace(-4.5, 4.5, 6)):
        fx.append(Fixture(name=f"wash{i+1}", kind="wash", address=addr, position=(float(x), 5.0, 1.0),
                          direction=(0.0, -1.0, -0.3), beam_min_deg=25.0, beam_max_deg=60.0)); addr += 7
    for i, x in enumerate((-3.5, -1.2, 1.2, 3.5)):
        fx.append(Fixture(name=f"blinder{i+1}", kind="blinder", address=addr, position=(float(x), 4.0, 2.5),
                          direction=(0.0, -0.2, 1.0))); addr += 2
    fx.append(Fixture(name="proj1", kind="projector", address=addr, position=(0.0, 3.0, 4.0), direction=(0.0, 0.0, -1.0),
                      beam_min_deg=30.0, beam_max_deg=55.0))
    return Rig(fx)


def hsv_hist(frames: np.ndarray, rig: Rig, n_h: int = 24, n_v: int = 12) -> np.ndarray:
    """(T, n_h + n_v) emitted-light-weighted hue histogram and value histogram per frame."""
    dec = decode_stream(rig, frames)
    lum = emitted_light(dec)                                  # (T, nf)
    T = lum.shape[0]
    out = np.zeros((T, n_h + n_v))
    hb = np.minimum((dec["hue"] * n_h).astype(int), n_h - 1)
    vb = np.minimum((np.clip(lum, 0, 1) * n_v).astype(int), n_v - 1)
    for t in range(T):
        w = lum[t] * dec["sat"][t]
        out[t, :n_h] = np.bincount(hb[t], weights=w, minlength=n_h)
        out[t, n_h:] = np.bincount(vb[t], minlength=n_v)
    out[:, :n_h] /= np.maximum(out[:, :n_h].sum(1, keepdims=True), 1e-6)
    out[:, n_h:] /= np.maximum(out[:, n_h:].sum(1, keepdims=True), 1e-6)
    return out


def channel_copy(frames: np.ndarray, src: Rig, dst: Rig) -> np.ndarray:
    """Copy the canonical fixtures' decoded parameters onto the unseen rig by index (the naive port)."""
    dec = decode_stream(src, frames)
    n_src, n_dst = len(src.fixtures), len(dst.fixtures)
    P = {}
    for k in ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom"):
        P[k] = np.stack([dec[k][:, j % n_src] for j in range(n_dst)], 1)
    P["gobo"] = np.zeros_like(P["dimmer"])
    return pack_universe(dst, P)


class Decoder:
    def __init__(self, ckpt: Path, device: str):
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = DecoderConfig(**blob["cfg"]); self.m = GroupDecoder(self.cfg).to(device).eval(); self.m.load_state_dict(blob["model"])
        self.mu = torch.as_tensor(blob["mu"], device=device); self.sd = torch.as_tensor(blob["sd"], device=device)
        self.cids = blob["corpus_ids"]; self.device = device; self.cond = blob["cond"]
        if self.cond:
            ob = torch.load(CKPT / "operator_head.pt", map_location=device, weights_only=False)
            self.op = LightTrunk(TrunkConfig(**ob["cfg"])).to(device).eval(); self.op.load_state_dict(ob["model"])
            self.omu = torch.as_tensor(ob["mu"], device=device); self.osd = torch.as_tensor(ob["sd"], device=device)

    @torch.no_grad()
    def params(self, x: np.ndarray, corpus: str = "jd", seed: int = 0) -> dict[str, np.ndarray]:
        xt = torch.from_numpy(x.astype(np.float32)).to(self.device)[None]
        if self.cond:
            xt = torch.cat([xt, self.op.trunk((xt - self.omu) / self.osd)], -1)
        p = self.m((xt - self.mu) / self.sd, torch.tensor([self.cids[corpus]], device=self.device))
        out = {k: v[0].float().cpu().numpy() for k, v in p.items() if k != "h"}
        if self.cfg.flow:
            g = torch.Generator(device=self.device).manual_seed(seed)
            look = self.m.sample_look(p["h"], generator=g)[0].float().cpu().numpy()
            out = assign_look(look, out); out["look"] = look
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="val_track")
    ap.add_argument("--ckpt", default=str(CKPT / "group_decoder_v1.pt"))
    ap.add_argument("--light-ckpt", default=str(CKPT / "light_trunk_jd_none_smooth.pt"))
    ap.add_argument("--tensor", default=str(CACHE_ROOT / "train_tensor_jd_none.npz"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--corpus-id", default="jd", help="which corpus embedding the decoder runs with (jd | rpmc)")
    ap.add_argument("--zero-spread", action="store_true", help="engine diagnostic: every effect on the beat (phase spread 0)")
    a = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z, meta = load_tensor(Path(a.tensor))
    sp = make_jd_splits(meta); idx = getattr(sp, a.split)
    rng = np.random.default_rng(0); idx = idx[rng.permutation(len(idx))[:a.n]]
    names = [meta[i]["name"] for i in idx]; by_name = {m["name"]: m for m in meta}
    lm = LightModel(a.light_ckpt); cond = ConductorHybrid(); rig = canonical_rig(); rig2 = unseen_rig()
    dec = Decoder(Path(a.ckpt), device); eng = GroupEngine(rig); eng2 = GroupEngine(rig2)
    ARMS = ("real", "conductor", "beat_trigger", "decoder", "donor", "engine_only")
    F = {k: [] for k in ARMS}; F["null"] = []; rows = []; transfer = []
    # decoder parameters per clip first (donors need them)
    clips = []
    for nm in names:
        d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{nm}.npz")
        mid = by_name[nm]["music_id"]
        vp = CACHE_ROOT / "justdance_visual" / f"{mid}.npz"
        if not vp.exists():
            continue
        bus_full = load_bus(CACHE_ROOT / "justdance_audio" / f"{mid}.npz")
        dur = d.duration_s; off = float(d.meta.get("start_s", 0.0))
        bus = slice_bus(bus_full, dur, offset_s=off)
        if bus.beats.size < 6:
            continue
        x = np.load(vp, allow_pickle=False)["x"].astype(np.float32)
        p_full = dec.params(x, a.corpus_id)                                     # causal: whole capture, then slice
        i0, i1 = int(round(off * HZ)), int(round((off + dur) * HZ))
        p = {k: v[i0:i1] for k, v in p_full.items()}
        if a.zero_spread:
            p["spread"] = np.zeros_like(p["spread"])
        if len(p["inten"]) < 50:
            continue
        clips.append({"name": nm, "d": d, "bus": bus, "dur": dur, "p": p})
    print(f"{len(clips)} clips prepared in {time.time()-t0:.0f}s", flush=True)

    def match_gain(frames_fn, ref_mean):
        fr = frames_fn(1.0); m = level_features(fr, rig)["mean"]
        g = float(np.clip(ref_mean / max(m, 1e-3), 0.3, 3.0))
        return frames_fn(g) if abs(g - 1) > 0.05 else fr, g
    for j, c in enumerate(clips):
        d, bus, dur, p = c["d"], c["bus"], c["dur"], c["p"]
        beats = bus.beats; bar = bus.channels["bar_phase"]; t10 = np.arange(len(p["inten"])) / HZ
        real = cond.render(d, bus).frames
        ref_mean = level_features(real, rig)["mean"]
        conductor = cond.render(lm.descriptors(bus, seed=0), bus).frames
        bt = render_beat_trigger(bus).frames
        decoder, g_dec = match_gain(lambda g: eng.render(p, t10, beats, bar, dur, gain=g).frames, ref_mean)
        dn = clips[(j + 1) % len(clips)]["p"]; n = len(p["inten"])
        pd = {k: (v[:n] if len(v) >= n else np.concatenate([v, np.repeat(v[-1:], n - len(v), 0)])) for k, v in dn.items()}
        donor, _ = match_gain(lambda g: eng.render(pd, t10, beats, bar, dur, gain=g).frames, ref_mean)
        eo, _ = match_gain(lambda g: eng.render(slow_knobs(p), t10, beats, bar, dur, gain=g).frames, ref_mean)
        arms = {"real": real, "conductor": conductor, "beat_trigger": bt, "decoder": decoder, "donor": donor, "engine_only": eo}
        for k, fr in arms.items():
            F[k].append(delta_features(fr, rig, 40.0))
        F["null"].append(delta_features(block_shuffle(real, 40.0, 1.0, rng), rig, 40.0))
        row = {"name": c["name"], "gain_decoder": g_dec}
        for k, fr in arms.items():
            ev, _ = light_onsets(fr, rig, 40.0)
            row[k] = bas_report(ev, beats, dur, rng=rng) | {"level": level_features(fr, rig)["mean"]}
        # rig transfer: decoder re-realised on the unseen rig vs channel-copy of the canonical DMX
        dec2 = eng2.render(p, t10, beats, bar, dur, gain=g_dec).frames
        copy2 = channel_copy(decoder, rig, rig2)
        h_ref = hsv_hist(decoder, rig); h_dec = hsv_hist(dec2, rig2); h_cp = hsv_hist(copy2, rig2)
        transfer.append({"decoder": float(np.abs(h_dec - h_ref).sum(1).mean()), "channel_copy": float(np.abs(h_cp - h_ref).sum(1).mean())})
        rows.append(row)
        if j % 10 == 0:
            print(f"  [{j+1}/{len(clips)}] {c['name']} gain {g_dec:.2f} bas dec {row['decoder']['bas']-row['decoder']['null_mean']:+.3f} "
                  f"cond {row['conductor']['bas']-row['conductor']['null_mean']:+.3f}  ev/clip dec {row['decoder']['n_events']} cond {row['conductor']['n_events']} real {row['real']['n_events']}  "
                  f"transfer dec {transfer[-1]['decoder']:.3f} copy {transfer[-1]['channel_copy']:.3f}  {time.time()-t0:.0f}s", flush=True)
    X = {k: np.array(v) for k, v in F.items()}
    sc = FeatureScaler(X["real"]); Z = {k: sc(v) for k, v in X.items()}
    fid = {k: frechet(Z[k], Z["real"]) for k in ARMS if k != "real"}
    fid["real_vs_block_shuffled"] = frechet(Z["real"], Z["null"]); floor = split_half_floor(Z["real"], rng=rng)

    def paired(key):
        v = np.array([r[key]["bas"] - r[key]["null_mean"] for r in rows]); n_nan = int(np.isnan(v).sum()); v = v[~np.isnan(v)]
        bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)]) if len(v) else np.array([np.nan])
        return {"delta": float(v.mean()) if len(v) else float("nan"), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_no_events": n_nan}
    bas = {k: paired(k) for k in ARMS}
    events = {k: float(np.mean([r[k]["n_events"] for r in rows])) for k in ARMS}
    levels = {k: float(np.mean([r[k]["level"] for r in rows])) for k in ARMS}
    tr = {"decoder": float(np.mean([t["decoder"] for t in transfer])), "channel_copy": float(np.mean([t["channel_copy"] for t in transfer]))}
    d_paired = np.array([r["decoder"]["bas"] - r["decoder"]["null_mean"] - (r["conductor"]["bas"] - r["conductor"]["null_mean"]) for r in rows]); d_paired = d_paired[~np.isnan(d_paired)]
    gain_dec = fid["conductor"] - fid["decoder"]; gain_donor = fid["conductor"] - fid["donor"]
    res = {"n_clips": len(rows), "split": a.split, "ckpt": a.ckpt, "corpus_id": a.corpus_id,
           "fid_light": fid | {"split_half_floor": floor}, "bas_minus_null": bas, "events_per_clip": events, "brightness_mean": levels,
           "rig_transfer_hist_distance": tr, "bas_decoder_minus_conductor": float(d_paired.mean()) if len(d_paired) else float("nan"), "flow": dec.cfg.flow,
           "gates": {
               "G9.3-fid": {"passed": bool(fid["decoder"] < fid["conductor"] and gain_dec > 2 * gain_donor),
                            "evidence": f"FID_light vs real: decoder {fid['decoder']:.2f}, conductor {fid['conductor']:.2f}, donor {fid['donor']:.2f}, engine-only {fid['engine_only']:.2f}, beat trigger {fid['beat_trigger']:.2f}; block-shuffle null {fid['real_vs_block_shuffled']:.2f}, split-half floor {floor['mean']:.2f}. Gain over the conductor {gain_dec:+.2f} vs donor gain {gain_donor:+.2f} (needs > 2x)."},
               "G9.3-bas": {"passed": bool(bas["decoder"]["delta"] >= bas["conductor"]["delta"] - 0.005 and events["decoder"] <= 1.1 * events["conductor"]),
                            "evidence": f"Light-BAS minus null: decoder {bas['decoder']['delta']:+.4f} CI {np.round(bas['decoder']['ci95'],4).tolist()}, conductor {bas['conductor']['delta']:+.4f}, real coach {bas['real']['delta']:+.4f}, beat trigger {bas['beat_trigger']['delta']:+.4f}, engine-only {bas['engine_only']['delta']:+.4f}; events per clip decoder {events['decoder']:.1f} conductor {events['conductor']:.1f} real {events['real']:.1f}."},
               "G9.3-ab": {"passed": False, "evidence": "Not measured: the brightness-matched human A/B needs raters (n >= 20 pairs)."},
               "G9.3-rig": {"passed": bool(tr["decoder"] < tr["channel_copy"]),
                            "evidence": f"unseen rig (8 heads, 6 washes, 4 blinders, no strobe): HSV-histogram distance to the canonical render, decoder re-realised {tr['decoder']:.3f} vs channel-copy {tr['channel_copy']:.3f} (L1 per frame, hue + value histograms)."}},
           "elapsed_s": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"decoder_eval_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "gates"}, indent=1, default=float))
    for g, v in res["gates"].items():
        print(f"{g}: {'PASS' if v['passed'] else 'FAIL'} -- {v['evidence']}")


if __name__ == "__main__":
    main()
