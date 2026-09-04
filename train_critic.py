#!/usr/bin/env python
"""H-K: a factorised audio-light critic with beat-trigger negatives, used as a best-of-N reranker
(PLAN.md 9.5, gates G9.5-bt / G9.5-additive / G9.5-bon).

Bar-synchronous tokens from the authored corpora (RPMC-L2, Just Dance visual): audio = the
bar-pooled 52-d operator features; light = the bar-pooled scene vector, its within-bar value
std, and the look-change count. Windows of 16 bars. InfoNCE with THREE negative families for
every positive: a circular shift of the same song's light by >= 2 bars, another song's light,
and BEAT-TRIGGER light synthesised from this very audio (the rule set of conductor/beat_trigger.py
on the 10 Hz features) -- without the third, the cheapest solution is a beat detector.

Two heads trained the same way: C_R sees only the redundant channels (rms / onset / beat in the
audio, value mean / std in the light); C_U sees everything. The reranking score is C_U - lambda C_R.
The ADDITIVE PROJECTION test (MAS.S60 lecture 07) fits s(a, l) ~ f(a) + g(l) to the critic's own
scores and checks whether best-of-8 changes order; then best-of-8 over the shipped model's flow
seeds, scored through the conductor on held-out Just Dance clips, against random-of-8, on
FID_light and Light-BAS (the human question needs raters).

    bash run_torch.sh train_critic.py --steps 3000 --n-clips 40
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightsem.audio.online import ema
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.decode import decode_stream, emitted_light
from lightsem.eval.bas import bas_report
from lightsem.eval.fid import FeatureScaler, frechet, split_half_floor
from lightsem.eval.light_features import block_shuffle, delta_features, light_onsets
from lightsem.model.infer import LightModel
from lightsem.model.jd_data import make_jd_splits
from lightsem.model.light_data import load_tensor
from lightsem.motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus
from probe_event_target import event_train
from train_group_decoder import load_corpus
from train_operator_head import split
from train_planner import bar_pool, bars_of

CKPT = Path("/mnt/lightsem-data/ckpt"); OUT = Path("/root/lightsem/outputs/stage9")
HZ = 10.0; WIN = 16
# redundant channel indices in the 52-d features (rms x3, onset x3, beat sin/cos/accent)
RED_AUDIO = [0, 1, 2] + [18, 19, 20] + [48, 49, 50]
LIGHT_DIM = 6 + 1 + 1
RED_LIGHT = [3, 4, 6]        # value, p90, value std


def beat_trigger_scene(x: np.ndarray) -> np.ndarray:
    """conductor/beat_trigger.py's rules on the 10 Hz features -> (T, 6) scene vector."""
    def n(v):
        lo, hi = np.percentile(v, 5), np.percentile(v, 95); return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    lvl = n(x[:, 0]); onset = n(x[:, 18]); bright = n(x[:, 21]); accent = np.clip(x[:, 50], 0, 1)
    jump = np.abs(np.diff(bright, prepend=bright[:1])) > 0.18
    k = np.cumsum(jump & ~np.concatenate([[False], jump[:-1]]))
    hue = np.array((0.60, 0.92, 0.08, 0.33, 0.75, 0.50))[k % 6]
    washes = 0.15 + 0.7 * lvl; beams = (0.25 + 0.75 * accent) * (0.35 + 0.65 * lvl); blind = onset ** 3
    value = 0.4 * washes + 0.4 * beams + 0.2 * blind; p90 = np.maximum.reduce([washes, beams, blind])
    return np.stack([np.cos(2 * np.pi * hue), np.sin(2 * np.pi * hue), np.full_like(value, 0.85) * (1 - 0.2 * blind), value, p90, np.ones_like(value) * 0.9], 1).astype(np.float32)


def light_tokens(y: np.ndarray, bar: np.ndarray) -> np.ndarray:
    ev = event_train(y)[0]; n = int(bar.max()) + 1
    out = np.zeros((n, LIGHT_DIM), np.float32)
    for b in range(n):
        m = bar == b
        if m.any():
            out[b, :6] = y[m].mean(0); out[b, 6] = y[m, 3].std(); out[b, 7] = np.log1p(ev[m].sum())
    return out


def dmx_scene(frames: np.ndarray, rig, fps: float = 40.0) -> np.ndarray:
    """DMX -> (T10, 6) scene vector (the beat_trigger_scene of run_stage9_gate at 10 Hz)."""
    dec = decode_stream(rig, frames); lum = emitted_light(dec)
    w = lum / np.maximum(lum.sum(1, keepdims=True), 1e-9); hue = dec["hue"]; sat = dec["sat"]
    hc = (w * np.cos(2 * np.pi * hue)).sum(1); hs = (w * np.sin(2 * np.pi * hue)).sum(1)
    sc = np.stack([hc, hs, (w * sat).sum(1), lum.mean(1), np.percentile(lum, 90, axis=1), np.sqrt(hc ** 2 + hs ** 2)], 1)
    t40 = np.arange(len(sc)) / fps; t10 = np.arange(int(len(sc) / fps * HZ)) / HZ
    return np.stack([np.interp(t10, t40, sc[:, k]) for k in range(6)], 1).astype(np.float32)


class Enc(nn.Module):
    def __init__(self, in_dim, d=128, n_layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.SiLU())
        self.pos = nn.Embedding(64, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True), n_layers)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        h = self.inp(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return F.normalize(self.out(self.enc(h).mean(1)), dim=-1)


class Critic(nn.Module):
    def __init__(self, a_dim, l_dim):
        super().__init__(); self.a = Enc(a_dim); self.l = Enc(l_dim); self.temp = nn.Parameter(torch.tensor(2.0))

    def score(self, a, l):
        return (self.a(a) * self.l(l)).sum(-1) * self.temp.exp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--n-clips", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--light-ckpt", default=str(CKPT / "light_trunk_jd_none_smooth.pt"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    songs = load_corpus("rpmc", a.limit) + load_corpus("jd", a.limit)
    for s in songs:
        bar = bars_of(s["phase"]); s["bar"] = bar; s["ta"] = bar_pool(s["x"], bar); s["tl"] = light_tokens(s["y"], bar)
        s["tb"] = light_tokens(beat_trigger_scene(s["x"]), bar)
    songs = [s for s in songs if len(s["ta"]) >= WIN + 4]
    tr, va, te = split(songs); print(f"{len(songs)} songs: train {len(tr)} test {len(te)}  {time.time()-t0:.0f}s", flush=True)
    A = np.concatenate([s["ta"] for s in tr[:300]]); L = np.concatenate([s["tl"] for s in tr[:300]])
    amu, asd = torch.tensor(A.mean(0), device=device), torch.tensor(A.std(0) + 1e-6, device=device)
    lmu, lsd = torch.tensor(L.mean(0), device=device), torch.tensor(L.std(0) + 1e-6, device=device)
    red_a = torch.tensor([j for j in range(A.shape[1]) if j % 52 in RED_AUDIO], device=device); red_l = torch.tensor(RED_LIGHT, device=device)
    crit = {"U": Critic(A.shape[1], LIGHT_DIM).to(device), "R": Critic(len(red_a), len(RED_LIGHT)).to(device)}
    rng = np.random.default_rng(0)

    def norm(a, l):
        return (a - amu) / asd, (l - lmu) / lsd

    def windows(sgs, n):
        A_, P_, S_, O_, B_ = [], [], [], [], []
        for _ in range(n):
            s = sgs[rng.integers(0, len(sgs))]; i = rng.integers(0, len(s["ta"]) - WIN)
            A_.append(s["ta"][i:i + WIN]); P_.append(s["tl"][i:i + WIN]); B_.append(s["tb"][i:i + WIN])
            sh = int(rng.integers(2, max(3, len(s["ta"]) - WIN))); j = (i + sh) % (len(s["ta"]) - WIN)
            S_.append(s["tl"][j:j + WIN])
            o = sgs[rng.integers(0, len(sgs))]; k = rng.integers(0, len(o["ta"]) - WIN); O_.append(o["tl"][k:k + WIN])
        f = lambda v: torch.from_numpy(np.stack(v)).to(device)  # noqa: E731
        return f(A_), f(P_), f(S_), f(O_), f(B_)

    def run(c, key, a_, l_):
        a_, l_ = norm(a_, l_)
        if key == "R":
            a_, l_ = a_[..., red_a], l_[..., red_l]
        return c.score(a_, l_)

    for key, c in crit.items():
        opt = torch.optim.AdamW(c.parameters(), 3e-4, weight_decay=0.01)
        for step in range(a.steps):
            A_, P_, S_, O_, B_ = windows(tr, a.batch)
            pos = run(c, key, A_, P_); negs = torch.stack([run(c, key, A_, S_), run(c, key, A_, O_), run(c, key, A_, B_)], 1)
            loss = -(pos - torch.logsumexp(torch.cat([pos[:, None], negs], 1), 1)).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if step % 500 == 0 or step == a.steps - 1:
                print(f"  C_{key} step {step:4d} loss {loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)
        c.eval()

    # ------------------------------------------------------------ held-out pair accuracies
    @torch.no_grad()
    def score(a_, l_):
        return (run(crit["U"], "U", a_, l_) - a.lam * run(crit["R"], "R", a_, l_)).cpu().numpy()

    @torch.no_grad()
    def score_long(ta_, tl_):
        """A whole clip scored as the mean over 16-bar windows (stride 8), the length it was trained on."""
        n = min(len(ta_), len(tl_)); starts = list(range(0, max(n - WIN, 0) + 1, WIN // 2)) or [0]
        A_ = torch.from_numpy(np.stack([ta_[i:i + WIN] if i + WIN <= n else np.pad(ta_[i:n], ((0, i + WIN - n), (0, 0))) for i in starts])).to(device)
        L_ = torch.from_numpy(np.stack([tl_[i:i + WIN] if i + WIN <= n else np.pad(tl_[i:n], ((0, i + WIN - n), (0, 0))) for i in starts])).to(device)
        return float(score(A_, L_).mean())
    acc = {"vs_beat_trigger": [], "vs_shift": [], "vs_other_song": [], "U_vs_beat_trigger": [], "R_vs_beat_trigger": []}
    with torch.no_grad():
        for _ in range(40):
            A_, P_, S_, O_, B_ = windows(te, 64)
            sp = score(A_, P_)
            acc["vs_beat_trigger"] += (sp > score(A_, B_)).tolist(); acc["vs_shift"] += (sp > score(A_, S_)).tolist(); acc["vs_other_song"] += (sp > score(A_, O_)).tolist()
            acc["U_vs_beat_trigger"] += (run(crit["U"], "U", A_, P_) > run(crit["U"], "U", A_, B_)).cpu().tolist()
            acc["R_vs_beat_trigger"] += (run(crit["R"], "R", A_, P_) > run(crit["R"], "R", A_, B_)).cpu().tolist()
    acc = {k: {"acc": float(np.mean(v)), "n": len(v), "ci95_lo": float(np.mean(v) - 1.96 * np.sqrt(np.mean(v) * (1 - np.mean(v)) / len(v)))} for k, v in acc.items()}
    print("held-out pair accuracy:", json.dumps({k: round(v["acc"], 3) for k, v in acc.items()}), flush=True)

    # ------------------------------------------------------------ best-of-8 on held-out JD clips + additive projection
    z, meta = load_tensor(CACHE_ROOT / "train_tensor_jd_none.npz"); sp_ = make_jd_splits(meta); idx = sp_.val_track
    idx = idx[rng.permutation(len(idx))[:a.n_clips]]; by_name = {m["name"]: m for m in meta}
    lm = LightModel(a.light_ckpt); cond = ConductorHybrid(); rig = canonical_rig()
    F_best, F_rand, F_real, F_null, rows, order_changes = [], [], [], [], [], []
    feat_rows = []
    for i_ in idx:
        nm = meta[i_]["name"]; d = load_descriptors(CACHE_ROOT / "justdance_motion" / f"{nm}.npz"); mid = by_name[nm]["music_id"]
        vp = CACHE_ROOT / "justdance_visual" / f"{mid}.npz"
        if not vp.exists():
            continue
        off, dur = float(d.meta.get("start_s", 0.0)), d.duration_s
        bus = slice_bus(load_bus(CACHE_ROOT / "justdance_audio" / f"{mid}.npz"), dur, offset_s=off)
        if bus.beats.size < 6:
            continue
        x = np.load(vp, allow_pickle=False)["x"].astype(np.float32)[int(round(off * HZ)):int(round((off + dur) * HZ))]
        if len(x) < 200:
            continue
        phase = np.mod(np.arctan2(x[:, 48], x[:, 49]) / (2 * np.pi), 1.0); bar = bars_of(phase); ta = bar_pool(x, bar)
        cands, scores, ltoks = [], [], []
        for s_ in range(a.seeds):
            fr = cond.render(lm.descriptors(bus, seed=s_), bus).frames; cands.append(fr)
            y = dmx_scene(fr, rig); n = min(len(y), len(bar)); tl = light_tokens(y[:n], bar[:n])
            n_b = min(len(ta), len(tl))
            scores.append(score_long(ta[:n_b], tl[:n_b])); ltoks.append(tl[:n_b])
        best = int(np.argmax(scores)); real = cond.render(d, bus).frames
        F_best.append(delta_features(cands[best], rig, 40.0)); F_rand.append(delta_features(cands[0], rig, 40.0)); F_real.append(delta_features(real, rig, 40.0))
        F_null.append(delta_features(block_shuffle(real, 40.0, 1.0, rng), rig, 40.0))
        ev_b, _ = light_onsets(cands[best], rig, 40.0); ev_r, _ = light_onsets(cands[0], rig, 40.0); ev_t, _ = light_onsets(real, rig, 40.0)
        rows.append({"best": bas_report(ev_b, bus.beats, dur, rng=rng), "rand": bas_report(ev_r, bus.beats, dur, rng=rng), "real": bas_report(ev_t, bus.beats, dur, rng=rng), "scores": scores})
        feat_rows.append((ta[:len(ltoks[0])].mean(0), [t.mean(0) for t in ltoks], scores))
    # additive projection: s ~ f(a) + g(l) fitted by ridge on (audio mean token, light mean token) over all candidates
    Xa = np.concatenate([[r[0]] * len(r[1]) for r in feat_rows]); Xl = np.concatenate([r[1] for r in feat_rows]); S = np.concatenate([r[2] for r in feat_rows])
    Xadd = np.concatenate([Xa, Xl], 1); Xadd = (Xadd - Xadd.mean(0)) / (Xadd.std(0) + 1e-6)
    w = np.linalg.solve(Xadd.T @ Xadd + 10 * np.eye(Xadd.shape[1]), Xadd.T @ (S - S.mean())); S_add = Xadd @ w + S.mean()
    k = 0
    for r in feat_rows:
        n = len(r[1]); order_changes.append(int(np.argmax(r[2])) != int(np.argmax(S_add[k:k + n]))); k += n
    Xb, Xr, Xt, Xn = map(np.array, (F_best, F_rand, F_real, F_null)); sc_ = FeatureScaler(Xt)
    fid = {"best_of_8": frechet(sc_(Xb), sc_(Xt)), "random_of_8": frechet(sc_(Xr), sc_(Xt)), "null": frechet(sc_(Xt), sc_(Xn)), "floor": split_half_floor(sc_(Xt), rng=rng)["mean"]}
    bas = {k: float(np.mean([r[k]["bas"] - r[k]["null_mean"] for r in rows])) for k in ("best", "rand", "real")}
    r2_add = float(1 - ((S - S_add) ** 2).sum() / ((S - S.mean()) ** 2).sum())
    res = {"n_songs": len(songs), "pair_accuracy": acc, "best_of_8": {"n_clips": len(rows), "fid_light": fid, "bas_minus_null": bas,
           "additive_projection_r2": r2_add, "order_change_frac": float(np.mean(order_changes))},
           "gates": {"G9.5-bt": {"passed": bool(acc["vs_beat_trigger"]["ci95_lo"] > 0.5 and acc["vs_beat_trigger"]["acc"] >= 0.6),
                                 "evidence": f"held-out pair accuracy (16-bar windows, {acc['vs_beat_trigger']['n']} pairs): real vs beat-trigger light on the same audio {acc['vs_beat_trigger']['acc']:.3f} "
                                             f"(C_U alone {acc['U_vs_beat_trigger']['acc']:.3f}, C_R alone {acc['R_vs_beat_trigger']['acc']:.3f}), real vs 2+-bar shift {acc['vs_shift']['acc']:.3f}, real vs other song {acc['vs_other_song']['acc']:.3f}; chance 0.5."},
                     "G9.5-additive": {"passed": bool(np.mean(order_changes) >= 0.3),
                                       "evidence": f"an additive model f(audio) + g(light) fitted to the critic's scores over {len(S)} candidates explains R2 {r2_add:.3f}; the best-of-8 argmax changes on {np.mean(order_changes)*100:.0f}% of clips (gate >= 30%)."},
                     "G9.5-bon": {"passed": False, "evidence": f"human 'fits the music' not run (needs raters). Proxies on {len(rows)} held-out JD clips: FID_light best-of-8 {fid['best_of_8']:.2f} vs random-of-8 {fid['random_of_8']:.2f} "
                                                              f"(null {fid['null']:.2f}, floor {fid['floor']:.2f}); BAS minus null best {bas['best']:+.4f} vs random {bas['rand']:+.4f} vs real coach {bas['real']:+.4f}."}},
           "measurements": [{"diagnostic_id": "critic-pair-accuracy", "before": 0.5, "after": round(acc["vs_beat_trigger"]["acc"], 4)},
                            {"diagnostic_id": "light-bas-minus-shift-null", "before": round(bas["rand"], 4), "after": round(bas["best"], 4)}],
           "config": {"critic": "train_critic.py: two dual encoders (2-layer Transformer d128 each side) on 16-bar windows; audio = bar-pooled 52-d features, light = bar-pooled scene vector + value std + look-change count; "
                      "InfoNCE with shift / other-song / beat-trigger negatives; C_R on rms/onset/beat vs value channels only; score = C_U - 0.5 C_R; best-of-8 over the shipped model's flow seeds through the hybrid conductor",
                      "train_corpora": "rpmc,jd", "n_songs": len(songs)},
           "notes": "", "elapsed_s": time.time() - t0}
    res["notes"] = f"H-K critic. {res['gates']['G9.5-bt']['evidence']} {res['gates']['G9.5-additive']['evidence']} {res['gates']['G9.5-bon']['evidence']}"
    torch.save({k: c.state_dict() for k, c in crit.items()} | {"amu": amu.cpu().numpy(), "asd": asd.cpu().numpy(), "lmu": lmu.cpu().numpy(), "lsd": lsd.cpu().numpy()}, CKPT / "critic_v1.pt")
    OUT.mkdir(exist_ok=True); (OUT / "critic.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["gates"], indent=1))


if __name__ == "__main__":
    main()
