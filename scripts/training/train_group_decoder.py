#!/usr/bin/env python
"""H-I: train the group-effects decoder on AUTHORED light (PLAN.md 9.3), never on the conductor.

Corpora, all in the operator-head format (52-d causal features at 10 Hz + a 6-d scene vector):
RPMC-L2 operators (699), Just Dance visual (332) and, only if gate G9.2 passed, Beat Saber
(300). One decoder, a learned corpus embedding (SongFormer's source embedding), the frozen
audio-only operator head's hidden state as extra conditioning, a random rig per window, and
two losses on the differentiable render: the scene vector (level dims gain-normalised inside
the window on BOTH sides -- absolute light is a recorded rule-out) and the look-change hazard
against the corpus's own scene-change events (the exp-016 target).

    bash run_torch.sh scripts/training/train_group_decoder.py --steps 6000 --corpora rpmc,jd --run-tag v1
"""
from __future__ import annotations

import argparse, glob, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lightsem.model.group_decoder import DecoderConfig, GroupDecoder, random_rig, render_scene, window_norm
from lightsem.model.light_trunk import LightTrunk, TrunkConfig
from scripts.diagnostics.probe_event_target import average_precision, event_train, hazard_feats, target
from scripts.training.train_operator_head import split

CACHE = Path("/mnt/lightsem-data/cache")
CKPT = Path("/mnt/lightsem-data/ckpt")
OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage9")
CORPORA = {"rpmc": (CACHE / "rpmc_causal", 0), "jd": (CACHE / "justdance_visual", 1), "bsaber": (CACHE / "beatsaver_visual", 2)}
WIN = 300
BEAT_COLS = (48, 49)          # beat_sin, beat_cos in the 52-d feature vector


def load_corpus(name: str, limit: int = 0):
    root, cid = CORPORA[name]
    songs = []
    for p in sorted(glob.glob(f"{root}/*.npz")):
        d = np.load(p, allow_pickle=False)
        x, y = d["x"].astype(np.float32), d["y"].astype(np.float32)
        if len(x) < WIN + 20 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        tr = d["ev_scene"].astype(np.float32) if "ev_scene" in d else event_train(y)[0]
        phase = np.mod(np.arctan2(x[:, BEAT_COLS[0]], x[:, BEAT_COLS[1]]) / (2 * np.pi), 1.0).astype(np.float32)
        songs.append({"key": Path(p).stem, "corpus": name, "cid": cid, "x": x, "y": y, "phase": phase,
                      "t": target(tr).astype(np.float32), "haz": hazard_feats(tr).astype(np.float32)})
        if limit and len(songs) >= limit:
            break
    return songs


@torch.no_grad()
def operator_state(songs, device, ckpt=CKPT / "operator_head.pt"):
    """Frozen audio-only operator head (RPMC) -> its 192-d trunk state per frame, float16."""
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = TrunkConfig(**blob["cfg"]); m = LightTrunk(cfg).to(device).eval(); m.load_state_dict(blob["model"])
    mu = torch.as_tensor(blob["mu"], device=device); sd = torch.as_tensor(blob["sd"], device=device)
    assert not blob.get("with_hazard"), "the conditioning head must be audio-only"
    for s in songs:
        x = (torch.from_numpy(s["x"]).to(device)[None] - mu) / sd
        s["cond"] = m.trunk(x)[0].half().cpu().numpy()
    return cfg.d_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--corpora", default="rpmc,jd")
    ap.add_argument("--run-tag", default="v1")
    ap.add_argument("--no-cond", action="store_true", help="no frozen operator-head state")
    ap.add_argument("--no-random-rig", action="store_true", help="ablation: canonical rig weights only")
    ap.add_argument("--look-weight", type=float, default=0.5)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--flow", action="store_true", help="v2: sample the look with a rectified flow (look-then-assign)")
    ap.add_argument("--self-hazard", action="store_true", help="H-O: hazard features of the cue train appended to the input (teacher-forced; own cues at inference)")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="songs per corpus (smoke)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    songs = []
    for c in args.corpora.split(","):
        songs += load_corpus(c, args.limit)
    cond_dim = 0 if args.no_cond else operator_state(songs, device)
    tr, va, te = split(songs)
    print(f"{len(songs)} songs ({args.corpora}): train {len(tr)} val {len(va)} test {len(te)}; cond {cond_dim}; {time.time()-t0:.0f}s", flush=True)

    def feats(s):
        parts = [s["x"]] + ([] if args.no_cond else [s["cond"].astype(np.float32)]) + ([s["haz"]] if args.self_hazard else [])
        return parts[0] if len(parts) == 1 else np.concatenate(parts, 1)
    sample = np.concatenate([feats(s)[:600] for s in tr[:300]])
    mu = torch.from_numpy(sample.mean(0)).to(device); sd = torch.from_numpy(sample.std(0) + 1e-6).to(device)
    ysample = np.concatenate([s["y"] for s in tr[:300]])
    # per-dim weights: hue/sat/conc on their corpus variance, value dims are window-normalised (unit)
    yv = ysample.var(0) + 1e-4
    wdim = torch.tensor([1 / yv[0], 1 / yv[1], 1 / yv[2], 1.0, 1.0, 1 / yv[5]], dtype=torch.float32, device=device)
    wdim = wdim / wdim.mean()
    pos = float(np.concatenate([s["t"] for s in tr]).mean()); pw = torch.tensor([(1 - pos) / pos], device=device)

    cfg = DecoderConfig(in_dim=songs[0]["x"].shape[1], cond_dim=cond_dim + (songs[0]["haz"].shape[1] if args.self_hazard else 0),
                        offline=args.offline, flow=args.flow, d_model=args.d_model, n_blocks=args.n_blocks)
    model = GroupDecoder(cfg).to(device)
    print(f"decoder {model.n_params()/1e6:.2f} M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(0)

    def batch(n):
        xs, ys, ps, ts, cs = [], [], [], [], []
        for _ in range(n):
            s = tr[rng.integers(0, len(tr))]
            i = rng.integers(0, len(s["x"]) - WIN)
            xs.append(feats(s)[i:i + WIN]); ys.append(s["y"][i:i + WIN]); ps.append(s["phase"][i:i + WIN])
            ts.append(s["t"][i:i + WIN]); cs.append(s["cid"])
        f = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
        return f(xs), f(ys), f(ps), f(ts), torch.tensor(cs, device=device)

    def losses(x, y, ph, t, c, train=True):
        p = model((x - mu) / sd, c)
        w = random_rig(x.shape[0], device) if (train and not args.no_random_rig) else \
            torch.tensor([[4.0, 6.0, 1.0, 2.0, 1.0]], device=device).repeat(x.shape[0], 1)
        yh = render_scene(p, ph, w)
        l_scene = (((window_norm(yh) - window_norm(y)) ** 2) * wdim).mean()
        l_look = F.binary_cross_entropy_with_logits(p["look_logit"], t, pos_weight=pw)
        if args.flow:
            l_scene = l_scene + model.look_flow.loss(p["h"], window_norm(y))
        return l_scene, l_look, p

    model.train()
    for step in range(args.steps):
        x, y, ph, t, c = batch(args.batch)
        l_scene, l_look, _ = losses(x, y, ph, t, c)
        loss = l_scene + args.look_weight * l_look
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:5d} scene {l_scene.item():.4f} look {l_look.item():.4f}  {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ held-out evaluation per corpus
    model.eval()

    @torch.no_grad()
    def own_cue_scores(s, block: int = 10, thr: float = 0.5, refractory: int = 20):
        """Autoregressive inference: the hazard features come from the decoder's OWN emitted cues, recomputed
        every `block` frames (1 s); a cue fires when the hazard exceeds `thr` after a refractory period."""
        T = len(s["x"]); base = s["x"] if args.no_cond else np.concatenate([s["x"], s["cond"].astype(np.float32)], 1)
        cues = np.zeros(T, np.float32); scores = np.zeros(T, np.float32); last = -10 ** 6
        for b0 in range(0, T, block):
            hz = hazard_feats(cues).astype(np.float32)          # from cues emitted so far (zeros ahead)
            x = torch.from_numpy(np.concatenate([base[:b0 + block], hz[:b0 + block]], 1)).to(device)[None]
            pl = torch.sigmoid(model((x - mu) / sd, torch.tensor([s["cid"]], device=device))["look_logit"][0, b0:b0 + block]).cpu().numpy()
            scores[b0:b0 + block] = pl
            for j, v in enumerate(pl):
                if v > thr and (b0 + j) - last >= refractory:
                    cues[b0 + j] = 1.0; last = b0 + j
        return scores
    res = {"n_songs": len(songs), "corpora": args.corpora, "cond_dim": cond_dim, "params_m": model.n_params() / 1e6, "self_hazard": args.self_hazard,
           "d_model": args.d_model, "n_blocks": args.n_blocks,
           "random_rig": not args.no_random_rig, "offline": args.offline, "per_corpus": {}}
    with torch.no_grad():
        for cname in args.corpora.split(","):
            rows = {"scene_mse": [], "scene_mse_shuffled_donor": [], "look_ap_lift": [], "look_ap_lift_hazard_ridge": [], "value_r": [],
                    "sample_scene_mse": [], "sample_value_std_ratio": [], "sample_value_r": []}
            tes = [s for s in te if s["corpus"] == cname]
            trs = [s for s in tr if s["corpus"] == cname]
            # rhythm baseline for the look-change events, fitted on train songs (train_operator_head's recipe)
            XtX = sum(s["haz"].T.astype(np.float64) @ s["haz"] for s in trs); Xty = sum(s["haz"].T.astype(np.float64) @ s["t"] for s in trs)
            w_h = np.linalg.solve(XtX + 30 * np.eye(XtX.shape[0]), Xty)
            for j, s in enumerate(tes):
                x = torch.from_numpy(feats(s)).to(device)[None]; y = torch.from_numpy(s["y"]).to(device)[None]
                ph = torch.from_numpy(s["phase"]).to(device)[None]; t = torch.from_numpy(s["t"]).to(device)[None]
                c = torch.tensor([s["cid"]], device=device)
                l_scene, _, p = losses(x, y, ph, t, c, train=False)
                rows["scene_mse"].append(l_scene.item())
                # shuffled donor: this song's audio decoded, scored against ANOTHER held-out song's scene
                d = tes[(j + 1) % len(tes)]; n = min(len(s["y"]), len(d["y"]))
                yh = render_scene({k: v[:, :n] for k, v in p.items()}, ph[:, :n], torch.tensor([[4.0, 6.0, 1.0, 2.0, 1.0]], device=device))
                yd = torch.from_numpy(d["y"][:n]).to(device)[None]
                rows["scene_mse_shuffled_donor"].append((((window_norm(yh) - window_norm(yd)) ** 2) * wdim).mean().item())
                if s["t"].sum() >= 3:
                    pl = torch.sigmoid(p["look_logit"][0]).cpu().numpy()
                    rows["look_ap_lift"].append(average_precision(pl, s["t"]) / s["t"].mean())
                    if args.self_hazard:
                        rows.setdefault("look_ap_lift_own_cues", []).append(average_precision(own_cue_scores(s), s["t"]) / s["t"].mean())
                    rows["look_ap_lift_hazard_ridge"].append(average_precision(s["haz"] @ w_h, s["t"]) / s["t"].mean())
                yh_full = render_scene(p, ph, torch.tensor([[4.0, 6.0, 1.0, 2.0, 1.0]], device=device))[0].cpu().numpy()
                rows["value_r"].append(float(np.corrcoef(yh_full[:, 3], s["y"][:, 3])[0, 1]))
                if args.flow:
                    g = torch.Generator(device=device).manual_seed(0)
                    lk = model.sample_look(p["h"], generator=g)
                    yn = window_norm(y)
                    rows["sample_scene_mse"].append((((lk - yn) ** 2) * wdim).mean().item())
                    lkn = lk[0].cpu().numpy(); ynn = yn[0].cpu().numpy()
                    rows["sample_value_std_ratio"].append(float(lkn[:, 3].std() / (ynn[:, 3].std() + 1e-6)))
                    rows["sample_value_r"].append(float(np.corrcoef(lkn[:, 3], ynn[:, 3])[0, 1]))
            res["per_corpus"][cname] = {k: float(np.nanmean(v)) for k, v in rows.items() if v} | {"n_test": len(tes)}
            r = res["per_corpus"][cname]
            print(f"  [{cname}] test {len(tes)}: scene MSE {r['scene_mse']:.4f} (shuffled donor {r['scene_mse_shuffled_donor']:.4f})  "
                  f"look-change lift {r['look_ap_lift']:.2f} vs rhythm {r['look_ap_lift_hazard_ridge']:.2f}  value r {r['value_r']:+.3f}", flush=True)
    res["train_s"] = time.time() - t0
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                "wdim": wdim.cpu().numpy(), "corpora": args.corpora, "cond": not args.no_cond, "flow": args.flow, "self_hazard": args.self_hazard,
                "corpus_ids": {k: v[1] for k, v in CORPORA.items()}}, CKPT / f"group_decoder_{args.run_tag}.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"group_decoder_{args.run_tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
