#!/usr/bin/env python
"""H-M: does one frozen music-foundation layer group add anything the bus lacks, once the
catalogue leak is subtracted?  (PLAN.md 9.7 item 5, gates G9.7-emb / G9.7-leak)

MERT-v1-330M (24 kHz, 75 Hz frames), two layer groups the plan names -- semantic (10-13) and
rhythm (15-16) -- mean-pooled to the 10 Hz operator grid and cached; PCA to 32 d fitted on
training songs; then exp-016's ridge on Just Dance visual scene-change events:

    bus | bus + PCA(embedding) | bus + same-width RANDOM PROJECTION of the bus (the width control)
    bus + PCA(embedding - its song mean)  (the residual after the catalogue hash is removed)

and the song-ID kNN leak probe: with each song's embedding split into two halves, how often does
the first half's mean find the second half of the SAME song as nearest neighbour among all
songs (1/n_songs is chance)? A high hit rate with no residual gain means the embedding is a hash
of the catalogue, exactly what the plan says to kill.

    bash run_torch.sh run_stage9_embedding.py extract        # GPU, caches cache/justdance_mert/<yt>.npz
    bash run.sh       run_stage9_embedding.py audit          # CPU, writes outputs/stage9/hm_embedding.json
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

CACHE = Path("/mnt/lightsem-data/cache"); OUT = Path("/root/lightsem/outputs/stage9")
EMB = CACHE / "justdance_mert"
MERT = "/mnt/lightsem-data/models/hf/models--m-a-p--MERT-v1-330M/snapshots/5240c2708a5acaee1007f43fb9735c7dcd0b78c9"
GROUPS = {"semantic": (9, 10, 11, 12), "rhythm": (14, 15)}    # hook index i = output of encoder layer i+1
HZ = 10.0


def extract():
    import torch, soundfile as sf, julius
    from transformers import AutoModel
    dev = "cuda"
    model = AutoModel.from_pretrained(MERT, trust_remote_code=True).to(dev).eval()
    caps: dict[int, torch.Tensor] = {}
    for i, layer in enumerate(model.encoder.layers):   # transformers 5 drops hidden_states from this custom model; hook the layers
        layer.register_forward_hook(lambda mod, inp, out, i=i: caps.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach()))
    EMB.mkdir(exist_ok=True)
    wavs = sorted((CACHE / "justdance_wav").glob("*.wav")); t0 = time.time(); n = 0
    for i, w in enumerate(wavs):
        out = EMB / f"{w.stem}.npz"
        if out.exists() or not (CACHE / "justdance_visual" / f"{w.stem}.npz").exists():
            continue
        x, sr = sf.read(str(w), dtype="float32", always_2d=True); x = torch.from_numpy(x.mean(1))
        if sr != 24000:
            x = julius.resample_frac(x, sr, 24000)
        T10 = len(np.load(CACHE / "justdance_visual" / f"{w.stem}.npz", allow_pickle=False)["x"])
        chunk, hop = 24000 * 30, 24000 * 30
        feats = {g: [] for g in GROUPS}
        with torch.no_grad():
            for s in range(0, len(x), hop):
                seg = x[s:s + chunk]
                if len(seg) < 24000:
                    break
                model(seg[None].to(dev))
                for g, layers in GROUPS.items():
                    h = torch.stack([caps[l][0] for l in layers]).mean(0)  # (F, 1024) at 75 Hz
                    F_ = h.shape[0]; n10 = int(round(len(seg) / 24000 * HZ))
                    idx = torch.linspace(0, F_, n10 + 1).long()
                    pooled = torch.stack([h[idx[k]:max(idx[k + 1], idx[k] + 1)].mean(0) for k in range(n10)])
                    feats[g].append(pooled.half().cpu().numpy())
        arr = {g: np.concatenate(v)[:T10] for g, v in feats.items()}
        np.savez(out, **arr, sr=24000)
        n += 1
        if n % 20 == 1:
            print(f"[{i+1}/{len(wavs)}] {w.stem} {arr['semantic'].shape} {time.time()-t0:.0f}s", flush=True)
    print(f"extracted {n} in {(time.time()-t0)/60:.1f} min", flush=True)


def audit():
    sys.path.insert(0, "/root/lightsem")
    from probe_event_target import audio_feats, average_precision, event_train, target
    from run_stage9_stems import folds_of, ridge_fit, ridge_apply, shift_target, SHIFTS
    t0 = time.time(); songs = []
    for p in sorted(EMB.glob("*.npz")):
        v = np.load(CACHE / "justdance_visual" / f"{p.stem}.npz", allow_pickle=False); e = np.load(p, allow_pickle=False)
        x, y = v["x"].astype(np.float32), v["y"].astype(np.float32); T = min(len(x), len(e["semantic"]))
        if T < 320:
            continue
        tr, _ = event_train(y[:T])
        songs.append({"key": p.stem, "xa": audio_feats(x[:T]), "x": x[:T], "t": target(tr).astype(np.float32),
                      "sem": e["semantic"][:T].astype(np.float32), "rhy": e["rhythm"][:T].astype(np.float32)})
    keys = [s["key"] for s in songs]; fold = folds_of(keys); print(f"{len(songs)} songs, {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    # ---- song-ID leak probe on the raw semantic embedding (song means of two halves)
    A = np.stack([s["sem"][: len(s["sem"]) // 2].mean(0) for s in songs]); B = np.stack([s["sem"][len(s["sem"]) // 2:].mean(0) for s in songs])
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6); Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-6)
    hit = float((np.argmax(An @ Bn.T, 1) == np.arange(len(songs))).mean())
    Ab = np.stack([s["x"][: len(s["x"]) // 2].mean(0) for s in songs]); Bb = np.stack([s["x"][len(s["x"]) // 2:].mean(0) for s in songs])
    Ab = (Ab - Ab.mean(0)) / (Ab.std(0) + 1e-6); Bb = (Bb - Bb.mean(0)) / (Bb.std(0) + 1e-6)
    hit_bus = float((np.argmax(Ab @ Bb.T, 1) == np.arange(len(songs))).mean())
    print(f"song-ID kNN probe: embedding halves match their own song {hit*100:.1f}% (bus scalars {hit_bus*100:.1f}%, chance {100/len(songs):.2f}%)", flush=True)
    # ---- per-fold PCA (train songs) and ridge arms
    res_ap = {}
    ARMS = ("bus", "bus+sem32", "bus+rhy32", "bus+sem32_resid", "bus+rhy32_resid", "bus+randproj32", "sem32_only")
    ap = {k: [] for k in ARMS}; ap_sh = {k: [] for k in ARMS}
    for f in range(5):
        tr = [i for i in range(len(songs)) if fold[i] != f]; te = [i for i in range(len(songs)) if fold[i] == f]
        def pca(field, resid):
            X = np.concatenate([songs[i][field] - (songs[i][field].mean(0) if resid else 0) for i in tr[:120]])
            mu = X.mean(0); U, S, Vt = np.linalg.svd(X[rng.permutation(len(X))[:20000]] - mu, full_matrices=False)
            return mu, Vt[:32].T
        P = {("sem", False): pca("sem", False), ("rhy", False): pca("rhy", False), ("sem", True): pca("sem", True), ("rhy", True): pca("rhy", True)}
        R = rng.standard_normal((songs[0]["xa"].shape[1], 32)) / np.sqrt(songs[0]["xa"].shape[1])
        def build(s, arm):
            if arm == "bus":
                return s["xa"]
            if arm == "bus+randproj32":
                return np.concatenate([s["xa"], s["xa"] @ R], 1)
            field = "sem" if "sem" in arm else "rhy"; resid = arm.endswith("_resid")
            mu, W = P[(field, resid)]
            e = s[field] - (s[field].mean(0) if resid else 0)
            z = (e - mu) @ W
            return z if arm == "sem32_only" else np.concatenate([s["xa"], z], 1)
        for arm in ARMS:
            p = ridge_fit(np.concatenate([build(songs[i], arm) for i in tr]), np.concatenate([songs[i]["t"] for i in tr]), 30.0)
            for i in te:
                s = songs[i]
                if s["t"].sum() < 3:
                    continue
                sc = ridge_apply(p, build(s, arm)); ap[arm].append(average_precision(sc, s["t"]))
                ap_sh[arm].append(np.mean([average_precision(sc, shift_target(s["t"], int(round(sh * HZ)))) for sh in SHIFTS]))
    base = np.array(ap["bus"])
    for arm in ARMS:
        v = np.array(ap[arm]); d = v - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        res_ap[arm] = {"ap": float(v.mean()), "delta_ap": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                       "ap_shift": float(np.mean(ap_sh[arm])), "n": int(len(v))}
        print(f"  {arm:18s} AP {v.mean():.4f}  dAP {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  shift {np.mean(ap_sh[arm]):.4f}", flush=True)
    best = max(("bus+sem32", "bus+rhy32"), key=lambda k: res_ap[k]["delta_ap"]); best_res = best + "_resid"
    rp = res_ap["bus+randproj32"]["delta_ap"]
    g_emb = {"passed": bool(res_ap[best_res]["delta_ap"] >= 0.02 and res_ap[best_res]["delta_ap"] - rp >= 0.02 and res_ap[best_res]["ci95"][0] > 0),
             "evidence": f"exp-016 ridge on JD visual events, {res_ap['bus']['n']} held-out songs, base AP {res_ap['bus']['ap']:.4f}: "
                         + "; ".join(f"{k} dAP {v['delta_ap']:+.4f} CI {[round(x, 4) for x in v['ci95']]}" for k, v in res_ap.items() if k != "bus")
                         + f". Best group {best}: raw {res_ap[best]['delta_ap']:+.4f}, after subtracting the song mean {res_ap[best_res]['delta_ap']:+.4f}, random projection {rp:+.4f} (gate: residual >= +0.02 over the bus and over the random projection)."}
    g_leak = {"passed": bool(res_ap[best_res]["delta_ap"] >= 0.5 * max(res_ap[best]["delta_ap"], 1e-6) or res_ap[best]["delta_ap"] < 0.02),
              "evidence": f"song-ID kNN probe: two halves of a song find each other {hit*100:.1f}% of the time on the raw embedding (bus scalars {hit_bus*100:.1f}%, chance {100/len(songs):.2f}%); "
                          f"gain kept after subtracting the song mean: {res_ap[best_res]['delta_ap']:+.4f} of {res_ap[best]['delta_ap']:+.4f}."}
    res = {"n_songs": len(songs), "arms": res_ap, "song_id_probe": {"embedding": hit, "bus": hit_bus, "chance": 1 / len(songs)},
           "gates": {"G9.7-emb": g_emb, "G9.7-leak": g_leak},
           "measurements": [{"diagnostic_id": "event-ap-lift-by-feature-family", "before": round(res_ap["bus"]["ap"] / 0.1, 4), "after": round(res_ap[best_res]["ap"] / 0.1, 4)}],
           "config": {"model": "MERT-v1-330M frozen, layers 10-13 (semantic) and 15-16 (rhythm) mean-pooled to 10 Hz, PCA 32 d fitted per fold on training songs",
                      "audit": "run_stage9_embedding.py: exp-016 ridge on JD visual scene-change events, 5-fold by song, random-projection width control, song-mean residual, song-ID kNN probe"},
           "notes": (f"H-M frozen MERT layer groups on JD visual events ({len(songs)} songs). {g_emb['evidence']} {g_leak['evidence']}"), "elapsed_s": time.time() - t0}
    # the diagnostic is a lift; AP / base rate with the corpus base rate
    rate = float(np.mean([s["t"].mean() for s in songs]))
    res["measurements"] = [{"diagnostic_id": "event-ap-lift-by-feature-family", "before": round(res_ap["bus"]["ap"] / rate, 4), "after": round(res_ap[best_res]["ap"] / rate, 4)}]
    OUT.mkdir(exist_ok=True); (OUT / "hm_embedding.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["gates"], indent=1))


if __name__ == "__main__":
    {"extract": extract, "audit": audit}[sys.argv[1]]()
