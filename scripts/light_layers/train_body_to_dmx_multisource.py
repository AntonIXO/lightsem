#!/usr/bin/env python
"""One-seed masked multi-source body-to-DMX diagnostic fit.

BeatSaver supplies body+audio+authored scene/event targets. RPMC-L2 supplies
audio+HSV scene targets only; body, angle and event losses are masked. This is
the smallest honest joint use of the two public target dialects.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch

from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig, masked_loss
from scripts.light_layers.train_body_to_dmx import load_pairs


def moments(arrays):
    x = np.concatenate(arrays); m, s = x.mean(0), x.std(0); s[s < 1e-5] = 1.; return m.astype(np.float32), s.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--body-root", type=Path, required=True); ap.add_argument("--ids", required=True); ap.add_argument("--rpmc-dir", type=Path, default=Path("/root/lightsem/data/cache/rpmc_train")); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--manifest", type=Path, default=Path("/mnt/lightsem-data/beatsaver/manifest.json")); ap.add_argument("--cache", type=Path, default=Path("/mnt/lightsem-data/cache")); ap.add_argument("--steps", type=int, default=1600); ap.add_argument("--window", type=int, default=128); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args(); out = args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); out.mkdir(parents=True)
    beat = load_pairs(args.body_root, args.ids.split(","), args.cache, args.manifest)
    if len(beat) < 4: raise RuntimeError("need at least four BeatSaver body pairs")
    beat.sort(key=lambda p: p["id"]); n_val = max(2, int(np.ceil(len(beat) * .2))); beat_train, beat_val = beat[:-n_val], beat[-n_val:]
    rpmc = sorted(args.rpmc_dir.glob("*.npz"));
    if len(rpmc) < 10: raise RuntimeError("need at least ten RPMC compact files")
    rpmc_train, rpmc_val = rpmc[:-max(10, len(rpmc)//5)], rpmc[-max(10, len(rpmc)//5):]
    bm, bs = moments([p["body"] for p in beat_train]); am, ass = moments([p["audio"] for p in beat_train]); rm, rs = moments([np.load(p, allow_pickle=False)["x"].astype(np.float32) for p in rpmc_train[:40]])
    cfg = BodyToDMXConfig(audio_dim=int(am.shape[0]), rpmc_audio_dim=int(rm.shape[0])); model = BodyToDMX(cfg).to(args.device); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    bt = [torch.tensor(v, device=args.device) for v in (bm, bs, am, ass, rm, rs)]
    history=[]
    for step in range(args.steps):
        use_rpmc = random.random() < .35
        if use_rpmc:
            p = rpmc_train[step % len(rpmc_train)]
            with np.load(p, allow_pickle=False) as z: audio, scene = z["x"].astype(np.float32), z["y"].astype(np.float32)
            n=len(audio); start=0 if n<=args.window else random.randrange(0,n-args.window); sl=slice(start,min(start+args.window,n)); audio=torch.tensor(audio[sl],device=args.device); scene=torch.tensor(scene[sl],device=args.device); body=torch.zeros(len(audio),cfg.body_dim,device=args.device)
            pred=model(body[None],((audio-bt[4])/bt[5])[None],audio_kind="rpmc",body_present=False); losses=masked_loss(pred,scene[None])
        else:
            p=beat_train[step%len(beat_train)]; n=len(p["body"]); start=0 if n<=args.window else random.randrange(0,n-args.window); sl=slice(start,min(start+args.window,n)); body=torch.tensor(p["body"][sl],device=args.device); audio=torch.tensor(p["audio"][sl],device=args.device); scene=torch.tensor(p["scene"][sl],device=args.device); event=torch.tensor(p["event"][sl],device=args.device); angle=torch.tensor(p["angle"][sl],device=args.device); pred=model(((body-bt[0])/bt[1])[None],((audio-bt[2])/bt[3])[None]); losses=masked_loss(pred,scene[None],event[None],angle=angle[None])
        opt.zero_grad(set_to_none=True); losses["total"].backward(); opt.step()
        if step % max(1,args.steps//20)==0 or step==args.steps-1: history.append({"step":step,"source":"rpmc" if use_rpmc else "beatsaver",**{k:float(v.detach().cpu()) for k,v in losses.items()}})
    model.eval()
    def eval_beat(rows):
        maes=[]
        with torch.no_grad():
            for p in rows:
                body=torch.tensor(p["body"],device=args.device); audio=torch.tensor(p["audio"],device=args.device); y=torch.tensor(p["scene"],device=args.device); q=model(((body-bt[0])/bt[1])[None],((audio-bt[2])/bt[3])[None])["scene"][0]; maes.append(float(torch.mean(torch.abs(q-y)).cpu()))
        return float(np.mean(maes))
    def eval_rpmc(rows):
        maes=[]
        with torch.no_grad():
            for p in rows:
                with np.load(p,allow_pickle=False) as z: audio,y=z["x"].astype(np.float32),z["y"].astype(np.float32)
                audio=torch.tensor(audio,device=args.device); y=torch.tensor(y,device=args.device); body=torch.zeros(len(audio),cfg.body_dim,device=args.device); q=model(body[None],((audio-bt[4])/bt[5])[None],audio_kind="rpmc",body_present=False)["scene"][0]; maes.append(float(torch.mean(torch.abs(q-y)).cpu()))
        return float(np.mean(maes))
    report={"schema_version":1,"status":"diagnostic_only","reason":"unseen-song source bodies lack active-baseline paired safety; RPMC target has no physical angle labels","config":cfg.__dict__,"params":model.n_params(),"seed":args.seed,"beatsaver_train_ids":[p["id"] for p in beat_train],"beatsaver_val_ids":[p["id"] for p in beat_val],"rpmc_train_files":len(rpmc_train),"rpmc_val_files":len(rpmc_val),"history":history,"beatsaver_val_scene_mae":eval_beat(beat_val),"rpmc_val_scene_mae":eval_rpmc(rpmc_val)}
    torch.save({"model":model.state_dict(),"config":cfg.__dict__,"stats":{"body_mean":bm.tolist(),"body_std":bs.tolist(),"audio_mean":am.tolist(),"audio_std":ass.tolist(),"rpmc_mean":rm.tolist(),"rpmc_std":rs.tolist()},"seed":args.seed},out/"model.pt"); (out/"REPORT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
