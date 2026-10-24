#!/usr/bin/env python3
"""Diagnostic-only stage-space aim arm on generated body-to-DMX supervision."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lightsem.dance.pipeline import accelerator_lock
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
from lightsem.experiments.audio_light_foundation.model import AudioToLightConfig
from lightsem.experiments.foundation_audio_light.direct_model import AudioOnlyDirectLightModel, DirectLightConfig
from lightsem.experiments.foundation_audio_light.safety_projection import SafetyConfig
from lightsem.experiments.foundation_audio_light.peft import trainable_parameter_report
from scripts.experiments.run_foundation_v3_audio_only import CONTRACT, load_row, loss_parts


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fd_penalty(aim: torch.Tensor) -> torch.Tensor:
    v = torch.diff(aim, dim=1) * 10.0
    a = torch.diff(aim, n=2, dim=1) * 100.0
    j = torch.diff(aim, n=3, dim=1) * 1000.0
    return sum(F.relu(torch.linalg.norm(x, dim=-1) - limit).square().mean()
               for x, limit in ((v, .18), (a, .5), (j, 1.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", type=Path, default=Path("/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013/demo_3b4ab_v1/demo.npz"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = json.loads(CONTRACT.read_text())
    row = next(load_row(r) for r in contract["rows"] if r["id"] == "3b4ab")
    with np.load(args.demo, allow_pickle=False) as z:
        target = np.asarray(z["angles"], np.float32)
    # Generated diagnostic DMX angles are normalized [0,1]; convert to the
    # model's centered [-1,1] aim convention and repeat 6 heads to 8 slots.
    target = 2.0 * target - 1.0
    target = target[:, np.arange(8) % target.shape[1]]
    n = min(len(row["audio"]), len(target))
    audio = row["audio"][:n]
    rgb = row["rgb"][:n]
    brightness = row["brightness"][:n]
    event = row["event"][:n]
    split = max(4, int(n * .8))
    fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)
    cfg = AudioToLightConfig(audio_dim=1116, width=96, heads=4, local_window=32,
                             latent_slots=8, adapter_kind="hierarchical",
                             adapter_bottleneck=32, adapter_heads=4, adapter_local_window=32)
    model = AudioOnlyDirectLightModel(DirectLightConfig(
        foundation=cfg,
        safety=SafetyConfig(dt=.1, max_angle=1.0, max_velocity=.18, max_acceleration=.5, max_jerk=1.5),
        dynamics_mode="smooth_jerk", dynamics_position_gain=.105,
        dynamics_velocity_gain=.25, dynamics_acceleration_gain=.25,
        dynamics_radial_bounds=True,
    )).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    windows = list(range(0, max(1, split - 128 + 1), 128))
    started = time.time(); history=[]
    with accelerator_lock():
        model.train()
        for step in range(args.steps):
            start = windows[step % len(windows)]; end = start + 128
            x = torch.from_numpy(audio[start:end][None]).float().to(device)
            t = {"audio": x,
                 "rgb": torch.from_numpy(rgb[start:end][None]).float().to(device),
                 "brightness": torch.from_numpy(brightness[start:end][None]).float().to(device),
                 "event": torch.from_numpy(event[start:end][None]).float().to(device),
                 "valid": torch.ones((1, end-start), dtype=torch.bool, device=device)}
            y = torch.from_numpy(target[start:end][None]).float().to(device)
            ts = torch.arange(end-start, device=device, dtype=torch.float32)[None] / 10.0
            out = model(t["audio"], fixtures[None], timestamps=ts, valid=t["valid"], duration_seconds=(end-start)/10.0)
            base = loss_parts(out, t)
            motion = F.smooth_l1_loss(out["aim"], y)
            raw = fd_penalty(out["aim"])
            total = base["total"] + 5.0 * motion + 20.0 * raw
            opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if step == 0 or (step+1) % max(1,args.steps//4)==0:
                history.append({"step":step+1,"total":float(total.detach().cpu()),"motion":float(motion.detach().cpu()),"raw_fd":float(raw.detach().cpu())})
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(audio[split:][None]).float().to(device)
            ts = torch.arange(len(audio)-split, device=device, dtype=torch.float32)[None]/10.0
            out = model(x, fixtures[None], timestamps=ts, valid=torch.ones((1,len(audio)-split),dtype=torch.bool,device=device), duration_seconds=(len(audio)-split)/10.0)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/"PREDICTIONS_auxiliary_holdout.npz", time=np.arange(len(audio)-split)/10.0,
                        aim=out["aim"].detach().cpu().numpy(), rgb=out["rgb"].detach().cpu().numpy(),
                        brightness=out["brightness"].detach().cpu().numpy(), event=out["event"].detach().cpu().numpy(),
                        audio_latent=out["audio_latent"].detach().cpu().numpy(), anchor=out["anchor"].detach().cpu().numpy())
    ck=args.out/"model.pt"; torch.save({"state_dict":model.state_dict(),"config":model.config_dict(),"seed":args.seed},ck)
    receipt={"schema":"lightsem.foundation_stage_space_auxiliary.v1","protocol_id":"foundation-stage-space-auxiliary-20261021-v1","status":"diagnostic_only","source":"generated body-to-DMX demo, not authored light ground truth","source_demo":str(args.demo.resolve()),"source_demo_sha256":sha(args.demo),"checkpoint":str(ck),"checkpoint_sha256":sha(ck),"trainable_parameters":trainable_parameter_report(model),"history":history,"train_frames":split,"holdout_frames":n-split,"runtime_seconds":time.time()-started,"promotion":"forbidden; does not qualify canonical 300-row or own-song audio-to-light behavior"}
    (args.out/"TRAINING.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2,sort_keys=True))

if __name__ == "__main__":
    main()
