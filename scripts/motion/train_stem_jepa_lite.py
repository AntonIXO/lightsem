#!/usr/bin/env python3
"""Train a compact held-out-source predictor on MUSDB stem targets."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.stereo_v1 import StereoAudioV1, resample_stream, sha256
from lightsem.experiments.audio_stereo.stem_jepa import StemJepaConfig, StemJepaLite, stem_jepa_loss


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing Stem-JEPA artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")


class Records:
    def __init__(self, manifest_path: Path, split: str):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("test_access", True):
            raise ValueError("Stem-JEPA manifest must prove test_access=false")
        rows = [row for row in manifest["entries"] if row.get("split") == split]
        if not rows:
            raise ValueError(f"No Stem-JEPA rows for split {split!r}")
        self.rows = []
        for row in rows:
            descriptor = StereoAudioV1.load(row["mixture_feature_path"])
            with np.load(row["target_path"], allow_pickle=False) as target:
                spatial = target["source_spatial_10"].reshape(len(target["time_10"]), 4, 5)
                event = target["source_event_40"].reshape(len(target["time_40"]), 4, 4)
                spatial40 = np.stack([resample_stream(spatial[:, i], target["time_10"], target["time_40"]) for i in range(4)], axis=1)
                # [log energy, active, onset L/R, ILD/width/pan/coherence/log energy]
                source = np.concatenate((event, spatial40), axis=-1).astype(np.float32)
            self.rows.append({"id": row["id"], "features": descriptor.features_40.astype(np.float32), "source": source})
        continuous = np.concatenate([np.concatenate((r["source"][:, :, :1], r["source"][:, :, 4:]), axis=-1).reshape(-1, 6) for r in self.rows], axis=0)
        self.cont_mean = continuous.mean(0).astype(np.float32); self.cont_std = np.maximum(continuous.std(0), 1e-3).astype(np.float32)
        onset = np.concatenate([r["source"][:, :, 2:4].reshape(-1, 2) for r in self.rows], axis=0)
        self.onset_threshold = np.percentile(onset, 85, axis=0).astype(np.float32)

    def batch(self, rng: np.random.Generator, batch_size: int, frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, queries, targets = [], [], []
        for _ in range(batch_size):
            row = self.rows[int(rng.integers(len(self.rows)))]
            source_id = int(rng.integers(4))
            start = 0 if len(row["features"]) <= frames else int(rng.integers(len(row["features"]) - frames + 1))
            stop = start + frames
            x = self._pad(row["features"][start:stop], frames)
            target = self._pad(row["source"][start:stop, source_id], frames)
            target = target.copy(); target[:, 2:4] = (target[:, 2:4] > self.onset_threshold).astype(np.float32)
            continuous = np.concatenate((target[:, :1], target[:, 4:]), axis=1)
            continuous = (continuous - self.cont_mean) / self.cont_std
            target[:, :1] = continuous[:, :1]; target[:, 4:] = continuous[:, 1:]
            query = np.zeros((frames, 4), np.float32); query[:, source_id] = 1.
            features.append(x); queries.append(query); targets.append(target)
        return np.asarray(features, np.float32), np.asarray(queries, np.float32), np.asarray(targets, np.float32)

    @staticmethod
    def _pad(value: np.ndarray, length: int) -> np.ndarray:
        if len(value) >= length: return value[:length]
        return np.pad(value, ((0, length - len(value)), (0, 0)), mode="edge")


def train(manifest_path: Path, out: Path, *, split: str, steps: int, batch_size: int, frames: int,
          seed: int, device: str) -> dict:
    if out.exists(): raise FileExistsError(f"Preserving existing Stem-JEPA root: {out}")
    records = Records(manifest_path, split); config = StemJepaConfig()
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = StemJepaLite(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    started = time.monotonic(); losses=[]; model.train()
    for step in range(steps):
        x, query, target = records.batch(rng, batch_size, frames)
        x = torch.from_numpy(x).to(device); query=torch.from_numpy(query).to(device); target=torch.from_numpy(target).to(device)
        valid=torch.ones(x.shape[:2],dtype=torch.bool,device=device); optimizer.zero_grad(set_to_none=True)
        if str(device).startswith("cuda"):
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): outputs=model(x,query,valid); loss,_=stem_jepa_loss(outputs,target,valid)
        else: outputs=model(x,query,valid); loss,_=stem_jepa_loss(outputs,target,valid)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite Stem-JEPA loss at {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
    out.mkdir(parents=True); checkpoint=out/'final.pt'
    torch.save({'model':model.state_dict(),'config':model.config_dict(),'stats':{'cont_mean':records.cont_mean,'cont_std':records.cont_std,'onset_threshold':records.onset_threshold},'steps':steps,'seed':seed,'split':split,'manifest':str(manifest_path),'manifest_sha256':sha256(manifest_path)},checkpoint)
    report={'schema_version':1,'status':'complete','arm':config.arm,'seed':seed,'steps':steps,'batch_size':batch_size,'frames':frames,'device':device,'elapsed_s':time.monotonic()-started,'loss_first':losses[0],'loss_last':losses[-1],'loss_min':min(losses),'finite':True,'test_access':False,'manifest':str(manifest_path),'manifest_sha256':sha256(manifest_path),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'model_config':model.config_dict(),'interpretation':'held-out-source prediction probe; no native dance qualification'}
    write_json(out/'REPORT.json',report); return report


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--split',choices=('train','dev'),default='train'); p.add_argument('--steps',type=int,default=6000); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--frames',type=int,default=320); p.add_argument('--seed',type=int,default=0); p.add_argument('--device',default='cuda'); a=p.parse_args(argv)
    r=train(a.manifest,a.out,split=a.split,steps=a.steps,batch_size=a.batch_size,frames=a.frames,seed=a.seed,device=a.device); print(json.dumps({k:r[k] for k in ('status','arm','steps','loss_first','loss_last')})); return 0

if __name__=='__main__': raise SystemExit(main())
