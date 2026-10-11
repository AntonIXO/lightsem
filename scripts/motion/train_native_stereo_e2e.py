#!/usr/bin/env python3
"""End-to-end native C training on an external spatial context.

This branch trains the native planner, late flow blocks, audio readout, and
zero-initialized stereo modules against paired native TRAIN poses. The first
six flow blocks and the legacy plan projection remain frozen by the native
safety contract. No translator is used: the selected external method context
lands directly in the trainable native network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.dynamic_dance.audio import model_features
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.kinematics import dynamics_losses, project_native, reprojection_loss
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from lightsem.experiments.native_repair.losses import raw_native_losses, WEIGHTS as RAW_WEIGHTS


ROOT = Path("/mnt/lightsem-data/experiments/lightsem/native_stereo_e2e_20261011_v1")
BASELINE = Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/learning/runs/joint/seed_0/final.pt")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Preserving existing stereo adapter receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha(path)}) + "\n")


def pool(value: np.ndarray, bins: int) -> np.ndarray:
    from scripts.motion.generate_native_stereo_demos import pool_native
    if bins != (len(value) + 14) // 15:
        raise ValueError("Coarse bin count mismatch")
    return pool_native(value, len(value))


class PilotRecords:
    def __init__(self, manifest_path: Path, context_manifest: Path, *, limit_frames: int = 480):
        manifest = json.loads(manifest_path.read_text())
        contexts = json.loads(context_manifest.read_text())
        if manifest.get('test_access') is not False or contexts.get('test_access') is not False:
            raise ValueError('TRAIN-only manifests required')
        if any(r.get('partition') != 'train' for r in manifest['entries']) or any(r.get('split') != 'train' for r in contexts['rows']):
            raise ValueError('Non-TRAIN data reached native training')
        self.stereo_dim = int(contexts.get('feature_dim', 64))
        self.schema = contexts.get('schema', 'legacy_external64')
        context_rows = {row["id"]: row for row in contexts["rows"]}
        if len(context_rows) != len(contexts['rows']) or set(context_rows) != {r['key'] for r in manifest['entries']}:
            raise ValueError('Native/context id sets differ')
        baseline_features = json.loads((Path("/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/features/COMPLETE.json")).read_text())
        feature_rows = {row["key"]: row for row in baseline_features["entries"]}
        self.rows = []
        for row in manifest["entries"]:
            feature = feature_rows.get(row["key"]) or feature_rows.get(row["feature_key"])
            context = context_rows.get(row["key"])
            if feature is None or context is None:
                raise ValueError(f"Missing paired audio/context for {row['key']}")
            if sha(Path(row['pose_cache'])) != row['pose_cache_sha256'] or sha(Path(context['context_path'])) != context['context_sha256']:
                raise ValueError('Changed pose/context cache')
            if context.get('audio_sha256') and context['audio_sha256'] != row['wav_sha256']:
                raise ValueError('Pose/context source audio differs')
            with np.load(row["pose_cache"], allow_pickle=False) as z:
                pose = z["pose"].astype(np.float32); mask = z["mask"].astype(bool); starts = z["starts"].astype(np.int32); target = z["time"].astype(np.float64)
            timeline = MusicTimeline.load(feature["timeline_path"])
            extra, _ = model_features(timeline, target)
            with np.load(feature["base92_path"], allow_pickle=False) as z:
                base = z["x30"].astype(np.float32)
                if not np.array_equal(z["time30"], target):
                    raise ValueError("Base92 and RTMPose clocks differ")
            with np.load(context["context_path"], allow_pickle=False) as z:
                stereo = np.stack([np.interp(target, z["time30"], z["fused30"][:, i]) for i in range(self.stereo_dim)], 1).astype(np.float32)
            if not np.isfinite(stereo).all():
                raise ValueError('Nonfinite training context')
            valid = np.isfinite(base).all(1) & np.isfinite(extra).all(1) & np.isfinite(stereo).all(1)
            self.rows.append({"key": row["key"], "group": row["group"], "base": base, "extra": extra.astype(np.float32),
                              "stereo": stereo, "pose": pose, "mask": mask, "valid": valid, "starts": starts, "length": len(target)})
        if not self.rows:
            raise ValueError("No stereo pilot records")
        values = np.concatenate([row["stereo"] for row in self.rows], axis=0)
        self.stereo_mean = values.mean(0).astype(np.float32)
        self.stereo_std = np.maximum(values.std(0), 1e-3).astype(np.float32)
        self.frames = limit_frames
        self.groups = {}
        for i, row in enumerate(self.rows): self.groups.setdefault(row["group"], []).append(i)
        self.groups = list(self.groups.values())

    def batch(self, rng: np.random.Generator, batch_size: int) -> dict:
        chosen = []
        for _ in range(batch_size):
            group = self.groups[int(rng.integers(len(self.groups)))]
            row = self.rows[group[int(rng.integers(len(group)))] ]
            starts = row["starts"] if len(row["starts"]) else np.array([0])
            start = int(starts[int(rng.integers(len(starts)))])
            chosen.append((row, start))
        b, t = len(chosen), self.frames
        result = {"base": np.zeros((b, t, 92), np.float32), "extra": np.zeros((b, t, 1061), np.float32),
                  "stereo": np.zeros((b, t, self.stereo_dim), np.float32), "valid": np.zeros((b, t), bool),
                  "indices": np.zeros((b, t), np.int64), "pose": np.zeros((b, t, 13, 2), np.float32),
                  "pose_mask": np.zeros((b, t, 13), bool), "coarse": np.zeros((b, 1024, 1061), np.float32),
                  "coarse_stereo": np.zeros((b, 1024, self.stereo_dim), np.float32), "coarse_valid": np.zeros((b, 1024), bool)}
        for i, (row, start) in enumerate(chosen):
            n = min(t, row["length"] - start); end = start + n; bins = (row["length"] + 14) // 15
            result["base"][i, :n] = row["base"][start:end]
            result["extra"][i, :n] = row["extra"][start:end]
            result["stereo"][i, :n] = row["stereo"][start:end]
            result["valid"][i, :n] = row["valid"][start:end]
            result["indices"][i] = np.minimum(start + np.arange(t), row["length"] - 1)
            result["pose"][i, :n] = row["pose"][start:end]; result["pose_mask"][i, :n] = row["mask"][start:end]
            result["coarse"][i, :bins] = pool(row["extra"], bins)
            result["coarse_stereo"][i, :bins] = pool(row["stereo"], bins)
            result["coarse_valid"][i, :bins] = True
        return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--context-manifest", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fusion", choices=("concat","gated_residual","role_specialized"), default="concat")
    args = p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Preserving existing end-to-end root: {args.out}")
    from lightsem.experiments.native_dynamics.evaluation import setup
    setup()
    torch.manual_seed(args.seed)
    records = PilotRecords(args.manifest, args.context_manifest)
    saved = torch.load(args.baseline, map_location="cpu", weights_only=False)
    cfg = Config(**{**saved["config"], "stereo_dim": records.stereo_dim, "stereo_fusion": args.fusion, "freeze_early_blocks": 6})
    model = NativeDynamicsModel(MusicConfig(**saved["base_config"]), saved["normalizers"], cfg)
    result = model.load_state_dict(saved["model"], strict=False)
    # Fusion candidates add their own zero-initialized modules on top of the
    # frozen baseline.  They are intentionally absent from the baseline
    # checkpoint and must be admitted here; all other missing/unexpected keys
    # remain a hard architecture mismatch.
    allowed_missing = (
        "flow.stereo_projection.",
        "flow.stereo_adapters.",
        "flow.stereo_gate.",
        "stereo_plan.",
    )
    if result.unexpected_keys or any(not key.startswith(allowed_missing) for key in result.missing_keys):
        raise ValueError(f"Baseline state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    # End-to-end means every native parameter outside the mandated early-flow
    # freeze participates in the loss. Re-apply the freeze after enabling all
    # parameters so the contract cannot be weakened by a config change.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for block in model.flow.blocks[:cfg.freeze_early_blocks]:
        for parameter in block.parameters():
            parameter.requires_grad_(False)
    for parameter in model.flow.plan_projection.parameters():
        parameter.requires_grad_(False)
    model.to(args.device).train()
    baseline_model = NativeDynamicsModel(MusicConfig(**saved["base_config"]), saved["normalizers"], Config(**saved["config"])).to(args.device).eval()
    baseline_load = baseline_model.load_state_dict(saved["model"], strict=True)
    if baseline_load.missing_keys or baseline_load.unexpected_keys:
        raise ValueError(f"Baseline model reload mismatch: {baseline_load}")
    baseline_model.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("End-to-end branch has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    started = time.monotonic(); losses=[]; raw_losses=[]
    for step in range(args.steps):
        batch = records.batch(rng, args.batch_size)
        values = {key: torch.from_numpy(value).to(args.device) for key, value in batch.items()}
        base_stats=saved["audio_normalizers"]; extra_stats=saved["extra_normalizers"]
        base=(values.pop("base")-torch.tensor(base_stats["mean"],device=args.device))/torch.tensor(base_stats["std"],device=args.device)
        extra=(values.pop("extra")-torch.tensor(extra_stats["mean"],device=args.device))/torch.tensor(extra_stats["std"],device=args.device)
        stereo=(values.pop("stereo")-torch.tensor(records.stereo_mean,device=args.device))/torch.tensor(records.stereo_std,device=args.device)
        coarse=(values.pop("coarse")-torch.tensor(extra_stats["mean"],device=args.device))/torch.tensor(extra_stats["std"],device=args.device)
        coarse_stereo=(values.pop("coarse_stereo")-torch.tensor(records.stereo_mean,device=args.device))/torch.tensor(records.stereo_std,device=args.device)
        fine=torch.cat((base,extra),-1)
        noise=model.flow.noise(args.batch_size,480,generator=torch.Generator(device=args.device).manual_seed(args.seed+step))
        valid=values["valid"]; coarse_valid=values["coarse_valid"]
        condition,_=model.condition(fine,coarse,coarse_valid,values["indices"],stereo,coarse_stereo)
        generated,_=model.rollout(condition,noise,gradient_steps=cfg.gradient_steps,teacher=None)
        with torch.no_grad():
            base_condition,_=baseline_model.condition(fine,coarse,coarse_valid,values["indices"])
            baseline_native,_=baseline_model.rollout(base_condition,noise,gradient_steps=0,teacher=None)
        projected,_=project_native(generated,valid)
        pose_loss=reprojection_loss(projected,values["pose"],values["pose_mask"]&valid[...,None])
        dyn=dynamics_losses(projected,values["pose"],(values["pose_mask"]&valid[...,None]))
        raw_prior=raw_native_losses(generated,baseline_native,valid)
        raw_total=sum(RAW_WEIGHTS[name]*value for name,value in raw_prior.items())
        # The raw prior is deliberately strong: end-to-end external conditioning
        # must learn the paired motion evidence without creating a new jitter mode.
        loss=pose_loss+.1*sum(dyn.values())+1.0*raw_total+.0001*sum(parameter.square().mean() for parameter in trainable)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu())); raw_losses.append(float(raw_total.detach().cpu()))
        if (step+1) % 100 == 0:
            print(f'update {step+1}/{args.steps} loss={losses[-1]:.6f} raw_prior={raw_losses[-1]:.6f}', flush=True)
    args.out.mkdir(parents=True)
    checkpoint=args.out/"final.pt"
    changed = [name for name, value in model.state_dict().items() if name in saved['model'] and not torch.equal(value.cpu(), saved['model'][name].cpu())]
    forbidden = [f'flow.blocks.{i}.' for i in range(6)] + ['flow.plan_projection.']
    if any(any(name.startswith(prefix) for prefix in forbidden) for name in changed):
        raise ValueError('Frozen weights changed')
    for prefix in ('planner.', 'flow.output.', 'flow.blocks.6.'):
        if not any(name.startswith(prefix) for name in changed):
            raise ValueError(f'Native weights did not learn: {prefix}')
    torch.save({"model":model.state_dict(),"config":asdict(cfg),"base_config":saved["base_config"],"normalizers":saved["normalizers"],"audio_normalizers":saved["audio_normalizers"],"extra_normalizers":saved["extra_normalizers"],"stereo_mean":records.stereo_mean,"stereo_std":records.stereo_std,"baseline_checkpoint":str(args.baseline),"baseline_sha256":sha(args.baseline),"context_manifest":str(args.context_manifest),"context_manifest_sha256":sha(args.context_manifest),"steps":args.steps,"seed":args.seed,"lr":args.lr,"fusion":args.fusion,"context_schema":records.schema,"training_manifest_sha256":sha(args.manifest),"changed_native_tensors":changed,"training_mode":"native_c_end_to_end_external_spatial","frozen_first_flow_blocks":cfg.freeze_early_blocks},checkpoint)
    report={"schema_version":1,"status":"complete","seed":args.seed,"steps":args.steps,"batch_size":args.batch_size,"lr":args.lr,"arm":json.loads(args.context_manifest.read_text()).get("method"),"baseline":str(args.baseline),"baseline_sha256":sha(args.baseline),"context_manifest":str(args.context_manifest),"context_manifest_sha256":sha(args.context_manifest),"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"elapsed_s":time.monotonic()-started,"loss_first":losses[0],"loss_last":losses[-1],"raw_prior_first":raw_losses[0],"raw_prior_last":raw_losses[-1],"finite":True,"fusion":args.fusion,"context_schema":records.schema,"changed_native_tensors":changed,"torch_version":torch.__version__,"training_mode":"native C end-to-end with external spatial context; first six flow blocks and legacy plan projection frozen","diagnostic_only":True,"scope":"paired native TRAIN only; raw safety gate required before mapper/render; no baseline promotion"}
    write(args.out/"REPORT.json",report); print(json.dumps(report,sort_keys=True)); return 0


if __name__=='__main__': raise SystemExit(main())
