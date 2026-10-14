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
    p.add_argument("--fusion", choices=("concat","gated_residual","role_specialized","dual_cross_attention"), default="concat")
    p.add_argument("--stereo-scale", type=float, default=0.6)
    p.add_argument("--counterfactual-weight", type=float, default=1.0)
    p.add_argument("--counterfactual-margin", type=float, default=0.008)
    p.add_argument("--quiet-weight", type=float, default=0.0,
                   help="Weight observed-2D low-speed loss; zero preserves the historical pilot")
    p.add_argument("--quiet-active-weight", type=float, default=0.0,
                   help="Preserve active baseline speed while reducing quiet excess motion")
    p.add_argument("--quiet-margin", type=float, default=0.02,
                   help="Observed projected-speed excess margin for the quiet gate")
    p.add_argument("--raw-prior-weight", type=float, default=1.0,
                   help="Weight paired raw-motion prior against the frozen baseline")
    p.add_argument("--freeze-incumbent", action="store_true",
                   help="Train only native/stereo conditioning adapters; freeze the incumbent backbone")
    p.add_argument("--quiet-adapter-only", action="store_true",
                   help="Add a zero-initialized quiet adapter and train only its parameters")
    p.add_argument("--quiet-energy-gate", action="store_true",
                   help="Gate the quiet adapter on low normalized mixture RMS")
    p.add_argument("--quiet-energy-quantile", type=float, default=.30,
                   help="TRAIN quantile defining the low-energy quiet-adapter gate")
    p.add_argument("--quiet-jerk-weight", type=float, default=0.0,
                   help="Weight low-energy root-relative jerk excess; zero preserves prior objectives")
    p.add_argument("--quiet-jerk-target-ratio", type=float, default=.85,
                   help="Allowed fraction of paired baseline quiet jerk before the excess penalty")
    p.add_argument("--quiet-jerk-floor", type=float, default=25.0,
                   help="Absolute jerk floor in m/s^3 for the quiet excess budget")
    p.add_argument("--quiet-adapter-scale", type=float, default=.2,
                   help="Maximum gated quiet adapter contribution")
    args = p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Preserving existing end-to-end root: {args.out}")
    from lightsem.experiments.native_dynamics.evaluation import setup
    setup()
    torch.manual_seed(args.seed)
    records = PilotRecords(args.manifest, args.context_manifest)
    saved = torch.load(args.baseline, map_location="cpu", weights_only=False)
    if not 0.0 < args.stereo_scale <= 1.0:
        raise ValueError("--stereo-scale must be in (0,1]")
    if args.counterfactual_weight < 0.0 or args.counterfactual_margin < 0.0:
        raise ValueError("Counterfactual settings must be nonnegative")
    if args.quiet_weight < 0.0:
        raise ValueError("Quiet loss weight must be nonnegative")
    if args.quiet_active_weight < 0.0 or args.quiet_margin < 0.0:
        raise ValueError("Quiet active weight and margin must be nonnegative")
    if args.raw_prior_weight < 0.0:
        raise ValueError("Raw prior weight must be nonnegative")
    if args.quiet_jerk_weight < 0.0:
        raise ValueError("Quiet jerk weight must be nonnegative")
    if not 0.0 < args.quiet_jerk_target_ratio <= 1.0:
        raise ValueError("Quiet jerk target ratio must be in (0,1]")
    if args.quiet_jerk_floor < 0.0:
        raise ValueError("Quiet jerk floor must be nonnegative")
    if not 0.0 < args.quiet_adapter_scale <= 1.0:
        raise ValueError("Quiet adapter scale must be in (0,1]")
    if not 0.0 < args.quiet_energy_quantile < 1.0:
        raise ValueError("Quiet energy quantile must be in (0,1)")
    quiet_threshold=0.0
    if args.quiet_energy_gate:
        mean=float(saved["extra_normalizers"]["mean"][1040]); std=float(saved["extra_normalizers"]["std"][1040])
        values=np.concatenate([row["extra"][:,1040] for row in records.rows])
        quiet_threshold=float(np.quantile((values-mean)/max(std,1e-6), args.quiet_energy_quantile))
    cfg = Config(**{**saved["config"], "stereo_dim": records.stereo_dim,
                    "stereo_fusion": args.fusion, "freeze_early_blocks": 6,
                    "stereo_adapter_scale": args.stereo_scale,
                    "stereo_plan_scale": args.stereo_scale,
                    "quiet_adapter": bool(args.quiet_adapter_only),
                    "quiet_adapter_scale": args.quiet_adapter_scale,
                    "quiet_energy_gate": bool(args.quiet_energy_gate),
                    "quiet_energy_threshold": quiet_threshold})
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
        "flow.stereo_cross_fast.",
        "flow.stereo_cross_slow.",
        "stereo_plan.",
        "flow.quiet_projection.",
        "flow.quiet_adapters.",
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
    if args.quiet_adapter_only:
        allowed = ("flow.quiet_projection.", "flow.quiet_adapters.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
    elif args.freeze_incumbent:
        allowed = ("flow.native_adapters.", "flow.stereo_adapters.", "stereo_plan.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
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
    started = time.monotonic(); losses=[]; raw_losses=[]; quiet_losses=[]; quiet_jerk_losses=[]
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
        # Drop exactly one learned stream while keeping audio, pose crop, and
        # noise fixed. The counterfactual term below gives both streams a
        # measurable root-relative effect instead of letting shared native
        # weights absorb the intervention.
        cf_stereo=stereo.clone(); dropped_stream='spain' if step % 2 == 0 else 'obsrnn'
        cf_coarse_stereo=coarse_stereo.clone()
        if dropped_stream == 'spain':
            cf_stereo[..., :64] = 0.; cf_coarse_stereo[..., :64] = 0.
        else:
            cf_stereo[..., 64:] = 0.; cf_coarse_stereo[..., 64:] = 0.
        cf_condition,_=model.condition(fine,coarse,coarse_valid,values["indices"],cf_stereo,cf_coarse_stereo)
        counterfactual,_=model.rollout(cf_condition,noise,gradient_steps=cfg.gradient_steps,teacher=None)
        with torch.no_grad():
            if baseline_model.config.stereo_dim:
                base_condition,_=baseline_model.condition(fine,coarse,coarse_valid,values["indices"],stereo,coarse_stereo)
            else:
                base_condition,_=baseline_model.condition(fine,coarse,coarse_valid,values["indices"])
            baseline_native,_=baseline_model.rollout(base_condition,noise,gradient_steps=0,teacher=None)
        projected,_=project_native(generated,valid)
        pose_loss=reprojection_loss(projected,values["pose"],values["pose_mask"]&valid[...,None])
        dyn=dynamics_losses(projected,values["pose"],(values["pose_mask"]&valid[...,None]))
        quiet_loss=generated.sum()*0
        quiet_jerk_loss=generated.sum()*0
        active_preserve_loss=generated.sum()*0
        if args.quiet_weight or args.quiet_jerk_weight:
            # Observed 2D is a masked front-end, not metric 3D truth.  Use it
            # only to identify low-speed holds and compare raw generated
            # projected speed on the same valid joints.  No generated smoothing
            # or post-hoc retiming enters this term.
            pose_mask=values["pose_mask"] & valid[...,None]
            gvel=(projected[:,1:]-projected[:,:-1])*30.
            tvel=(values["pose"][:,1:]-values["pose"][:,:-1])*30.
            edge=pose_mask[:,1:] & pose_mask[:,:-1]
            gspeed=torch.sqrt(gvel.square().sum(-1)+1e-8)
            tspeed=torch.sqrt(tvel.square().sum(-1)+1e-8)
            observed_quiet=(tspeed < 0.10) & edge
            observed_active=(tspeed >= 0.10) & edge
            counts=observed_quiet.sum((-1,-2))
            enough=counts >= 4
            # Penalize only excess speed above the observed quiet proxy. A
            # symmetric regression here encourages a uniformly slowed body.
            quiet_excess=torch.relu(gspeed-tspeed-args.quiet_margin)
            quiet_frame_loss=torch.where(observed_quiet, quiet_excess.square(), torch.zeros_like(quiet_excess))
            quiet_loss=(quiet_frame_loss.sum((-1,-2)) / counts.clamp_min(1)).masked_select(enough).mean() if enough.any() else generated.sum()*0
            base_projected,_=project_native(baseline_native,valid)
            base_speed=torch.sqrt(((base_projected[:,1:]-base_projected[:,:-1])*30.).square().sum(-1)+1e-8)
            active_counts=observed_active.sum((-1,-2))
            active_enough=active_counts >= 4
            active_frame_loss=torch.where(observed_active, torch.nn.functional.smooth_l1_loss(
                torch.log1p(gspeed), torch.log1p(base_speed), reduction='none'), torch.zeros_like(gspeed))
            active_preserve_loss=(active_frame_loss.sum((-1,-2)) / active_counts.clamp_min(1)).masked_select(active_enough).mean() if active_enough.any() else generated.sum()*0
            if args.quiet_jerk_weight:
                # Penalize only high-frequency articulated jerk in low-energy
                # frames. Root translation is removed so this cannot be paid
                # for by suppressing musical travel. The detached incumbent
                # sets a local budget; active windows remain governed by the
                # separate observed-speed preservation term.
                from lightsem.experiments.motion_study.music_physical import motion_kinematics
                generated_positions = motion_kinematics(generated, valid)["positions"]
                baseline_positions = motion_kinematics(baseline_native, valid)["positions"].detach()
                generated_relative = generated_positions - generated_positions[:, :, :1]
                baseline_relative = baseline_positions - baseline_positions[:, :, :1]
                generated_jerk = generated_relative
                baseline_jerk = baseline_relative
                jerk_valid = valid
                quiet_energy = extra[..., 1040] <= quiet_threshold
                for _ in range(3):
                    generated_jerk = torch.diff(generated_jerk, dim=1) * 30.
                    baseline_jerk = torch.diff(baseline_jerk, dim=1) * 30.
                    jerk_valid = jerk_valid[:, :-1] & jerk_valid[:, 1:]
                    quiet_energy = quiet_energy[:, :-1] & quiet_energy[:, 1:]
                generated_norm = generated_jerk.square().sum(-1).add(1e-8).sqrt().mean(-1)
                baseline_norm = baseline_jerk.square().sum(-1).add(1e-8).sqrt().mean(-1)
                budget = args.quiet_jerk_target_ratio * baseline_norm.detach() + args.quiet_jerk_floor
                excess = torch.relu(generated_norm - budget) / budget.clamp_min(1.)
                quiet_jerk_mask = jerk_valid & quiet_energy
                quiet_jerk_loss = torch.where(quiet_jerk_mask, excess.square(), torch.zeros_like(excess)).sum() / quiet_jerk_mask.sum().clamp_min(1)
        raw_prior=raw_native_losses(generated,baseline_native,valid)
        raw_total=sum(RAW_WEIGHTS[name]*value for name,value in raw_prior.items())
        cf_projected,_=project_native(counterfactual,valid)
        # Fixed projected upper body: head, shoulders, elbows, wrists. The
        # target is a small torso-unit response plus a term proportional to the
        # removed stream's actual variation, so silent contexts are not forced
        # into arbitrary motion.
        response=torch.sqrt((projected[:,:,:7]-cf_projected[:,:,:7]).square().sum(-1)+1e-8).mean((1,2))
        stream_delta=(stereo-cf_stereo).abs().mean((1,2)).detach()
        response_target=args.counterfactual_margin + .002*stream_delta.clamp(max=4.)
        counterfactual_loss=torch.relu(response_target-response).mean()
        # The raw prior is deliberately strong: end-to-end external conditioning
        # must learn the paired motion evidence without creating a new jitter mode.
        loss=pose_loss+.1*sum(dyn.values())+args.quiet_weight*quiet_loss+args.quiet_jerk_weight*quiet_jerk_loss+args.quiet_active_weight*active_preserve_loss+args.raw_prior_weight*raw_total+.0001*sum(parameter.square().mean() for parameter in trainable)
        loss=loss+args.counterfactual_weight*counterfactual_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        losses.append(float(loss.detach().cpu())); raw_losses.append(float(raw_total.detach().cpu()))
        quiet_losses.append(float(quiet_loss.detach().cpu()))
        quiet_jerk_losses.append(float(quiet_jerk_loss.detach().cpu()))
        if (step+1) % 100 == 0:
            print(f'update {step+1}/{args.steps} loss={losses[-1]:.6f} raw_prior={raw_losses[-1]:.6f} quiet={quiet_losses[-1]:.6f} quiet_jerk={quiet_jerk_losses[-1]:.6f} active={float(active_preserve_loss.detach().cpu()):.6f}', flush=True)
    args.out.mkdir(parents=True)
    checkpoint=args.out/"final.pt"
    changed = [name for name, value in model.state_dict().items()
               if name not in saved['model'] or not torch.equal(value.cpu(), saved['model'][name].cpu())]
    forbidden = [f'flow.blocks.{i}.' for i in range(6)] + ['flow.plan_projection.']
    if any(any(name.startswith(prefix) for prefix in forbidden) for name in changed):
        raise ValueError('Frozen weights changed')
    required_prefixes = ('flow.quiet_adapters.',) if args.quiet_adapter_only else (('flow.native_adapters.', 'flow.stereo_adapters.', 'stereo_plan.') if args.freeze_incumbent else ('planner.', 'flow.output.', 'flow.blocks.6.'))
    for prefix in required_prefixes:
        if not any(name.startswith(prefix) for name in changed):
            raise ValueError(f'Native weights did not learn: {prefix}')
    torch.save({"model":model.state_dict(),"config":asdict(cfg),"base_config":saved["base_config"],"normalizers":saved["normalizers"],"audio_normalizers":saved["audio_normalizers"],"extra_normalizers":saved["extra_normalizers"],"stereo_mean":records.stereo_mean,"stereo_std":records.stereo_std,"baseline_checkpoint":str(args.baseline),"baseline_sha256":sha(args.baseline),"context_manifest":str(args.context_manifest),"context_manifest_sha256":sha(args.context_manifest),"steps":args.steps,"seed":args.seed,"lr":args.lr,"fusion":args.fusion,"stereo_scale":args.stereo_scale,"counterfactual_weight":args.counterfactual_weight,"counterfactual_margin":args.counterfactual_margin,"context_schema":records.schema,"training_manifest_sha256":sha(args.manifest),"changed_native_tensors":changed,"training_mode":"native_end_to_end_with_counterfactual_stereo_response","frozen_first_flow_blocks":cfg.freeze_early_blocks},checkpoint)
    report={"schema_version":1,"status":"complete","seed":args.seed,"steps":args.steps,"batch_size":args.batch_size,"lr":args.lr,"arm":json.loads(args.context_manifest.read_text()).get("method"),"baseline":str(args.baseline),"baseline_sha256":sha(args.baseline),"context_manifest":str(args.context_manifest),"context_manifest_sha256":sha(args.context_manifest),"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"elapsed_s":time.monotonic()-started,"loss_first":losses[0],"loss_last":losses[-1],"raw_prior_first":raw_losses[0],"raw_prior_last":raw_losses[-1],"quiet_weight":args.quiet_weight,"quiet_active_weight":args.quiet_active_weight,"quiet_margin":args.quiet_margin,"quiet_jerk_weight":args.quiet_jerk_weight,"quiet_jerk_target_ratio":args.quiet_jerk_target_ratio,"quiet_jerk_floor":args.quiet_jerk_floor,"quiet_adapter_scale":args.quiet_adapter_scale,"quiet_jerk_loss_first":quiet_jerk_losses[0],"quiet_jerk_loss_last":quiet_jerk_losses[-1],"raw_prior_weight":args.raw_prior_weight,"freeze_incumbent":args.freeze_incumbent,"quiet_adapter_only":args.quiet_adapter_only,"quiet_energy_gate":args.quiet_energy_gate,"quiet_energy_quantile":args.quiet_energy_quantile,"quiet_energy_threshold":quiet_threshold,"quiet_loss_first":quiet_losses[0],"quiet_loss_last":quiet_losses[-1],"finite":True,"fusion":args.fusion,"stereo_scale":args.stereo_scale,"counterfactual_weight":args.counterfactual_weight,"counterfactual_margin":args.counterfactual_margin,"context_schema":records.schema,"changed_native_tensors":changed,"torch_version":torch.__version__,"training_mode":"native end-to-end with zero-initialized gated quiet adapter, counterfactual stereo response, excess-only observed-2D quiet loss, low-energy root-relative jerk excess, and active baseline preservation","diagnostic_only":True,"scope":"paired native TRAIN only; raw safety gate required before mapper/render; no baseline promotion"}
    write(args.out/"REPORT.json",report); print(json.dumps(report,sort_keys=True)); return 0


if __name__=='__main__': raise SystemExit(main())
