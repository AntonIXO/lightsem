#!/usr/bin/env python3
"""Bridge the open appearance prior into the frozen legacy audio-to-light model.

This is an appearance-transfer experiment, not physical-light supervision:
RPMC-L2 and Roskilde only provide the frozen two-channel SpeedNet prior.  The
legacy MuQ+Base92 model remains the only source of canonical appearance
predictions and its bounded dynamics projection remains in the path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lightsem.experiments.audio_light_foundation.model import AudioToLightConfig
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
from lightsem.experiments.foundation_audio_light.direct_model import (
    AudioOnlyDirectLightModel,
    DirectLightConfig,
)
from lightsem.experiments.foundation_audio_light.safety_projection import (
    SafetyConfig,
    bounded_dynamics_projection,
    smooth_bounded_dynamics_projection,
)
from scripts.experiments.evaluate_foundation_v3_own_songs import align_features, load_model
from scripts.experiments.run_cyclic_motion_prior import SpeedNet, own_features
from scripts.experiments.run_foundation_v3_audio_only import load_row, loss_parts
from scripts.experiments.run_foundation_v3_audio_only import audit_output as audit_training_output


DEFAULT_CONTRACT = Path(
    "/mnt/lightsem-data/experiments/lightsem/"
    "audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json"
)
DEFAULT_BASE_CHECKPOINT = Path(
    "/mnt/lightsem-data/experiments/lightsem/"
    "foundation_v3_jerk_safe_20261020_v1/model.pt"
)
DEFAULT_PRIOR_CHECKPOINT = Path(
    "/mnt/lightsem-data/experiments/lightsem/"
    "foundation_cyclic_motion_prior_open_combo_20261021_v2/speed_model.pt"
)
DEFAULT_MUQ = Path(
    "/mnt/lightsem-data/experiments/lightsem/"
    "audio_pipeline_integration_20261005/features/muq"
)
DEFAULT_BASE92 = Path(
    "/mnt/lightsem-data/experiments/lightsem/"
    "native_bf16_v7b/demo_features/features/base92"
)
DEFAULT_SHONGLE_AUDIO = Path(
    "/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8/"
    "audio_reference/shpongle.m4a"
)
DEFAULT_CURE_AUDIO = Path(
    "/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8/"
    "audio_reference/cure_for_me.m4a"
)
DEFAULT_PATCH = Path(
    "/mnt/lightsem-data/experiments/lightsem/"
    "foundation_v3_motion_active_8beam_demo_20261020/shpongle/patch_8_beams.json"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return ((x - x.mean(0, keepdims=True)) / np.maximum(x.std(0, keepdims=True), 1e-5)).astype(
        np.float32
    )


def interp_time(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if len(x) == n:
        return x
    if len(x) < 2 or n < 1:
        raise ValueError(f"cannot interpolate shape {x.shape} to {n}")
    old = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    flat = np.stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])], 1)
    return flat.astype(np.float32)


def common_mir_features(wav_path: Path) -> np.ndarray:
    """Match the 36-D recipe used by the open-data prior."""
    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    hop = 2205
    n_fft = 4096
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop).T
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr).T[:, :12]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).reshape(-1, 1)
    rms = librosa.feature.rms(S=stft, frame_length=n_fft).T
    delta = np.diff(mfcc[:, :9], axis=0, prepend=mfcc[:1, :9])
    n = min(len(mfcc), len(chroma), len(onset), len(rms), len(delta))
    feature = np.concatenate([mfcc[:n], chroma[:n], onset[:n], rms[:n], delta[:n]], 1)
    if feature.shape[1] != 36:
        raise ValueError(f"expected 36-D common MIR, got {feature.shape}")
    return zscore(np.nan_to_num(feature))


def load_prior(checkpoint: Path, device: torch.device) -> SpeedNet:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SpeedNet(int(state.get("dim", 36))).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def prior_controls(
    prior: SpeedNet,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        output = prior(torch.from_numpy(features.T[None]).to(device)).cpu().numpy()[0]
    return output.astype(np.float32)


def cached_controls(
    row: dict[str, Any],
    cache_dir: Path,
    prior: SpeedNet,
    device: torch.device,
) -> np.ndarray:
    path = cache_dir / f"{row['id']}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as bundle:
            return np.asarray(bundle["controls"], np.float32)
    features = common_mir_features(Path(row["wav_path"]))
    controls = prior_controls(prior, features, device)
    np.savez_compressed(path, controls=controls, feature_dim=np.int32(features.shape[1]))
    return controls


class PriorConditionedDirectLightModel(nn.Module):
    """Frozen legacy model plus an exact-zeroable prior residual at hidden state."""

    def __init__(self, base: AudioOnlyDirectLightModel, prior_dim: int = 2):
        super().__init__()
        self.base = base
        width = base.foundation.config.width
        self.prior_to_hidden = nn.Sequential(
            nn.Linear(prior_dim, max(16, width // 3)),
            nn.SiLU(),
            nn.Linear(max(16, width // 3), width),
        )
        nn.init.zeros_(self.prior_to_hidden[-1].weight)
        nn.init.zeros_(self.prior_to_hidden[-1].bias)

    def _zeroable_shift(self, prior_controls: torch.Tensor) -> torch.Tensor:
        shift = self.prior_to_hidden(prior_controls)
        zero = torch.zeros(
            (prior_controls.shape[0], 1, prior_controls.shape[-1]),
            dtype=prior_controls.dtype,
            device=prior_controls.device,
        )
        return shift - self.prior_to_hidden(zero)

    def forward(
        self,
        audio: torch.Tensor,
        fixtures: torch.Tensor,
        *,
        prior_controls: torch.Tensor,
        timestamps=None,
        valid=None,
        state=None,
        duration_seconds=None,
    ) -> dict[str, torch.Tensor]:
        if prior_controls.shape[:2] != audio.shape[:2] or prior_controls.shape[-1] != 2:
            raise ValueError(
                f"prior_controls must be [B,T,2] aligned with audio, got "
                f"{tuple(prior_controls.shape)} vs {tuple(audio.shape)}"
            )
        if isinstance(state, dict):
            foundation_state = state.get("foundation")
            motion_state = state.get("motion")
        else:
            foundation_state = state
            motion_state = None
        tokens = self.base.foundation.audio(
            audio,
            timestamps=timestamps,
            valid=valid,
            duration_seconds=duration_seconds,
        )
        hidden, new_state = self.base.foundation.context(
            tokens,
            valid,
            foundation_state,
            return_state=True,
        )
        hidden = hidden + self._zeroable_shift(prior_controls)
        latent_seed = self.base.foundation.context.latent_seed[None].expand(hidden.shape[0], -1, -1)
        out = self.base.foundation._decode(hidden, fixtures, latent_seed)
        raw = out["aim"] * float(self.base.config.residual_gain)
        if self.base.config.dynamics_mode == "hard_clipped":
            projection = bounded_dynamics_projection
        elif self.base.config.dynamics_mode == "smooth_jerk":
            projection = smooth_bounded_dynamics_projection
        else:
            raise ValueError(f"unknown dynamics_mode={self.base.config.dynamics_mode!r}")
        (delta_angle, velocity, acceleration, jerk), new_motion_state = projection(
            raw,
            motion_state,
            self.base.config.safety,
            **(
                {
                    "drive_scale": self.base.config.dynamics_drive_scale,
                    "position_gain": self.base.config.dynamics_position_gain,
                    "velocity_gain": self.base.config.dynamics_velocity_gain,
                    "acceleration_gain": self.base.config.dynamics_acceleration_gain,
                    "radial_bounds": self.base.config.dynamics_radial_bounds,
                }
                if self.base.config.dynamics_mode == "smooth_jerk"
                else {}
            ),
        )
        anchor = self.base._anchor(fixtures)
        out["raw_aim"] = out["aim"]
        out["anchor"] = anchor
        out["aim"] = anchor[:, None] + delta_angle
        out["velocity"] = velocity
        out["acceleration"] = acceleration
        out["jerk"] = jerk
        out["audio_latent"] = hidden
        out["state"] = {"foundation": new_state, "motion": new_motion_state}
        return out


def make_row(row: dict[str, Any], prior_controls: np.ndarray) -> dict[str, Any]:
    loaded = load_row(row)
    controls = interp_time(prior_controls, len(loaded["audio"]))
    loaded["wav_path"] = row["wav_path"]
    loaded["prior_controls"] = controls
    return loaded


def batch(rows: list[dict[str, Any]], indices: list[tuple[int, int]], window: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "audio": torch.from_numpy(
            np.stack([rows[index]["audio"][start : start + window] for index, start in indices])
        ).float().to(device),
        "prior_controls": torch.from_numpy(
            np.stack([rows[index]["prior_controls"][start : start + window] for index, start in indices])
        ).float().to(device),
        "rgb": torch.from_numpy(
            np.stack([rows[index]["rgb"][start : start + window] for index, start in indices])
        ).float().to(device),
        "brightness": torch.from_numpy(
            np.stack([rows[index]["brightness"][start : start + window] for index, start in indices])
        ).float().to(device),
        "event": torch.from_numpy(
            np.stack([rows[index]["event"][start : start + window] for index, start in indices])
        ).float().to(device),
        "valid": torch.ones((len(indices), window), dtype=torch.bool, device=device),
    }


def appearance_metric(output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, float]:
    rgb = float((output["rgb"] - target["rgb"]).abs().mean().detach().cpu())
    brightness = float((output["brightness"] - target["brightness"]).abs().mean().detach().cpu())
    event = float(
        F.binary_cross_entropy_with_logits(output["event"], target["event"], reduction="mean")
        .detach()
        .cpu()
    )
    return {"rgb_l1": rgb, "brightness_l1": brightness, "event_bce": event, "composite": (rgb + brightness + event) / 3.0}


def evaluate_rows(
    model: AudioOnlyDirectLightModel | PriorConditionedDirectLightModel,
    rows: list[dict[str, Any]],
    device: torch.device,
    use_prior: bool,
    window: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    model.eval()
    totals = {"rgb_l1": 0.0, "brightness_l1": 0.0, "event_bce": 0.0, "composite": 0.0}
    count = 0
    per_row: dict[str, dict[str, float]] = {}
    fixtures = torch.from_numpy(make_fixture_metadata()[None]).float().to(device)
    with torch.no_grad():
        for row in rows:
            accum = {key: 0.0 for key in totals}
            n_chunks = 0
            for start in range(0, len(row["audio"]), window):
                end = min(len(row["audio"]), start + window)
                if end - start < 4:
                    continue
                audio = torch.from_numpy(row["audio"][start:end][None]).float().to(device)
                target = {
                    "rgb": torch.from_numpy(row["rgb"][start:end][None]).float().to(device),
                    "brightness": torch.from_numpy(row["brightness"][start:end][None]).float().to(device),
                    "event": torch.from_numpy(row["event"][start:end][None]).float().to(device),
                }
                valid = torch.ones((1, end - start), dtype=torch.bool, device=device)
                timestamps = torch.arange(end - start, device=device, dtype=torch.float32)[None] / 10.0
                if use_prior:
                    prior_controls = torch.from_numpy(row["prior_controls"][start:end][None]).float().to(device)
                    output = model(
                        audio,
                        fixtures,
                        prior_controls=prior_controls,
                        timestamps=timestamps,
                        valid=valid,
                        duration_seconds=(end - start) / 10.0,
                    )
                else:
                    output = model(
                        audio,
                        fixtures,
                        timestamps=timestamps,
                        valid=valid,
                        duration_seconds=(end - start) / 10.0,
                    )
                metric = appearance_metric(output, target)
                for key in accum:
                    accum[key] += metric[key]
                n_chunks += 1
            if not n_chunks:
                continue
            row_metric = {key: value / n_chunks for key, value in accum.items()}
            per_row[row["id"]] = row_metric
            for key in totals:
                totals[key] += row_metric[key]
            count += 1
    if count:
        totals = {key: value / count for key, value in totals.items()}
    return totals, per_row


def infer_model(
    model: AudioOnlyDirectLightModel | PriorConditionedDirectLightModel,
    audio: np.ndarray,
    device: torch.device,
    prior_controls: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    fixtures = torch.from_numpy(make_fixture_metadata()[None]).float().to(device)
    state = None
    keys = ("aim", "velocity", "acceleration", "jerk", "rgb", "brightness", "event", "audio_latent", "anchor")
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    with torch.no_grad():
        for start in range(0, len(audio), 128):
            end = min(len(audio), start + 128)
            x = torch.from_numpy(audio[start:end][None]).float().to(device)
            valid = torch.ones((1, end - start), dtype=torch.bool, device=device)
            timestamps = torch.arange(start, end, device=device, dtype=torch.float32)[None] / 10.0
            if prior_controls is None:
                output = model(
                    x,
                    fixtures,
                    timestamps=timestamps,
                    valid=valid,
                    state=state,
                    duration_seconds=len(audio) / 10.0,
                )
            else:
                pc = torch.from_numpy(prior_controls[start:end][None]).float().to(device)
                output = model(
                    x,
                    fixtures,
                    prior_controls=pc,
                    timestamps=timestamps,
                    valid=valid,
                    state=state,
                    duration_seconds=len(audio) / 10.0,
                )
            state = output["state"]
            for key in keys:
                chunks[key].append(output[key].detach().cpu().numpy())
    return {key: np.concatenate(values, 1) for key, values in chunks.items()}


def flatten_prediction(output: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value[0] for key, value in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--prior-checkpoint", type=Path, default=DEFAULT_PRIOR_CHECKPOINT)
    parser.add_argument("--shpongle-muq", type=Path, default=DEFAULT_MUQ / "shpongle.npz")
    parser.add_argument("--shpongle-base92", type=Path, default=DEFAULT_BASE92 / "shpongle.npz")
    parser.add_argument("--cure-muq", type=Path, default=DEFAULT_MUQ / "cure_for_me.npz")
    parser.add_argument("--cure-base92", type=Path, default=DEFAULT_BASE92 / "cure_for_me.npz")
    parser.add_argument("--shpongle-audio", type=Path, default=DEFAULT_SHONGLE_AUDIO)
    parser.add_argument("--cure-audio", type=Path, default=DEFAULT_CURE_AUDIO)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing artifact root: {args.out}")
    for path in (
        args.contract,
        args.base_checkpoint,
        args.prior_checkpoint,
        args.shpongle_muq,
        args.shpongle_base92,
        args.cure_muq,
        args.cure_base92,
        args.shpongle_audio,
        args.cure_audio,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    args.out.mkdir(parents=True, exist_ok=False)
    cache_dir = args.out / "common_mir_cache"
    cache_dir.mkdir()

    contract = json.loads(args.contract.read_text())
    prior = load_prior(args.prior_checkpoint, device)
    rows = []
    for source_row in contract["rows"]:
        loaded = load_row(source_row)
        loaded["wav_path"] = source_row["wav_path"]
        controls = cached_controls(source_row, cache_dir, prior, device)
        loaded["prior_controls"] = interp_time(controls, len(loaded["audio"]))
        rows.append(loaded)
    groups = np.asarray(sorted({row["wav_sha256"] for row in rows}))
    groups = groups[np.random.default_rng(args.seed).permutation(len(groups))]
    split = max(1, int(0.8 * len(groups)))
    train_groups = set(groups[:split])
    train_rows = [row for row in rows if row["wav_sha256"] in train_groups]
    val_rows = [row for row in rows if row["wav_sha256"] not in train_groups]
    train_windows = [
        (index, start)
        for index, row in enumerate(train_rows)
        for start in range(0, max(1, len(row["audio"]) - args.window + 1), args.window)
        if start + args.window <= len(row["audio"])
    ]
    if not train_windows or not val_rows:
        raise RuntimeError(f"invalid split: train_windows={len(train_windows)}, val_rows={len(val_rows)}")

    base = load_model(args.base_checkpoint, device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    treatment = PriorConditionedDirectLightModel(base).to(device)
    optimizer = torch.optim.AdamW(treatment.prior_to_hidden.parameters(), lr=2e-3, weight_decay=1e-4)
    fixtures = torch.from_numpy(make_fixture_metadata()).float().to(device)
    history: list[dict[str, float | int]] = []
    treatment.train()
    for step in range(args.steps):
        selected = [train_windows[(step * args.batch_size + j) % len(train_windows)] for j in range(args.batch_size)]
        target = batch(train_rows, selected, args.window, device)
        timestamps = torch.arange(args.window, device=device, dtype=torch.float32)[None].expand(args.batch_size, -1) / 10.0
        output = treatment(
            target["audio"],
            fixtures[None].expand(args.batch_size, -1, -1),
            prior_controls=target["prior_controls"],
            timestamps=timestamps,
            valid=target["valid"],
            duration_seconds=args.window / 10.0,
        )
        parts = loss_parts(output, target)
        optimizer.zero_grad(set_to_none=True)
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(treatment.prior_to_hidden.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 8) == 0:
            history.append({"step": step + 1, **{key: float(value.detach().cpu()) for key, value in parts.items()}})

    baseline_metric, baseline_rows = evaluate_rows(base, val_rows, device, False, args.window)
    treatment_metric, treatment_rows = evaluate_rows(treatment, val_rows, device, True, args.window)
    zero_metric, _ = evaluate_rows(
        treatment,
        [{**row, "prior_controls": np.zeros_like(row["prior_controls"])} for row in val_rows],
        device,
        True,
        args.window,
    )

    zero_probe = val_rows[0]
    probe_n = min(args.window, len(zero_probe["audio"]))
    probe_audio = torch.from_numpy(zero_probe["audio"][:probe_n][None]).float().to(device)
    probe_prior = torch.from_numpy(zero_probe["prior_controls"][:probe_n][None]).float().to(device)
    probe_zero = torch.zeros_like(probe_prior)
    valid = torch.ones((1, probe_n), dtype=torch.bool, device=device)
    timestamps = torch.arange(probe_n, device=device, dtype=torch.float32)[None] / 10.0
    with torch.no_grad():
        control = base(probe_audio, fixtures[None], timestamps=timestamps, valid=valid, duration_seconds=probe_n / 10.0)
        treatment_out = treatment(
            probe_audio,
            fixtures[None],
            prior_controls=probe_prior,
            timestamps=timestamps,
            valid=valid,
            duration_seconds=probe_n / 10.0,
        )
        zero_out = treatment(
            probe_audio,
            fixtures[None],
            prior_controls=probe_zero,
            timestamps=timestamps,
            valid=valid,
            duration_seconds=probe_n / 10.0,
        )
    zero_diff = max(float((control[key] - zero_out[key]).abs().max().cpu()) for key in ("aim", "rgb", "brightness", "event"))
    live_diff = float((control["rgb"] - treatment_out["rgb"]).abs().mean().cpu())

    song_sources = {
        "shpongle": (args.shpongle_audio, args.shpongle_muq, args.shpongle_base92),
        "cure_for_me": (args.cure_audio, args.cure_muq, args.cure_base92),
    }
    own_reports: dict[str, Any] = {}
    own_predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, (audio_path, muq_path, base92_path) in song_sources.items():
        audio, times = align_features(muq_path, base92_path)
        own_features_36 = own_features(audio_path)
        controls = interp_time(prior_controls(prior, own_features_36, device), len(audio))
        baseline_output = flatten_prediction(infer_model(base, audio, device))
        treatment_output = flatten_prediction(infer_model(treatment, audio, device, controls))
        baseline_audit = audit_training_output(
            {key: torch.from_numpy(value[None]) for key, value in baseline_output.items()}, 10.0
        )
        treatment_audit = audit_training_output(
            {key: torch.from_numpy(value[None]) for key, value in treatment_output.items()}, 10.0
        )
        own_predictions[f"{name}_baseline"] = baseline_output
        own_predictions[f"{name}_treatment"] = treatment_output
        own_reports[name] = {
            "frames": int(len(audio)),
            "duration_s": float(times[-1]),
            "audio_sha256": sha(audio_path),
            "muq_sha256": sha(muq_path),
            "base92_sha256": sha(base92_path),
            "baseline": baseline_audit,
            "treatment": treatment_audit,
            "prior_speed_mean": float(controls[:, 0].mean()),
            "prior_speed_std": float(controls[:, 0].std()),
            "prior_turn_mean": float(controls[:, 1].mean()),
            "prior_turn_std": float(controls[:, 1].std()),
        }

    n = min(
        len(own_predictions["shpongle_treatment"]["rgb"]),
        len(own_predictions["cure_for_me_treatment"]["rgb"]),
    )
    own_swap = {
        "baseline_rgb": float(
            np.mean(
                np.abs(
                    own_predictions["shpongle_baseline"]["rgb"][:n]
                    - own_predictions["cure_for_me_baseline"]["rgb"][:n]
                )
            )
        ),
        "treatment_rgb": float(
            np.mean(
                np.abs(
                    own_predictions["shpongle_treatment"]["rgb"][:n]
                    - own_predictions["cure_for_me_treatment"]["rgb"][:n]
                )
            )
        ),
        "baseline_aim": float(
            np.mean(
                np.abs(
                    own_predictions["shpongle_baseline"]["aim"][:n]
                    - own_predictions["cure_for_me_baseline"]["aim"][:n]
                )
            )
        ),
        "treatment_aim": float(
            np.mean(
                np.abs(
                    own_predictions["shpongle_treatment"]["aim"][:n]
                    - own_predictions["cure_for_me_treatment"]["aim"][:n]
                )
            )
        ),
    }

    raw_pass = all(own_reports[name]["treatment"]["raw_safety_pass"] for name in own_reports)
    collapse_pass = all(own_reports[name]["treatment"]["output_collapse_pass"] for name in own_reports)
    conditioning_pass = own_swap["treatment_rgb"] > 0.001
    appearance_gain = baseline_metric["composite"] - treatment_metric["composite"]
    status = (
        "diagnostic_candidate"
        if appearance_gain >= 0.005 and zero_diff <= 1e-6 and live_diff > 1e-4 and raw_pass and collapse_pass and conditioning_pass
        else "rejected_appearance_gain"
        if appearance_gain < 0.005
        else "rejected_prior_integration"
        if zero_diff > 1e-6 or live_diff <= 1e-4
        else "rejected_raw_safety"
        if not raw_pass
        else "output_collapsed"
        if not collapse_pass
        else "unconditioned_or_shared_prior"
    )

    checkpoint = args.out / "prior_bridge_adapter.pt"
    torch.save(
        {
            "adapter_state_dict": treatment.prior_to_hidden.state_dict(),
            "seed": args.seed,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "base_checkpoint_sha256": sha(args.base_checkpoint),
            "prior_checkpoint": str(args.prior_checkpoint.resolve()),
            "prior_checkpoint_sha256": sha(args.prior_checkpoint),
            "feature_contract": "36-D common MIR at 10 Hz; MFCC13+chroma12+onset1+RMS1+MFCC-delta9",
            "zero_ablation": "prior shift subtracts its exact zero-input value",
        },
        checkpoint,
    )
    for label, values in own_predictions.items():
        np.savez_compressed(
            args.out / f"PREDICTIONS_{label}.npz",
            time=np.arange(len(values["aim"])) / 10.0,
            **values,
        )
    receipt = {
        "schema": "lightsem.foundation_open_prior_bridge.v1",
        "protocol_id": "foundation-open-prior-bridge-20261022-v1",
        "status": status,
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "window": args.window,
        "batch_size": args.batch_size,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha(args.base_checkpoint),
        "prior_checkpoint": str(args.prior_checkpoint.resolve()),
        "prior_checkpoint_sha256": sha(args.prior_checkpoint),
        "contract": str(args.contract.resolve()),
        "contract_sha256": sha(args.contract),
        "train_groups": len(train_groups),
        "validation_groups": len(groups) - len(train_groups),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_windows": len(train_windows),
        "history": history,
        "baseline_heldout": baseline_metric,
        "treatment_heldout": treatment_metric,
        "zero_prior_heldout": zero_metric,
        "heldout_gain": appearance_gain,
        "prior_live_probe": {"max_zero_ablation_abs_diff": zero_diff, "mean_live_rgb_abs_diff": live_diff},
        "own_song_audio_swap": own_swap,
        "songs": own_reports,
        "raw_safety_pass": raw_pass,
        "output_collapse_pass": collapse_pass,
        "conditioning_pass": conditioning_pass,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha(checkpoint),
        "motion_target_claim": False,
        "native_dmx": False,
        "promotion": "appearance diagnostic only; physical motion and authored-light promotion remain blocked",
    }
    (args.out / "TRAINING.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
