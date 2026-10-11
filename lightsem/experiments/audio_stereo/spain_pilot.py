"""Reduced-width SpaIn-Net pilot.

This is a clean Torch implementation of the paper's D1-CAT mechanism: each
source has its own spectral encoder, a positional encoding of its requested
panning angle is concatenated to the stereo magnitude spectrum, and the source
streams are joined by the two X-UMX bridging operations around source BLSTMs.
It intentionally has no checkpoint or fallback motion path.  The model is
randomly initialized until the supervised train loop below fits it on a
prepared synthetic re-panning corpus.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


REFERENCE_REPO = "https://github.com/darius522/spain-net"
REFERENCE_COMMIT = "7b4d9adf50273806db9ac3ea183da37e03e8af3e"
REFERENCE_PAPER = "https://arxiv.org/abs/2202.07523"


@dataclass(frozen=True)
class SpaInNetConfig:
    sample_rate: int = 16_000
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    width: int = 128
    position_dim: int = 128
    num_layers: int = 2
    sources: int = 4

    def __post_init__(self) -> None:
        if self.sample_rate not in (16_000, 24_000):
            raise ValueError("SpaIn-Net pilot supports 16 kHz or 24 kHz")
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length > self.n_fft:
            raise ValueError("Invalid STFT dimensions")
        if self.width != 128:
            raise ValueError("The faithful reduced-width pilot uses width=128")
        if self.position_dim != 128 or self.num_layers != 2 or self.sources != 4:
            raise ValueError("Pilot contract requires position_dim=128, 2 BLSTM layers, 4 sources")
        if self.position_dim % 2:
            raise ValueError("position_dim must be even")


def equal_power_repan(stems: Tensor, angles_deg: Tensor) -> tuple[Tensor, Tensor]:
    """Render mono source stems at supplied CPP angles.

    ``stems`` is ``B,S,N`` (four source signals) and angles are degrees in
    ``[-45,45]``.  The returned mixture and target are ``B,2,N`` and
    ``B,S,2,N`` respectively.  This is the constant-power law used by the
    reference implementation, expressed with radians for torch's trigonometry.
    """
    if stems.ndim != 3 or stems.shape[1] != 4:
        raise ValueError("stems must have shape B,4,N")
    if angles_deg.shape != stems.shape[:2]:
        raise ValueError("angles must have shape B,4")
    alpha = torch.deg2rad(angles_deg.to(stems.dtype)).clamp(-torch.pi / 4, torch.pi / 4)
    left = (2.0**-0.5) * (torch.cos(alpha) + torch.sin(alpha))
    right = (2.0**-0.5) * (torch.cos(alpha) - torch.sin(alpha))
    target = torch.stack((stems * left[..., None], stems * right[..., None]), dim=2)
    return target.sum(dim=1), target


def spain_reconstruction_loss(prediction: Tensor, target: Tensor, mixture: Tensor,
                              mixture_weight: float = 0.1, eps: float = 1e-8) -> Tensor:
    """Experimental gain-normalized source and mixture reconstruction loss.

    Small absolute waveform MSE on quiet clips is not evidence of separation.
    Normalization makes the error comparable across gains; it does not prove
    source separation, avoid all collapse, or supervise dance quality.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have shape [B,4,2,N]")
    if mixture.shape != (prediction.shape[0], 2, prediction.shape[-1]):
        raise ValueError("mixture must have shape [B,2,N]")
    target_energy = target.square().mean(dim=(1, 2, 3)).clamp_min(eps)
    source_error = (prediction - target).square().mean(dim=(1, 2, 3)) / target_energy
    mixture_energy = mixture.square().mean(dim=(1, 2)).clamp_min(eps)
    consistency = (prediction.sum(dim=1) - mixture).square().mean(dim=(1, 2)) / mixture_energy
    return source_error.mean() + float(mixture_weight) * consistency.mean()


def slots_to_native_context(slots: Tensor, frame_count: int) -> Tensor:
    """Resample B,S,T,D slots without mixing time and descriptor coordinates."""
    if slots.ndim != 4 or frame_count < 1:
        raise ValueError("Expected B,S,T,D slots and a positive frame count")
    batch, sources, frames, width = slots.shape
    channels = slots.permute(0, 1, 3, 2).reshape(batch, sources * width, frames)
    return F.interpolate(channels, size=frame_count, mode="linear",
                         align_corners=False).transpose(1, 2)


class SpaInNet(nn.Module):
    """D1-CAT SpaIn-Net with differentiable waveform reconstruction."""

    def __init__(self, config: SpaInNetConfig | None = None):
        super().__init__()
        self.config = config or SpaInNetConfig()
        bins = self.config.n_fft // 2 + 1
        self.bins = bins
        self.register_buffer("window", torch.hann_window(self.config.win_length), persistent=False)
        self.position = _AnglePositionEncoding(self.config.position_dim)

        # D1-CAT: each target source receives its own encoder stream with the
        # angle embedding concatenated to the stereo magnitude spectrum.
        self.source_encoders = nn.ModuleList(
            [_FeatureEncoder(2 * bins + self.config.position_dim, self.config.width)
             for _ in range(self.config.sources)]
        )
        self.source_blstm = nn.ModuleList(
            [nn.LSTM(self.config.width, self.config.width // 2,
                     num_layers=self.config.num_layers, bidirectional=True,
                     dropout=0.0, batch_first=True)
             for _ in range(self.config.sources)]
        )
        # The first bridge averages source encodings before the source BLSTMs;
        # the second bridge averages encoder/BLSTM skip features before the
        # source-specific magnitude-mask decoders.
        self.source_decoders = nn.ModuleList(
            [_MaskDecoder(2 * self.config.width, 2 * bins) for _ in range(self.config.sources)]
        )

    def forward(self, wav: Tensor, pan_angles: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Separate ``B,2,N`` stereo waveforms using ``B,4`` pan slots.

        Returns ``(B,4,2,N)`` separated stereo waveforms and inspectable
        features. ``native_descriptor`` is ``B,T30,64``: four source slots,
        each deterministically pooled to 16 dimensions (no random projection).
        """
        if wav.ndim != 3 or wav.shape[1] != 2:
            raise ValueError("wav must have shape B,2,N")
        if pan_angles.ndim != 2 or pan_angles.shape != (wav.shape[0], self.config.sources):
            raise ValueError("pan_angles must have shape B,4")
        batch, _, n_samples = wav.shape
        wav = wav.float()
        window = self.window.to(device=wav.device, dtype=wav.dtype)
        stft = torch.stft(wav.reshape(batch * 2, n_samples), n_fft=self.config.n_fft,
                          hop_length=self.config.hop_length, win_length=self.config.win_length,
                          window=window, center=True, return_complex=True)
        stft = stft.reshape(batch, 2, self.bins, -1)
        magnitude = stft.abs()
        frames = magnitude.shape[-1]
        spectral = magnitude.permute(0, 3, 1, 2).reshape(batch, frames, 2 * self.bins)
        angle_embedding = self.position(pan_angles).to(spectral)

        source_features: list[Tensor] = []
        for index, encoder in enumerate(self.source_encoders):
            source_features.append(encoder(torch.cat((spectral, angle_embedding[:, None, index].expand(-1, frames, -1)), dim=-1)))
        cross_one = torch.stack(source_features, dim=1).mean(dim=1)
        skip_features: list[Tensor] = []
        for feature, blstm in zip(source_features, self.source_blstm):
            recurrent = blstm(cross_one)[0]
            skip_features.append(torch.cat((feature, recurrent), dim=-1))
        cross_two = torch.stack(skip_features, dim=1).mean(dim=1)
        masks = torch.stack([decoder(cross_two) for decoder in self.source_decoders], dim=1)
        # Masks are non-negative magnitude masks, as in X-UMX. Mixture phase
        # is used for the differentiable waveform reconstruction.
        estimated_magnitude = masks * magnitude.permute(0, 3, 1, 2)[:, None]
        phase = stft[:, None] / magnitude[:, None].clamp_min(1e-8)
        estimated_stft = estimated_magnitude.permute(0, 1, 3, 4, 2) * phase
        separated = torch.istft(
            estimated_stft.reshape(batch * self.config.sources * 2, self.bins, frames),
            n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.win_length, window=window, center=True, length=n_samples,
        ).reshape(batch, self.config.sources, 2, n_samples)

        slot16 = F.adaptive_avg_pool1d(
            torch.stack(source_features, dim=1).reshape(batch * self.config.sources * frames, 1, self.config.width),
            16,
        ).reshape(batch, self.config.sources, frames, 16)
        t30 = max(1, round(n_samples / self.config.sample_rate * 30))
        descriptor = slots_to_native_context(slot16, t30)
        features = {
            "native_descriptor": descriptor,
            "source_slots": slot16,
            "source_features": torch.stack(source_features, dim=1),
            "cross_two": cross_two,
            "masks": masks,
            "mixture_magnitude": magnitude,
            "pan_embedding": angle_embedding,
        }
        return separated, features

    def config_dict(self) -> dict[str, Any]:
        return {"schema_version": 2, "architecture": "spain_inspired_encoded_angle_xumx",
                "descriptor_layout": "B,S,T,D -> B,S*D,T -> B,T30,S*D",
                "pretrained": False, "config": asdict(self.config),
                "reference_repo": REFERENCE_REPO, "reference_commit": REFERENCE_COMMIT,
                "reference_paper": REFERENCE_PAPER,
                "spatial_semantics": "source panning angle, not physical 3D location"}


class _FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, width, bias=False)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        value = self.linear(x.reshape(-1, shape[-1]))
        value = torch.tanh(self.norm(value))
        return value.reshape(shape[0], shape[1], -1)


class _MaskDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.norm = nn.BatchNorm1d(input_dim // 2)
        self.output = nn.Linear(input_dim // 2, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        value = self.linear(x.reshape(-1, shape[-1]))
        value = F.relu(self.norm(value))
        value = F.relu(self.output(value)).reshape(shape[0], shape[1], -1)
        return value.view(shape[0], shape[1], 2, -1)


class _AnglePositionEncoding(nn.Module):
    """The paper's signed-angle sinusoidal encoding, vectorized."""

    def __init__(self, dimension: int, resolution: float = 45.0):
        super().__init__()
        index = torch.arange(dimension, dtype=torch.float32)
        self.register_buffer("positive_exponent", 2 * torch.floor(index / 2) / dimension, persistent=False)
        self.register_buffer("negative_exponent", 2 * torch.floor(torch.flip(index, dims=(0,)) / 2) / dimension, persistent=False)
        self.resolution = resolution

    def forward(self, angles_deg: Tensor) -> Tensor:
        angle = angles_deg.clamp(-self.resolution, self.resolution)
        exponent = torch.where(angle[..., None] >= 0, self.positive_exponent, self.negative_exponent)
        value = angle.abs()[..., None] / torch.pow(self.resolution, exponent)
        encoded = torch.empty_like(value)
        encoded[..., 0::2] = torch.sin(value[..., 0::2])
        encoded[..., 1::2] = torch.cos(value[..., 1::2])
        return encoded


def _manifest_entries(path: Path) -> list[Path]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("files", []))
        entries = [item["path"] if isinstance(item, dict) else item for item in payload]
    else:
        entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [Path(item) if Path(item).is_absolute() else path.parent / item for item in entries]


def _load_stems(path: Path) -> Tensor:
    with np.load(path, allow_pickle=False) as data:
        key = "stems" if "stems" in data else "sources" if "sources" in data else None
        if key is None:
            raise KeyError(f"{path} must contain stems or sources")
        stems = torch.from_numpy(np.asarray(data[key], dtype=np.float32))
    if stems.ndim == 4 and stems.shape[0] == 1:
        stems = stems[0]
    if stems.ndim == 3 and stems.shape[1] == 2:
        stems = stems.mean(dim=1)
    if stems.ndim != 2 or stems.shape[0] != 4:
        raise ValueError(f"{path} stems must be [4,N] or [4,2,N]")
    return stems


def train_manifest(manifest: str | Path, *, steps: int = 1, segment_samples: int = 4096,
                   learning_rate: float = 1e-3, seed: int = 0,
                   config: SpaInNetConfig | None = None) -> tuple[SpaInNet, list[float]]:
    """Train only on manifest-listed NPZ stems with synthetic CPP repanning."""
    random.seed(seed)
    torch.manual_seed(seed)
    model = SpaInNet(config).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    paths = _manifest_entries(Path(manifest))
    if not paths:
        raise ValueError("manifest contains no NPZ items")
    history: list[float] = []
    for step in range(steps):
        stems = _load_stems(paths[step % len(paths)])
        if stems.shape[-1] < segment_samples:
            stems = F.pad(stems, (0, segment_samples - stems.shape[-1]))
        start = random.randint(0, stems.shape[-1] - segment_samples)
        stems = stems[:, start:start + segment_samples][None]
        angles = torch.empty(1, 4).uniform_(-45.0, 45.0)
        mixture, target = equal_power_repan(stems, angles)
        prediction, _ = model(mixture, angles)
        loss = spain_reconstruction_loss(prediction, target, mixture)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reduced-width SpaIn-Net D1-CAT pilot")
    parser.add_argument("manifest", type=Path, help="JSON/JSONL manifest of NPZ files containing four stems")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--segment-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, history = train_manifest(args.manifest, steps=args.steps,
                                     segment_samples=args.segment_samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict(),
                "loss_history": history}, args.output)


if __name__ == "__main__":
    main()
