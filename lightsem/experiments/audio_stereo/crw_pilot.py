"""A small, faithful StereoCRW reproduction for the stereo pilot.

The original StereoCRW code (commit ``c08488505042fcdc2788d339ba23901929e7b6d4``)
uses a shared ResNet9 over short complex-STFT audio patches and trains a
left-to-right-to-left cycle correspondence objective.  This module keeps that
mechanism while removing the old torchaudio API and its implicit complex tensor
layout.  It is a reproduction probe: no FMA checkpoint is bundled or loaded.

The canonical FMA geometry is 16 kHz audio, 1024-sample patches, four-sample
stride, and 49 patches.  The native conditioner features are deterministic
summaries of the learned correspondence distribution and its confidence; they
are not a random projection or a DSP relabeling.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet


ORIGINAL_REPO_COMMIT = "c08488505042fcdc2788d339ba23901929e7b6d4"
PRETRAINED_WEIGHTS_AVAILABLE = False


@dataclass(frozen=True)
class StereoCRWConfig:
    """Geometry and objective parameters from the StereoCRW pilot recipe."""

    sample_rate: int = 16_000
    patch_size: int = 1_024
    patch_stride: int = 4
    patch_num: int = 49
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 8
    feature_dim: int = 128
    tau: float = 0.05
    cycle_num: int = 1
    smooth_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.patch_size <= 0 or self.patch_stride <= 0:
            raise ValueError("StereoCRW sample and patch dimensions must be positive")
        if self.patch_num < 2 or self.n_fft <= 0 or self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("Invalid StereoCRW patch/STFT geometry")
        if self.win_length > self.n_fft or self.hop_length > self.n_fft:
            raise ValueError("win_length and hop_length cannot exceed n_fft")
        if self.feature_dim <= 0 or self.tau <= 0 or self.cycle_num <= 0:
            raise ValueError("Invalid StereoCRW feature/objective dimensions")

    @property
    def required_samples(self) -> int:
        return self.patch_size + self.patch_stride * (self.patch_num - 1)

    def config_dict(self) -> dict:
        return {**asdict(self), "schema_version": 1,
                "required_samples": self.required_samples,
                "original_repo_commit": ORIGINAL_REPO_COMMIT,
                "backbone": "torchvision ResNet9 BasicBlock [1,1,1,1]",
                "stft_complex_layout": "torch.view_as_real -> channel-first real/imag",
                "pretrained_weights": PRETRAINED_WEIGHTS_AVAILABLE,
                "reproduction_scope": "new seed-0 self-supervised pilot; no FMA pretrained claim"}


def _as_waveforms(waveforms: Tensor) -> Tensor:
    """Normalize input to finite ``[batch, 2, samples]`` float tensors."""
    value = torch.as_tensor(waveforms)
    if value.ndim == 2:
        if value.shape[0] == 2:
            value = value.unsqueeze(0)
        elif value.shape[1] == 2:
            value = value.T.unsqueeze(0)
        else:
            raise ValueError("StereoCRW expects [2,T], [T,2], or [B,2,T]")
    elif value.ndim == 3 and value.shape[1] == 2:
        pass
    elif value.ndim == 3 and value.shape[2] == 2:
        value = value.transpose(1, 2)
    else:
        raise ValueError("StereoCRW expects [2,T], [T,2], or [B,2,T]")
    value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError("StereoCRW waveform contains non-finite values")
    return value.contiguous()


def extract_patches(waveforms: Tensor, config: StereoCRWConfig = StereoCRWConfig()) -> Tensor:
    """Extract the canonical overlapping waveform patches.

    Returns ``[B, 2, patch_num, patch_size]``.  The source implementation
    unfolds with the four-sample stride and uses exactly 49 patches; extra
    samples are deliberately ignored so the correspondence clock is fixed.
    """
    value = _as_waveforms(waveforms)
    if value.shape[-1] < config.required_samples:
        raise ValueError(
            f"StereoCRW needs at least {config.required_samples} samples, got {value.shape[-1]}"
        )
    patches = value.unfold(-1, config.patch_size, config.patch_stride)
    return patches[..., :config.patch_num, :].contiguous()


class StereoCRWPatchEncoder(nn.Module):
    """Shared ResNet9 encoder for real/imaginary STFT patch images."""

    def __init__(self, config: StereoCRWConfig = StereoCRWConfig()):
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.hann_window(config.win_length), persistent=False)
        backbone = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=config.feature_dim)
        backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = backbone

    def stft_images(self, patches: Tensor) -> Tensor:
        """Convert ``[N,patch_size]`` to ``[N,2,freq,time]`` real/imag images."""
        if patches.ndim != 2 or patches.shape[-1] != self.config.patch_size:
            raise ValueError("Expected [N, patch_size] waveform patches")
        window = self.window.to(device=patches.device, dtype=patches.dtype)
        spec = torch.stft(
            patches, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
            win_length=self.config.win_length, window=window, center=True,
            return_complex=True,
        )
        real_imag = torch.view_as_real(spec)  # [N, F, T, 2], explicit modern layout
        return real_imag.permute(0, 3, 1, 2).contiguous()

    def encode_patches(self, patches: Tensor) -> Tensor:
        if patches.ndim != 3 or patches.shape[-1] != self.config.patch_size:
            raise ValueError("Expected [B, patch_num, patch_size] waveform patches")
        images = self.stft_images(patches.reshape(-1, patches.shape[-1]))
        encoded = self.backbone(images)
        return encoded.reshape(patches.shape[0], patches.shape[1], self.config.feature_dim)

    def forward(self, waveforms: Tensor) -> Tensor:
        patches = extract_patches(waveforms, self.config)
        left = self.encode_patches(patches[:, 0])
        right = self.encode_patches(patches[:, 1])
        return torch.stack((left, right), dim=1)


def correspondence_distribution(left: Tensor, right: Tensor, tau: float = 0.05) -> tuple[Tensor, Tensor]:
    """Return row-normalized left-to-right and right-to-left affinities."""
    if left.ndim != 3 or right.ndim != 3 or left.shape != right.shape:
        raise ValueError("StereoCRW embeddings must both have shape [B,P,D]")
    if tau <= 0:
        raise ValueError("StereoCRW temperature must be positive")
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    logits = torch.matmul(left, right.transpose(-1, -2)) / tau
    return F.softmax(logits, dim=-1), F.softmax(logits.transpose(-1, -2), dim=-1)


def _offset_tensor(shift_offset: int | Tensor, batch: int, device: torch.device) -> Tensor:
    offset = torch.as_tensor(shift_offset, device=device, dtype=torch.long)
    if offset.ndim == 0:
        offset = offset.repeat(batch)
    if offset.shape != (batch,):
        raise ValueError("shift_offset must be a scalar or [batch] integer tensor")
    return offset


def shifted_cycle_targets(length: int, shift_offset: int | Tensor, *, device: torch.device,
                          batch: int = 1) -> tuple[Tensor, Tensor]:
    """Build original StereoCRW offset labels and valid rows.

    The pinned implementation defines a positive ``shift_offset`` as a target
    index subtraction (``label = row - shift_offset``), then removes the
    corresponding boundary rows.  Keeping this convention explicit prevents a
    silent sign inversion in counterfactual delay tests.
    """
    offsets = _offset_tensor(shift_offset, batch, device)
    max_abs = int(offsets.abs().max().item())
    if max_abs >= length:
        raise ValueError("shift_offset leaves no valid cycle rows")
    rows = torch.arange(max_abs, length - max_abs, device=device, dtype=torch.long)
    labels = rows.unsqueeze(0) - offsets.unsqueeze(1)
    if (labels < 0).any() or (labels >= length).any():
        raise ValueError("shift_offset produces out-of-range cycle labels")
    return rows, labels


def _cycle_nll(affinity_a: Tensor, affinity_b: Tensor, shift_offset: int | Tensor,
               cycle_num: int) -> tuple[Tensor, Tensor, Tensor]:
    b, length, _ = affinity_a.shape
    rows, labels = shifted_cycle_targets(length, shift_offset, device=affinity_a.device, batch=b)
    # ``correspondence_distribution`` already returns row-normalized
    # probabilities.  The original implementation applies softmax to logits
    # immediately before this multiplication; applying it a second time would
    # silently flatten the learned distribution.
    cycle = torch.linalg.matrix_power(affinity_a @ affinity_b, cycle_num)
    selected = cycle[:, rows, :].reshape(-1, length)
    target = labels.reshape(-1)
    loss = F.nll_loss(torch.log(selected.clamp_min(torch.finfo(selected.dtype).tiny)), target)
    return loss, cycle, target


def cycle_loss(left: Tensor, right: Tensor, *, tau: float = 0.05, shift_offset: int | Tensor = 0,
               cycle_num: int = 1, bidirectional: bool = True) -> Tensor:
    """Compute the StereoCRW left-right-left cycle NLL."""
    aff_lr, aff_rl = correspondence_distribution(left, right, tau)
    loss_lr, _, _ = _cycle_nll(aff_lr, aff_rl, shift_offset, cycle_num)
    if not bidirectional:
        return loss_lr
    loss_rl, _, _ = _cycle_nll(aff_rl, aff_lr, shift_offset, cycle_num)
    return .5 * (loss_lr + loss_rl)


def _smoothness_loss(affinity: Tensor) -> Tensor:
    """Original-style smoothness penalty on expected correspondence motion."""
    b, length, _ = affinity.shape
    index = torch.arange(length, device=affinity.device, dtype=affinity.dtype)
    expected = affinity @ index
    displacement = expected - index
    if length < 2:
        return displacement.square().mean() * 0.
    return (displacement[:, 1:] - displacement[:, :-1]).abs().mean()


def stereo_crw_loss(left: Tensor, right: Tensor, *, tau: float = 0.05,
                    shift_offset: int | Tensor = 0, cycle_num: int = 1,
                    bidirectional: bool = True, smooth_weight: float = 0.) -> tuple[Tensor, dict[str, Tensor]]:
    """Return total loss and auditable cycle/affinity components."""
    if smooth_weight < 0:
        raise ValueError("smooth_weight must be nonnegative")
    aff_lr, aff_rl = correspondence_distribution(left, right, tau)
    loss_lr, cycle_lr, labels_lr = _cycle_nll(aff_lr, aff_rl, shift_offset, cycle_num)
    if bidirectional:
        loss_rl, cycle_rl, labels_rl = _cycle_nll(aff_rl, aff_lr, shift_offset, cycle_num)
        cycle_component = .5 * (loss_lr + loss_rl)
    else:
        loss_rl = torch.zeros_like(loss_lr)
        cycle_rl = torch.empty(0, device=left.device)
        labels_rl = torch.empty(0, dtype=torch.long, device=left.device)
        cycle_component = loss_lr
    smooth = _smoothness_loss(aff_lr) if smooth_weight else torch.zeros_like(cycle_component)
    total = cycle_component + smooth_weight * smooth
    return total, {
        "cycle": cycle_component, "cycle_left": loss_lr, "cycle_right": loss_rl,
        "smooth": smooth, "affinity_l2r": aff_lr, "affinity_r2l": aff_rl,
        "cycle_matrix_l2r": cycle_lr, "cycle_matrix_r2l": cycle_rl,
        "labels_l2r": labels_lr, "labels_r2l": labels_rl, "total": total,
    }


def _correspondence_descriptors(affinity: Tensor) -> Tensor:
    """Per-patch displacement, spread, entropy, and confidence descriptors."""
    b, length, _ = affinity.shape
    index = torch.arange(length, device=affinity.device, dtype=affinity.dtype)
    expected = affinity @ index
    displacement = expected - index
    variance = (affinity * (index.view(1, 1, -1) - expected.unsqueeze(-1)).square()).sum(-1)
    entropy = -(affinity.clamp_min(torch.finfo(affinity.dtype).tiny) *
                affinity.clamp_min(torch.finfo(affinity.dtype).tiny).log()).sum(-1)
    top = torch.topk(affinity, k=min(2, length), dim=-1).values
    margin = top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else 0.)
    distance = (index.view(1, -1) - index.view(-1, 1)).abs()
    local = (distance <= 1).to(dtype=affinity.dtype).view(1, length, length)
    local_mass = (affinity * local).sum(-1).clamp(0., 1.)
    confidence = top[..., 0]
    scale = max(float(length - 1), 1.)
    entropy_scale = torch.log(torch.tensor(float(length), device=affinity.device,
                                            dtype=affinity.dtype)).clamp_min(1.)
    return torch.stack((displacement / scale, displacement.abs() / scale,
                        variance / (scale * scale), entropy / entropy_scale,
                        confidence, margin, local_mass, torch.tanh(displacement / 2.),), dim=-1)


def native_conditioner_features(affinity: Tensor, *, per_patch: bool = False) -> Tensor:
    """Extract deterministic 64-D correspondence/confidence features.

    Eight per-patch descriptors are summarized by eight fixed statistics
    (mean, standard deviation, min, max, p10, p50, p90, and final-minus-first).
    With ``per_patch=True`` the same descriptor channels are expanded with a
    fixed eight-term temporal polynomial basis, producing ``[B,P,64]`` for
    native-clock interpolation.
    """
    if affinity.ndim != 3 or affinity.shape[-1] != affinity.shape[-2]:
        raise ValueError("Expected square [B,P,P] correspondence distribution")
    desc = _correspondence_descriptors(affinity)
    if per_patch:
        p = desc.shape[1]
        x = torch.linspace(-1., 1., p, device=desc.device, dtype=desc.dtype)
        basis = torch.stack((torch.ones_like(x), x, x.square(), x ** 3,
                             torch.sin(torch.pi * x), torch.cos(torch.pi * x),
                             torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x)), dim=-1)
        return (desc.unsqueeze(-1) * basis.view(1, p, 1, 8)).reshape(desc.shape[0], p, 64)
    q = torch.quantile(desc, torch.tensor((.10, .50, .90), device=desc.device, dtype=desc.dtype), dim=1)
    stats = torch.stack((desc.mean(1), desc.std(1, unbiased=False), desc.amin(1), desc.amax(1),
                         q[0], q[1], q[2], desc[:, -1] - desc[:, 0]), dim=1)
    return stats.reshape(desc.shape[0], 64)


class StereoCRW(nn.Module):
    """Shared-encoder StereoCRW model and native conditioner probe."""

    def __init__(self, config: StereoCRWConfig = StereoCRWConfig()):
        super().__init__()
        self.config = config
        self.encoder = StereoCRWPatchEncoder(config)

    def forward(self, waveforms: Tensor, *, shift_offset: int | Tensor = 0) -> dict[str, Tensor]:
        encoded = self.encoder(waveforms)
        left, right = encoded[:, 0], encoded[:, 1]
        affinity_l2r, affinity_r2l = correspondence_distribution(left, right, self.config.tau)
        total, details = stereo_crw_loss(
            left, right, tau=self.config.tau, shift_offset=shift_offset,
            cycle_num=self.config.cycle_num, smooth_weight=self.config.smooth_weight,
        )
        return {
            "left_embeddings": left, "right_embeddings": right,
            "affinity_l2r": affinity_l2r, "affinity_r2l": affinity_r2l,
            "native_features": native_conditioner_features(affinity_l2r),
            "native_patch_features": native_conditioner_features(affinity_l2r, per_patch=True),
            "loss": total, **{key: value for key, value in details.items()
                              if key not in ("affinity_l2r", "affinity_r2l")},
        }

    def config_dict(self) -> dict:
        return self.config.config_dict()


StereoCRWModel = StereoCRW
StereoCRWEncoder = StereoCRWPatchEncoder


def training_step(model: StereoCRW, optimizer: torch.optim.Optimizer, waveforms: Tensor,
                  *, shift_offset: int | Tensor = 0) -> dict[str, float]:
    """Minimal harness hook: one update on a batch of real stereo waveforms."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(waveforms, shift_offset=shift_offset)
    if not torch.isfinite(output["loss"]):
        raise FloatingPointError("Nonfinite StereoCRW loss")
    output["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(output["loss"].detach()),
            "cycle": float(output["cycle"].detach()),
            "smooth": float(output["smooth"].detach())}


__all__ = [
    "ORIGINAL_REPO_COMMIT", "PRETRAINED_WEIGHTS_AVAILABLE", "StereoCRWConfig",
    "StereoCRWPatchEncoder", "StereoCRWEncoder", "StereoCRW",
    "StereoCRWModel", "extract_patches", "correspondence_distribution",
    "shifted_cycle_targets", "cycle_loss", "stereo_crw_loss",
    "native_conditioner_features", "training_step",
]
