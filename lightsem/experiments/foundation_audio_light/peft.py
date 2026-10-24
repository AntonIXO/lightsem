"""Small, explicit parameter-efficient adapters for frozen music features.

The adapters operate *after* the authenticated MuQ feature cache.  They never
pretend to fine-tune MuQ weights that are not present in the process.  Every
residual branch is zero-initialised, so the ``none``/adapter comparison starts
from the same frozen-feature baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterConfig:
    kind: str = "none"
    width: int = 96
    bottleneck: int = 32
    heads: int = 4
    local_window: int = 32
    hierarchy_windows: tuple[int, ...] = (1, 4, 16)

    def __post_init__(self) -> None:
        if self.kind not in {"none", "local_conv", "causal_transformer", "hierarchical"}:
            raise ValueError(f"unknown adapter kind: {self.kind}")
        if self.width < 8 or self.bottleneck < 1:
            raise ValueError("width and bottleneck must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.local_window < 1 or any(w < 1 for w in self.hierarchy_windows):
            raise ValueError("adapter windows must be positive")


class _ZeroResidual(nn.Module):
    """A residual bottleneck whose initial output is exactly zero."""

    def __init__(self, width: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class LocalConvAdapter(_ZeroResidual):
    """Causal depthwise local adapter for onset/short-gesture detail."""

    def __init__(self, width: int, bottleneck: int, kernel_size: int = 5) -> None:
        super().__init__(width, bottleneck)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.local = nn.Conv1d(width, width, kernel_size, groups=width, bias=False)
        nn.init.zeros_(self.local.weight)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("adapter input must be B,T,D")
        # Left padding keeps this branch causal.  The zero-initialised filter
        # keeps the baseline exact until the arm is trained.
        y = x.transpose(1, 2)
        y = torch.nn.functional.pad(y, (self.kernel_size - 1, 0))
        y = self.local(y).transpose(1, 2)
        out = x + self.residual(y)
        return _mask_valid(out, valid)


class CausalTransformerAdapter(_ZeroResidual):
    """Bounded causal self-attention adapter."""

    def __init__(self, width: int, bottleneck: int, heads: int, local_window: int) -> None:
        super().__init__(width, bottleneck)
        self.local_window = local_window
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("adapter input must be B,T,D")
        b, t, _ = x.shape
        outputs = []
        for start in range(0, t, self.local_window):
            end = min(t, start + self.local_window)
            chunk = x[:, start:end]
            n = chunk.shape[1]
            mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=x.device), diagonal=1)
            chunk_valid = valid[:, start:end] if valid is not None else None
            key_padding = (~chunk_valid).clone() if chunk_valid is not None else None
            if key_padding is not None:
                key_padding[:, 0] = False
            attended, _ = self.attention(chunk, chunk, chunk, attn_mask=mask,
                                         key_padding_mask=key_padding, need_weights=False)
            outputs.append(self.attn_norm(chunk + attended))
        y = torch.cat(outputs, dim=1) if outputs else x.new_zeros(x.shape)
        out = x + self.residual(y)
        return _mask_valid(out, valid)


def _causal_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Causal moving mean with correct normalization at the left boundary."""
    if window <= 1:
        return x
    b, t, d = x.shape
    csum = torch.cumsum(x, dim=1)
    padded = torch.cat((x.new_zeros((b, 1, d)), csum), dim=1)
    end = torch.arange(1, t + 1, device=x.device)
    begin = (end - window).clamp_min(0)
    sums = padded[:, end] - padded[:, begin]
    counts = (end - begin).to(x.dtype).view(1, t, 1)
    return sums / counts


class HierarchicalTemporalAdapter(nn.Module):
    """SoniDo-inspired local/middle/coarse causal feature adapter."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.local = LocalConvAdapter(config.width, config.bottleneck)
        self.branches = nn.ModuleList(
            [_ZeroResidual(config.width, config.bottleneck) for _ in config.hierarchy_windows]
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("adapter input must be B,T,D")
        local = self.local(x, valid)
        residual = x.new_zeros(x.shape)
        for window, branch in zip(self.config.hierarchy_windows, self.branches):
            residual = residual + branch.residual(_causal_mean(local, int(window)))
        return _mask_valid(local + residual, valid)


class IdentityAdapter(nn.Module):
    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        return _mask_valid(x, valid)


def _mask_valid(x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return x
    if valid.shape != x.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be a B,T bool mask")
    return torch.where(valid[..., None], x, torch.zeros_like(x))


def build_temporal_adapter(config: AdapterConfig) -> nn.Module:
    if config.kind == "none":
        return IdentityAdapter()
    if config.kind == "local_conv":
        return LocalConvAdapter(config.width, config.bottleneck)
    if config.kind == "causal_transformer":
        return CausalTransformerAdapter(config.width, config.bottleneck, config.heads,
                                        config.local_window)
    return HierarchicalTemporalAdapter(config)


class LoRALinear(nn.Module):
    """LoRA wrapper for a linear layer, with a zero initial residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(x))


def inject_lora(module: nn.Module, *, rank: int = 8, alpha: float = 16.0,
                name_fragments: Iterable[str] = ("q", "k", "v", "out")) -> list[str]:
    """Replace matching Linear children and return their qualified names.

    MuQ versions expose different attention naming conventions.  The caller
    must record the returned names in the receipt; an empty list is a hard
    configuration error for a requested LoRA arm.
    """
    fragments = tuple(str(x).lower() for x in name_fragments)
    replaced: list[str] = []
    targets = []
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
                continue
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            if not any(fragment in qualified.lower() for fragment in fragments):
                continue
            targets.append((parent, child_name, child, qualified))
    for parent, child_name, child, qualified in targets:
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        replaced.append(qualified)
    return replaced


def trainable_parameter_report(module: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable),
            "fraction": float(trainable / max(total, 1))}

