"""Target-independent, offline audio students for the light-layer campaign.

The only observed input is an aligned 40 Hz acoustic feature sequence. The
optional frozen rhythmic logits are just two of those input features. There is
no pose/video input, beat detector, palette sampler, or forced-flash policy.

Slow context averages non-overlapping groups of four audio frames, then uses
an offline Conformer at 10 Hz. The accent head also sees the original acoustic
features at 40 Hz. Shared and independent models have identical temporal
support. The independent model changes only width, not depth/window/pooling.

Forward handles a whole input up to ``max_frames40``. Longer inputs fail rather
than silently truncate or apply an unverified chunking approximation. Attention
still allocates a quadratic mask; the length limit is not a memory guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal, Sequence

import torch
from torch import nn

from lightsem.model.light_trunk import ConformerBlock, TrunkConfig


@dataclass(frozen=True)
class LightLayersConfig:
    in_dim: int = 94
    d_model: int = 128
    n_blocks: int = 3
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15
    attn_window: int = 160
    dropout: float = 0.1
    palette_size: int = 32
    visual_aux_dim: int = 0
    pose_aux_dim: int = 0
    max_frames40: int = 12_000

    def __post_init__(self) -> None:
        for name in ("in_dim", "d_model", "n_blocks", "n_heads", "ff_mult",
                     "conv_kernel", "attn_window", "palette_size", "max_frames40"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % 4 or self.d_model % self.n_heads:
            raise ValueError("d_model must be a multiple of four and of n_heads")
        if self.conv_kernel % 2 != 1:
            raise ValueError("conv_kernel must be odd for symmetric offline support")
        if self.attn_window < 2 or self.attn_window % 2:
            raise ValueError("attn_window must be even and at least two")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.palette_size < 2:
            raise ValueError("palette_size must be at least two")
        if type(self.visual_aux_dim) is not int or self.visual_aux_dim not in (0, 6, 96):
            raise ValueError("visual_aux_dim must be 0, 6, or 96")
        if type(self.pose_aux_dim) is not int or self.pose_aux_dim not in (0, 4):
            raise ValueError("pose_aux_dim must be 0 or 4")
        if self.max_frames40 % 4:
            raise ValueError("max_frames40 must be divisible by four")

    @property
    def null_palette_id(self) -> int:
        """A learned null state, not a palette output class or a padding row."""
        return self.palette_size

    @property
    def context_radius10(self) -> int:
        """Structural slow-context radius in frames, excluding song boundaries.

        The reused attention uses strict inequalities. Its even window W sees
        offsets -(W/2-1) through +(W/2-1), i.e. W-1 positions, not W positions.
        Each serial convolution adds (kernel-1)/2 frames on either side.
        This states support only; no chunking equivalence is asserted.
        """
        return self.n_blocks * (self.attn_window // 2 - 1 + (self.conv_kernel - 1) // 2)

    def trunk_config(self) -> TrunkConfig:
        return TrunkConfig(
            in_dim=self.in_dim, d_model=self.d_model, n_blocks=self.n_blocks,
            n_heads=self.n_heads, ff_mult=self.ff_mult, conv_kernel=self.conv_kernel,
            attn_window=self.attn_window, dropout=self.dropout, offline=True,
        )


@dataclass(frozen=True)
class ParameterMatch:
    """All trainable parameters, including enabled auxiliary/state heads.

    These are active training parameters, not a FLOP or latency measurement.
    No padding tensors are added to improve the reported match.
    """

    shared_d_model: int
    independent_d_model: int
    shared_params: int
    independent_params: int
    relative_difference: float
    candidate_widths: tuple[int, ...]
    tolerance: float = 0.05


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AudioEncoder(nn.Module):
    """Aligned average pooling followed by an offline 10 Hz Conformer."""

    def __init__(self, cfg: LightLayersConfig):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(cfg.in_dim),
                                 nn.Linear(cfg.in_dim, cfg.d_model), nn.SiLU())
        self.blocks = nn.ModuleList(ConformerBlock(cfg.trunk_config())
                                    for _ in range(cfg.n_blocks))

    def forward(self, audio40: torch.Tensor) -> torch.Tensor:
        b, t, f = audio40.shape
        # Context frame j is aligned with raw frames [4*j, 4*j+4).
        pooled10 = audio40.reshape(b, t // 4, 4, f).mean(dim=2)
        context10 = self.inp(pooled10)
        for block in self.blocks:
            context10 = block(context10)
        return context10


class _LightLayers(nn.Module):
    def __init__(self, cfg: LightLayersConfig, *, independent: bool):
        super().__init__()
        self.cfg = cfg
        self.independent = independent
        names = ("accent", "chromatic", "palette") if independent else ("shared",)
        self.encoders = nn.ModuleDict({name: AudioEncoder(cfg) for name in names})
        self.accent_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.in_dim, cfg.d_model), nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.chromatic_head = nn.Linear(cfg.d_model, 1)
        self.palette_state_embedding = nn.Embedding(cfg.palette_size + 1, cfg.d_model)
        self.palette_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model), nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.palette_size),
        )
        self.visual_aux_head = (nn.Linear(cfg.d_model, cfg.visual_aux_dim)
                                if cfg.visual_aux_dim else None)
        self.pose_aux_head = (nn.Linear(cfg.d_model, cfg.pose_aux_dim)
                              if cfg.pose_aux_dim else None)

    def _validate_audio(self, audio40: torch.Tensor) -> None:
        if not isinstance(audio40, torch.Tensor) or audio40.ndim != 3:
            raise ValueError("audio40 must be a tensor of shape (B, T40, in_dim)")
        b, t, f = audio40.shape
        if b <= 0 or t <= 0 or t % 4:
            raise ValueError("audio40 needs a nonempty batch and T40 divisible by four")
        if f != self.cfg.in_dim:
            raise ValueError(f"expected {self.cfg.in_dim} acoustic features, got {f}")
        if t > self.cfg.max_frames40:
            raise ValueError(f"T40={t} exceeds max_frames40={self.cfg.max_frames40}; "
                             "chunking is not implemented")
        if not audio40.is_floating_point():
            raise ValueError("audio40 must be floating point")

    def forward(self, audio40: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw logits and palette context; no thresholding or sampling.

        ``accent_logit``: (B, T40).
        ``chromatic_change_logit``: (B, T40/4).
        ``context10``: (B, T40/4, d_model), for ``palette_logits``.
        ``visual_aux``/``pose_aux``: (B, T40/4, configured dimension), only
        returned in training mode when their configured dimension is nonzero.
        Auxiliary heads use the chromatic branch in the independent model.
        """
        self._validate_audio(audio40)
        if self.independent:
            accent10 = self.encoders["accent"](audio40)
            chromatic10 = self.encoders["chromatic"](audio40)
            palette10 = self.encoders["palette"](audio40)
        else:
            accent10 = chromatic10 = palette10 = self.encoders["shared"](audio40)
        accent_input = torch.cat((accent10.repeat_interleave(4, dim=1), audio40), dim=-1)
        result = {
            "accent_logit": self.accent_head(accent_input).squeeze(-1),
            "chromatic_change_logit": self.chromatic_head(chromatic10).squeeze(-1),
            "context10": palette10,
        }
        if self.training:
            if self.visual_aux_head is not None:
                result["visual_aux"] = self.visual_aux_head(chromatic10)
            if self.pose_aux_head is not None:
                result["pose_aux"] = self.pose_aux_head(chromatic10)
        return result

    def palette_logits(
        self,
        context10: torch.Tensor,
        generated_previous_token_ids: torch.Tensor | None = None,
        *,
        state_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict K raw logits from audio context and explicit previous state.

        Previous ids have shape (B, T10), dtype int64, and range [0, K], where
        K is the null id. Pass None for all-null state. A same-shaped boolean
        ``state_drop_mask`` replaces selected ids by null. There is no random
        state dropout, hidden recurrent state, target input, or decoding loop.
        The caller supplies generated previous ids at inference and handles
        any teacher forcing, state corruption, or free-running generation.
        """
        if (not isinstance(context10, torch.Tensor) or context10.ndim != 3
                or context10.shape[-1] != self.cfg.d_model
                or min(context10.shape[:2]) <= 0 or not context10.is_floating_point()):
            raise ValueError("context10 must be nonempty floating (B, T10, d_model)")
        shape = context10.shape[:2]
        if generated_previous_token_ids is None:
            ids = torch.full(shape, self.cfg.null_palette_id, dtype=torch.long,
                             device=context10.device)
        else:
            ids = generated_previous_token_ids
            if (not isinstance(ids, torch.Tensor) or ids.shape != shape
                    or ids.dtype != torch.long or ids.device != context10.device):
                raise ValueError("previous ids must be int64 (B, T10) on the context device")
            if bool(((ids < 0) | (ids > self.cfg.null_palette_id)).any()):
                raise ValueError("previous ids must lie in [0, palette_size] inclusive")
        if state_drop_mask is not None:
            if (not isinstance(state_drop_mask, torch.Tensor)
                    or state_drop_mask.shape != shape or state_drop_mask.dtype != torch.bool
                    or state_drop_mask.device != context10.device):
                raise ValueError("state_drop_mask must be boolean (B, T10) on the context device")
            ids = ids.masked_fill(state_drop_mask, self.cfg.null_palette_id)
        state = self.palette_state_embedding(ids)
        return self.palette_head(torch.cat((context10, state), dim=-1))


class SharedLightLayers(_LightLayers):
    """One audio encoder shared by accent, chromatic-change, and palette heads."""

    def __init__(self, cfg: LightLayersConfig | None = None):
        super().__init__(cfg if cfg is not None else LightLayersConfig(), independent=False)


def match_independent_width(
    cfg: LightLayersConfig,
    allowed_widths: Sequence[int] | None = None,
) -> ParameterMatch:
    """Choose a narrower multiple-of-four width using only parameter counts.

    The default candidates are every positive multiple of four below the
    reference width that is also divisible by n_heads. An explicit candidate
    set must obey those constraints. All other configuration fields stay fixed.
    The smallest absolute count error wins; ties choose the smaller width.
    Failure to reach 5% is an error, not permission to change temporal context.
    Counting on the meta device does not allocate weights or consume RNG state.
    """
    if allowed_widths is None:
        candidates = tuple(w for w in range(4, cfg.d_model, 4) if w % cfg.n_heads == 0)
    else:
        candidates = tuple(allowed_widths)
        if any(type(w) is not int or w <= 0 or w >= cfg.d_model
               or w % 4 or w % cfg.n_heads for w in candidates):
            raise ValueError("candidate widths must be narrower positive multiples of four "
                             "and of n_heads")
        candidates = tuple(sorted(set(candidates)))
    if not candidates:
        raise ValueError("no allowed narrower widths")
    with torch.device("meta"):
        shared_params = count_trainable_parameters(_LightLayers(cfg, independent=False))
        counts = {
            width: count_trainable_parameters(
                _LightLayers(replace(cfg, d_model=width), independent=True))
            for width in candidates
        }
    width = min(candidates, key=lambda w: (abs(counts[w] - shared_params), w))
    difference = abs(counts[width] - shared_params) / shared_params
    if difference > 0.05:
        raise ValueError(f"no width matches within 5%: closest width {width} gives "
                         f"{counts[width]} vs shared {shared_params} ({difference:.2%})")
    return ParameterMatch(cfg.d_model, width, shared_params, counts[width], difference, candidates)


class IndependentLightLayers(_LightLayers):
    """Three real encoders; width chosen against the supplied shared config.

    ``reference_cfg`` is the shared configuration. ``cfg`` is the selected
    narrower configuration. ``parameter_match`` records the count-only choice.
    Compute cost has not been matched or measured by this procedure.
    """

    def __init__(self, cfg: LightLayersConfig | None = None,
                 *, allowed_widths: Sequence[int] | None = None):
        reference = cfg if cfg is not None else LightLayersConfig()
        match = match_independent_width(reference, allowed_widths)
        super().__init__(replace(reference, d_model=match.independent_d_model), independent=True)
        self.reference_cfg = reference
        self.parameter_match = match


def build_light_layers(
    cfg: LightLayersConfig | None = None,
    *,
    architecture: Literal["shared", "independent"] = "shared",
    allowed_widths: Sequence[int] | None = None,
) -> SharedLightLayers | IndependentLightLayers:
    if architecture == "shared":
        if allowed_widths is not None:
            raise ValueError("allowed_widths is only valid for the independent architecture")
        return SharedLightLayers(cfg)
    if architecture == "independent":
        return IndependentLightLayers(cfg, allowed_widths=allowed_widths)
    raise ValueError("architecture must be 'shared' or 'independent'")
