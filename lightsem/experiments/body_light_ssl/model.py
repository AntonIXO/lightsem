"""A small action-conditioned JEPA-style encoder for body-to-light data.

This is a controlled first loop rather than a native-lighting checkpoint.  The
student sees frozen audio, a body stream and semantic light state.  Body and
light are masked at selected timestamps; the predictor must match an EMA target
encoder in latent space.  A second causal head predicts the next latent while
conditioned on the previous semantic light action.  The latter is the minimum
needed before calling the model a world-model probe: an action-free encoder is
not evidence that light changes the dancer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ControlledJepaConfig:
    audio_dim: int
    body_dim: int
    light_dim: int
    width: int = 96
    embedding_dim: int = 64
    heads: int = 4
    layers: int = 2
    feedforward_mult: int = 4
    dropout: float = 0.0
    ema_decay: float = 0.996
    mask_ratio: float = 0.35
    action_semantics: str = "semantic_light_state"

    def __post_init__(self) -> None:
        if min(self.audio_dim, self.body_dim, self.light_dim) <= 0:
            raise ValueError("audio_dim, body_dim and light_dim must be positive")
        if self.width < 8 or self.embedding_dim < 4 or self.width % self.heads:
            raise ValueError("width must be divisible by heads and dimensions must be positive")
        if self.heads < 1 or self.layers < 1 or self.feedforward_mult < 1:
            raise ValueError("heads, layers and feedforward_mult must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")
        if self.action_semantics.lower() in {"dmx", "raw_dmx", "raw-dmx", "universe"}:
            raise ValueError("the SSL loop consumes semantic light state, not raw DMX")

    def as_dict(self) -> dict:
        return {**asdict(self), "schema_version": 1,
                "inference_contract": "audio+body+semantic_light_state",
                "native_body_contract": "audited 30 Hz body; no replacement generation"}


def _time_features(timestamps: torch.Tensor) -> torch.Tensor:
    """Build clock features only from the caller's physical timestamps."""
    seconds = timestamps.to(dtype=torch.float32)
    return torch.stack((seconds / 60.0,
                        torch.sin(2.0 * torch.pi * seconds / 60.0),
                        torch.cos(2.0 * torch.pi * seconds / 60.0)), dim=-1)


def _validate_inputs(audio: torch.Tensor, body: torch.Tensor, light: torch.Tensor,
                     timestamps: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if audio.ndim != 3 or body.ndim != 3 or light.ndim != 3:
        raise ValueError("audio, body and light must be B,T,D tensors")
    if audio.shape[:2] != body.shape[:2] or audio.shape[:2] != light.shape[:2]:
        raise ValueError("audio, body and light must share B,T")
    if timestamps.shape != audio.shape[:2]:
        raise ValueError("timestamps must be B,T; an external physical clock is required")
    if not torch.isfinite(audio).all() or not torch.isfinite(body).all() or not torch.isfinite(light).all():
        raise ValueError("inputs must be finite")
    if not torch.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if valid is None:
        valid = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
    if valid.shape != audio.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be a B,T bool tensor")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("each sequence needs at least one valid frame")
    return valid


def make_mask(valid: torch.Tensor, ratio: float, *, generator: torch.Generator | None = None,
              block: int = 4) -> torch.Tensor:
    """Make reproducible temporal span masks without touching invalid frames."""
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be B,T bool")
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be in (0, 1)")
    if block < 1:
        raise ValueError("block must be positive")
    b, t = valid.shape
    # Sampling on CPU keeps a caller-supplied generator reproducible across CPU
    # and CUDA smoke runs; the result is copied back to the input device.
    random = torch.rand((b, t), generator=generator, device="cpu")
    # Sample span starts at ratio/block so the expanded mask remains close to
    # the requested coverage instead of growing toward one for long spans.
    starts = random < min(1.0, ratio / max(block, 1))
    mask = starts.clone()
    if block > 1 and t > 1:
        expanded = starts.clone()
        for offset in range(1, block):
            expanded[:, offset:] |= starts[:, :-offset]
        mask = expanded
    mask &= valid.cpu()
    for row in range(b):
        if valid[row].any() and not mask[row].any():
            first = int(torch.nonzero(valid[row], as_tuple=False)[0])
            mask[row, first] = True
    return mask.to(valid.device)


class _TemporalEncoder(nn.Module):
    def __init__(self, config: ControlledJepaConfig):
        super().__init__()
        input_dim = config.audio_dim + config.body_dim + config.light_dim
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, config.width), nn.SiLU())
        self.clock = nn.Sequential(nn.Linear(3, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.feedforward_mult * config.width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.embedding_dim)

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor, valid: torch.Tensor,
                *, causal: bool = False) -> torch.Tensor:
        h = self.input(x) + self.clock(_time_features(timestamps).to(dtype=x.dtype))
        t = x.shape[1]
        attn_mask = None
        if causal:
            attn_mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=x.device), diagonal=1)
        key_padding = ~valid
        # Avoid an all-masked row in padded examples. Its value is discarded
        # after the encoder, while the valid history remains well-defined.
        key_padding = key_padding.clone()
        key_padding[:, 0] = False
        h = self.transformer(h, mask=attn_mask, src_key_padding_mask=key_padding)
        h = self.output(self.norm(h))
        return torch.where(valid[..., None], h, torch.zeros_like(h))


class ControlledJepaModel(nn.Module):
    """Student/target encoders with masked and action-conditioned predictions."""

    def __init__(self, config: ControlledJepaConfig):
        super().__init__()
        self.config = config
        self.context_encoder = _TemporalEncoder(config)
        self.target_encoder = deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.embedding_dim), nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.future_predictor = nn.Sequential(
            nn.LayerNorm(config.embedding_dim + config.light_dim),
            nn.Linear(config.embedding_dim + config.light_dim, config.width), nn.SiLU(),
            nn.Linear(config.width, config.embedding_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(config.body_dim + config.light_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def _pack(self, audio: torch.Tensor, body: torch.Tensor, light: torch.Tensor,
              mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            if mask.shape != audio.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("mask must be B,T bool")
            body_light = torch.cat((body, light), dim=-1).clone()
            body_light = torch.where(mask[..., None], self.mask_token.to(body_light), body_light)
        else:
            body_light = torch.cat((body, light), dim=-1)
        return torch.cat((audio, body_light), dim=-1)

    def forward(self, audio: torch.Tensor, body: torch.Tensor, light: torch.Tensor,
                timestamps: torch.Tensor, valid: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        valid = _validate_inputs(audio, body, light, timestamps, valid)
        if mask is None:
            mask = torch.zeros_like(valid)
        if mask.shape != valid.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be B,T bool")
        mask = mask & valid
        masked_input = self._pack(audio, body, light, mask)
        clean_input = self._pack(audio, body, light, None)
        context = self.context_encoder(masked_input, timestamps, valid, causal=False)
        with torch.no_grad():
            target = self.target_encoder(clean_input, timestamps, valid, causal=False)
        predicted = self.predictor(context)

        # The causal branch only sees history up to t and the observed semantic
        # action at t; it predicts the target latent at t+1.  No future body or
        # future light is allowed into this path.
        causal_context = self.context_encoder(masked_input, timestamps, valid, causal=True)
        future_input = torch.cat((causal_context[:, :-1], light[:, :-1]), dim=-1)
        future_predicted = self.future_predictor(future_input)
        return {
            "context": context,
            "causal_context": causal_context,
            "target": target,
            "predicted": predicted,
            "future_predicted": future_predicted,
            "future_target": target[:, 1:].detach(),
            "mask": mask,
            "valid": valid,
        }

    @torch.no_grad()
    def ema_update(self, decay: float | None = None) -> None:
        decay = self.config.ema_decay if decay is None else float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        for target, source in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)

    def config_dict(self) -> dict:
        return {"architecture": "masked_temporal_jepa+causal_action_predictor", **self.config.as_dict()}
