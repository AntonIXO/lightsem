"""Multirate audio projection used by the continuous-attention pilot.

The encoder accepts features that are already extracted by the repository's
pinned audio pipeline.  It adds explicit rate and clock embeddings, rather
than silently resampling or replacing MuQ.  A caller can provide high-rate,
event-rate and phrase-rate streams; the pilot uses one aligned stream and
keeps the other streams optional.
"""

from dataclasses import dataclass
import torch
from torch import nn
from .peft import AdapterConfig, build_temporal_adapter


@dataclass(frozen=True)
class AudioEncoderConfig:
    input_dim: int
    width: int = 96
    dropout: float = 0.0
    local_rate_hz: float = 10.0
    phrase_rate_hz: float = 1.0
    adapter_kind: str = "none"
    adapter_bottleneck: int = 32
    adapter_heads: int = 4
    adapter_local_window: int = 32

    def __post_init__(self):
        if self.input_dim < 1 or self.width < 8:
            raise ValueError("audio dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.local_rate_hz <= 0 or self.phrase_rate_hz <= 0:
            raise ValueError("rates must be positive")


class AudioEncoder(nn.Module):
    """Typed projection of frozen audio features.

    ``forward`` returns ``B,T,width`` and preserves the supplied timestamps.
    The timestamp channels are normalized physical seconds, not a playback
    clock; the model cannot invent a separate temporal source from them.
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width, config.width),
        )
        self.rate_embedding = nn.Parameter(torch.zeros(3, config.width))
        nn.init.normal_(self.rate_embedding, std=0.02)
        self.timestamp = nn.Sequential(
            nn.Linear(4, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        self.temporal_adapter = build_temporal_adapter(AdapterConfig(
            kind=config.adapter_kind,
            width=config.width,
            bottleneck=config.adapter_bottleneck,
            heads=config.adapter_heads,
            local_window=config.adapter_local_window,
        ))

    def forward(self, features, *, timestamps=None, rate="local", valid=None, duration_seconds=None):
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("expected B,T,input_dim audio features")
        b, t, _ = features.shape
        if valid is None:
            valid = torch.ones((b, t), dtype=torch.bool, device=features.device)
        if valid.shape != (b, t) or valid.dtype != torch.bool:
            raise ValueError("valid must be B,T bool")
        if timestamps is None:
            timestamps = torch.arange(t, device=features.device, dtype=features.dtype)
            timestamps = timestamps[None].expand(b, -1) / self.config.local_rate_hz
        if timestamps.shape != (b, t):
            raise ValueError("timestamps must be B,T")
        rate_index = {"local": 0, "event": 1, "phrase": 2}.get(rate)
        if rate_index is None:
            raise ValueError("rate must be local, event or phrase")
        seconds = timestamps.float()
        if duration_seconds is None:
            duration = seconds[:, -1:].clamp_min(1.0)
        else:
            duration = torch.as_tensor(duration_seconds, device=seconds.device, dtype=seconds.dtype)
            duration = duration.reshape(1, 1).expand(b, 1).clamp_min(1.0)
        phase = seconds / duration
        clock = torch.stack((
            seconds / 60.0,
            torch.sin(2 * torch.pi * phase),
            torch.cos(2 * torch.pi * phase),
            phase,
        ), -1).to(features.dtype)
        out = self.projection(features) + self.timestamp(clock)
        out = out + self.rate_embedding[rate_index].view(1, 1, -1)
        return self.temporal_adapter(out, valid)
