"""Streaming local-attention + recurrent-state + latent-memory backbone."""

from dataclasses import dataclass
from typing import NamedTuple
import torch
from torch import nn


@dataclass(frozen=True)
class LongContextConfig:
    width: int = 96
    heads: int = 4
    local_window: int = 32
    latent_slots: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        if self.width < 8 or self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.local_window < 1 or self.latent_slots < 1:
            raise ValueError("window and latent slots must be positive")


class MemoryState(NamedTuple):
    recurrent: torch.Tensor
    latent: torch.Tensor


class LongContextMemory(nn.Module):
    """A bounded streaming context operator.

    Local attention is restricted to each supplied chunk.  A GRU state and a
    fixed latent array cross chunk boundaries, so chunked inference is not an
    independent reset at every window.  ``forward`` accepts a prior state and
    returns the new state when ``return_state`` is true.
    """

    def __init__(self, config: LongContextConfig):
        super().__init__()
        self.config = config
        self.local = nn.MultiheadAttention(config.width, config.heads,
                                           dropout=config.dropout, batch_first=True)
        self.norm_local = nn.LayerNorm(config.width)
        # A GRUCell loop is intentionally used instead of a fused GRU kernel:
        # the same sequence of operations must be reproducible when inference
        # is split at chunk boundaries.
        self.recurrent = nn.GRUCell(config.width, config.width)
        self.norm_recurrent = nn.LayerNorm(config.width)
        self.latent_seed = nn.Parameter(torch.randn(config.latent_slots, config.width) * 0.02)
        self.to_latent = nn.MultiheadAttention(config.width, config.heads,
                                               dropout=config.dropout, batch_first=True)
        self.from_latent = nn.MultiheadAttention(config.width, config.heads,
                                                 dropout=config.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(config.width), nn.Linear(config.width, 4 * config.width),
            nn.GELU(), nn.Linear(4 * config.width, config.width)
        )

    def initial_state(self, batch_size, *, device=None, dtype=None):
        device = device or self.latent_seed.device
        dtype = dtype or self.latent_seed.dtype
        recurrent = torch.zeros(1, batch_size, self.config.width, device=device, dtype=dtype)
        latent = self.latent_seed.to(device=device, dtype=dtype)[None].expand(batch_size, -1, -1).clone()
        return MemoryState(recurrent, latent)

    def _local_attention(self, x, valid):
        t = x.shape[1]
        causal = torch.triu(torch.ones((t, t), dtype=torch.bool, device=x.device), diagonal=1)
        key_padding = ~valid
        # At least one valid key avoids an all-masked softmax NaN on padded rows.
        key_padding = key_padding.clone()
        key_padding[:, 0] = False
        attended, _ = self.local(x, x, x, attn_mask=causal,
                                 key_padding_mask=key_padding, need_weights=False)
        return torch.where(valid[..., None], self.norm_local(x + attended), torch.zeros_like(x))

    def forward(self, x, valid=None, state=None, *, return_state=False):
        if x.ndim != 3 or x.shape[-1] != self.config.width:
            raise ValueError("expected B,T,width context tokens")
        b, t, _ = x.shape
        if valid is None:
            valid = torch.ones((b, t), dtype=torch.bool, device=x.device)
        if valid.shape != (b, t) or valid.dtype != torch.bool:
            raise ValueError("valid must be B,T bool")
        if state is None:
            state = self.initial_state(b, device=x.device, dtype=x.dtype)
        if state.recurrent.shape != (1, b, self.config.width):
            raise ValueError("recurrent state has the wrong shape")
        if state.latent.shape != (b, self.config.latent_slots, self.config.width):
            raise ValueError("latent state has the wrong shape")
        # The caller controls chunk size.  We retain a strict local window by
        # splitting oversized input, while preserving one state across chunks.
        outputs = []
        recurrent = state.recurrent
        latent = state.latent
        for start in range(0, t, self.config.local_window):
            end = min(t, start + self.config.local_window)
            chunk = x[:, start:end]
            chunk_valid = valid[:, start:end]
            local = self._local_attention(chunk, chunk_valid)
            hidden = recurrent[0]
            recurrent_steps = []
            for step in range(local.shape[1]):
                proposed = self.recurrent(local[:, step], hidden)
                valid_step = chunk_valid[:, step, None]
                hidden = torch.where(valid_step, proposed, hidden)
                recurrent_steps.append(hidden)
            recurrent_out = torch.stack(recurrent_steps, 1)
            recurrent = hidden[None]
            recurrent_out = torch.where(chunk_valid[..., None], self.norm_recurrent(recurrent_out), torch.zeros_like(recurrent_out))
            latent_delta, _ = self.to_latent(latent, recurrent_out, recurrent_out,
                                             key_padding_mask=~chunk_valid, need_weights=False)
            latent = latent + torch.tanh(latent_delta)
            latent_context, _ = self.from_latent(recurrent_out, latent, latent, need_weights=False)
            out = recurrent_out + latent_context + self.ff(recurrent_out)
            outputs.append(torch.where(chunk_valid[..., None], out, torch.zeros_like(out)))
        result = torch.cat(outputs, dim=1) if outputs else x.new_zeros(x.shape)
        new_state = MemoryState(recurrent, latent)
        return (result, new_state) if return_state else result
