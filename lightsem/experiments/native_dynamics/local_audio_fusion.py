"""Bounded local audio readout for motion-state queries at native 30 Hz.

The attention and FiLM controls have identical parameters and input support.
Audio keys/values are computed once per rollout, with global frame indices
preserved when the vector field is evaluated in chunks. No temporal oscillator,
output filter, or playback clock is introduced.
"""
import torch
from torch import nn
import torch.nn.functional as F


class LocalAudioFusion(nn.Module):
    OFFSETS = (-60, -30, -15, -5, 0, 5, 15, 30, 60)

    def __init__(self, audio_dim, motion_dim, width=64, heads=4, mode='attention'):
        super().__init__()
        if mode not in ('attention', 'film', 'hybrid') or heads < 1 or width % heads:
            raise ValueError('Invalid local fusion mode/dimensions')
        self.mode, self.width, self.heads = mode, width, heads
        self.audio = nn.Linear(audio_dim, width)
        self.query = nn.Linear(motion_dim, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.relative_position = nn.Parameter(torch.zeros(len(self.OFFSETS), width))
        self.output = nn.Linear(width * (2 if mode == 'hybrid' else 1), motion_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def prepare(self, audio, valid=None):
        if audio.ndim != 3 or audio.shape[1] == 0:
            raise ValueError('Expected nonempty B,T,F audio')
        if valid is None:
            valid = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        if valid.shape != audio.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('Expected Boolean B,T audio validity')
        clean = torch.where(valid[..., None], audio, 0.)
        h = F.silu(self.audio(clean))
        n = audio.shape[1]
        index = torch.arange(n, device=audio.device)[:, None] + audio.new_tensor(self.OFFSETS, dtype=torch.long)[None]
        inside = (index >= 0) & (index < n)
        index = index.clamp(0, n - 1)
        mask = inside[None] & valid[:, index] & valid[:, :, None]
        key = self.key(h)[:, index] + self.relative_position[None, None]
        value = self.value(h)[:, index]
        # B,T,H,K,D, including mask before a dot product can encounter padding.
        shape = (*key.shape[:3], self.heads, self.width // self.heads)
        key = torch.where(mask[..., None], key, 0.).reshape(shape).transpose(2, 3)
        value = torch.where(mask[..., None], value, 0.).reshape(shape).transpose(2, 3)
        return key, value, mask

    def forward(self, motion, context):
        key, value, mask = context
        if motion.shape[:2] != mask.shape[:2]:
            raise ValueError('Motion and cached audio clocks differ')
        query = self.query(motion).reshape(*motion.shape[:2], self.heads, 1, self.width // self.heads)
        attended = F.scaled_dot_product_attention(query, key, value,
            attn_mask=mask[:, :, None, None, :], dropout_p=0.).squeeze(-2)
        if self.mode in ('attention', 'hybrid'):
            # SDPA returns zero for all-masked rows; suppress output bias too.
            attention_hidden = attended
        else:
            attention_hidden = None
        if self.mode in ('film', 'hybrid'):
            count = mask.sum(-1).clamp_min(1)[:, :, None, None]
            shift = value.sum(-2) / count
            scale = key.sum(-2) / count
            film_hidden = torch.tanh(query.squeeze(-2)) * torch.sigmoid(scale) + shift
        else:
            film_hidden = None
        hidden = attention_hidden if self.mode == 'attention' else film_hidden
        if self.mode == 'hybrid':
            hidden = torch.cat((attention_hidden, film_hidden), -1)
        result = self.output(hidden.flatten(-2))
        return torch.where(mask.any(-1, keepdim=True), result, 0.)
