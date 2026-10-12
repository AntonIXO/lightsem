"""Small causal body/audio -> rig-independent light model.

The model deliberately predicts an observable group-level light state.  It does
not invent pan/tilt supervision from RPMC-L2 histograms; a later capability
renderer is responsible for expanding the state to DMX.  The multiplicative
fusion block is kept explicit so it can be compared with its nearest additive
projection during evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class BodyToDMXConfig:
    body_dim: int = 144
    # BeatSaver's causal operator features are 52-D; the native LightSem
    # 92-channel bus can be added in a later adapter without changing targets.
    audio_dim: int = 52
    d_model: int = 128
    layers: int = 3
    kernel: int = 7
    scene_dim: int = 6
    event_dim: int = 1
    angle_dim: int = 12                 # six canonical moving-head pan/tilt pairs


class CausalBlock(nn.Module):
    def __init__(self, d: int, kernel: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, kernel, groups=d)
        self.mix = nn.Conv1d(d, 2 * d, 1)
        self.out = nn.Conv1d(d, d, 1)
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is B,T,D.  Left padding is structural, so future perturbations
        # cannot affect an earlier output.
        y = self.norm(x).transpose(1, 2)
        y = F.pad(y, (self.kernel - 1, 0))
        y = self.conv(y)
        y = F.glu(self.mix(y), dim=1)
        y = self.out(y).transpose(1, 2)
        return x + y


class BodyToDMX(nn.Module):
    """Causal multiplicative fusion with scene and authored-event heads."""

    def __init__(self, cfg: BodyToDMXConfig | None = None):
        super().__init__()
        self.cfg = cfg or BodyToDMXConfig()
        d = self.cfg.d_model
        self.body = nn.Sequential(nn.LayerNorm(self.cfg.body_dim), nn.Linear(self.cfg.body_dim, d), nn.SiLU())
        self.audio = nn.Sequential(nn.LayerNorm(self.cfg.audio_dim), nn.Linear(self.cfg.audio_dim, d), nn.SiLU())
        self.additive = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU())
        self.mul_a = nn.Linear(d, d)
        self.mul_b = nn.Linear(d, d)
        self.fuse = nn.Linear(3 * d, d)
        self.blocks = nn.ModuleList([CausalBlock(d, self.cfg.kernel) for _ in range(self.cfg.layers)])
        self.norm = nn.LayerNorm(d)
        self.scene = nn.Linear(d, self.cfg.scene_dim)
        self.event = nn.Linear(d, self.cfg.event_dim)
        self.angle = nn.Linear(d, self.cfg.angle_dim)

    def encode(self, body: torch.Tensor, audio: torch.Tensor, *, additive: bool = False) -> torch.Tensor:
        b, a = self.body(body), self.audio(audio)
        if additive:
            h = self.additive(torch.cat([b, a], dim=-1))
        else:
            # Low-rank bilinear interaction without materialising d*d tensors.
            h = self.fuse(torch.cat([b, a, torch.tanh(self.mul_a(b)) * torch.sigmoid(self.mul_b(a))], dim=-1))
        for block in self.blocks:
            h = block(h)
        return self.norm(h)

    def forward(self, body: torch.Tensor, audio: torch.Tensor, *, additive: bool = False) -> dict[str, torch.Tensor]:
        h = self.encode(body, audio, additive=additive)
        return {"h": h, "scene": self.scene(h), "event_logit": self.event(h), "angle_logit": self.angle(h)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def causality_probe(model: BodyToDMX, body: torch.Tensor, audio: torch.Tensor, cut: int) -> float:
    model.eval()
    ref = model(body, audio)["scene"]
    body2, audio2 = body.clone(), audio.clone()
    body2[:, cut:] = torch.randn_like(body2[:, cut:])
    audio2[:, cut:] = torch.randn_like(audio2[:, cut:])
    changed = model(body2, audio2)["scene"]
    return float((ref[:, :cut] - changed[:, :cut]).abs().max().cpu())


def masked_loss(pred: dict[str, torch.Tensor], scene: torch.Tensor, event: torch.Tensor,
                scene_mask: torch.Tensor | None = None, angle: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if scene_mask is None:
        scene_mask = torch.ones(scene.shape[:-1], dtype=torch.bool, device=scene.device)
    m = scene_mask.unsqueeze(-1).expand_as(scene)
    scene_l = F.smooth_l1_loss(pred["scene"][m], scene[m]) if m.any() else scene.sum() * 0
    event_l = F.binary_cross_entropy_with_logits(pred["event_logit"].squeeze(-1), event.float())
    if angle is None:
        angle_l = scene.sum() * 0
    else:
        angle_l = F.mse_loss(torch.sigmoid(pred["angle_logit"]), angle.float())
    return {"scene": scene_l, "event": event_l, "angle": angle_l, "total": scene_l + 0.25 * event_l + 0.5 * angle_l}

