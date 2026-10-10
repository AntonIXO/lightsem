"""Compact Stem-JEPA-style source-query predictor.

This is a LightSem reimplementation probe, not a claim of reproducing the
original Stem-JEPA training corpus.  It predicts a held-out source's local
semantic target from the mixture and a source-ID query, with explicit source
activity and stereo-image heads kept separate from the compatibility loss.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import _TemporalTower


STEM_JEPA_SOURCE_DIM = 9  # log energy, activity, L/R onsets, ILD/width/pan/coherence/log energy


@dataclass(frozen=True)
class StemJepaConfig:
    arm: str = "stem_jepa_lite"
    input_dim: int = 29
    query_dim: int = 4
    source_dim: int = STEM_JEPA_SOURCE_DIM
    width: int = 64
    embedding_dim: int = 64

    def __post_init__(self) -> None:
        if self.arm != "stem_jepa_lite" or min(self.input_dim, self.query_dim,
                                               self.source_dim, self.width, self.embedding_dim) <= 0:
            raise ValueError("Invalid StemJepa configuration")


class StemJepaLite(nn.Module):
    def __init__(self, config: StemJepaConfig = StemJepaConfig()):
        super().__init__()
        self.config = config
        self.tower = _TemporalTower(config.input_dim + config.query_dim, config.width, depth=4)
        self.embedding = nn.Linear(config.width, config.embedding_dim)
        self.predictor = nn.Sequential(nn.Linear(config.embedding_dim, config.embedding_dim), nn.SiLU(),
                                       nn.Linear(config.embedding_dim, config.embedding_dim))
        self.target_projection = nn.Linear(config.source_dim, config.embedding_dim)
        self.source_head = nn.Linear(config.width, config.source_dim)
        self.activity_head = nn.Linear(config.width, 1)
        self.onset_head = nn.Linear(config.width, 2)
        self.spatial_head = nn.Linear(config.width, 5)

    def forward(self, features: torch.Tensor, query: torch.Tensor,
                valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or query.shape[:2] != features.shape[:2] or query.shape[-1] != self.config.query_dim:
            raise ValueError("StemJepa inputs must be B,T,F and B,T,source_query")
        x = torch.cat((features, query), dim=-1)
        hidden = self.tower(x, valid)
        embedding = self.embedding(hidden)
        return {
            "embedding": embedding,
            "prediction": self.predictor(embedding),
            "source": self.source_head(hidden),
            "activity_logits": self.activity_head(hidden).squeeze(-1),
            "onset_logits": self.onset_head(hidden),
            "spatial": self.spatial_head(hidden),
        }

    def config_dict(self) -> dict:
        return {**asdict(self.config), "schema_version": 1,
                "source_target": "MUSDB stem-derived local semantic target",
                "spatial_semantics": "stereo image descriptors, not physical 3D location"}


def stem_jepa_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor,
                   valid: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if valid is None:
        valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    mask = valid[..., None]
    source_loss = ((outputs["source"] - target) ** 2).masked_fill(~mask, 0.).sum() / mask.sum().clamp_min(1)
    # The target embedding is a normalized source-semantic target; prediction
    # is separate from the semantic heads so compatibility cannot replace
    # observed-source detection.
    embedding_dim = outputs["prediction"].shape[-1]
    if target.shape[-1] > embedding_dim:
        raise ValueError("Stem target width exceeds embedding width")
    target_embedding = F.normalize(F.pad(target, (0, embedding_dim - target.shape[-1])), dim=-1)
    prediction_loss = ((F.normalize(outputs["prediction"], dim=-1) - target_embedding) ** 2).masked_fill(~mask, 0.).sum() / mask.sum().clamp_min(1)
    activity = target[..., 1]
    activity_loss = F.binary_cross_entropy_with_logits(outputs["activity_logits"], activity,
                                                       reduction="none").masked_fill(~valid, 0.).sum() / valid.sum().clamp_min(1)
    onset_target = target[..., 2:4]
    onset_loss = F.binary_cross_entropy_with_logits(outputs["onset_logits"], onset_target,
                                                    reduction="none").masked_fill(~mask, 0.).sum() / mask.sum().clamp_min(1)
    spatial_loss = F.mse_loss(outputs["spatial"], target[..., 4:9])
    total = source_loss + .5 * prediction_loss + .5 * activity_loss + .5 * onset_loss + .5 * spatial_loss
    return total, {"source": source_loss, "prediction": prediction_loss,
                   "activity": activity_loss, "onset": onset_loss,
                   "spatial": spatial_loss, "total": total}
