"""Source-aware stereo semantics for the spatial-conditioning pilot.

The semantic branch is trained against authenticated MUSDB stem-derived
targets.  It learns to infer source activity and source stereo image from the
mastered mixture, then fuses fast events, slow spatial evidence, and pairwise
source relations into a compact context.  The expensive stem analysis is an
offline teacher/target construction step; the native dancer receives only
the cached context tensor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import _TemporalTower


STEM_NAMES = ("drums", "bass", "vocals", "other")
SOURCE_EVENT_NAMES = tuple(
    f"{stem}_{name}" for stem in STEM_NAMES
    for name in ("log_energy", "active", "onset_left", "onset_right")
)
SOURCE_SPATIAL_NAMES = tuple(
    f"{stem}_{name}" for stem in STEM_NAMES
    for name in ("ild_db", "width_fraction", "pan_balance", "coherence", "log_energy")
)
SOURCE_RELATION_NAMES = tuple(
    f"{left}_{right}_{name}" for i, left in enumerate(STEM_NAMES)
    for right in STEM_NAMES[i + 1:]
    for name in ("coaccent", "lag_s", "lag_confidence")
)


@dataclass(frozen=True)
class SemanticStereoConditionerConfig:
    arm: str = "semantic_factorized"
    input_dim: int = 29
    source_event_dim: int = len(SOURCE_EVENT_NAMES)
    source_spatial_dim: int = len(SOURCE_SPATIAL_NAMES)
    source_relation_dim: int = len(SOURCE_RELATION_NAMES)
    width: int = 64
    context_dim: int = 64
    spatial_hop: int = 4
    relation_hop: int = 20

    def __post_init__(self) -> None:
        if self.arm != "semantic_factorized":
            raise ValueError("Semantic conditioner arm must be semantic_factorized")
        if min(self.input_dim, self.source_event_dim, self.source_spatial_dim,
               self.source_relation_dim, self.width, self.context_dim) <= 0:
            raise ValueError("Semantic conditioner dimensions must be positive")
        if self.spatial_hop < 1 or self.relation_hop < self.spatial_hop:
            raise ValueError("Invalid semantic stream rates")


class SemanticStereoConditioner(nn.Module):
    """Infer source-aware event, spatial, and relation streams from a mix."""

    def __init__(self, config: SemanticStereoConditionerConfig = SemanticStereoConditionerConfig()):
        super().__init__()
        self.config = config
        self.shared = _TemporalTower(config.input_dim, config.width)
        self.event_tower = _TemporalTower(config.width, config.width, depth=2)
        self.spatial_tower = _TemporalTower(config.width, config.width, depth=2)
        self.event_head = nn.Linear(config.width, config.source_event_dim)
        self.activity_head = nn.Linear(config.width, len(STEM_NAMES))
        self.onset_head = nn.Linear(config.width, len(STEM_NAMES) * 2)
        self.spatial_head = nn.Linear(config.width, config.source_spatial_dim)
        self.relation_head = nn.Linear(config.width, config.source_relation_dim)
        self.fuse = nn.Linear(config.width * 3, config.context_dim)

    def forward(self, features: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("Expected B,T,stereo features")
        if valid is None:
            valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        shared = self.shared(features, valid)
        event_hidden = self.event_tower(shared, valid)
        spatial_hidden = self.spatial_tower(shared, valid)
        event = self.event_head(event_hidden)
        spatial_hidden_10 = F.avg_pool1d(
            spatial_hidden.transpose(1, 2), self.config.spatial_hop,
            self.config.spatial_hop, ceil_mode=True).transpose(1, 2)
        spatial = self.spatial_head(spatial_hidden_10)
        activity_logits = self.activity_head(event_hidden)
        onset_logits = self.onset_head(event_hidden)
        spatial_up = F.interpolate(spatial_hidden_10.transpose(1, 2), size=features.shape[1],
                                   mode="linear", align_corners=False).transpose(1, 2)
        relation_hidden_2 = F.avg_pool1d(
            spatial_hidden.transpose(1, 2), self.config.relation_hop,
            self.config.relation_hop, ceil_mode=True).transpose(1, 2)
        relation = self.relation_head(relation_hidden_2)
        relation_up = F.interpolate(relation_hidden_2.transpose(1, 2), size=features.shape[1],
                                    mode="linear", align_corners=False).transpose(1, 2)
        fused = F.silu(self.fuse(torch.cat((event_hidden, spatial_up, relation_up), dim=-1)))
        return {
            "event": event,
            "activity_logits": activity_logits,
            "onset_logits": onset_logits,
            "spatial": spatial,
            "relation": relation,
            "fused": fused,
            "event_hidden": event_hidden,
            "spatial_hidden": spatial_hidden_10,
            "relation_hidden": relation_hidden_2,
        }

    def config_dict(self) -> dict:
        return {**asdict(self.config), "schema_version": 1,
                "source_event_names": list(SOURCE_EVENT_NAMES),
                "source_spatial_names": list(SOURCE_SPATIAL_NAMES),
                "source_relation_names": list(SOURCE_RELATION_NAMES),
                "spatial_semantics": "source stereo image inferred from mastered mixture"}


def _match(value: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(value.shape[1], target.shape[1])
    return value[:, :length], target[:, :length]


def semantic_conditioner_loss(outputs: dict[str, torch.Tensor], event_target: torch.Tensor,
                              spatial_target: torch.Tensor, relation_target: torch.Tensor,
                              valid: torch.Tensor | None = None,
                              activity_target: torch.Tensor | None = None,
                              onset_target: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked semantic regression with a separate binary source-activity term."""
    if valid is None:
        valid = torch.ones(event_target.shape[:2], dtype=torch.bool, device=event_target.device)
    event_pred, event_target = _match(outputs["event"], event_target)
    spatial_pred, spatial_target = _match(outputs["spatial"], spatial_target)
    relation_pred, relation_target = _match(outputs["relation"], relation_target)
    event_valid = valid[:, :event_pred.shape[1]][..., None]
    event_reg = ((event_pred - event_target) ** 2).masked_fill(~event_valid, 0.)
    event_reg = event_reg.sum() / event_valid.sum().clamp_min(1)
    active_index = torch.arange(event_target.shape[-1], device=event_target.device) % 4 == 1
    event_valid_full = event_valid.expand_as(event_pred)
    active_logits = outputs["activity_logits"][:, :event_pred.shape[1]]
    if activity_target is None:
        active_target = event_target[..., active_index].clamp(0., 1.)
    else:
        active_target = activity_target[:, :event_pred.shape[1]].clamp(0., 1.)
    active_valid = event_valid_full[..., :len(STEM_NAMES)]
    active_bce = F.binary_cross_entropy_with_logits(active_logits, active_target,
                                                    reduction="none").masked_fill(~active_valid, 0.)
    active_bce = active_bce.sum() / active_valid.sum().clamp_min(1)
    if onset_target is None:
        onset_target = (event_target.reshape(*event_target.shape[:-1], len(STEM_NAMES), 4)[..., 2:4]
                        .reshape(*event_target.shape[:-1], len(STEM_NAMES) * 2) > .5).float()
    onset_logits = outputs["onset_logits"][:, :event_pred.shape[1]]
    onset_valid = event_valid_full[..., :onset_logits.shape[-1]]
    onset_bce = F.binary_cross_entropy_with_logits(onset_logits, onset_target[:, :onset_logits.shape[1]],
                                                   reduction="none").masked_fill(~onset_valid, 0.)
    onset_bce = onset_bce.sum() / onset_valid.sum().clamp_min(1)
    spatial_loss = F.mse_loss(spatial_pred, spatial_target)
    relation_loss = F.mse_loss(relation_pred, relation_target)
    total = event_reg + .5 * active_bce + .75 * spatial_loss + .5 * relation_loss + .75 * onset_bce
    return total, {"event_regression": event_reg, "source_activity_bce": active_bce,
                   "onset_bce": onset_bce, "spatial": spatial_loss,
                   "relation": relation_loss, "total": total}
