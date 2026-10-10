"""Small, versioned stereo representation arms for the six-hour pilot.

The model is intentionally compact: MuQ remains frozen and supplies the
existing content representation, while this module learns only the additional
stereo/event/spatial interface.  ``StereoNativeAdapter`` is zero initialized,
so attaching it to a native-C checkpoint is an exact no-op until it is trained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ...audio.stereo_v1 import EVENT_NAMES, FEATURE_NAMES, RELATION_NAMES, SPATIAL_NAMES


@dataclass(frozen=True)
class StereoConditionerConfig:
    arm: str = "factorized"
    input_dim: int = len(FEATURE_NAMES)
    event_dim: int = len(EVENT_NAMES)
    spatial_dim: int = len(SPATIAL_NAMES)
    relation_dim: int = len(RELATION_NAMES)
    width: int = 64
    fused_dim: int = 64
    spatial_hop: int = 4  # 40Hz -> 10Hz
    relation_hop: int = 20  # 40Hz -> 2Hz

    def __post_init__(self) -> None:
        if self.arm not in ("fused", "factorized"):
            raise ValueError("Stereo arm must be fused or factorized")
        if min(self.input_dim, self.event_dim, self.spatial_dim, self.relation_dim,
               self.width, self.fused_dim) <= 0:
            raise ValueError("Stereo dimensions must be positive")
        if self.spatial_hop < 1 or self.relation_hop < self.spatial_hop:
            raise ValueError("Invalid stereo stream rates")


class _TemporalTower(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int = 3):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 3, padding=1) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))

    def forward(self, value: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.input.in_features:
            raise ValueError("Expected B,T,stereo feature tensor")
        x = F.silu(self.input(value))
        for block, norm in zip(self.blocks, self.norms):
            residual = x
            x = block(F.silu(x).transpose(1, 2)).transpose(1, 2)
            x = norm(residual + .2 * x)
        if valid is not None:
            if valid.shape != value.shape[:2] or valid.dtype != torch.bool:
                raise ValueError("Invalid stereo validity mask")
            x = torch.where(valid[..., None], x, 0.)
        return x


class StereoConditioner(nn.Module):
    """R1 fused or R2 factorized event/spatial stereo conditioner."""

    def __init__(self, config: StereoConditionerConfig = StereoConditionerConfig()):
        super().__init__()
        self.config = config
        self.shared = _TemporalTower(config.input_dim, config.width)
        self.event_tower = _TemporalTower(config.width, config.width, depth=2)
        self.spatial_tower = _TemporalTower(config.width, config.width, depth=2)
        self.event_head = nn.Linear(config.width, config.event_dim)
        self.spatial_head = nn.Linear(config.width, config.spatial_dim)
        self.fuse = nn.Linear(config.width * 2, config.fused_dim)
        self.relation_head = nn.Linear(config.fused_dim, config.relation_dim)

    def forward(self, features: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("Expected B,T,stereo features")
        if valid is None:
            valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        shared = self.shared(features, valid)
        if self.config.arm == "fused":
            # R1 has one shared temporal representation and separate readout
            # heads.  It is the capacity-matched early-fusion control.
            event_hidden = shared
            spatial_hidden = shared
        else:
            # R2 keeps distinct temporal towers for fast events and slower
            # spatial/phrase evidence before the cross-stream fusion layer.
            event_hidden = self.event_tower(shared, valid)
            spatial_hidden = self.spatial_tower(shared, valid)
        event = self.event_head(event_hidden)
        spatial_hidden_10 = F.avg_pool1d(
            spatial_hidden.transpose(1, 2), self.config.spatial_hop,
            self.config.spatial_hop, ceil_mode=True).transpose(1, 2)
        spatial = self.spatial_head(spatial_hidden_10)
        spatial_up = F.interpolate(spatial_hidden_10.transpose(1, 2), size=features.shape[1],
                                   mode="linear", align_corners=False).transpose(1, 2)
        fused = F.silu(self.fuse(torch.cat((event_hidden, spatial_up), dim=-1)))
        relation_hidden = F.avg_pool1d(
            fused.transpose(1, 2), self.config.relation_hop,
            self.config.relation_hop, ceil_mode=True).transpose(1, 2)
        relation = self.relation_head(relation_hidden)
        return {
            "event": event,
            "spatial": spatial,
            "relation": relation,
            "fused": fused,
            "event_hidden": event_hidden,
            "spatial_hidden": spatial_hidden_10,
        }

    def config_dict(self) -> dict:
        return {**asdict(self.config), "schema_version": 1,
                "feature_names": list(FEATURE_NAMES), "event_names": list(EVENT_NAMES),
                "spatial_names": list(SPATIAL_NAMES), "relation_names": list(RELATION_NAMES),
                "spatial_semantics": "stereo image descriptors, not physical 3D location"}


class StereoNativeAdapter(nn.Module):
    """Zero-initialized projections for native planner and flow conditioning."""

    def __init__(self, fused_dim: int, flow_width: int, planner_dim: int = 80,
                 relation_dim: int = len(RELATION_NAMES)):
        super().__init__()
        if min(fused_dim, flow_width, planner_dim, relation_dim) <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.fused_dim = fused_dim
        self.flow_width = flow_width
        self.planner_dim = planner_dim
        self.relation_dim = relation_dim
        self.flow = nn.Linear(fused_dim, flow_width)
        self.planner = nn.Linear(relation_dim, planner_dim)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)
        nn.init.zeros_(self.planner.weight)
        nn.init.zeros_(self.planner.bias)

    def forward(self, streams: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused = streams.get("fused")
        relation = streams.get("relation")
        if fused is None or relation is None:
            raise ValueError("Stereo streams must contain fused and relation tensors")
        if fused.ndim != 3 or relation.ndim != 3:
            raise ValueError("Stereo streams must be B,T,C")
        if fused.shape[-1] != self.fused_dim or relation.shape[-1] != self.relation_dim:
            raise ValueError("Stereo adapter input width mismatch")
        return {"flow_bias": self.flow(fused), "planner_context": self.planner(relation)}

    def config_dict(self) -> dict:
        return {"schema_version": 1, "fused_dim": self.fused_dim,
                "flow_width": self.flow_width, "planner_dim": self.planner_dim,
                "relation_dim": self.relation_dim, "zero_initialized": True}


def _match(value: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(value.shape[1], target.shape[1])
    return value[:, :length], target[:, :length]


def conditioner_loss(outputs: dict[str, torch.Tensor], events: torch.Tensor,
                     spatial: torch.Tensor, relation: torch.Tensor,
                     valid: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Descriptor reconstruction loss used for the compact pilot probes.

    These are representation-engineering losses, not source truth.  A future
    supervised arm can replace the targets with authenticated stem/event labels
    while retaining the same stream contract.
    """
    if valid is None:
        valid = torch.ones(events.shape[:2], dtype=torch.bool, device=events.device)
    event_pred, event_target = _match(outputs["event"], events)
    event_valid = valid[:, :event_pred.shape[1]][..., None]
    spatial_pred, spatial_target = _match(outputs["spatial"], spatial)
    relation_pred, relation_target = _match(outputs["relation"], relation)
    event_loss = ((event_pred - event_target) ** 2).masked_fill(~event_valid, 0.).sum() / event_valid.sum().clamp_min(1)
    spatial_loss = F.mse_loss(spatial_pred, spatial_target)
    relation_loss = F.mse_loss(relation_pred, relation_target)
    total = event_loss + .75 * spatial_loss + .5 * relation_loss
    return total, {"event": event_loss, "spatial": spatial_loss, "relation": relation_loss, "total": total}
