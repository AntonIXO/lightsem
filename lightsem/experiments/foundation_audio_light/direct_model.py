"""Audio-only bounded direct music-to-light model.

The spatial prior comes only from fixture metadata.  It is a rig geometry
prior, not a second motion source.  Temporal changes are produced by the audio
backbone and are passed through the differentiable bounded-dynamics projection
before they can reach a mapper or renderer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from ..audio_light_foundation.model import AudioToLightFoundation, AudioToLightConfig
from .safety_projection import (
    SafetyConfig,
    bounded_dynamics_projection,
    smooth_bounded_dynamics_projection,
)


@dataclass(frozen=True)
class DirectLightConfig:
    foundation: AudioToLightConfig = AudioToLightConfig()
    anchor_pan_scale: float = 0.90
    anchor_tilt_scale: float = 0.25
    residual_gain: float = 0.10
    safety: SafetyConfig = SafetyConfig(dt=0.1, max_angle=0.4, max_velocity=0.18,
                                       max_acceleration=0.5, max_jerk=1.5)
    dynamics_mode: str = "hard_clipped"
    dynamics_drive_scale: float = 4.0
    dynamics_position_gain: float = 1.0
    dynamics_velocity_gain: float = 2.0
    dynamics_acceleration_gain: float = 2.0
    dynamics_radial_bounds: bool = False


class AudioOnlyDirectLightModel(nn.Module):
    """Direct audio→fixture controls with no body input at inference."""

    def __init__(self, config: DirectLightConfig = DirectLightConfig()):
        super().__init__()
        self.config = config
        self.foundation = AudioToLightFoundation(config.foundation)

    def _anchor(self, fixtures: torch.Tensor) -> torch.Tensor:
        if fixtures.ndim != 3 or fixtures.shape[-1] < 2:
            raise ValueError("fixtures must be B,N,F with at least x/y metadata")
        pan = torch.atan2(fixtures[..., 1], fixtures[..., 0]) / torch.pi
        tilt = fixtures[..., 1]
        return torch.stack((pan * self.config.anchor_pan_scale,
                            tilt * self.config.anchor_tilt_scale), dim=-1)

    def forward(self, audio: torch.Tensor, fixtures: torch.Tensor, *, timestamps=None,
                valid=None, state=None, duration_seconds=None):
        if isinstance(state, dict):
            foundation_state = state.get("foundation")
            motion_state = state.get("motion")
        else:
            foundation_state = state
            motion_state = None
        out = self.foundation(audio, fixtures, timestamps=timestamps, valid=valid,
                              state=foundation_state, duration_seconds=duration_seconds)
        raw = out["aim"] * float(self.config.residual_gain)
        if self.config.dynamics_mode == "hard_clipped":
            projection = bounded_dynamics_projection
        elif self.config.dynamics_mode == "smooth_jerk":
            projection = smooth_bounded_dynamics_projection
        else:
            raise ValueError(f"unknown dynamics_mode={self.config.dynamics_mode!r}")
        (delta_angle, velocity, acceleration, jerk), new_motion_state = projection(
            raw,
            motion_state,
            self.config.safety,
            **({
                "drive_scale": self.config.dynamics_drive_scale,
                "position_gain": self.config.dynamics_position_gain,
                "velocity_gain": self.config.dynamics_velocity_gain,
                "acceleration_gain": self.config.dynamics_acceleration_gain,
                "radial_bounds": self.config.dynamics_radial_bounds,
            } if self.config.dynamics_mode == "smooth_jerk" else {}),
        )
        anchor = self._anchor(fixtures)
        out["raw_aim"] = out["aim"]
        out["anchor"] = anchor
        out["aim"] = anchor[:, None] + delta_angle
        out["velocity"] = velocity
        out["acceleration"] = acceleration
        out["jerk"] = jerk
        out["state"] = {"foundation": out["state"], "motion": new_motion_state}
        return out

    def config_dict(self) -> dict[str, Any]:
        return {
            "architecture": "audio_only+hierarchical_adapter+fixture_anchor+bounded_dynamics",
            "foundation": self.foundation.config_dict(),
            "anchor_pan_scale": self.config.anchor_pan_scale,
            "anchor_tilt_scale": self.config.anchor_tilt_scale,
            "residual_gain": self.config.residual_gain,
            "safety": asdict(self.config.safety),
            "dynamics_mode": self.config.dynamics_mode,
            "dynamics_drive_scale": self.config.dynamics_drive_scale,
            "dynamics_position_gain": self.config.dynamics_position_gain,
            "dynamics_velocity_gain": self.config.dynamics_velocity_gain,
            "dynamics_acceleration_gain": self.config.dynamics_acceleration_gain,
            "dynamics_radial_bounds": self.config.dynamics_radial_bounds,
            "body_inference_input": False,
            "native_dmx": False,
        }

