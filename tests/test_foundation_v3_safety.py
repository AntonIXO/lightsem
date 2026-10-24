import torch

from lightsem.experiments.foundation_audio_light.direct_model import DirectLightConfig
from lightsem.experiments.foundation_audio_light.safety_projection import (
    SafetyConfig,
    smooth_bounded_dynamics_projection,
)


def test_smooth_native_parameterization_keeps_training_graph_and_finite_state():
    proposal = torch.zeros(1, 64, 4, 2, requires_grad=True)
    proposal.data[:, 5::11] = 0.03
    config = SafetyConfig(dt=0.1, max_angle=0.4, max_velocity=0.18,
                          max_acceleration=0.5, max_jerk=1.5)
    values, state = smooth_bounded_dynamics_projection(proposal, config=config)
    angle, velocity, acceleration, jerk = values
    assert angle.shape == proposal.shape
    assert all(torch.isfinite(value).all() for value in values)
    assert all(torch.isfinite(value).all() for value in state)
    angle.square().mean().backward()
    assert proposal.grad is not None
    assert torch.isfinite(proposal.grad).all()


def test_new_direct_model_config_is_explicit_and_legacy_default_is_preserved():
    assert DirectLightConfig().dynamics_mode == "hard_clipped"
    assert DirectLightConfig(dynamics_mode="smooth_jerk").dynamics_mode == "smooth_jerk"
