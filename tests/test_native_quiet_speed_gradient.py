"""A genuinely moving generated trajectory must receive a quiet-target gradient."""
import pytest
import torch

from lightsem.experiments.native_dynamics.kinematics import dynamics_losses


@pytest.mark.parametrize('initial_speed', [1., 10.])
def test_quiet_observation_can_slow_a_fast_generated_wrist(initial_speed):
    speed = torch.tensor(initial_speed, requires_grad=True)
    time = torch.arange(180) / 30
    direction = torch.zeros(1, 1, 13, 2)
    direction[..., 5, 0] = 1
    generated = time[None, :, None, None] * direction * speed
    observed = torch.zeros_like(generated)
    valid = torch.ones(generated.shape[:3], dtype=torch.bool)
    loss = dynamics_losses(generated, observed, valid)['low_speed']
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(speed.grad)
    assert speed.grad > 1, 'The quiet loss must push an initially fast trajectory slower'


def test_unobserved_wrist_does_not_receive_a_quiet_target():
    speed = torch.tensor(1., requires_grad=True)
    direction = torch.zeros(1, 1, 13, 2)
    direction[..., 5, 0] = 1
    generated = torch.arange(180)[None, :, None, None] / 30 * direction * speed
    valid = torch.ones(generated.shape[:3], dtype=torch.bool)
    valid[..., 5] = False
    loss = dynamics_losses(generated, torch.zeros_like(generated), valid)['low_speed']
    loss.backward()
    assert torch.isfinite(speed.grad)
    assert speed.grad == 0, 'Missing observations must not become zero-speed targets'
