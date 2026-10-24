"""Differentiable bounds for proposed fixture motion."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SafetyConfig:
    dt: float = 1.0 / 40.0
    max_angle: float = 1.0
    max_velocity: float = 2.5
    max_acceleration: float = 12.0
    max_jerk: float = 80.0


def bounded_dynamics_projection(jerk_proposal, previous=None, config=SafetyConfig()):
    """Integrate bounded jerk with the historical hard state bounds.

    This is retained for sealed historical checkpoints.  New native safety
    work must use :func:`smooth_bounded_dynamics_projection`: hard clipping of
    an integrated state can create finite-difference jerk spikes at the clip
    boundary even when the parameterized jerk is bounded.
    """
    if jerk_proposal.ndim < 2 or jerk_proposal.shape[-1] != 2:
        raise ValueError("expected ... ,2 jerk proposal")
    dt = config.dt
    time_dim = -3 if jerk_proposal.ndim >= 3 else -2
    step_shape = jerk_proposal.select(time_dim, 0).shape
    if previous is None:
        angle = jerk_proposal.new_zeros(step_shape)
        velocity = jerk_proposal.new_zeros(step_shape)
        acceleration = jerk_proposal.new_zeros(step_shape)
    else:
        angle, velocity, acceleration = previous
    angles, velocities, accelerations, jerks = [], [], [], []
    for i in range(jerk_proposal.shape[time_dim]):
        jerk = jerk_proposal.select(time_dim, i).tanh() * config.max_jerk
        acceleration = (acceleration + dt * jerk).clamp(-config.max_acceleration, config.max_acceleration)
        velocity = (velocity + dt * acceleration).clamp(-config.max_velocity, config.max_velocity)
        angle = (angle + dt * velocity).clamp(-config.max_angle, config.max_angle)
        angles.append(angle); velocities.append(velocity); accelerations.append(acceleration); jerks.append(jerk)
    return (torch.stack(angles, time_dim), torch.stack(velocities, time_dim),
            torch.stack(accelerations, time_dim), torch.stack(jerks, time_dim)), (angle, velocity, acceleration)


def smooth_bounded_dynamics_projection(
    jerk_proposal,
    previous=None,
    config=SafetyConfig(),
    *,
    drive_scale: float = 4.0,
    position_gain: float = 1.0,
    velocity_gain: float = 2.0,
    acceleration_gain: float = 2.0,
    radial_bounds: bool = False,
):
    """Integrate a smooth jerk-limited native angular state.

    The state is bounded by smooth ``tanh`` feedback rather than hard clipping.
    Hard clipping is not a harmless safety operation here: it changes the
    emitted angle without changing the stored acceleration/jerk state and can
    manufacture large finite-difference derivatives at the boundary.  The
    smooth feedback is part of the native model parameterisation and remains
    in the training graph; the saved angle stream is audited directly.
    """
    if jerk_proposal.ndim < 2 or jerk_proposal.shape[-1] != 2:
        raise ValueError("expected ... ,2 jerk proposal")
    if min(drive_scale, position_gain, velocity_gain, acceleration_gain) < 0:
        raise ValueError("smooth dynamics gains must be non-negative")
    dt = config.dt
    time_dim = -3 if jerk_proposal.ndim >= 3 else -2
    step_shape = jerk_proposal.select(time_dim, 0).shape
    if previous is None:
        angle = jerk_proposal.new_zeros(step_shape)
        velocity = jerk_proposal.new_zeros(step_shape)
        acceleration = jerk_proposal.new_zeros(step_shape)
    else:
        angle, velocity, acceleration = previous
    angles, velocities, accelerations, jerks = [], [], [], []
    for i in range(jerk_proposal.shape[time_dim]):
        proposal = jerk_proposal.select(time_dim, i)
        previous_angle = angle
        previous_velocity = velocity
        drive = drive_scale * proposal
        drive = drive - position_gain * torch.tanh(angle / config.max_angle)
        drive = drive - velocity_gain * torch.tanh(velocity / config.max_velocity)
        drive = drive - acceleration_gain * torch.tanh(acceleration / config.max_acceleration)
        jerk = config.max_jerk * torch.tanh(drive)
        acceleration = acceleration + dt * jerk
        if radial_bounds:
            acceleration_norm = torch.linalg.norm(acceleration, dim=-1, keepdim=True)
            acceleration = acceleration * (config.max_acceleration * torch.tanh(acceleration_norm / config.max_acceleration) / acceleration_norm.clamp_min(1e-8))
        velocity = velocity + dt * acceleration
        if radial_bounds:
            velocity_norm = torch.linalg.norm(velocity, dim=-1, keepdim=True)
            velocity = velocity * (config.max_velocity * torch.tanh(velocity_norm / config.max_velocity) / velocity_norm.clamp_min(1e-8))
        candidate_angle = angle + dt * velocity
        if radial_bounds:
            angle_norm = torch.linalg.norm(candidate_angle, dim=-1, keepdim=True)
            angle = candidate_angle * (config.max_angle * torch.tanh(angle_norm / config.max_angle) / angle_norm.clamp_min(1e-8))
            velocity = (angle - previous_angle) / dt
            acceleration = (velocity - previous_velocity) / dt
        else:
            angle = candidate_angle
        angles.append(angle); velocities.append(velocity); accelerations.append(acceleration); jerks.append(jerk)
    return (torch.stack(angles, time_dim), torch.stack(velocities, time_dim),
            torch.stack(accelerations, time_dim), torch.stack(jerks, time_dim)), (angle, velocity, acceleration)
