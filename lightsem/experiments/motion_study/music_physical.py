"""Opt-in physical objectives for a future native-motion ablation.

Not wired into MusicModel. No combined objective or suggested weights. Inputs use
geometry.py's row-6D convention, Y-up metres, and outgoing root velocity at 30 Hz.
Only the clean estimate is normalized; reference features are physical native139.
These engineering losses do not measure music conditioning or dance quality.
"""
import torch

from .geometry import DIM, FEET, FPS, forward_kinematics, rotation_6d_to_matrix


def clean_estimate(x_t, velocity, time):
    """Reconstruct normalized endpoint x_1 = x_t + (1-t)*v in FP32.

    x_t/velocity: (B,T,139); time: scalar, (B,), or (B,1,1).
    This is an endpoint estimate, not the vector-field target itself.
    """
    if x_t.ndim != 3 or x_t.shape[-1] != DIM or velocity.shape != x_t.shape:
        raise ValueError('Expected matching (B,T,139) state and velocity')
    with torch.autocast(device_type=x_t.device.type, enabled=False):
        time = torch.as_tensor(time, device=x_t.device, dtype=torch.float32)
        if time.ndim == 1 and time.shape == (x_t.shape[0],):
            time = time[:, None, None]
        if time.ndim != 0 and time.shape != (x_t.shape[0], 1, 1):
            raise ValueError('Expected scalar, (B,), or (B,1,1) time')
        if not bool(torch.isfinite(time).all() & (time >= 0).all() & (time <= 1).all()):
            raise ValueError('Time must be finite and in [0,1]')
        return x_t.float() + (1 - time) * velocity.float()


def _validate_mask(features, valid):
    if features.ndim != 3 or features.shape[-1] != DIM or features.shape[1] < 1:
        raise ValueError('Expected (B,T,139) features with T >= 1')
    if valid.dtype != torch.bool or valid.shape != features.shape[:2]:
        raise ValueError('Expected boolean (B,T) valid mask')
    if valid.device != features.device:
        raise ValueError('Features and mask must share a device')


def motion_kinematics(features, valid):
    """Differentiable FP32 FK and edge velocities with explicit validity.

    A derivative exists only when BOTH adjacent frames are valid. Root x/z starts
    at zero in each contiguous valid segment; velocity of its last frame is not
    integrated. No movement is invented across a gap. Invalid positions/edges are
    returned as zero. A one-frame recording has positions but no velocity edges.
    """
    _validate_mask(features, valid)
    with torch.autocast(device_type=features.device.type, enabled=False):
        value = torch.where(valid[..., None], features.float(), 0.)
        edge = valid[:, :-1] & valid[:, 1:]
        steps = torch.where(edge[..., None], value[:, :-1, :2] / FPS, 0.)
        horizontal = torch.cat((torch.zeros_like(value[:, :1, :2]), steps.cumsum(1)), 1)
        starts = valid & ~torch.cat((torch.zeros_like(valid[:, :1]), valid[:, :-1]), 1)
        indices = torch.arange(value.shape[1], device=value.device)[None].expand_as(valid)
        anchors = torch.where(starts, indices, 0).cummax(1).values
        horizontal = horizontal - horizontal.gather(1, anchors[..., None].expand(-1, -1, 2))
        root = torch.stack((horizontal[..., 0], value[..., 2], horizontal[..., 1]), -1)
        root = torch.where(valid[..., None], root, 0.)
        rotations = rotation_6d_to_matrix(value[..., 3:135].reshape(*value.shape[:2], 22, 6))
        positions = forward_kinematics(rotations, root)
        positions = torch.where(valid[..., None, None], positions, 0.)
        velocities = torch.where(edge[..., None, None], torch.diff(positions, dim=1) * FPS, 0.)
        return {'root': root, 'positions': positions, 'velocities': velocities, 'edges': edge}


def _masked_mean(value, mask):
    """Mean over selected scalar entries; empty selections give differentiable zero."""
    mask = mask.expand_as(value)
    return torch.where(mask, value, 0.).sum() / mask.sum().clamp_min(1)


def physical_losses(clean_normalized, reference_features, mean, std, valid):
    """Return separate scalar, FP32 physical losses, without a total or weights.

    position: mean squared joint distance (m^2).
    velocity: mean squared joint velocity error ((m/s)^2).
    planted_foot_velocity: mean squared generated foot speed ((m/s)^2), selected
      ONLY by detached reference contacts >= .5 at BOTH ends of a valid edge.
    root_displacement: mean squared horizontal displacement error (m^2), measured
      from each valid segment's origin, preserving reference travel, not shrinking
      all root movement toward zero. Not an absolute global-position constraint.
    floor_penetration: mean squared depth below y=0 across all joints (m^2).

    Means pool valid frames/edges/joints across the batch. Short holds contribute;
    empty selections yield zero. Contact labels in generated features are never
    used. This is not the separate generated-contact coverage/evaluation gate.
    Floor loss can be nonzero even at identity if a reference penetrates the floor.
    mean/std must be finite native139 vectors, with strictly positive std.
    """
    _validate_mask(clean_normalized, valid)
    if reference_features.shape != clean_normalized.shape:
        raise ValueError('Reference must match (B,T,139) clean estimate')
    with torch.autocast(device_type=clean_normalized.device.type, enabled=False):
        mean = torch.as_tensor(mean, device=clean_normalized.device, dtype=torch.float32)
        std = torch.as_tensor(std, device=clean_normalized.device, dtype=torch.float32)
        if mean.shape != (DIM,) or std.shape != (DIM,):
            raise ValueError('Expected (139,) normalizers')
        if not bool(torch.isfinite(mean).all() & torch.isfinite(std).all() & (std > 0).all()):
            raise ValueError('Normalizers must be finite and std positive')
        clean = torch.where(valid[..., None], clean_normalized.float(), 0.) * std + mean
        reference = torch.where(valid[..., None], reference_features.detach().float(), 0.)
        generated = motion_kinematics(clean, valid)
        target = motion_kinematics(reference, valid)
        edges = generated['edges']
        contacts = reference[..., 135:139] >= .5
        planted = edges[..., None] & contacts[:, :-1] & contacts[:, 1:]
        return {
            'position': _masked_mean((generated['positions'] - target['positions']).square().sum(-1), valid[..., None]),
            'velocity': _masked_mean((generated['velocities'] - target['velocities']).square().sum(-1), edges[..., None]),
            'planted_foot_velocity': _masked_mean(generated['velocities'][:, :, FEET].square().sum(-1), planted),
            'root_displacement': _masked_mean((generated['root'][..., [0, 2]] - target['root'][..., [0, 2]]).square().sum(-1), valid),
            'floor_penetration': _masked_mean((-generated['positions'][..., 1]).clamp_min(0).square(), valid[..., None]),
        }
