"""22-joint articulated motion, Y up, metres, 30 Hz; no mesh-model dependency.

Fixed offsets/parents follow the public SMPL skeleton used by EDGE/LODGE:
https://github.com/Stanford-TML/EDGE/blob/main/vis.py
This is a canonical skeleton, not subject-specific SMPL mesh reconstruction.
Rotation 6D stores the first TWO ROWS, matching the FineDance/PyTorch3D convention.
"""
import numpy as np
import torch
import torch.nn.functional as F

FPS = 30
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
NAMES = ('pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
         'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
         'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
         'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
OFFSETS = np.asarray([
    [0, 0, 0], [.05858135, -.08228004, -.01766408], [-.06030973, -.09051332, -.01354254],
    [.00443945, .12440352, -.03838522], [.04345142, -.38646945, .008037],
    [-.04325663, -.38368791, -.00484304], [.00448844, .1379564, .02682033],
    [-.01479032, -.42687458, -.037428], [.01905555, -.4200455, -.03456167],
    [-.00226458, .05603239, .00285505], [.04105436, -.06028581, .12204243],
    [-.03483987, -.06210566, .13032329], [-.0133902, .21163553, -.03346758],
    [.07170245, .11399969, -.01889817], [-.08295366, .11247234, -.02370739],
    [.01011321, .08893734, .05040987], [.12292141, .04520509, -.019046],
    [-.11322832, .04685326, -.00847207], [.2553319, -.01564902, -.02294649],
    [-.26012748, -.01436928, -.03126873], [.26570925, .01269811, -.00737473],
    [-.26910836, .00679372, -.00602676]], dtype=np.float32)
FEET = (7, 8, 10, 11)
BODY_HEIGHT = 1.7  # fixed canonical adult scale used for normalized engineering gates
DIM = 139


def rotation_6d_to_matrix(value):
    """Row convention with a valid SO(3) fallback for zero/parallel decoder vectors."""
    a, b = value[..., :3], value[..., 3:]
    norm = torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    axis = torch.zeros_like(a)
    axis[..., 0] = 1
    x = torch.where(norm > 1e-6, a/norm.clamp_min(1e-6), axis)
    orthogonal = b - (x * b).sum(-1, keepdim=True) * x
    fallback = F.one_hot(x.abs().argmin(-1), 3).to(x.dtype)
    fallback = fallback - (fallback*x).sum(-1, keepdim=True)*x
    orthogonal = torch.where(torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True) > 1e-6,
                             orthogonal, fallback)
    y = F.normalize(orthogonal, dim=-1, eps=1e-6)
    return torch.stack((x, y, torch.linalg.cross(x, y, dim=-1)), dim=-2)


def matrix_to_rotation_6d(value):
    return value[..., :2, :].reshape(*value.shape[:-2], 6)


def forward_kinematics(rotations, root, offsets=None):
    """Local (...,22,3,3) rotations -> world (...,22,3) joints."""
    offsets = torch.as_tensor(OFFSETS, device=root.device, dtype=root.dtype) if offsets is None else offsets
    positions, world_rotations = [root], [rotations[..., 0, :, :]]
    for joint, parent in enumerate(PARENTS[1:], 1):
        positions.append(positions[parent] + (world_rotations[parent] @ offsets[joint, :, None]).squeeze(-1))
        world_rotations.append(world_rotations[parent] @ rotations[..., joint, :, :])
    return torch.stack(positions, -2)


def root_from_features(features, origin=None):
    """139 layout: vx,vz,height, 22x6 local rotations, four contact probabilities.

    Velocity at frame t carries the displacement FROM t TO t+1. The first root
    position is the supplied x/z origin; no chunk-wise integration is permitted.
    """
    v = features[..., :2] / FPS
    start = torch.zeros_like(v[..., :1, :])
    horizontal = torch.cat((start, torch.cumsum(v[..., :-1, :], dim=-2)), dim=-2)
    if origin is not None:
        horizontal = horizontal + origin[..., None, :]
    return torch.stack((horizontal[..., 0], features[..., 2], horizontal[..., 1]), dim=-1)


def joints_from_features(features):
    rotations = rotation_6d_to_matrix(features[..., 3:135].reshape(*features.shape[:-1], 22, 6))
    return forward_kinematics(rotations, root_from_features(features))


def contacts_from_joints(joints):
    """Stationary near-floor ankles/toes; finite differences in physical units."""
    feet = joints[..., FEET, :]
    speed = torch.linalg.vector_norm(torch.diff(feet, dim=-3) * FPS, dim=-1)
    # Conservative endpoint assignment; either adjacent motion breaks a contact.
    speed = torch.maximum(torch.cat((speed[..., :1, :], speed), dim=-2),
                          torch.cat((speed, speed[..., -1:, :]), dim=-2))
    heights = feet[..., 1]
    thresholds = heights.new_tensor([.12, .12, .055, .055])
    return ((heights <= thresholds) & (heights >= -.04) & (speed <= .20)).float()


def prepare_native(raw):
    """Canonicalize once per recording, keeping travel, holds and all valid frames."""
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or raw.shape[1] != 315 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError('Expected finite FineDance (T,315), at least three frames')
    value = torch.from_numpy(raw)
    rotations = rotation_6d_to_matrix(value[:, 3:].reshape(-1, 52, 6)[:, :22])
    root = value[:, :3].clone()
    yaw = torch.atan2(rotations[0, 0, 0, 2], rotations[0, 0, 2, 2])
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    frame = torch.stack((c, c*0, s, c*0, c*0+1, c*0, -s, c*0, c)).reshape(3, 3)
    root = (root - root[0] * root.new_tensor([1, 0, 1])) @ frame.T
    rotations[:, 0] = frame @ rotations[:, 0]
    joints = forward_kinematics(rotations, root)
    floor = torch.quantile(joints[:, FEET, 1].amin(-1), .05)
    root[:, 1] -= floor
    joints = forward_kinematics(rotations, root)
    contacts = contacts_from_joints(joints)
    velocity = torch.diff(root[:, [0, 2]], dim=0) * FPS
    velocity = torch.cat((velocity, velocity[-1:]), dim=0)
    features = torch.cat((velocity, root[:, 1:2], matrix_to_rotation_6d(rotations).reshape(-1, 132), contacts), -1)
    reconstructed = joints_from_features(features)
    if not torch.allclose(reconstructed, joints, atol=2e-5, rtol=1e-5):
        raise RuntimeError('Root/rotation conversion did not round trip')
    return {'features': features.numpy(), 'joints': joints.numpy(), 'root': root.numpy(),
            'rotations_6d': matrix_to_rotation_6d(rotations).numpy(), 'contacts': contacts.numpy(),
            'fps': np.asarray(FPS), 'floor_offset': np.asarray(float(floor)), 'initial_yaw': np.asarray(float(yaw))}
