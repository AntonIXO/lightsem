"""Four CPU, body-driven relational transfer experiments.

These are original skeleton-only adaptations of ideas from R2ET (2303.08658),
MeshRet (2410.20986), Contact-Aware Retargeting (2109.07431), and OmniRetarget
(2509.26633). They neither run those models nor assume a skinned source mesh or
a humanoid light rig. Returned coordinates are deformations from one fixed
canonical pose; fixture geometry and emitted-DMX checks belong to the caller.
"""
from __future__ import annotations

import numpy as np

from .common import FieldResult, all_attribution, softmax, skeleton_laplacian, unit
from .skeleton import PARENTS


ANATOMICAL_SCALE_M = 1.8  # Fixed across songs; never fit to a clip's range.
_SENSORS = np.array([20, 21, 18, 19, 4, 5, 10, 11])
_STRESS_METRIC = np.array([1., .65, .35])
_STRESS_SCALE = .75
_STRESS_ANCHOR = .8
_STRESS_STEPS = 16
_PARTNERS = np.array([0, 9, 15, 20, 21, 4, 5, 10, 11])


def _sensor_readout():
    """Fixed anatomical graph distances, with exact identities at the sensors."""
    distances = np.full((22, 22), np.inf)
    np.fill_diagonal(distances, 0.)
    for child, parent in enumerate(PARENTS[1:], 1):
        distances[child, parent] = distances[parent, child] = 1.
    for k in range(22):
        distances = np.minimum(distances, distances[:, k, None] + distances[None, k])
    weights = np.exp(-distances[:, _SENSORS])
    weights /= weights.sum(axis=1, keepdims=True)
    for i, joint in enumerate(_SENSORS):
        weights[joint] = 0.
        weights[joint, i] = 1.
    return weights


_READOUT = _sensor_readout()


def _fit_constellation(points):
    """Anchored SMACOF, batched over frames, with no temporal initialization.

    Minimize sum_{i<j} (||q_i-q_j||-.75*d_ij)^2 + .8||q-a||^2,
    where a=diag(1,.65,.35)*p. The anisotropic anchor makes this different from
    the trivial exact isometry q=.75*p. Anchoring fixes translation/orientation
    and chirality; distances alone would leave those indeterminate.
    """
    p = points / ANATOMICAL_SCALE_M
    anchor = p * _STRESS_METRIC
    distance = np.linalg.norm(p[:, :, None] - p[:, None, :], axis=-1) * _STRESS_SCALE
    k = p.shape[1]
    inverse = np.linalg.inv((k + _STRESS_ANCHOR) * np.eye(k) - np.ones((k, k)))
    q = anchor.copy()
    for _ in range(_STRESS_STEPS):
        delta = q[:, :, None] - q[:, None, :]
        ratio = distance / np.maximum(np.linalg.norm(delta, axis=-1), 1e-9)
        rhs = (ratio[..., None] * delta).sum(axis=2) + _STRESS_ANCHOR * anchor
        q = np.einsum('ij,tjc->tic', inverse, rhs)
    error = np.linalg.norm(q[:, :, None] - q[:, None, :], axis=-1) - distance
    return q * ANATOMICAL_SCALE_M, float(np.sqrt(np.mean(error ** 2)))


def distance_constellation(features):
    """06: coupled distance stress on eight sensors and a fixed 22-role readout."""
    p, reference = features.positions, features.reference
    origin = reference[0]
    solved, stress = _fit_constellation(p[:, _SENSORS] - origin)
    rest, rest_stress = _fit_constellation((reference[_SENSORS] - origin)[None])
    # Retain a small, local source residual for joints absent from the eight
    # sensor constellation; no role is silently replaced by an interpolant.
    source_interp = np.einsum('jk,tkc->tjc', _READOUT, p[:, _SENSORS])
    rest_interp = _READOUT @ reference[_SENSORS]
    local = (p - source_interp) - (reference - rest_interp)
    deformation = np.einsum('jk,tkc->tjc', _READOUT, solved - rest)
    deformation += .25 * local * _STRESS_METRIC
    attribution = [sorted({0, j, *map(int, _SENSORS)}) for j in range(22)]
    return FieldResult(deformation, attribution, {
        'algorithm': '06', 'name': 'distance_constellation', 'paper': '2303.08658',
        'adaptation': 'Anchored eight-sensor stress; original skeleton-only adaptation.',
        'anatomical_scale_m': ANATOMICAL_SCALE_M, 'sensor_joints': _SENSORS.tolist(),
        'stress_iterations': _STRESS_STEPS, 'distance_gain': _STRESS_SCALE,
        'anchor_weight': _STRESS_ANCHOR, 'anchor_axis_gain': _STRESS_METRIC.tolist(),
        'local_residual_gain': .25, 'normalized_stress_rms': stress,
        'reference_normalized_stress_rms': rest_stress,
        'limitation': 'Eight sensors summarize geometry; downstream projection can lose it.',
    })


def _interaction_field(p, reference):
    """Near/far vectors in bone observer frames; identities never switch."""
    parent = np.maximum(PARENTS, 0)
    bones = p - p[:, parent]
    bones[:, 0] = p[:, 9] - p[:, 0]
    rb = reference - reference[parent]
    rb[0] = reference[9] - reference[0]
    tangent = unit(bones, unit(rb))
    # A torso frame supplies roll about each observer bone. Degeneracy is
    # handled using fixed geometric axes, never previous-frame state.
    torso_up = unit(p[:, 12] - p[:, 0])
    shoulder = p[:, 16] - p[:, 17]
    torso_right = unit(shoulder - np.sum(shoulder * torso_up, axis=-1)[:, None] * torso_up,
                       np.array([1., 0., 0.]))
    torso_depth = unit(np.cross(torso_right, torso_up), np.array([0., 0., 1.]))
    cross = np.cross(tangent, torso_depth[:, None])
    alternate = np.cross(tangent, torso_right[:, None])
    use_alternate = np.linalg.norm(cross, axis=-1, keepdims=True) < 1e-7
    cross = np.where(use_alternate, alternate, cross)
    side = unit(cross, np.array([1., 0., 0.]))
    normal = unit(np.cross(side, tangent), np.array([0., 0., 1.]))
    vectors = (p[:, None, _PARTNERS] - p[:, :, None]) / ANATOMICAL_SCALE_M
    coordinates = np.stack((np.einsum('tjkc,tjc->tjk', vectors, side),
                            np.einsum('tjkc,tjc->tjk', vectors, tangent),
                            np.einsum('tjkc,tjc->tjk', vectors, normal)), axis=-1)
    distance = np.linalg.norm(vectors, axis=-1)
    valid = np.arange(22)[:, None] != _PARTNERS[None]
    near = softmax(np.where(valid, -6. * distance, -np.inf), axis=-1)
    far = softmax(np.where(valid, 6. * distance, -np.inf), axis=-1)
    local = np.sum((.35 * near + .65 * far)[..., None] * coordinates, axis=2)
    return local * ANATOMICAL_SCALE_M, int(np.count_nonzero(use_alternate))


def near_far_observer_field(features):
    """07: skeletal observer-frame interaction vectors, plus a world residual."""
    field, degenerate = _interaction_field(features.positions, features.reference)
    rest, _ = _interaction_field(features.reference[None], features.reference)
    # Pure observer-frame vectors are invariant to a rigid body turn. Retain
    # that information explicitly instead of silently discarding it.
    points = .8 * (field - rest) + .2 * features.displacement
    attribution = [sorted({0, 9, 12, 16, 17, j, max(PARENTS[j], 0), *map(int, _PARTNERS)})
                   for j in range(22)]
    return FieldResult(points, attribution, {
        'algorithm': '07', 'name': 'near_far_observer_field', 'paper': '2410.20986',
        'adaptation': 'Bone-frame skeletal vectors; no mesh sensors or learned MeshRet.',
        'anatomical_scale_m': ANATOMICAL_SCALE_M, 'partner_joints': _PARTNERS.tolist(),
        'soft_distance_beta': 6., 'near_weight': .35, 'far_weight': .65,
        'observer_gain': .8, 'world_residual_gain': .2,
        'alternate_frame_count': degenerate,
        'limitation': 'Pooled vectors compress relations; degenerate observer frames can change sharply.',
    })


_BRIDGES = ((20, 21, .10, .45), (20, 9, .18, .55),
            (21, 9, .18, .55), (20, 15, .16, .45), (21, 15, .16, .45))
_BRIDGE_GROUP = (9, 15, 20, 21)
_BRIDGE_WEIGHT_EPS = 1e-5


def _smooth_proximity(distance, inner, outer):
    s = np.clip((outer - distance) / (outer - inner), 0., 1.)
    return s * s * (3. - 2. * s)


def _bridge_field(p, velocity, reference):
    # Instantaneous graph consensus: (I+L(w))Q=P. For an isolated pair,
    # q_a-q_b=(p_a-p_b)/(1+2w), so w=c/(2(1-c)) contracts by 1-c without
    # swapping the two identities. The epsilon keeps exact proximity finite.
    # Solve the fixed four-role component, not 22 unrelated target coordinates.
    matrix = np.broadcast_to(np.eye(len(_BRIDGE_GROUP)),
                             (len(p), len(_BRIDGE_GROUP), len(_BRIDGE_GROUP))).copy()
    proximity = np.zeros(p.shape[:2])
    for a, b, inner, outer in _BRIDGES:
        c = _smooth_proximity(np.linalg.norm(p[:, a] - p[:, b], axis=-1), inner, outer)
        weight = c / (2. * np.maximum(1. - c, _BRIDGE_WEIGHT_EPS))
        i, j = _BRIDGE_GROUP.index(a), _BRIDGE_GROUP.index(b)
        matrix[:, i, i] += weight
        matrix[:, j, j] += weight
        matrix[:, i, j] -= weight
        matrix[:, j, i] -= weight
        proximity[:, a] = np.maximum(proximity[:, a], c)
        proximity[:, b] = np.maximum(proximity[:, b], c)
    q = p.copy()
    q[:, _BRIDGE_GROUP] = np.linalg.solve(matrix, p[:, _BRIDGE_GROUP])
    support_count = 0
    for toe, ankle in ((10, 7), (11, 8)):
        floor = reference[toe, 1]
        height = np.maximum(p[:, toe, 1] - floor, 0.)
        speed = np.linalg.norm(velocity[:, toe], axis=-1)
        confidence = _smooth_proximity(height, .02, .12) * _smooth_proximity(speed, .03, .15)
        # Source-defined contact intervals have a fixed source entry anchor.
        # No post-contact integration, elapsed-time envelope or release clock.
        active = confidence >= .75
        entries = active & ~np.r_[False, active[:-1]]
        entry_index = np.maximum.accumulate(np.where(entries, np.arange(len(p)), 0))
        blend = np.where(active, confidence, 0.)
        for joint in (toe, ankle):
            anchor = p[entry_index, joint].copy()
            anchor[:, 1] = reference[joint, 1]
            q[:, joint] = (1. - blend[:, None]) * p[:, joint] + blend[:, None] * anchor
            proximity[:, joint] = confidence
        support_count += int(np.count_nonzero(active))
    return q, proximity, support_count


def proximity_bridges(features):
    """08: body-proximity encounters and source-derived foot-support holds."""
    q, activity, support_count = _bridge_field(features.positions, features.velocity, features.reference)
    rest, _, _ = _bridge_field(features.reference[None], np.zeros((1, 22, 3)), features.reference)
    attribution = [sorted({0, j}) for j in range(22)]
    for joint in _BRIDGE_GROUP:
        attribution[joint] = [0, *_BRIDGE_GROUP]
    attribution[7] = [0, 7, 10]
    attribution[8] = [0, 8, 11]
    return FieldResult(q - rest, attribution, {
        'algorithm': '08', 'name': 'proximity_bridges', 'paper': '2109.07431',
        'adaptation': 'Skeleton proximity/support proxies; no surface-contact claim.',
        'revision': 'v3_shared_absolute_graph_consensus',
        'anatomical_scale_m': ANATOMICAL_SCALE_M,
        'bridge_solver': '(I + L(w)) Q = P',
        'bridge_weight': 'c / (2 * max(1-c, 1e-5))',
        'bridge_weight_epsilon': _BRIDGE_WEIGHT_EPS,
        'shared_group_joints': list(_BRIDGE_GROUP),
        'shared_group_readout': 'One shared similarity transform of absolute coordinates.',
        'bridges': [list(pair) for pair in _BRIDGES],
        'support_height_m': [.02, .12], 'support_speed_m_s': [.03, .15],
        'support_confidence_threshold': .75, 'support_joint_frames': support_count,
        'limitation': 'Skeleton support is a proxy and entry anchors are source-history dependent.',
    }, activity=activity, absolute_points=q, absolute_reference=rest[0],
       shared_groups=(_BRIDGE_GROUP,))


def _poisson_decoder():
    source = skeleton_laplacian()
    source /= np.diag(source)[:, None]
    # A virtual role graph, not a claim that fixtures share skeletal topology.
    # Add bilateral/interaction links to change the differential decoder.
    adjacency = -skeleton_laplacian()
    np.fill_diagonal(adjacency, 0.)
    for a, b in ((20, 21), (18, 19), (16, 17), (4, 5), (7, 8),
                 (10, 11), (20, 9), (21, 9), (0, 15)):
        adjacency[a, b] = adjacency[b, a] = .75
    target = np.eye(22) - adjacency / adjacency.sum(axis=1, keepdims=True)
    ridge = .2
    decoder = np.linalg.solve(target.T @ target + ridge * np.eye(22), target.T)
    return source, target, decoder, ridge


_SOURCE_LAPLACIAN, _ROLE_LAPLACIAN, _POISSON_DECODER, _POISSON_RIDGE = _poisson_decoder()


def screened_poisson(features):
    """09: transplant skeletal differential coordinates to a different role graph."""
    differential = np.einsum('ij,tjc->tic', _SOURCE_LAPLACIAN, features.displacement)
    points = np.einsum('ij,tjc->tic', _POISSON_DECODER, differential)
    # Differential coordinates annihilate translations. An explicit, shared
    # root-height component retains body lift with no synthetic motion.
    points += .25 * features.displacement[:, :1]
    return FieldResult(points, all_attribution(), {
        'algorithm': '09', 'name': 'screened_poisson', 'paper': '2509.26633',
        'adaptation': 'Fixed skeleton-to-role graph differential decoder; no interaction mesh.',
        'anatomical_scale_m': ANATOMICAL_SCALE_M, 'ridge': _POISSON_RIDGE,
        'root_height_gain': .25, 'interaction_edge_weight': .75,
        'limitation': 'Graph coupling spreads attribution and ridge can attenuate articulation.',
    })


FUNCTIONS = {
    '06': distance_constellation,
    '07': near_far_observer_field,
    '08': proximity_bridges,
    '09': screened_poisson,
}
