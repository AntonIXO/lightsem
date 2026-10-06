"""Five deterministic anatomical transfer fields, evaluated entirely on CPU.

These are LightSem adaptations, not implementations of the authors' networks:
01/03 borrow orientation/FK reasoning from Villegas et al. (2018), arXiv
1804.05653, sections 3.1 and 4.1; 02 isolates articulation explicitly; 03/05
borrow topology/contact preservation from Aberman et al. (2020), arXiv
2005.05732, sections 4.3/5.2; 04 borrows masked body-part pooling from Hu et al.
(2023), arXiv 2306.08006, section IV-B, equation 5.

Every descriptor is evaluated against the fixed Features.reference, never a
song mean, a chosen first frame, or an adaptive running state. The returned
22 field samples retain their anatomical indices. Fixed neutral offsets and
body-dependent spatial transformations supply no autonomous temporal motion.
The common spatial/fixture mapper, rather than this module, produces N lights.
"""
from __future__ import annotations

import numpy as np

from .common import BODY_PARTS, Features, FieldResult, softmax, unit
from .skeleton import PARENTS


_PARENT = np.maximum(np.asarray(PARENTS), 0)
_CHILD = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10,
          8: 11, 9: 12, 12: 15, 13: 16, 14: 17, 16: 18, 17: 19,
          18: 20, 19: 21}


def _meta(identifier, name, equation, parameters, references):
    return {'id': identifier, 'name': name, 'equation': equation,
            'parameters': parameters, 'references': references,
            'implementation': 'deterministic LightSem adaptation; no trained paper model',
            'neutral': 'fixed canonical reference; no per-song fitting',
            'temporal_driver': 'supplied body only'}


def _bone_attribution():
    result = [sorted({0, int(_PARENT[j]), j}) for j in range(22)]
    result[0] = [0, 9]
    return result


def bone_direction(features: Features) -> FieldResult:
    """01: 0.9 m * (unit bone - neutral unit bone) + 0.15 * displacement.

The small positional term carries body height and translation within a part;
the main term exposes orientations hidden by endpoint pooling. A zero-length
bone has its fixed reference direction, including when the entire body is
degenerate. Root orientation uses the pelvis-to-spine3 vector from Features.
"""
    neutral = unit(features.reference_bones)
    direction = unit(features.bones, fallback=neutral)
    points = .9 * (direction - neutral) + .15 * features.displacement
    return FieldResult(points, _bone_attribution(), _meta(
        '01', 'Bone direction', '0.9*(unit(b)-unit(b_ref))+0.15*(p-p_ref)',
        {'orientation_metres': .9, 'position_gain': .15}, ['1804.05653']))


def _triplet(j):
    if j == 0:
        return 0, 3, 9
    if j in _CHILD:
        return int(_PARENT[j]), j, _CHILD[j]
    parent = int(_PARENT[j])
    return int(_PARENT[parent]), parent, j


def _bend_descriptor(positions, reference):
    descriptors = []
    for j in range(22):
        a, b, c = _triplet(j)
        incoming = unit(positions[..., b, :] - positions[..., a, :],
                        fallback=unit(reference[b] - reference[a]))
        outgoing = unit(positions[..., c, :] - positions[..., b, :],
                        fallback=unit(reference[c] - reference[b]))
        cross = np.cross(incoming, outgoing)
        # Regularization prevents a straight limb from defining a noisy plane.
        normal = cross / np.sqrt(np.sum(cross * cross, axis=-1, keepdims=True) + .05 ** 2)
        flexion = .5 * (1 - np.clip(np.sum(incoming * outgoing, axis=-1), -1, 1))
        axis = np.eye(3)[j % 3]  # Fixed spatial encoding, never an animation phase.
        descriptors.append(.65 * normal + .6 * flexion[..., None] * axis + .3 * outgoing)
    return np.stack(descriptors, axis=-2)


def bend_plane(features: Features) -> FieldResult:
    """02: regularized bend-plane normal + flexion + outgoing orientation.

For incoming/outgoing unit vectors u,v, n=(u cross v)/sqrt(|u cross v|²+.05²)
and f=(1-u.v)/2. Each sample is .65*n+.6*f*e_(j mod 3)+.3*v, minus its neutral
descriptor, plus .1 position displacement. The fixed per-joint axis distributes
the fourth articulation descriptor over the three-dimensional field. This is
a lossy encoding, not an invertible representation of all joint rotations.
"""
    points = (_bend_descriptor(features.positions, features.reference)
              - _bend_descriptor(features.reference, features.reference)
              + .1 * features.displacement)
    attribution = [sorted({0, j, *_triplet(j)}) for j in range(22)]
    return FieldResult(points, attribution, _meta(
        '02', 'Bend plane and flexion',
        'delta(.65*n+.6*f*e_(j%3)+.3*unit(outgoing))+.1*displacement',
        {'plane_regularizer': .05, 'plane_gain': .65, 'flexion_gain': .6,
         'orientation_gain': .3, 'position_gain': .1}, ['1804.05653']))


def virtual_sculpture(features: Features) -> FieldResult:
    """03: FK with fixed nonuniform target bone lengths and all 22 joints.

q_j=q_parent+length_ref_j*gain_j*unit(source_bone_j). The pelvis carries its
source height. Gains expand upper/lower arms by 1.8/2.8, thigh/shin by 1.4/1.9,
foot by 1.2, and torso/collar/head by 1.4. Both the absolute FK pose and the
fixed expanded neutral pose are supplied as one shared structural group. The
installation must apply one similarity transform to that group: individual
fixture neutral offsets would destroy the intended skeletal geometry.
The deformation channel remains absolute pose minus the expanded neutral.
Attribution follows ancestor chains. No song-dependent normalization is used.
"""
    gains = np.full(22, 1.4)
    gains[[18, 19]] = 1.8
    gains[[20, 21]] = 2.8
    gains[[7, 8]] = 1.9
    gains[[10, 11]] = 1.2
    neutral = unit(features.reference_bones)
    directions = unit(features.bones, fallback=neutral)
    lengths = np.linalg.norm(features.reference_bones, axis=-1) * gains
    absolute = np.zeros_like(features.positions)
    absolute[:, 0] = features.positions[:, 0]
    reference = np.zeros_like(features.reference)
    reference[0] = features.reference[0]
    attribution = [[0]]
    for j in range(1, 22):
        parent = int(_PARENT[j])
        absolute[:, j] = absolute[:, parent] + lengths[j] * directions[:, j]
        reference[j] = reference[parent] + lengths[j] * neutral[j]
        attribution.append(sorted({*attribution[parent], parent, j}))
    return FieldResult(absolute - reference, attribution, _meta(
        '03', 'Virtual articulated sculpture',
        'absolute_j=absolute_parent+L_ref_j*g_j*unit(b_j); points=absolute-expanded_neutral',
        {'edge_gains': gains.tolist(), 'root_height_gain': 1.,
         'geometry_readout': 'one shared similarity transform for all 22 absolute FK points'},
        ['1804.05653', '2005.05732']), absolute_points=absolute,
        absolute_reference=reference, shared_groups=(tuple(range(22)),))


def _pool_descriptor(positions, reference):
    result = np.zeros_like(positions)
    for indices in BODY_PARTS:
        ix = np.asarray(indices)
        chain = positions[..., ix, :]
        local = chain - chain[..., :1, :]
        lengths = np.linalg.norm(np.diff(reference[ix], axis=0), axis=-1)
        total_length = max(float(lengths.sum()), 1e-8)
        # Uniform semantic chain coordinates are fixed for every song/frame.
        ranks = np.linspace(0., 1., len(ix))
        curvature = np.zeros(chain.shape[:-1])
        edges = np.diff(chain, axis=-2)
        neutral_edges = unit(np.diff(reference[ix], axis=0))
        direction = unit(edges, fallback=neutral_edges)
        curvature[..., 1:-1] = 1 - np.clip(
            np.sum(direction[..., :-1, :] * direction[..., 1:, :], axis=-1), -1, 1)
        reach = np.linalg.norm(local, axis=-1) / total_length
        salience = 1.5 * curvature + .5 * reach
        for query, j in zip(ranks, ix):
            score = -(ranks - query) ** 2 / (2 * .25 ** 2) + salience
            weights = softmax(score, axis=-1)
            result[..., j, :] = (np.sum(weights[..., None] * local, axis=-2)
                                 + .5 * chain[..., 0, :])
    return result


def pose_part_pooling(features: Features) -> FieldResult:
    """04: deterministic pose-aware attention masked strictly to five parts.

w_ij=softmax(-(r_j-s_i)^2/(2*.25²)+1.5*curvature_j+.5*reach_j).
Each query pools part-local positions and adds half its proximal position;
the canonical descriptor is subtracted. Curvature/reach are instantaneous
pose functions, so a held pose cannot trigger attention drift. No learned
attention weights or temporal attention from PAN are claimed here.
"""
    points = (_pool_descriptor(features.positions, features.reference)
              - _pool_descriptor(features.reference, features.reference))
    attribution = [None] * 22
    for part in BODY_PARTS:
        for j in part:
            attribution[j] = sorted({0, *part})
    return FieldResult(points, attribution, _meta(
        '04', 'Pose-aware body-part pooling',
        'delta(sum(softmax(proximity+1.5*curvature+.5*reach)*part_local)+.5*proximal)',
        {'query_sigma': .25, 'curvature_gain': 1.5, 'reach_gain': .5,
         'proximal_gain': .5, 'body_parts': [list(p) for p in BODY_PARTS]},
        ['2306.08006']))


def _support(positions, velocity, reference):
    feet = positions[..., [10, 11], :]
    floor = float(np.mean(reference[[10, 11], 1]))
    height = feet[..., 1] - floor
    speed_squared = np.sum(velocity[..., [10, 11], :] ** 2, axis=-1)
    confidence = np.exp(-height ** 2 / .08 ** 2 - speed_squared / .20 ** 2)
    support = ((np.sum(confidence[..., None] * feet, axis=-2)
                + .05 * positions[..., 0, :])
               / (np.sum(confidence, axis=-1, keepdims=True) + .05))
    return support


def support_relative(features: Features) -> FieldResult:
    """05: pose/velocity-derived support frame, with continuous flight behavior.

c_k=exp(-((foot_height-floor)/.08)^2-(foot_speed/.20)^2), where floor is fixed
from the canonical feet. support=(sum(c_k*foot_k)+.05*pelvis)/(sum(c_k)+.05).
Map p_j-support, subtract its canonical value, and restore .4 of pelvis height
displacement so flight does not erase vertical gestures. No hard contact
switch, first-frame calibration, clock, or previous solver state is used.
"""
    support = _support(features.positions, features.velocity, features.reference)
    reference_support = _support(features.reference, np.zeros_like(features.reference),
                                 features.reference)
    points = (features.positions - support[:, None, :]
              - (features.reference - reference_support))
    points += .4 * features.displacement[:, 0:1, :]
    attribution = [sorted({0, j, 10, 11}) for j in range(22)]
    return FieldResult(points, attribution, _meta(
        '05', 'Support-relative body transfer',
        'delta(p_j-weighted_support)+.4*pelvis_displacement',
        {'foot_height_scale_metres': .08, 'foot_speed_scale_m_s': .20,
         'pelvis_support_weight': .05, 'root_height_gain': .4,
         'fixed_floor_height': float(np.mean(features.reference[[10, 11], 1]))},
        ['2005.05732']))


FUNCTIONS = {'01': bone_direction, '02': bend_plane, '03': virtual_sculpture,
             '04': pose_part_pooling, '05': support_relative}
