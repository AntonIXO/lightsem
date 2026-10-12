"""Stable fixture identities and capability-specific reads of anatomical fields.

The default physical rig matches V1. Extra fixtures append immutable identities;
the source role never depends on a song, frame index, or requested total count.
"""
from __future__ import annotations
import numpy as np
from .common import ROLE_ORDER, NAMES, derivative


def _count(n):
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 10000:
        raise ValueError('Fixture counts must be integers in 0..10000')
    return n


def _fraction(n):
    """Van der Corput spatial coordinates; the input is a fixture ID, never time."""
    result, denominator = 0., 1.
    while n:
        denominator *= 2
        n, digit = divmod(n, 2)
        result += digit / denominator
    return result


def make_outputs(beams=8, lasers=5, bulbs=50):
    outputs = []
    for kind, count, base, role_offset in (('beam', beams, 8, 0), ('laser', lasers, 5, 8)):
        for index in range(_count(count)):
            # The five reference laser positions already contain the first
            # three dyadic interior coordinates (1/2,1/4,3/4). Skip them when
            # appending rays so new virtual fixtures do not share a mount.
            extra_index = index - base + (4 if kind == 'laser' else 1)
            x = -6.5 + 13 * index / (base - 1) if index < base else -6.5 + 13 * _fraction(extra_index)
            # Type-specific infinite IDs interleave beyond the default set.
            role_id = index + role_offset if index < base else 13 + 2 * (index - base) + (kind == 'laser')
            cycle, slot = divmod(role_id, len(ROLE_ORDER))
            role = ROLE_ORDER[slot]
            secondary = ROLE_ORDER[(slot + 7) % len(ROLE_ORDER)]
            # A moving head must not be attached to the translation-only
            # pelvis slot. Keep the pelvis available to the garland/global
            # field, while the first 13 moving outputs read the neck's
            # articulated vertical motion instead.
            if role == 0 and index < base:
                role = 12
            blend = 0. if cycle == 0 else .15 + .3 * _fraction(cycle)
            outputs.append({'name': f'{kind}_{index+1:03d}', 'kind': kind,
                'position': [float(x), 6. if kind == 'beam' else 3.8, -1. if kind == 'beam' else 1.5],
                'pan_range_deg': 540., 'tilt_range_deg': 270.,
                'role_index': int(role), 'secondary_role': int(secondary), 'role_blend': float(blend),
                'body_region': NAMES[role], 'spatial_slot': int(role_id),
                'anchor': [float(.90*x), .55 if kind == 'beam' else 1.15, -3.7 if kind == 'beam' else -5.2]})
    for index in range(_count(bulbs)):
        coordinate = index / 49 if index < 50 else _fraction(index - 49)
        x = 16 * (coordinate - .5)
        outputs.append({'name': f'bulb_{index+1:03d}', 'kind': 'bulb',
            'position': [float(x), float(5.7-.7*(x/8)**2), 3.],
            'field_coordinate': float(coordinate), 'body_region': 'anatomical_spatial_field'})
    if not outputs:
        raise ValueError('At least one output required')
    return outputs


def adapt_field(features, field, outputs):
    if not outputs or len({o['name'] for o in outputs}) != len(outputs):
        raise ValueError('Nonempty installation with unique fixture IDs required')
    signal = field.points
    frames, count = len(signal), len(outputs)
    targets = np.empty((frames, count, 3))
    sources = np.zeros_like(targets)
    levels = np.empty((frames, count))
    velocity = derivative(signal, features.fps)
    activity = field.activity
    if activity is None:
        activity = np.tanh(np.linalg.norm(velocity, axis=-1) / 1.5)
    # Static anatomical slots; moving source coordinates paint the garland.
    anchors = np.empty(22)
    anchors[np.asarray(ROLE_ORDER)] = np.linspace(.05, .95, 22)
    centers = np.clip(anchors[None] + .19*np.tanh((signal[..., 0] + .35*signal[..., 2] + .25*signal[..., 1])/.65), .01, .99)
    # A structural field is already a joint configuration. Independent fixture
    # offsets would destroy its distances and can even invert hand proximity.
    # Map each declared structure with ONE source-driven similarity transform.
    shared = np.zeros((frames, 22, 3))
    shared_mask = np.zeros(22, dtype=bool)
    shared_dependencies = {}
    shared_scales = []
    stage_center = np.array([0., 2.4, -3.5])
    stage_radius = np.array([7.5, 2.2, 2.3])
    rotation = np.array([-1., 1., -1.])  # fixed 180-degree spatial Y rotation
    for group in field.shared_groups:
        indices = np.asarray(group, dtype=int)
        reference_center = field.absolute_reference[indices].mean(axis=0)
        delta = (field.absolute_points[:, indices] - reference_center) * rotation * 2.
        extent = np.max(np.abs(delta) / stage_radius, axis=(1, 2))
        scale = 1. / np.maximum(1., extent)
        shared[:, indices] = stage_center + delta * scale[:, None, None]
        shared_mask[indices] = True
        dependencies = sorted({j for role in group for j in field.attribution[role]})
        for role in group:
            shared_dependencies[role] = dependencies
        shared_scales.append(scale)
        # Fixed lamps sample the same structural relationship; separate neutral
        # offsets must not invert proximity in the garland either.
        coordinates = shared[:, indices]
        centers[:, indices] = .5 + .44*np.tanh(
            coordinates[..., 0]/4. + .15*(coordinates[..., 1]-stage_center[1])/stage_radius[1]
            + .10*(coordinates[..., 2]-stage_center[2])/stage_radius[2])
    attribution = []
    for i, output in enumerate(outputs):
        if output['kind'] == 'bulb':
            delta = (float(output['field_coordinate']) - centers) / .038
            field_value = np.exp(-.5*delta*delta) * (.10 + .90*activity)
            levels[:, i] = .025 + .875*(-np.expm1(-.65*field_value.sum(axis=1)))
            targets[:, i] = output['position']
            attribution.append(sorted(set(j for a in field.attribution for j in a)))
            continue
        if output['kind'] not in ('beam', 'laser'):
            raise ValueError('Unknown fixture capability')
        a, b, blend = output['role_index'], output['secondary_role'], output['role_blend']
        point = (1-blend)*signal[:, a] + blend*signal[:, b]
        sources[:, i] = point
        # Complementary spatial bases; only the body changes in time.
        sign = -1. if output['spatial_slot'] % 2 else 1.
        transformed = np.column_stack((point[:, 0]+.30*sign*point[:, 2],
                                       point[:, 1]+.20*sign*point[:, 0],
                                       point[:, 2]-.25*point[:, 1]))
        raw = np.asarray(output['anchor']) + transformed * [2.6, 2.4, 2.2]
        # The previous tanh stage compressed every articulated signal before
        # angular decoding.  Use a fixed physical stage projection instead:
        # preserve the affine body-to-stage displacement and clip only at the
        # declared installation boundary.  This is spatial, source-driven and
        # independent of time, song, output count, and fixture brightness.
        delta = raw - np.asarray(output['anchor'])
        stage_scale = np.array([7.5, 5., 2.8])
        targets[:, i] = np.asarray(output['anchor']) + stage_scale * np.tanh(delta / (1.5 * stage_scale))
        if shared_mask[a] and blend == 0:
            targets[:, i] = shared[:, a]
        elif blend and (shared_mask[a] or shared_mask[b]):
            def role_target(role):
                if shared_mask[role]:
                    return shared[:, role]
                value = signal[:, role]
                transformed_role = np.column_stack((value[:, 0]+.30*sign*value[:, 2],
                    value[:, 1]+.20*sign*value[:, 0], value[:, 2]-.25*value[:, 1]))
                raw_role = np.asarray(output['anchor']) + transformed_role*[2.6, 2.4, 2.2]
                delta_role = raw_role - np.asarray(output['anchor'])
                stage_scale = np.array([7.5, 5., 2.8])
                return np.asarray(output['anchor']) + stage_scale * np.tanh(delta_role / (1.5 * stage_scale))
            targets[:, i] = (1-blend)*role_target(a) + blend*role_target(b)
        drive = (1-blend)*activity[:, a] + blend*activity[:, b]
        levels[:, i] = .08 + .72*drive
        attribution.append(sorted(set(shared_dependencies.get(a, field.attribution[a])) |
            (set(shared_dependencies.get(b, field.attribution[b])) if blend else set())))
    frame_scale = np.ones(frames)
    if not np.isfinite(targets).all() or not np.isfinite(levels).all():
        raise ValueError('Nonfinite capability mapping')
    result = {'targets': targets, 'dimmer': levels, 'source_points': sources,
            'field_points': signal, 'field_activity': activity,
            'body_relative': features.positions, 'bulb_field_centers': centers,
            'attribution': attribution, 'method': field.meta,
            'stage_projection_scale': frame_scale,
            'mapping_projection': 'affine_group_similarity_clip_v3'}
    if field.shared_groups:
        result.update(field_absolute_points=field.absolute_points,
                      shared_stage_points=shared, shared_role_mask=shared_mask,
                      shared_similarity_scale=np.stack(shared_scales, axis=1))
        result['method'] = {**field.meta, 'shared_structure_readout': {
            'groups': [list(group) for group in field.shared_groups],
            'stage_center': stage_center.tolist(), 'bounding_radius': stage_radius.tolist(),
            'spatial_rotation_diagonal': rotation.tolist(), 'nominal_scale': 2.,
            'minimum_fit_scale': float(np.min(shared_scales)),
            'scope': 'One similarity transform per whole structure; no per-fixture offsets inside a group.'}}
    result['mapping_projection'] = 'bounded_softclip_v4'
    return result
