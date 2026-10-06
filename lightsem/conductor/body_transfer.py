"""Retarget a saved dance to arbitrary virtual light outputs.

No audio, beat phase, temporal oscillator, random source or motion generator is
accepted here. Every changing target is a spatial transform of the supplied
body. Extra fixtures sample articulated limb chains, not copies of six DMX
channels. Colors are provided by the independently selected P2 color layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Joint ordering is the canonical native139 / FineDance 22-joint skeleton.
CHAINS = ((16, 18, 20), (1, 4, 7, 10), (2, 5, 8, 11), (17, 19, 21))
CHAIN_NAMES = ('left_arm', 'left_leg', 'right_leg', 'right_arm')


@dataclass(frozen=True)
class TransferConfig:
    fps: float = 40.
    target_x_gain: float = 6.
    target_height_gain: float = 4.
    target_depth_gain: float = 3.
    target_bound_x: float = 7.5
    target_bound_u: float = 12.5
    body_speed_scale: float = 2.
    maximum_speed_deg_s: float = 90.
    maximum_acceleration_deg_s2: float = 360.
    apply_motion_limits: bool = True  # False emits requested angles directly for simulation.


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10000:
        raise ValueError('Output counts must be integers in 0..10000')
    return value


def source_weights(chain, fraction):
    """A point on a fixed articulated chain, with stable anatomical identity."""
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Chain fraction must be in [0,1]')
    position = fraction * (len(chain) - 1)
    left = min(int(position), len(chain) - 2)
    blend = position - left
    weights = np.zeros(22)
    weights[chain[left]] += 1 - blend
    weights[chain[left + 1]] += blend
    return weights


def build_outputs(beams=8, lasers=5, bulbs=50):
    """Output geometry and body attachments; all constants are time-independent.

    Laser means a simulated thin ray. This API emits no hardware packets.
    The per-output matrix spreads related gestures spatially without a delay.
    """
    outputs = []
    for kind, count in (('beam', beams), ('laser', lasers)):
        count = _count(count)
        for index in range(count):
            chain_index = index % len(CHAINS)
            within = index // len(CHAINS)
            population = (count - 1 - chain_index) // len(CHAINS) + 1
            fraction = 1. if population == 1 else .55 + .45 * within / (population - 1)
            x = 0. if count == 1 else -6.5 + 13 * index / (count - 1)
            # This is a fixed spatial basis, not a time- or beat-driven rotation.
            shear = 0. if population == 1 else -.25 + .5 * within / (population - 1)
            outputs.append({
                'name': f'{kind}_{index + 1:03d}', 'kind': kind,
                'position': [x, 6. if kind == 'beam' else 3.8, -1. if kind == 'beam' else 1.5],
                'body_region': CHAIN_NAMES[chain_index], 'chain_fraction': fraction,
                'joint_weights': source_weights(CHAINS[chain_index], fraction).tolist(),
                'spatial_shear': shear,
                'pan_range_deg': 540., 'tilt_range_deg': 270.,
            })
    count = _count(bulbs)
    for index in range(count):
        coordinate = .5 if count == 1 else index / (count - 1)
        x = 16 * (coordinate - .5)
        outputs.append({'name': f'bulb_{index + 1:03d}', 'kind': 'bulb',
                        'position': [x, 5.7 - .7 * (x / 8) ** 2, 3.],
                        'field_coordinate': coordinate,
                        'body_region': 'four_limb_spatial_field'})
    if not outputs:
        raise ValueError('At least one output is required')
    return outputs


def sample_body(world, source_fps=30., target_fps=40.):
    """Fixed physical timestamps; no retiming, crop-wise root integration or RNG."""
    world = np.asarray(world, dtype=float)
    if world.ndim != 3 or world.shape[1:] != (22, 3) or len(world) < 2 or not np.isfinite(world).all():
        raise ValueError('Expected finite full body (T>=2,22,3)')
    if min(source_fps, target_fps) <= 0 or not np.isfinite([source_fps, target_fps]).all():
        raise ValueError('Positive finite frame rates required')
    times = np.arange(round(len(world) / source_fps * target_fps)) / target_fps
    native = np.arange(len(world)) / source_fps
    flat = world.reshape(len(world), -1)
    result = np.stack([np.interp(times, native, flat[:, j]) for j in range(flat.shape[1])], 1)
    return times, result.reshape(-1, 22, 3)


def raw_transfer(body, outputs, config=None):
    """Return desired targets/levels and traceable source channels before limits.

    Horizontal root drift is omitted from the local installation; root height,
    body orientation and limb articulation remain. Four current limb locations
    paint a spatial intensity field on the fixed bulbs. There is no chase clock.
    """
    cfg = config or TransferConfig()
    body = np.asarray(body, float)
    if body.ndim != 3 or body.shape[1:] != (22, 3) or len(body) < 2 or not np.isfinite(body).all():
        raise ValueError('Expected finite (T>=2,22,3) body')
    if not outputs or cfg.fps <= 0 or cfg.body_speed_scale <= 0:
        raise ValueError('Outputs and positive timing/scales are required')
    # Subtract horizontal translation only. No smoothing away fast limb motion.
    relative = body - body[:, 0:1] * np.array([1., 0., 1.])
    velocity = np.diff(relative, axis=0, prepend=relative[:1]) * cfg.fps
    speed = np.linalg.norm(velocity, axis=-1)
    targets = np.zeros((len(body), len(outputs), 3))
    levels = np.zeros((len(body), len(outputs)))
    sources = np.zeros_like(targets)
    tips = relative[:, [20, 10, 11, 21]]
    # Project x plus a small signed depth term, then bound to the bulb strip.
    centers = .5 + .48 * np.tanh((tips[..., 0] + .35 * tips[..., 2]) / .8)
    tip_drive = .15 + .85 * np.tanh(speed[:, [20, 10, 11, 21]] / cfg.body_speed_scale)
    for index, output in enumerate(outputs):
        if output['kind'] == 'bulb':
            distance = (output['field_coordinate'] - centers) / .075
            field = np.exp(-.5 * distance ** 2) * tip_drive
            levels[:, index] = .04 + .86 * (1 - np.prod(1 - field, axis=1))
            targets[:, index] = output['position']
            continue
        if output['kind'] not in ('beam', 'laser'):
            raise ValueError('Unknown output kind')
        weights = np.asarray(output['joint_weights'], float)
        if weights.shape != (22,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.):
            raise ValueError('Body attachments require normalized nonnegative joint weights')
        point = np.einsum('tjc,j->tc', relative, weights)
        sources[:, index] = point
        shear = float(output['spatial_shear'])
        x_raw = cfg.target_x_gain * (point[:, 0] + shear * (point[:, 1] - 1.))
        x = cfg.target_bound_x * np.tanh(x_raw / cfg.target_bound_x)
        # Preserve limb height and depth on a continuous floor/back-wall chart.
        u_raw = 6.25 + cfg.target_height_gain * (point[:, 1] - 1.) - cfg.target_depth_gain * point[:, 2]
        half = cfg.target_bound_u / 2
        u = half + half * np.tanh((u_raw - half) / half)
        targets[:, index, 0] = x
        targets[:, index, 1] = np.maximum(u - 7., 0.)
        targets[:, index, 2] = 1. - np.minimum(u, 7.)
        local_speed = np.linalg.norm(np.einsum('tjc,j->tc', velocity, weights), axis=1)
        levels[:, index] = .28 + .57 * np.tanh(local_speed / cfg.body_speed_scale)
    if not np.isfinite(targets).all() or not np.isfinite(levels).all():
        raise ValueError('Nonfinite body transfer')
    return {'targets': targets, 'dimmer': levels, 'source_points': sources,
            'body_relative': relative, 'bulb_field_centers': centers,
            'limb_speed': speed[:, [20, 10, 11, 21]]}


def constrained_transfer(body, outputs, left_rgb, right_rgb, config=None):
    """Fit body targets to emitted 16-bit pan/tilt; pack arbitrary DMX universes.

    Mechanical fitting is a projection of supplied targets, not a replacement
    trajectory. The preview and audit must decode these bytes, not raw targets.
    """
    cfg = config or TransferConfig()
    return compile_intent(raw_transfer(body, outputs, cfg), outputs, left_rgb, right_rgb, cfg)


def compile_intent(result, outputs, left_rgb, right_rgb, config=None):
    """Compile supplied spatial intent through the same V1 mechanics and patch.

    This boundary permits alternative body mappings without changing V1 raw
    transfer. It accepts no audio, animation clock, or replacement controller.
    """
    from types import SimpleNamespace
    from lightsem.experiments.motion_study.offline_angles import project_angles
    from lightsem.experiments.beat_conditioned.beat_conditioned_render import _angles
    cfg = config or TransferConfig()
    targets = np.asarray(result['targets'], dtype=float)
    levels = np.asarray(result['dimmer'], dtype=float)
    frames = len(targets)
    if not outputs or targets.shape != (frames, len(outputs), 3) or frames < 2:
        raise ValueError('Expected targets (T>=2,N,3) and nonempty outputs')
    if levels.shape != (frames, len(outputs)) or not np.isfinite(targets).all() or not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1)):
        raise ValueError('Finite targets and dimmers in [0,1] required')
    if not isinstance(cfg.apply_motion_limits, bool):
        raise ValueError('apply_motion_limits must be an explicit boolean')
    limits = [cfg.fps] + ([cfg.maximum_speed_deg_s, cfg.maximum_acceleration_deg_s2] if cfg.apply_motion_limits else [])
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError('Positive finite timing and mechanical limits required')
    if len({output['name'] for output in outputs}) != len(outputs):
        raise ValueError('Fixture identities must be unique')
    for index, output in enumerate(outputs):
        if output['kind'] not in ('beam', 'laser', 'bulb'):
            raise ValueError('Unknown fixture capability')
        position = np.asarray(output['position'], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Finite fixture position required')
        if output['kind'] != 'bulb':
            ranges = np.asarray([output['pan_range_deg'], output['tilt_range_deg']], dtype=float)
            if not np.isfinite(ranges).all() or np.any(ranges <= 0):
                raise ValueError('Positive finite angular ranges required')
            if np.any(np.linalg.norm(targets[:, index] - position, axis=1) < 1e-12):
                raise ValueError('Moving fixture target cannot coincide with its origin')
    left_rgb, right_rgb = np.asarray(left_rgb), np.asarray(right_rgb)
    if left_rgb.shape != (frames, 3) or right_rgb.shape != left_rgb.shape:
        raise ValueError('One RGB color per side at every body timestamp required')
    if left_rgb.dtype != np.uint8 or right_rgb.dtype != np.uint8:
        raise ValueError('Use actual quantized color-layer RGB')
    controls, patch = [], []
    universe, address = 0, 0
    constraints = SimpleNamespace(fps=cfg.fps, maximum_speed_deg_s=cfg.maximum_speed_deg_s,
        maximum_acceleration_deg_s2=cfg.maximum_acceleration_deg_s2,
        quiet_target_speed_deg_s=cfg.maximum_speed_deg_s)
    for index, output in enumerate(outputs):
        width = 4 if output['kind'] == 'bulb' else 8
        if address + width > 512:
            universe += 1
            address = 0
        values = np.zeros((frames, width), np.uint8)
        offset = 0
        if output['kind'] != 'bulb':
            pan, tilt = _angles(targets[:, index] - np.asarray(output['position']))
            desired = np.column_stack((pan, tilt))
            ranges = np.array([output['pan_range_deg'], output['tilt_range_deg']])
            if not cfg.apply_motion_limits:
                # No temporal optimizer, smoothing, speed or acceleration cap.
                # Keep only representable 16-bit pan/tilt encoding.
                ticks = np.clip(np.rint((.5 + desired / ranges) * 65535), 0, 65535).astype(np.int64)
            elif np.all(desired == desired[:1]):
                # Exact optimum for a constant target. Avoid solver tolerance
                # moving a stationary body across a 16-bit rounding boundary.
                ticks = np.broadcast_to(np.clip(np.rint((.5 + desired[:1] / ranges) * 65535), 0, 65535), desired.shape).astype(np.int64)
            else:
                ticks = project_angles(desired, ranges, np.ones(frames), constraints)
            values[:, :4] = np.stack((ticks[:, 0] >> 8, ticks[:, 0] & 255,
                                      ticks[:, 1] >> 8, ticks[:, 1] & 255), 1)
            offset = 4
        values[:, offset] = np.rint(levels[:, index] * 255).astype(np.uint8)
        values[:, offset + 1:offset + 4] = left_rgb if output['position'][0] < 0 else right_rgb
        controls.append(values)
        patch.append({**output, 'universe': universe, 'address': address + 1, 'n_channels': width,
                      'channels': (['pan_coarse', 'pan_fine', 'tilt_coarse', 'tilt_fine'] if offset else []) + ['dimmer', 'red', 'green', 'blue']})
        address += width
    dmx = np.zeros((frames, universe + 1, 512), np.uint8)
    for output, values in zip(patch, controls):
        start = output['address'] - 1
        dmx[:, output['universe'], start:start + output['n_channels']] = values
    return dmx, patch, result


def decode(dmx, patch):
    """Decode exactly the controls that the virtual output renderer consumes."""
    dmx = np.asarray(dmx)
    if dmx.dtype != np.uint8 or dmx.ndim != 3 or dmx.shape[-1] != 512:
        raise ValueError('Expected (T,universes,512) uint8 DMX')
    angles = np.zeros((len(dmx), len(patch), 2))
    directions = np.zeros((len(dmx), len(patch), 3))
    rgb = np.zeros((len(dmx), len(patch), 3))
    dimmer = np.zeros((len(dmx), len(patch)))
    for index, output in enumerate(patch):
        a = output['address'] - 1
        values = dmx[:, output['universe'], a:a + output['n_channels']].astype(float)
        offset = 0
        if output['kind'] != 'bulb':
            ticks = values[:, [0, 2]] * 256 + values[:, [1, 3]]
            angles[:, index] = (ticks / 65535 - .5) * [output['pan_range_deg'], output['tilt_range_deg']]
            pan, tilt = np.deg2rad(angles[:, index]).T
            directions[:, index] = np.column_stack((np.sin(pan) * np.cos(tilt),
                                                    -np.cos(pan) * np.cos(tilt), np.sin(tilt)))
            offset = 4
        dimmer[:, index] = values[:, offset] / 255
        rgb[:, index] = values[:, offset + 1:offset + 4] / 255
    return {'angles_deg': angles, 'directions': directions, 'rgb': rgb, 'dimmer': dimmer}
