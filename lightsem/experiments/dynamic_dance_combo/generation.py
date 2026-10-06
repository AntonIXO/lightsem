"""Frozen-checkpoint D7 generation with bounded D8 global-context transfer.

This is an experimental inference composition, not a trained joint model. The
same D8 weights evaluate global and local plans. Its decoder was trained with
global plans, not zero-plan dropout; local plans are a separately supervised
branch of the same planner. Both requested compositions start from one output
of this generator and are recomposed/retimed by the caller.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import torch

from lightsem.experiments.dynamic_dance.audio import model_features
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline, sha256
from lightsem.experiments.dynamic_dance.learning import PLAN_CHANNELS, plan_condition
from lightsem.experiments.dynamic_dance.training import (
    array_sha256, load_learned, planner_predict, timeline_provenance,
    validate_audio_binding,
)
from lightsem.experiments.motion_study.geometry import joints_from_features

RESIDUAL_CAP = .25
FLOW_STEPS = 32
DISPLAY_SEED = 0
FPS = 30.


def _finite_tensor(name, value):
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite FP32')


def bounded_residual(base, residual):
    """Bound each frame's normalized-native channel RMS, with no motion floor."""
    if base.shape != residual.shape or base.ndim != 3 or base.shape[-1] != 139:
        raise ValueError('Expected matching B,T,139 vector fields')
    _finite_tensor('D7 vector field', base)
    _finite_tensor('D8 planning residual', residual)
    base_rms = base.square().mean(-1).sqrt()
    before = residual.square().mean(-1).sqrt()
    scale = torch.minimum(torch.ones_like(before), RESIDUAL_CAP * base_rms / (before + 1e-8))
    bounded = residual * scale[..., None]
    after = bounded.square().mean(-1).sqrt()
    return bounded, {'d7_field_rms': base_rms, 'residual_rms_before': before,
                     'residual_rms_after': after, 'cap_scale': scale,
                     'cap_active': before > RESIDUAL_CAP * base_rms}


class ResidualTransferField:
    """Evaluate all component fields at the same evolving combined state.

    Cap after exact-halo evaluation of each full field. No reductions over time
    enter the cap, so neither chunk length nor neighboring unrelated frames can
    change a frame's residual scale.
    """
    def __init__(self, d7, d8, d7_audio, global_audio, local_audio, *, chunked=True, core_frames=960):
        if d7_audio.shape != global_audio.shape or global_audio.shape != local_audio.shape:
            raise ValueError('All packed audio tensors must have identical shapes')
        if d7_audio.ndim != 3 or core_frames < 1:
            raise ValueError('Expected B,T,F audio and positive core size')
        for name, value in (('D7 audio', d7_audio), ('global audio', global_audio), ('local audio', local_audio)):
            _finite_tensor(name, value)
        if not torch.equal(d7_audio[..., :-PLAN_CHANNELS], global_audio[..., :-PLAN_CHANNELS]) or not torch.equal(global_audio[..., :-PLAN_CHANNELS], local_audio[..., :-PLAN_CHANNELS]):
            raise ValueError('Only the plan may differ between component conditions')
        if bool(torch.any(d7_audio[..., -PLAN_CHANNELS:] != 0)):
            raise ValueError('D7 must expose its unused zero plan slots')
        self.d7, self.d8 = d7, d8
        self.d7_audio, self.global_audio, self.local_audio = d7_audio, global_audio, local_audio
        self.chunked, self.core_frames = chunked, core_frames

    def _field(self, model, audio, state, time):
        if self.chunked:
            return model.vector_field_chunked(audio, state, time, core_frames=self.core_frames)
        return model.vector_field(audio, state, time)

    def reference(self, state, time):
        return self._field(self.d7, self.d7_audio, state, time)

    def __call__(self, state, time):
        base = self.reference(state, time)
        global_field = self._field(self.d8, self.global_audio, state, time)
        local_field = self._field(self.d8, self.local_audio, state, time)
        bounded, telemetry = bounded_residual(base, global_field-local_field)
        return base+bounded, telemetry


@torch.inference_mode()
def integrate_shared_noise(field, initial_noise, *, steps=FLOW_STEPS, include_reference=True):
    """One full-sequence state per arm; common explicit noise, no chunk restart."""
    if initial_noise.ndim != 3 or initial_noise.shape[-1] != 139 or initial_noise.shape[:2] != field.d7_audio.shape[:2]:
        raise ValueError('Initial noise must match B,T,139 generation dimensions')
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('Positive integer Euler steps required')
    _finite_tensor('Initial noise', initial_noise)
    reference = initial_noise.clone() if include_reference else None
    combined = initial_noise.clone()
    history = {}
    with torch.autocast(device_type=initial_noise.device.type, enabled=False):
        for index in range(steps):
            time = initial_noise.new_full((len(initial_noise),), index/steps)
            if include_reference:
                reference_velocity = field.reference(reference, time)
                _finite_tensor('Reference velocity', reference_velocity)
            combined_velocity, observation = field(combined, time)
            if include_reference:
                reference = reference + reference_velocity/steps
                _finite_tensor('Reference state', reference)
            combined = combined + combined_velocity/steps
            _finite_tensor('Combined state', combined)
            for key, value in observation.items():
                history.setdefault(key, []).append(value.detach().cpu().numpy())
    return reference, combined, {key: np.stack(value) for key, value in history.items()}


def validate_compatible(d7, d7_state, d8, d8_state):
    """Reject mixing fields that represent different normalized coordinates."""
    if asdict(d7.config) != asdict(d8.config) or d7.config.arm != 'flow' or d7.config.steps != FLOW_STEPS:
        raise ValueError('D7/D8 must share the frozen 32-step flow architecture')
    if d7.dynamic.arm != 'D7' or d8.dynamic.arm != 'D8':
        raise ValueError('Require actual D7 and D8 checkpoint arms')
    if {k: v for k, v in asdict(d7.dynamic).items() if k != 'arm'} != {k: v for k, v in asdict(d8.dynamic).items() if k != 'arm'}:
        raise ValueError('D7/D8 adapter architectures differ')
    if asdict(d7.physical_weights) != asdict(d8.physical_weights):
        raise ValueError('D7/D8 physical training safeguards differ')
    for name in ('mean', 'std', 'delta_scale', 'noise_kernel', 'frequencies'):
        if not torch.equal(getattr(d7, name), getattr(d8, name)):
            raise ValueError(f'D7/D8 {name} differs')
    for key in ('normalizers', 'audio_normalizers', 'extra_normalizers'):
        left, right = d7_state[key], d8_state[key]
        if left.keys() != right.keys() or any(not np.array_equal(np.asarray(left[name]), np.asarray(right[name])) for name in left):
            raise ValueError(f'D7/D8 {key} differs')
    if d7_state['feature_names'] != d8_state['feature_names']:
        raise ValueError('D7/D8 feature schemas differ')
    if d7.codec.config_dict() != d8.codec.config_dict() or d7.halo != d8.halo:
        raise ValueError('D7/D8 decode geometry or temporal halo differs')
    return {'identical_coordinates': True, 'identical_audio_normalizers': True,
            'identical_decoder': True, 'halo_frames': d7.halo,
            'receptive_field_frames': 2*d7.halo+1, 'flow_steps': FLOW_STEPS}


def _interpolate(time, values, target):
    time, values = np.asarray(time), np.asarray(values)
    if time.ndim != 1 or len(time) != len(values) or len(time) == 0 or np.any(np.diff(time) <= 0):
        raise ValueError('Expected strictly increasing feature timestamps')
    if not np.isfinite(time).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite feature interpolation input')
    return np.stack([np.interp(target, time, values[:, i]) for i in range(values.shape[-1])], -1).astype(np.float32)


def _normalise(values, norms):
    mean, std = np.asarray(norms['mean']), np.asarray(norms['std'])
    if mean.shape != (values.shape[-1],) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid frozen feature normalizers')
    return ((values-mean)/std).astype(np.float32)


@torch.inference_mode()
def generate_full(oldroot, timeline, base92, seconds, device, drawseed=7001, *,
                  initial_noise=None, plan_intervention=None, output_common_reference=True):
    """Return complete-song arrays and provenance, without writing artifacts.

    ``timeline`` and ``base92`` are authenticated paths. Native frame count is
    ceil(seconds*30); final media duration is owned by the renderer and exact
    input audio. There is no hidden 180-second prefix clamp.
    """
    oldroot, timeline_path, base92_path = Path(oldroot), Path(timeline), Path(base92)
    timeline = MusicTimeline.load(timeline_path)
    binding = validate_audio_binding(timeline, base92_path)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Positive finite complete-song duration required')
    if 'duration_s' not in timeline.metadata or abs(float(timeline.metadata['duration_s'])-seconds) > 1/FPS+1e-8:
        raise ValueError('Requested duration must match the complete analyzed waveform')
    frames = int(math.ceil(seconds*FPS))
    if frames < 3:
        raise ValueError('At least three native frames required')
    time30 = np.arange(frames, dtype=np.float64)/FPS
    if timeline.time[0] > 1e-8 or timeline.time[-1] < time30[-1]-1/FPS-1e-8:
        raise ValueError('Timeline does not cover the complete generated song')
    device = torch.device(device)
    d7, s7, p7 = load_learned(oldroot, 'D7', DISPLAY_SEED, device=device)
    d8, s8, p8 = load_learned(oldroot, 'D8', DISPLAY_SEED, device=device)
    planner, sp, pp = load_learned(oldroot, 'planner', DISPLAY_SEED, device=device)
    compatibility = validate_compatible(d7, s7, d8, s8)
    raw, names = model_features(timeline, target_time=timeline.time)
    if names != s7['feature_names'] or names != sp['feature_names']:
        raise ValueError('Full-song feature schema differs from frozen checkpoints')
    if s7['extra_normalizers'].keys() != sp['extra_normalizers'].keys() or any(not np.array_equal(np.asarray(value), np.asarray(sp['extra_normalizers'][key])) for key, value in s7['extra_normalizers'].items()):
        raise ValueError('Planner uses different frozen audio feature normalization')
    predictions = planner_predict(planner, raw, sp['extra_normalizers'], device=device)
    coarse_time = np.arange(len(predictions['global']), dtype=np.float64)*.5
    unshifted_global_raw = predictions['global'].numpy().copy()
    if plan_intervention == 'shift_global_half':
        predictions['global'] = torch.roll(predictions['global'], len(predictions['global'])//2, dims=0)
    elif plan_intervention is not None:
        raise ValueError('Unknown planning intervention')
    coarse = {branch: plan_condition(predictions[branch]).numpy() for branch in ('local', 'global')}
    plans = {branch: _interpolate(coarse_time, values, time30) for branch, values in coarse.items()}
    with np.load(base92_path, allow_pickle=False) as arrays:
        if 'x30' in arrays and 'time30' in arrays:
            base, base_time = arrays['x30'].copy(), arrays['time30'].copy()
        elif 'x' in arrays and 'time' in arrays:
            base, base_time = arrays['x'].copy(), arrays['time'].copy()
        else:
            raise ValueError('Base92 requires explicit timestamps')
    if base.ndim != 2 or base.shape[-1] != 92 or len(base_time) < 1 or base_time[0] > time30[0]+1e-8 or base_time[-1] < time30[-1]-1/FPS-1e-8:
        raise ValueError('Base92 must cover the complete generated song')
    shared = np.concatenate((_normalise(_interpolate(base_time, base, time30), s7['audio_normalizers']),
                             _normalise(_interpolate(timeline.time, raw, time30), s7['extra_normalizers'])), -1)
    packed = {branch: torch.from_numpy(np.concatenate((shared, plan), -1))[None].to(device)
              for branch, plan in {**plans, 'D7': np.zeros((frames, PLAN_CHANNELS), np.float32)}.items()}
    if initial_noise is None:
        generator = torch.Generator(device=device).manual_seed(drawseed)
        initial = d7.noise(1, frames, generator=generator, device=device)
        noise_origin = 'generated once at complete-song length from declared draw seed'
    else:
        initial = torch.as_tensor(initial_noise, device=device)
        if initial.ndim == 2:
            initial = initial[None]
        if initial.shape != (1, frames, 139):
            raise ValueError('Explicit initial noise must match the complete-song frame count')
        _finite_tensor('Explicit initial noise', initial)
        initial = initial.clone()
        noise_origin = 'caller supplied explicit noise; actual values authenticated independently of draw-seed label'
    field = ResidualTransferField(d7, d8, packed['D7'], packed['global'], packed['local'])
    reference, combined, history = integrate_shared_noise(field, initial, include_reference=output_common_reference)
    outputs = {}
    for name, normalized in (('reference', reference), ('combined', combined)):
        if normalized is None:
            continue
        native = d7.codec.decode(normalized.float()*d7.std+d7.mean)
        world = joints_from_features(native)
        _finite_tensor(name+' native', native)
        _finite_tensor(name+' world', world)
        outputs[name] = {'native': native[0].cpu().numpy(), 'world': world[0].cpu().numpy(),
                         'contacts': native[0, :, 135:].cpu().numpy()}
    telemetry = {'initial_noise': initial[0].cpu().numpy(), 'coarse_time': coarse_time,
                 'coarse_global_raw_unshifted': unshifted_global_raw,
                 **{f'coarse_{branch}_raw': value.numpy() for branch, value in predictions.items()},
                 **{f'coarse_{branch}_plan': value for branch, value in coarse.items()},
                 **{f'{branch}_plan': value for branch, value in plans.items()},
                 **{key: value[:, 0] for key, value in history.items()}}
    metadata = {'kind': 'D7_with_D8_global_minus_local_residual', 'display_seed': DISPLAY_SEED,
                'draw_seed': drawseed, 'native_fps': FPS, 'frames': frames,
                'initial_noise_origin': noise_origin, 'initial_noise_time_indices': [0, frames-1],
                'output_common_reference': bool(output_common_reference),
                'plan_intervention': plan_intervention,
                'audio_duration_seconds': float(seconds), 'native_frame_extent_seconds': float(time30[-1]),
                'native_frame_count_duration_seconds': frames/FPS, 'newly_generated_motion': True,
                'source_clock': 'fresh native30Hz flow; shared explicit full-sequence initial noise',
                'full_song_noise_caveat': 'A longer full-song noise tensor does not reproduce a former 180-second prefix merely by retaining its draw seed.',
                'formula': 'v7 + (v8(global_plan)-v8(local_plan))*min(1,.25*RMS_channels(v7)/(RMS_channels(delta)+1e-8))',
                'residual_cap': RESIDUAL_CAP, 'residual_cap_scope': 'each frame, normalized native139 channel RMS; no nonzero floor',
                'interpretation': 'Experimental inference residual transfer; D8 decoder trained global plans only, no plan dropout; local branch is supervised by the same planner objectives.',
                'integration': '32 FP32 Euler steps, one full state per arm, exact-halo field evaluation, no chunk restarts',
                'compatibility': compatibility, **timeline_provenance(timeline, timeline_path),
                'base92_sha256': sha256(base92_path), 'base92_analyzed_waveform_sha256': binding['analyzed_waveform_sha256'],
                'checkpoints': {name: {'path': str(path.resolve()), 'sha256': sha256(path)} for name, path in (('D7', p7), ('D8', p8), ('planner', pp))},
                'initial_noise_sha256': array_sha256(initial), 'shared_audio_sha256': array_sha256(shared),
                'global_plan_sha256': array_sha256(plans['global']), 'local_plan_sha256': array_sha256(plans['local']),
                'cap_active_fraction': float(history['cap_active'].mean()),
                'residual_rms_before_mean': float(history['residual_rms_before'].mean()),
                'residual_rms_after_mean': float(history['residual_rms_after'].mean()),
                'combined_minus_reference_native_rms': float(np.sqrt(np.mean((outputs['combined']['native'].astype(np.float64)-outputs['reference']['native'])**2))) if output_common_reference else None,
                'combined_minus_reference_world_rms_m': float(np.sqrt(np.mean((outputs['combined']['world'].astype(np.float64)-outputs['reference']['world'])**2))) if output_common_reference else None,
                'output_array_sha256': {arm: {key: array_sha256(value) for key, value in arrays.items()} for arm, arrays in outputs.items()}}
    return {'time': time30, **outputs, 'telemetry': telemetry, 'metadata': metadata}
