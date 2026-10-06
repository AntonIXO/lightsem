"""CPU engineering checks; passing these does not qualify musical quality."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightsem.experiments.dynamic_dance.learning import DynamicFlow, DynamicFlowConfig
from lightsem.experiments.dynamic_dance_combo import generation as g
from lightsem.experiments.motion_study.music_model import MusicConfig


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def models():
    torch.manual_seed(902)
    norms = {'mean': [0.]*139, 'std': [1.]*139, 'delta_scale': [1.]*139}
    cfg = MusicConfig(width=4, frames=17, dilations=(1,), steps=32)
    left = DynamicFlow(cfg, norms, DynamicFlowConfig(extra_dim=2, arm='D7', adapter_width=4)).eval()
    right = DynamicFlow(cfg, norms, DynamicFlowConfig(extra_dim=2, arm='D8', adapter_width=4)).eval()
    with torch.no_grad():
        for model in (left, right):
            model.output.weight.normal_(std=.15)
            model.output.bias.normal_(std=.03)
            for layer in model.adapters:
                layer.weight.normal_(std=.3)
    state = {'normalizers': norms, 'audio_normalizers': {'mean': [0.]*92, 'std': [1.]*92},
             'extra_normalizers': {'mean': [0.]*2, 'std': [1.]*2}, 'feature_names': ['a', 'b']}
    return left, right, copy.deepcopy(state), copy.deepcopy(state)


def field_for(left, right, frames=29, *, same_plan=False, **kwargs):
    shared = torch.randn(1, frames, 94)
    global_plan = torch.randn(1, frames, 34)
    local_plan = global_plan.clone() if same_plan else torch.randn(1, frames, 34)
    packed = [torch.cat((shared, x), -1) for x in (torch.zeros_like(global_plan), global_plan, local_plan)]
    return g.ResidualTransferField(left, right, *packed, **kwargs)


def test_bound_is_per_frame_no_floor_and_never_exceeds_quarter():
    base = torch.ones(1, 3, 139)
    base[:, 1] = 0
    base[:, 2] = .4
    raw = torch.ones_like(base)*2
    bounded, telemetry = g.bounded_residual(base, raw)
    assert torch.count_nonzero(bounded[:, 1]) == 0
    assert torch.all(telemetry['residual_rms_after'] <= .25*telemetry['d7_field_rms']+1e-7)
    assert torch.allclose(bounded[0, 0], torch.full((139,), .25))
    assert torch.allclose(bounded[0, 2], torch.full((139,), .1))
    changed = base.clone(); changed[:, 2] *= 1000
    revised, _ = g.bounded_residual(changed, raw)
    assert torch.equal(revised[:, :2], bounded[:, :2])


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_field_fails(bad):
    base = torch.ones(1, 2, 139)
    raw = base.clone(); raw[0, 0, 0] = bad
    with pytest.raises(ValueError, match='finite FP32'):
        g.bounded_residual(base, raw)


def test_matching_plans_cancel_exactly_and_reference_matches_existing_generator():
    left, right, _, _ = models()
    field = field_for(left, right, same_plan=True, core_frames=7)
    initial = left.noise(1, 29, generator=torch.Generator().manual_seed(7001))
    before = initial.clone()
    reference, combined, hist = g.integrate_shared_noise(field, initial)
    assert torch.equal(initial, before)
    assert torch.equal(reference, combined)
    assert np.count_nonzero(hist['residual_rms_before']) == 0
    expected = left.generate(field.d7_audio, initial_noise=initial, chunked=True)
    decoded = left.codec.decode(reference*left.std+left.mean)
    assert torch.allclose(expected, decoded, atol=2e-7, rtol=1e-6)


def test_nonzero_global_context_changes_generation_and_exact_halo_matches_whole():
    left, right, _, _ = models()
    chunked = field_for(left, right, core_frames=7)
    whole = g.ResidualTransferField(left, right, chunked.d7_audio, chunked.global_audio, chunked.local_audio, chunked=False)
    initial = torch.randn(1, 29, 139)
    reference, combined, hist = g.integrate_shared_noise(chunked, initial, steps=3)
    ref_whole, combo_whole, whole_hist = g.integrate_shared_noise(whole, initial, steps=3)
    assert not torch.equal(reference, combined)
    assert np.max(hist['residual_rms_after']) > 0
    assert torch.allclose(reference, ref_whole, atol=3e-7, rtol=1e-6)
    assert torch.allclose(combined, combo_whole, atol=3e-7, rtol=1e-6)
    np.testing.assert_allclose(hist['residual_rms_after'], whole_hist['residual_rms_after'], atol=3e-7, rtol=1e-5)


def test_omitted_reference_does_not_change_combination():
    left, right, _, _ = models()
    field = field_for(left, right)
    initial = torch.randn(1, 29, 139)
    _, first, hist = g.integrate_shared_noise(field, initial, steps=3)
    reference, second, second_hist = g.integrate_shared_noise(field, initial, steps=3, include_reference=False)
    assert reference is None
    assert torch.equal(first, second)
    for key in hist:
        np.testing.assert_array_equal(hist[key], second_hist[key])


def test_different_nonplan_audio_is_rejected():
    left, right, _, _ = models()
    field = field_for(left, right)
    bad = field.local_audio.clone(); bad[..., 0] += .1
    with pytest.raises(ValueError, match='Only the plan'):
        g.ResidualTransferField(left, right, field.d7_audio, field.global_audio, bad)


@pytest.mark.parametrize('what', ['mean', 'std', 'noise_kernel', 'audio_normalizers', 'features', 'arm'])
def test_incompatible_checkpoints_rejected(what):
    left, right, s7, s8 = models()
    assert g.validate_compatible(left, s7, right, s8)['identical_coordinates']
    if what in ('mean', 'std', 'noise_kernel'):
        with torch.no_grad(): getattr(right, what).flatten()[0] += .01
    elif what == 'audio_normalizers':
        s8['audio_normalizers']['std'][0] = 2.
    elif what == 'features':
        s8['feature_names'][0] = 'wrong'
    else:
        right.dynamic = DynamicFlowConfig(extra_dim=2, arm='D8_flat', adapter_width=4)
    with pytest.raises(ValueError):
        g.validate_compatible(left, s7, right, s8)


@pytest.fixture
def full_inputs(tmp_path, monkeypatch):
    left, right, s7, s8 = models()
    seconds = .55
    frames = math.ceil(seconds*30)
    time = np.arange(frames)/30.
    timeline_path = tmp_path/'timeline.npz'; timeline_path.write_bytes(b'authenticated elsewhere in fixture')
    timeline_path.with_suffix('.json').write_text('{}')
    base_path = tmp_path/'base92.npz'; np.savez(base_path, x30=np.zeros((frames, 92), np.float32), time30=time)
    checkpoint_paths = {}
    for name in ('D7', 'D8', 'planner'):
        checkpoint_paths[name] = tmp_path/(name+'.pt'); checkpoint_paths[name].write_bytes(name.encode())
    timeline = SimpleNamespace(time=time, metadata={'duration_s': seconds, 'analyzed_waveform_sha256': 'a'*64})
    monkeypatch.setattr(g.MusicTimeline, 'load', lambda path: timeline)
    monkeypatch.setattr(g, 'validate_audio_binding', lambda tl, base: {'analyzed_waveform_sha256': 'a'*64})
    monkeypatch.setattr(g, 'load_learned', lambda root, arm, seed, device: ({'D7': left, 'D8': right, 'planner': None}[arm], {'D7': s7, 'D8': s8, 'planner': s7}[arm], checkpoint_paths[arm]))
    monkeypatch.setattr(g, 'model_features', lambda tl, target_time: (np.zeros((len(target_time), 2), np.float32), ['a', 'b']))
    raw_local = torch.arange(68, dtype=torch.float32).reshape(2, 34)*.01
    raw_global = raw_local + torch.tensor([.2, -.1])[:, None]
    monkeypatch.setattr(g, 'planner_predict', lambda *args, **kwargs: {'local': raw_local.clone(), 'global': raw_global.clone()})
    return tmp_path, timeline_path, base_path, seconds, left


def test_full_generation_ceils_tail_uses_explicit_noise_and_records_actual_shift(full_inputs):
    root, timeline_path, base_path, seconds, left = full_inputs
    # One maximum-track field, then a time prefix: independently drawing a
    # shorter field would not supply matching per-channel values.
    maximum = left.noise(1, 40, generator=torch.Generator().manual_seed(7001))
    explicit = maximum[:, :17].detach().numpy().copy()
    result = g.generate_full(root, timeline_path, base_path, seconds, 'cpu', initial_noise=explicit)
    shifted = g.generate_full(root, timeline_path, base_path, seconds, 'cpu', initial_noise=explicit,
                              plan_intervention='shift_global_half', output_common_reference=False)
    assert len(result['time']) == 17
    assert result['time'][-1] < seconds
    np.testing.assert_array_equal(result['telemetry']['initial_noise'], explicit[0])
    np.testing.assert_array_equal(result['telemetry']['local_plan'], shifted['telemetry']['local_plan'])
    np.testing.assert_array_equal(shifted['telemetry']['coarse_global_raw'], np.roll(result['telemetry']['coarse_global_raw'], 1, axis=0))
    np.testing.assert_array_equal(shifted['telemetry']['coarse_global_plan'], np.roll(result['telemetry']['coarse_global_plan'], 1, axis=0))
    assert 'reference' not in shifted
    assert shifted['metadata']['combined_minus_reference_world_rms_m'] is None
    assert result['metadata']['initial_noise_sha256'] == shifted['metadata']['initial_noise_sha256']
    assert result['metadata']['global_plan_sha256'] != shifted['metadata']['global_plan_sha256']
    assert result['telemetry']['residual_rms_after'].shape == (32, 17)
    for arm in ('reference', 'combined'):
        assert result[arm]['world'].shape == (17, 22, 3)
        assert result[arm]['native'].shape == (17, 139)
        assert result[arm]['contacts'].shape == (17, 4)
        assert np.isfinite(result[arm]['world']).all()


def test_explicit_wrong_length_fails_instead_of_crop_or_redraw(full_inputs):
    root, timeline, base, seconds, _ = full_inputs
    with pytest.raises(ValueError, match='frame count'):
        g.generate_full(root, timeline, base, seconds, 'cpu', initial_noise=np.zeros((16, 139), np.float32))


def test_seed_generation_repeats_exactly(full_inputs):
    root, timeline, base, seconds, _ = full_inputs
    first = g.generate_full(root, timeline, base, seconds, 'cpu')
    second = g.generate_full(root, timeline, base, seconds, 'cpu')
    for arm in ('reference', 'combined'):
        for key in first[arm]: np.testing.assert_array_equal(first[arm][key], second[arm][key])
    assert first['metadata']['initial_noise_sha256'] == second['metadata']['initial_noise_sha256']


def test_rejects_prefix_as_complete_song(full_inputs):
    root, timeline, base, _, _ = full_inputs
    with pytest.raises(ValueError, match='complete analyzed waveform'):
        g.generate_full(root, timeline, base, .2, 'cpu')
