"""Functional contracts for local audio-to-motion fusion, including padding."""
import pytest
import torch
from lightsem.experiments.native_dynamics.local_audio_fusion import LocalAudioFusion
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from lightsem.experiments.motion_study.music_model import MusicConfig

torch.set_num_threads(1)


def model(mode='none'):
    norms = {'mean': [0.]*139, 'std': [1.]*139, 'delta_scale': [1.]*139}
    return NativeDynamicsModel(MusicConfig(width=16, frames=48, dilations=(1, 2)), norms,
        Config(extra_dim=10, planner_width=16, audio_fusion=mode, audio_cross_width=16))


@pytest.mark.parametrize('mode', ['attention', 'film'])
def test_zero_initialization_preserves_incumbent_generation(mode):
    torch.manual_seed(0)
    incumbent, candidate = model(), model(mode)
    torch.nn.init.normal_(incumbent.flow.output.weight, std=.01)
    missing = candidate.load_state_dict(incumbent.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('flow.local_audio.') for k in missing.missing_keys)
    fine, coarse = torch.randn(1, 48, 102), torch.randn(1, 8, 10)
    mask, idx = torch.ones(1, 8, dtype=torch.bool), torch.arange(48)[None]
    noise = incumbent.flow.noise(1, 48)
    a, _ = incumbent.generate(fine, coarse, mask, idx, noise)
    b, _ = candidate.generate(fine, coarse, mask, idx, noise)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['attention', 'film'])
def test_audio_padding_does_not_leak_and_no_valid_audio_is_zero(mode):
    module = LocalAudioFusion(10, 16, 16, 4, mode)
    torch.nn.init.normal_(module.output.weight, std=.1)
    module.output.bias.data.fill_(.2)
    audio, motion = torch.randn(2, 150, 10), torch.randn(2, 150, 16)
    valid = torch.ones(2, 150, dtype=torch.bool); valid[0, 80:] = False; valid[1] = False
    changed = audio.clone(); changed[~valid] = float('nan')
    out = module(motion, module.prepare(changed, valid))
    ref = module(motion, module.prepare(audio, valid))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert torch.isfinite(out).all() and torch.count_nonzero(out[1]) == 0
    out.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())


@pytest.mark.parametrize('mode', ['attention', 'film'])
def test_local_audio_support_and_motion_query_gradients(mode):
    module = LocalAudioFusion(10, 16, 16, 4, mode)
    torch.nn.init.normal_(module.output.weight, std=.1)
    audio = torch.randn(1, 150, 10, requires_grad=True)
    motion = torch.randn(1, 150, 16, requires_grad=True)
    output = module(motion, module.prepare(audio))
    output[:, 75].square().sum().backward()
    assert audio.grad[:, 75].abs().sum() > 0
    assert motion.grad[:, 75].abs().sum() > 0
    assert audio.grad[:, :15].abs().sum() == 0
    assert audio.grad[:, 136:].abs().sum() == 0
    altered = audio.detach().clone(); altered[:, 75] += 5
    assert not torch.equal(output[:, 75], module(motion, module.prepare(altered))[:, 75])


@pytest.mark.parametrize('mode', ['attention', 'film'])
def test_rollout_and_chunked_generation_use_identical_audio_support(mode):
    torch.manual_seed(12)
    m = model(mode).eval()
    torch.nn.init.normal_(m.flow.output.weight, std=.01)
    torch.nn.init.normal_(m.flow.local_audio.output.weight, std=.1)
    fine, coarse = torch.randn(1, 150, 102), torch.randn(1, 12, 10)
    mask, idx = torch.ones(1, 12, dtype=torch.bool), torch.arange(150)[None]
    noise = m.flow.noise(1, 150)
    condition, _ = m.condition(fine, coarse, mask, idx)
    train, _ = m.rollout(condition, noise, gradient_steps=4)
    whole, _ = m.generate(fine, coarse, mask, idx, noise, core_frames=960)
    chunk, _ = m.generate(fine, coarse, mask, idx, noise, core_frames=17)
    torch.testing.assert_close(train, whole, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunk, whole, rtol=1e-5, atol=1e-5)
    train.square().mean().backward()
    for module in (m.flow.local_audio.audio, m.flow.local_audio.query, m.flow.local_audio.key):
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0


def test_controls_have_equal_trainable_capacity():
    a, b = model('attention'), model('film')
    assert [(k, tuple(v.shape)) for k, v in a.named_parameters()] == [(k, tuple(v.shape)) for k, v in b.named_parameters()]


def test_missing_context_fails_instead_of_bypassing_trained_fusion():
    m = model('attention')
    condition = torch.randn(1, 10, m.flow.conditioning_dim)
    base, bias = m.flow.prepare_conditioning(condition)
    with pytest.raises(ValueError, match='silent bypass'):
        m.flow.prepared_vector_field(base, bias, torch.randn(1, 10, 139), torch.tensor([.5]))
