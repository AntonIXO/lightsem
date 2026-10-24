import torch
import pytest

from lightsem.experiments.body_light_ssl import (
    ControlledJepaConfig,
    ControlledSSLLoop,
    ControlledJepaModel,
    audit_input_sensitivity,
    controlled_ssl_loss,
    make_mask,
)


def _batch(batch=2, frames=12, audio_dim=7, body_dim=9, light_dim=5):
    generator = torch.Generator().manual_seed(123)
    audio = torch.randn(batch, frames, audio_dim, generator=generator)
    body = torch.randn(batch, frames, body_dim, generator=generator)
    light = torch.randn(batch, frames, light_dim, generator=generator)
    timestamps = torch.arange(frames, dtype=torch.float32)[None].expand(batch, -1) / 10.0
    valid = torch.ones(batch, frames, dtype=torch.bool)
    return audio, body, light, timestamps, valid


def test_config_rejects_raw_dmx_semantics():
    with pytest.raises(ValueError, match="semantic light"):
        ControlledJepaConfig(audio_dim=2, body_dim=3, light_dim=4, action_semantics="raw_dmx")


def test_mask_is_reproducible_and_valid_only():
    valid = torch.tensor([[True, True, True, False, False]])
    a = make_mask(valid, .4, generator=torch.Generator().manual_seed(5))
    b = make_mask(valid, .4, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert not bool(a[~valid].any())
    assert bool(a[0, :3].any())


def test_jepa_outputs_and_loss_are_finite():
    audio, body, light, timestamps, valid = _batch()
    model = ControlledJepaModel(ControlledJepaConfig(audio_dim=7, body_dim=9, light_dim=5, width=32, embedding_dim=16))
    mask = make_mask(valid, .4, generator=torch.Generator().manual_seed(9))
    outputs = model(audio, body, light, timestamps, valid, mask)
    loss, terms = controlled_ssl_loss(outputs)
    assert outputs["predicted"].shape == (2, 12, 16)
    assert outputs["future_predicted"].shape == (2, 11, 16)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_masked_context_does_not_leak_masked_body_or_light():
    audio, body, light, timestamps, valid = _batch()
    model = ControlledJepaModel(ControlledJepaConfig(audio_dim=7, body_dim=9, light_dim=5, width=32, embedding_dim=16)).eval()
    mask = torch.zeros_like(valid); mask[:, 3:6] = True
    first = model(audio, body, light, timestamps, valid, mask)
    body_alt, light_alt = body.clone(), light.clone()
    body_alt[:, 3:6] += 100.0; light_alt[:, 3:6] -= 100.0
    second = model(audio, body_alt, light_alt, timestamps, valid, mask)
    torch.testing.assert_close(first["context"], second["context"], rtol=0, atol=1e-6)


def test_loop_step_and_input_sensitivity():
    audio, body, light, timestamps, valid = _batch()
    config = ControlledJepaConfig(audio_dim=7, body_dim=9, light_dim=5, width=32, embedding_dim=16)
    loop = ControlledSSLLoop(config, seed=0)
    metrics = loop.step(audio, body, light, timestamps, valid)
    assert metrics["step"] == 1.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    audit = audit_input_sensitivity(loop.model, audio, body, light, timestamps, valid)
    assert audit["finite"] == 1.0
    assert audit["embedding_std"] > 1e-6
    assert audit["body_context_delta"] > 1e-5
    assert audit["light_future_delta"] > 1e-5
