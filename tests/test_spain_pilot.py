import numpy as np
import torch

from lightsem.experiments.audio_stereo.spain_pilot import (
    SpaInNet,
    SpaInNetConfig,
    equal_power_repan,
    spain_reconstruction_loss,
    slots_to_native_context,
    train_manifest,
)


def test_equal_power_repan_preserves_constant_power():
    stems = torch.ones(1, 4, 32)
    angles = torch.tensor([[-45.0, -15.0, 15.0, 45.0]])
    mixture, target = equal_power_repan(stems, angles)
    assert mixture.shape == (1, 2, 32)
    assert target.shape == (1, 4, 2, 32)
    assert torch.allclose(target.square().sum(dim=2), stems.square(), atol=1e-5)


def test_reconstruction_loss_rejects_zero_source_solution():
    stems = torch.ones(1, 4, 32) * .1
    mixture, target = equal_power_repan(stems, torch.tensor([[-45., -15., 15., 45.]]))
    zero = torch.zeros_like(target)
    assert float(spain_reconstruction_loss(zero, target, mixture)) > .9
    assert float(spain_reconstruction_loss(target, target, mixture)) < 1e-5


def test_native_context_preserves_source_time_channel_identity():
    slots = torch.arange(2 * 4 * 7 * 16, dtype=torch.float32).reshape(2, 4, 7, 16)
    actual = slots_to_native_context(slots, 7)
    for source in range(4):
        for channel in range(16):
            assert torch.equal(actual[:, :, source * 16 + channel], slots[:, source, :, channel])


def test_constant_slot_channels_stay_constant_after_time_resampling():
    channels = torch.arange(64, dtype=torch.float32).reshape(1, 4, 1, 16)
    slots = channels.expand(1, 4, 19, 16)
    result = slots_to_native_context(slots, 30)
    assert torch.allclose(result, torch.arange(64).expand(1, 30, 64).float())


def test_normalized_loss_gain_invariance_and_gradient():
    torch.manual_seed(9)
    stems = torch.randn(1, 4, 64) * .02
    mix, target = equal_power_repan(stems, torch.tensor([[-40., -15., 10., 30.]]))
    losses = []
    for gain in (.1, 1., 3.):
        prediction = torch.zeros_like(target, requires_grad=True)
        loss = spain_reconstruction_loss(prediction, gain * target, gain * mix)
        loss.backward()
        assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
        losses.append(loss.detach())
    assert torch.allclose(torch.stack(losses), torch.full((3,), 1.1), atol=1e-5)


def test_spain_forward_returns_waveforms_and_native_slots():
    torch.manual_seed(3)
    model = SpaInNet(SpaInNetConfig(n_fft=64, hop_length=16, win_length=64))
    model.eval()
    wav = torch.randn(1, 2, 256)
    angles = torch.tensor([[-45.0, -10.0, 10.0, 45.0]])
    separated, features = model(wav, angles)
    assert separated.shape == (1, 4, 2, 256)
    assert features["native_descriptor"].shape == (1, 1, 64)
    assert torch.isfinite(separated).all()
    assert torch.isfinite(features["native_descriptor"]).all()
    assert features["masks"].shape[:3] == (1, 4, 17)
    _, mirrored = model(wav, -angles)
    assert not torch.allclose(features["pan_embedding"], mirrored["pan_embedding"])


def test_spain_backward_and_manifest_smoke(tmp_path):
    rng = np.random.default_rng(7)
    stem_path = tmp_path / "song.npz"
    np.savez(stem_path, stems=rng.standard_normal((4, 192)).astype(np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text('["song.npz"]')
    config = SpaInNetConfig(n_fft=64, hop_length=16, win_length=64)
    model, history = train_manifest(manifest, steps=1, segment_samples=128, config=config)
    assert len(history) == 1 and np.isfinite(history[0])
    assert model.config_dict()["pretrained"] is False
