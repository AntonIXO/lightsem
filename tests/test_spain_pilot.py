import numpy as np
import torch

from lightsem.experiments.audio_stereo.spain_pilot import (
    SpaInNet,
    SpaInNetConfig,
    equal_power_repan,
    train_manifest,
)


def test_equal_power_repan_preserves_constant_power():
    stems = torch.ones(1, 4, 32)
    angles = torch.tensor([[-45.0, -15.0, 15.0, 45.0]])
    mixture, target = equal_power_repan(stems, angles)
    assert mixture.shape == (1, 2, 32)
    assert target.shape == (1, 4, 2, 32)
    assert torch.allclose(target.square().sum(dim=2), stems.square(), atol=1e-5)


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
