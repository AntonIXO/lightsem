import numpy as np
import torch

from lightsem.conductor.body_to_dmx import scene_to_dmx
from lightsem.dmx import club_rig
from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig, causality_probe


def test_body_to_dmx_model_is_causal_and_finite():
    cfg = BodyToDMXConfig()
    model = BodyToDMX(cfg).eval()
    body = torch.randn(1, 32, cfg.body_dim)
    audio = torch.randn(1, 32, cfg.audio_dim)
    out = model(body, audio)
    assert out["scene"].shape == (1, 32, 6)
    assert torch.isfinite(out["scene"]).all()
    assert causality_probe(model, body, audio, 16) == 0.0


def test_scene_renderer_tracks_body_and_emits_decodable_dmx():
    T = 24
    scene = np.tile(np.array([[1., 0., .8, .7, .7, .5]]), (T, 1))
    body = np.zeros((T, 22, 3), np.float32)
    body[:, 2, 0] = np.linspace(-1., 1., T)
    stream = scene_to_dmx(scene, body, club_rig(), np.zeros(T))
    assert stream.shape == (T, 512) and stream.dtype == np.uint8
    assert np.isfinite(stream).all()
    assert not np.array_equal(stream[0], stream[-1])

