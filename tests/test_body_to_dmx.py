import numpy as np
import torch

from lightsem.conductor.body_to_dmx import scene_to_dmx
from lightsem.dmx import club_rig, Fixture, Rig
from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig, causality_probe


def test_body_to_dmx_model_is_causal_and_finite():
    cfg = BodyToDMXConfig()
    model = BodyToDMX(cfg).eval()
    body = torch.randn(1, 32, cfg.body_dim)
    audio = torch.randn(1, 32, cfg.audio_dim)
    out = model(body, audio)
    assert out["scene"].shape == (1, 32, 6)
    assert out["angle_logit"].shape == (1, 32, 12)
    assert torch.isfinite(out["scene"]).all()
    assert causality_probe(model, body, audio, 16) == 0.0
    rpmc = torch.randn(1, 32, cfg.rpmc_audio_dim)
    rpmc_out = model(torch.zeros_like(body), rpmc, audio_kind="rpmc", body_present=False)
    assert rpmc_out["scene"].shape == (1, 32, 6) and torch.isfinite(rpmc_out["scene"]).all()


def test_scene_renderer_tracks_body_and_emits_decodable_dmx():
    T = 24
    scene = np.tile(np.array([[1., 0., .8, .7, .7, .5]]), (T, 1))
    body = np.zeros((T, 22, 3), np.float32)
    body[:, 2, 0] = np.linspace(-1., 1., T)
    stream = scene_to_dmx(scene, body, club_rig(), np.zeros(T))
    assert stream.shape == (T, 512) and stream.dtype == np.uint8
    assert np.isfinite(stream).all()
    assert not np.array_equal(stream[0], stream[-1])
    angles = np.full((T, 6, 2), .5)
    angled = scene_to_dmx(scene, body, club_rig(), np.zeros(T), angles=angles)
    assert angled.shape == stream.shape and np.isfinite(angled).all()
    rig8 = Rig([Fixture(f"mh{i}", "moving_head", 1 + 12 * i, position=(float(i), 5., 0.)) for i in range(8)])
    stream8 = scene_to_dmx(scene, body, rig8, np.zeros(T), angles=angles)
    assert stream8.shape == (T, 512) and np.isfinite(stream8).all()

