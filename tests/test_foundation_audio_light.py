import torch

from lightsem.experiments.foundation_audio_light import FoundationAudioLightModel


def _inputs(batch=2, frames=64, fixtures=5):
    g = torch.Generator().manual_seed(0)
    audio = torch.randn(batch, frames, 52, generator=g)
    body = torch.randn(batch, frames, 66, generator=g)
    meta = torch.randn(batch, fixtures, 8, generator=g)
    timestamps = torch.arange(frames).float()[None].expand(batch, -1) / 10.
    return audio, body, meta, timestamps


def test_hybrid_outputs_are_finite_and_arbitrary_n():
    model = FoundationAudioLightModel().eval()
    audio, body, meta, timestamps = _inputs(fixtures=7)
    out = model(audio, body, meta, timestamps=timestamps, duration_seconds=6.4)
    assert out["brightness"].shape == (2, 64, 7, 1)
    assert out["angle"].shape == (2, 64, 7, 2)
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))


def test_streaming_state_matches_full_context():
    model = FoundationAudioLightModel().eval()
    audio, body, meta, timestamps = _inputs()
    full = model(audio, body, meta, timestamps=timestamps, duration_seconds=6.4)
    state = None
    pieces = []
    for start in range(0, 64, 32):
        part = model(audio[:, start:start + 32], body[:, start:start + 32], meta,
                     timestamps=timestamps[:, start:start + 32], duration_seconds=6.4,
                     state=state)
        state = part["state"]
        pieces.append(part["scene"])
    assert torch.allclose(full["scene"], torch.cat(pieces, 1), atol=1e-5, rtol=1e-5)


def test_fixture_queries_are_permutation_equivariant():
    model = FoundationAudioLightModel().eval()
    audio, body, meta, timestamps = _inputs(fixtures=5)
    perm = torch.tensor([2, 4, 0, 3, 1])
    a = model(audio, body, meta, timestamps=timestamps, duration_seconds=6.4)
    b = model(audio, body, meta[:, perm], timestamps=timestamps, duration_seconds=6.4)
    assert torch.allclose(a["brightness"], b["brightness"][:, :, torch.argsort(perm)], atol=1e-5, rtol=1e-5)


def test_frozen_body_blocks_motion_proposal():
    model = FoundationAudioLightModel().eval()
    audio, body, meta, timestamps = _inputs()
    frozen = body[:, :1].expand_as(body)
    out = model(audio, frozen, meta, timestamps=timestamps, duration_seconds=6.4)
    assert float(out["angle"].abs().max().detach()) == 0.0
