"""Regression tests for native CLI stage-boundary contracts.

These tests intentionally exercise the command orchestration with all expensive
generation, mapping, and rendering work replaced by small fakes.  They protect
the safety boundary where a failed native body must not be delivered as media.
"""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from lightsem.dance import cli


def _render_args(tmp_path: Path) -> Namespace:
    """Build the subset of ``render`` arguments consumed by ``do_render``."""

    return Namespace(
        wav=tmp_path / "song.wav",
        aist=None,
        baseline=None,
        d4max=False,
        d6=False,
        device="cpu",
        checkpoint=None,
        d6_checkpoint=None,
        features=None,
        base92=None,
        audio_python=None,
        audio_device=None,
        audio_cache=None,
        seed=None,
        out=tmp_path / "failed.mp4",
        colors=None,
        start=0.0,
        duration=None,
        beams=None,
        lasers=None,
        bulbs=None,
    )


def test_do_render_refuses_failure_diagnostic_before_mapping_or_encoding(tmp_path, monkeypatch):
    """A failed native body must stop before any DMX or video side effects."""

    args = _render_args(tmp_path)
    args.wav.write_bytes(b"test audio placeholder")
    config = {"lighting": {"counts": {"beams": 8, "lasers": 5, "bulbs": 50}}}
    generation = {"baseline_id": "fixed_c", "status": "failure_diagnostic"}

    monkeypatch.setattr(
        cli,
        "_generate",
        lambda _args, _root: (object(), object(), object(), config, generation),
    )

    def unexpected_map(*_args, **_kwargs):
        raise AssertionError("failed generation reached DMX mapping")

    def unexpected_render(*_args, **_kwargs):
        raise AssertionError("failed generation reached video rendering")

    monkeypatch.setattr("lightsem.dance.lighting.map_lights", unexpected_map)
    monkeypatch.setattr("lightsem.dance.render.render_video", unexpected_render)

    with pytest.raises((RuntimeError, ValueError), match="(?i)failure|unqualified|diagnostic"):
        cli.do_render(args)

    assert not args.out.exists(), "a rejected generation must not leave media behind"


def test_compile_world_uses_the_configured_mapper_id(monkeypatch):
    """The mapper implementation must follow the selected config, not a literal ``01``."""

    pytest.importorskip("torch")
    from dataclasses import asdict

    from lightsem.conductor.body_transfer import TransferConfig
    from lightsem.conductor.body_transfer_v2 import make_outputs, map_body as real_map_body
    from lightsem.conductor.body_transfer_v2.common import REFERENCE
    from lightsem.experiments.dynamic_dance_combo import campaign

    captured = {}

    def spy_map_body(body, outputs, algorithm_id, fps=40.0):
        captured["algorithm_id"] = str(algorithm_id)
        return real_map_body(body, outputs, algorithm_id, fps)

    monkeypatch.setattr("lightsem.conductor.body_transfer_v2.map_body", spy_map_body)
    frames = 3
    world = np.repeat(REFERENCE[None], frames, axis=0)
    time = np.arange(frames, dtype=float) / 40.0
    outputs = make_outputs(beams=1, lasers=0, bulbs=0)
    colors = np.tile(np.array([20, 40, 80], dtype=np.uint8), (frames, 1))
    loudness = {
        "time": np.arange(8, dtype=float) / 40.0,
        "momentary": np.full(8, -20.0),
        "short_term": np.full(8, -20.0),
    }

    campaign.compile_world(
        world,
        time,
        outputs,
        colors,
        colors,
        loudness,
        asdict(TransferConfig(apply_motion_limits=False)),
        mapper_id="03",
    )

    assert captured["algorithm_id"] == "03"
