"""Focused guards for D4 zero-information handling and clock receipts."""
import importlib.util
from pathlib import Path

import numpy as np

from lightsem.experiments.d4max_d8.motifs import compose
from lightsem.experiments.dynamic_dance.contracts import DancePlan, DanceResult


_SPEC = importlib.util.spec_from_file_location(
    "retiming_fixtures_for_d4d6", Path(__file__).with_name("test_dynamic_dance_retiming.py")
)
_FIXTURES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURES)


def test_zero_information_audio_keeps_own_song_body_in_identity_mode():
    body = _FIXTURES.source_body(16)
    music = _FIXTURES.music(16)
    music.muq[...] = 0
    music.sources[...] = 0
    music.mixture[...] = 0

    result, plan, bank = compose(body, music)

    np.testing.assert_array_equal(result.world, body)
    np.testing.assert_array_equal(result.source_time, result.time)
    assert bank["audio_information_available"] is False
    assert result.metadata["time_map_classification"] == "identity"
    assert result.metadata["source_order_preserved"] is True
    assert plan.metadata["diagnostic_reason"].startswith("zero-information")


def test_dance_diagnostics_arrays_roundtrip_without_changing_primary_schema(tmp_path):
    time = np.arange(6, dtype=np.float64) / 30
    diagnostics = {"composition_clock": np.linspace(0, 1, 6), "seam_rms": np.array([.1, .2])}
    result = DanceResult(time, np.zeros((6, 22, 3)), np.zeros((6, 4)), diagnostics=diagnostics)
    result_path = tmp_path / "body.npz"
    result.save(result_path)
    loaded = DanceResult.load(result_path)
    np.testing.assert_array_equal(loaded.diagnostics["composition_clock"], diagnostics["composition_clock"])
    np.testing.assert_array_equal(loaded.diagnostics["seam_rms"], diagnostics["seam_rms"])

    plan = DancePlan(time, time.copy(), np.ones((6, 1)), diagnostics=diagnostics)
    plan_path = tmp_path / "plan.npz"
    plan.save(plan_path)
    loaded_plan = DancePlan.load(plan_path)
    np.testing.assert_array_equal(loaded_plan.diagnostics["composition_clock"], diagnostics["composition_clock"])
