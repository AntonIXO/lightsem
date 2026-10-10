"""Clock-domain and provenance contracts for dynamic-dance retiming."""

import numpy as np

from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
from lightsem.experiments.dynamic_dance.retiming import (
    PARENTS,
    _raw_seam_diagnostics,
    d3_source_regional_retiming,
    d4_motif_recomposition,
)


def _body(seconds=4.0, fps=30.0):
    n = int(seconds * fps)
    t = np.arange(n) / fps
    body = np.zeros((n, 22, 3), dtype=np.float64)
    body[:, 0, 0] = 0.08 * t
    body[:, 0, 1] = 1.0
    for joint, parent in enumerate(PARENTS[1:], 1):
        # A nonconstant, finite source body with no independent clock per limb.
        offset = np.array((.03 * (joint % 3 + 1), .06, .02 * (joint % 2 + 1)))
        body[:, joint] = body[:, parent] + offset
    return body


def _timeline(seconds=4.0, fps=30.0):
    n = int(seconds * fps)
    t = np.arange(n) / fps
    sources = np.ones((n, 4, 4), dtype=np.float64)
    sources[:, :, 0] += .2 * np.sin(t[:, None] * 1.7)
    sources[:, :, 1] += np.exp(-((t[:, None] - 1.5) / .15) ** 2)
    sources[:, :, 2] = sources[:, :, 1]
    mixture = np.ones((n, 4), dtype=np.float64)
    sections = [
        {"start": 0.0, "end": 2.0, "label": "a", "confidence": .9},
        {"start": 2.0, "end": seconds, "label": "b", "confidence": .9},
    ]
    return MusicTimeline(
        time=t, muq=np.zeros((n, 1024), dtype=np.float32), muq_time=t,
        sources=sources, mixture=mixture,
        beats=np.arange(.5, seconds, .5), downbeats=np.asarray((0., 2.)),
        sections=sections, recurrence=np.eye(2), metadata={"audio_path": "test"},
    ).validate()


def test_regional_anchor_records_keep_output_and_native_body_domains_distinct():
    result, plan = d3_source_regional_retiming(_body(), _timeline())
    assert result.metadata["time_domain"] == "output_song_seconds"
    assert result.metadata["source_time_domain"] == "native_body_seconds"
    assert result.metadata["anchor_time_domain"] == "output_song_seconds"
    assert plan.anchors
    for anchor in plan.anchors:
        assert anchor["time_domain"] == "output_song_seconds"
        assert anchor["source_time_domain"] == "native_body_seconds"


def test_identity_recomposition_is_chronological_and_explicitly_classified():
    body, timeline = _body(), _timeline()
    result, _ = d4_motif_recomposition(body, timeline, control="identity")
    assert result.metadata["source_order_preserved"] is True
    assert result.metadata["time_map_classification"].startswith("identity;")
    np.testing.assert_array_equal(result.source_time, timeline.time)
    np.testing.assert_allclose(result.world, body, atol=1e-6)


def test_raw_seam_diagnostic_reports_unfiltered_output_and_clock_jump():
    body = _body(seconds=1.0)
    output = body.copy()
    output[15:, :, 0] += .4
    source_time = np.arange(len(body), dtype=np.float64) / 30.
    source_time[15:] += .7
    rows = [
        {"target_seconds": [0., .5], "source_seconds": [0., .5]},
        {"target_seconds": [.5, 1.0], "source_seconds": [1.2, 1.7]},
    ]
    diagnostics = _raw_seam_diagnostics(output, source_time, rows, 30.)
    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["target_frame"] == 15
    assert row["raw_output_jump_rms_m"] > 0
    assert row["source_clock_jump_is_provenance_only"] is True
    assert np.isclose(row["source_clock_delta_s"], .7 + 1 / 30.)
