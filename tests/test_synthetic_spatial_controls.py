from tools.synthetic_spatial_controls import FixtureLimits, audit, generate

def test_synthetic_stream_has_expected_verdicts():
    result = generate(duration_s=2.0, fps=40.0, limits=FixtureLimits())
    assert not result["requested_audit"]["physically_admissible"]
    assert result["feasible_audit"]["physically_admissible"]
    assert result["requested_fast_deg"].shape == result["feasible_projected_deg"].shape
    assert result["requested_audit"]["finite"]
    assert result["feasible_audit"]["finite"]

def test_audit_rejects_out_of_range_values():
    limits = FixtureLimits()
    result = generate(duration_s=1.0, fps=40.0, limits=limits)
    bad = result["feasible_projected_deg"].copy()
    bad[0, 0, 0] = limits.pan_max_deg + 1
    report = audit(bad, 40.0, limits)
    assert not report["in_range"]
    assert not report["physically_admissible"]
