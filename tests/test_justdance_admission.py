import numpy as np

from scripts.motion.audit_justdance_admission_20261007 import _apply_jump_gate


def _pose_with_one_missing_frame():
    k = np.zeros((600, 17, 2), dtype=np.float64)
    # Shoulders and hips define a stable torso; the remaining joints are finite.
    k[..., 0] = 500.0
    k[..., 1] = 300.0
    k[:, 5, 1] = k[:, 6, 1] = 300.0
    k[:, 11, 1] = k[:, 12, 1] = 200.0
    c = np.ones((600, 17), dtype=np.float64)
    box = np.tile(np.asarray([450., 0., 550., 400.]), (600, 1))
    k[100] = np.nan
    c[100] = np.nan
    return k, c, box


def test_missing_transition_is_not_counted_as_tracking_jump():
    k, c, box = _pose_with_one_missing_frame()
    meta = {'height': 1000., 'fps_eff': 30.}
    old = _apply_jump_gate(k, c, box, meta, np.arange(600) / 30., finite_transition_only=False)
    fixed = _apply_jump_gate(k, c, box, meta, np.arange(600) / 30., finite_transition_only=True)

    assert fixed['missing_transition_flags'] > 0
    assert fixed['tracking_jump_edges'] < old['tracking_jump_edges']
    assert fixed['eligible_windows'] > 0
