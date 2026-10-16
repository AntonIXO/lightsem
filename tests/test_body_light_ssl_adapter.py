import numpy as np

from lightsem.experiments.body_light_ssl.adapter import SSLAdapter


def test_bounded_residual_respects_speed_and_acceleration_caps():
    desired = np.zeros((20, 8), dtype=np.float32)
    desired[5:, :] = 1.0
    result = SSLAdapter._bounded_residual(desired, max_speed=2.0, max_accel=4.0)
    speed = np.diff(result, axis=0) * 40.0
    accel = np.diff(result, n=2, axis=0) * 40.0 ** 2
    assert np.max(np.abs(speed)) <= 2.0 + 1e-5
    assert np.max(np.abs(accel)) <= 4.0 + 1e-4
    assert np.isfinite(result).all()


def test_bounded_residual_is_causal():
    desired = np.zeros((12, 2), dtype=np.float32)
    first = SSLAdapter._bounded_residual(desired, max_speed=20.0, max_accel=40.0)
    desired[-1] = 1.0
    second = SSLAdapter._bounded_residual(desired, max_speed=20.0, max_accel=40.0)
    np.testing.assert_array_equal(first[:-1], second[:-1])
