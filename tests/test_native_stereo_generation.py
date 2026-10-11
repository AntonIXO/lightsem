import numpy as np

from scripts.motion.generate_native_stereo_demos import pool_native


def test_pool_native_matches_native_partial_bin_contract():
    value = np.arange(17 * 2, dtype=np.float32).reshape(17, 2)
    pooled = pool_native(value, 17)
    expected = np.vstack((value[:15].mean(0), value[15:17].mean(0)))
    assert pooled.shape == (2, 2)
    np.testing.assert_allclose(pooled, expected)


def test_pool_native_does_not_repeat_last_frame_into_partial_bin():
    value = np.zeros((16, 1), dtype=np.float32)
    value[-1] = 9.0
    pooled = pool_native(value, 16)
    assert pooled[-1, 0] == 9.0
