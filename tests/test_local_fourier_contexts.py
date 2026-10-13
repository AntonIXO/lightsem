import numpy as np

from scripts.motion.build_local_fourier_contexts import local_fourier


def test_local_fourier_context_is_finite_and_native_width():
    time = np.arange(180, dtype=np.float32) / 30.0
    values = np.stack([np.sin(2 * np.pi * (i + 1) * time) for i in range(8)], axis=1)
    result = local_fourier(values)
    assert result.shape == (180, 128)
    assert np.isfinite(result).all()


def test_local_fourier_context_has_no_nyquist_violation_in_declared_recipe():
    # The 30 Hz descriptor grid has a strict 15 Hz Nyquist limit. The short
    # window's highest declared component is 14 Hz; this guards the corrected
    # recipe against silently reintroducing an aliased 16 Hz component.
    source = open("scripts/motion/build_local_fourier_contexts.py", encoding="utf-8").read()
    assert '"15": [2.0, 4.0, 8.0, 14.0]' in source
