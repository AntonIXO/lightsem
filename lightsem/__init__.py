"""LightSem — semantic audio → stage lighting.

Structure-aware (L3) audio→light mapping: the model reads musical *structure*
(intro / build-up / drop / breakdown / outro) and maps it into a continuous
light-scene space, rather than thresholding an FFT band (L2).

Design constraint carried from the probe phase (see docs/RULE_OUTS.md):
    Do NOT map waveform → DMX. Map structure → scene.
"""

__version__ = "0.1.0"

KP_FPS = 60.0
SR = 22050
HOP = 512

__all__ = ["KP_FPS", "SR", "HOP", "__version__"]
