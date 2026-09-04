"""Audio stage: features, structural segmentation, anticipation."""

from .anticipation import (
    CAUSAL,
    ORACLE,
    AnticipationCurves,
    compute,
    evaluate_anticipation,
    tension_curve,
)
from .features import AudioFeatures, extract
from .structure import ROLES, Segment, detect_boundaries, segment

__all__ = [
    "AudioFeatures",
    "extract",
    "Segment",
    "ROLES",
    "segment",
    "detect_boundaries",
    "AnticipationCurves",
    "compute",
    "tension_curve",
    "evaluate_anticipation",
    "CAUSAL",
    "ORACLE",
]
