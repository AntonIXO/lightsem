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
from .stereo_v1 import (
    EVENT_NAMES,
    FEATURE_NAMES,
    RELATION_NAMES,
    SPATIAL_NAMES,
    StereoAudioV1,
    extract_stereo_v1,
    metamorphic_check,
    resample_stream,
)
from .structure import ROLES, Segment, detect_boundaries, segment

__all__ = [
    "AudioFeatures",
    "extract",
    "StereoAudioV1",
    "extract_stereo_v1",
    "metamorphic_check",
    "resample_stream",
    "FEATURE_NAMES",
    "EVENT_NAMES",
    "SPATIAL_NAMES",
    "RELATION_NAMES",
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
