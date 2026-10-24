"""Continuous audio/body-to-light foundation-model research components.

The package is deliberately a decoder pilot.  It consumes an already audited
native body and frozen audio features; it does not generate a replacement
body or bypass the native raw-motion safety gate.
"""

from .muq import MuQContract, validate_muq_features, validate_base92, configure_muq_scope, load_live_muq
from .retention import compare_features

try:
    from .audio_encoder import AudioEncoder, AudioEncoderConfig
    from .long_context import LongContextConfig, LongContextMemory, MemoryState
    from .fixture_queries import FixtureQueryDecoder
    from .modality_fusion import FoundationAudioLightModel, FoundationConfig
    from .peft import AdapterConfig, build_temporal_adapter, inject_lora, trainable_parameter_report
except ModuleNotFoundError as error:
    # Layer-selection and provenance probes are intentionally CPU-only.  They
    # remain importable when the Torch environment is unavailable.
    if error.name != "torch":
        raise

__all__ = [
    "AudioEncoder", "AudioEncoderConfig", "LongContextConfig",
    "LongContextMemory", "MemoryState", "FixtureQueryDecoder",
    "FoundationAudioLightModel", "FoundationConfig",
    "AdapterConfig", "build_temporal_adapter", "inject_lora", "trainable_parameter_report",
    "MuQContract", "validate_muq_features", "validate_base92", "configure_muq_scope", "load_live_muq",
    "compare_features",
]
