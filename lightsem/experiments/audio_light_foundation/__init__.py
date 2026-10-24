"""Target-led audio-to-light foundation pilot with dance as a privileged teacher."""

from .model import AudioToLightFoundation, AudioToLightConfig
from .data import load_rows, make_fixture_metadata

__all__ = ["AudioToLightFoundation", "AudioToLightConfig", "load_rows", "make_fixture_metadata"]
