"""Experimental audio-only light layers. No production defaults are changed.

Model exports are lazy so NumPy-only target/overlay tools can import sibling
submodules in the non-Torch environment.
"""

from importlib import import_module

__all__ = [
    "IndependentLightLayers", "LightLayersConfig", "ParameterMatch",
    "SharedLightLayers", "build_light_layers", "count_trainable_parameters",
    "match_independent_width",
]


def __getattr__(name: str):
    if name in __all__:
        return getattr(import_module(".model", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
