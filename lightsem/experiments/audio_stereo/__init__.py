"""StereoAudioV1 representation and native-conditioning pilot.

The package keeps the diagnostics path importable in the non-Torch runner;
model symbols are loaded lazily when a Torch caller requests them.
"""

__all__ = ["StereoConditioner", "StereoConditionerConfig", "StereoNativeAdapter", "conditioner_loss"]


def __getattr__(name):
    if name in __all__:
        from .model import StereoConditioner, StereoConditionerConfig, StereoNativeAdapter, conditioner_loss
        return {
            "StereoConditioner": StereoConditioner,
            "StereoConditionerConfig": StereoConditionerConfig,
            "StereoNativeAdapter": StereoNativeAdapter,
            "conditioner_loss": conditioner_loss,
        }[name]
    raise AttributeError(name)
