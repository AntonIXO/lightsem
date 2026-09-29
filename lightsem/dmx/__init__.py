"""DMX stage: fixture model, club rig, structure→scene mapping, universe stream."""

from .fixtures import UNIVERSE_SIZE, Fixture, FixtureState, Rig, club_rig


def __getattr__(name):
    # Renderers and network bridges only need fixture bytes. Keep audio-analysis
    # dependencies optional (in particular inside Blender's Python runtime),
    # while retaining the original public imports for scene/conductor callers.
    if name in {"SCENES", "Scene", "SceneEngine"}:
        from . import scenes
        value = getattr(scenes, name)
    elif name in {"ArtNetSender", "DmxStream", "build_cue_list", "render_stream"}:
        from . import universe
        value = getattr(universe, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value

__all__ = [
    "Fixture",
    "FixtureState",
    "Rig",
    "club_rig",
    "UNIVERSE_SIZE",
    "Scene",
    "SCENES",
    "SceneEngine",
    "DmxStream",
    "render_stream",
    "build_cue_list",
    "ArtNetSender",
]
