import pytest

from lightsem.dance.baseline import validate_checkpoint_state


def _state():
    names = [f"feature_{i}" for i in range(1061)]
    return {
        "model": {}, "base_config": {},
        "config": {"arm": "joint", "extra_dim": 1061, "steps": 32},
        "identity": {"arm": "joint", "seed": 3, "protocol_sha256": "p"},
        "normalizers": {"mean": [0.0] * 139, "std": [1.0] * 139},
        "audio_normalizers": {"mean": [0.0] * 92, "std": [1.0] * 92},
        "extra_normalizers": {"mean": [0.0] * 1061, "std": [1.0] * 1061},
        "feature_names": names,
    }


def test_checkpoint_provenance_accepts_self_describing_state():
    assert validate_checkpoint_state(_state(), expected_seed=3)["identity"]["seed"] == 3


@pytest.mark.parametrize("change", [
    lambda s: s["identity"].update(seed=4),
    lambda s: s["config"].update(arm="dynamics"),
    lambda s: s["feature_names"].__setitem__(0, s["feature_names"][1]),
    lambda s: s["extra_normalizers"]["std"].__setitem__(0, 0.0),
])
def test_checkpoint_provenance_rejects_identity_or_input_drift(change):
    state = _state()
    change(state)
    with pytest.raises(ValueError):
        validate_checkpoint_state(state, expected_seed=3)
