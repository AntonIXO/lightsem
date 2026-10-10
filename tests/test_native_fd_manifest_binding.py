"""The full-data loader must not silently fall back to an old FD cache."""

import json

import pytest

from lightsem.experiments.native_dynamics.dataset import fd_manifest_path


def test_full_fit_requires_an_explicit_finedance_manifest(tmp_path):
    (tmp_path / "TRAINING_PROTOCOL.json").write_text(json.dumps({
        "campaign_id": "native-full-data-speed-test",
    }))
    with pytest.raises(ValueError, match="legacy subset fallback"):
        fd_manifest_path(tmp_path)


def test_legacy_subset_fallback_is_still_explicitly_available_for_old_protocols(tmp_path):
    (tmp_path / "TRAINING_PROTOCOL.json").write_text(json.dumps({
        "campaign_id": "native-dynamics-jd-reference",
    }))
    assert fd_manifest_path(tmp_path).name == "manifest.json"
