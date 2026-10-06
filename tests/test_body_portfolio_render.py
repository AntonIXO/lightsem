"""Source authentication and GPU exclusion checks; no GPU/media processes."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rendering import render_body_portfolio as render


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_cpu_command_explicitly_excludes_gpu():
    plan = {'review_start_s': 60, 'review_duration_s': 120}
    command = render.command_for(Path('/tmp/example.mp4'), {'audio': '/tmp/example.wav'}, plan, 'libx264')
    assert 'libx264' in command
    assert not any('nvenc' in part or part == '-gpu' for part in command)
    with pytest.raises(ValueError, match='explicit'):
        render.validate_mode('schematic', 'av1_nvenc')


def test_gpu_receipt_required_before_lock_or_device_access(tmp_path, monkeypatch):
    monkeypatch.setattr(render.subprocess, 'check_output', lambda *a, **k: pytest.fail('GPU access before receipt'))
    with pytest.raises(FileNotFoundError):
        with render.gpu_admission(tmp_path):
            pytest.fail('admitted without release')
    write_json(tmp_path / 'records/GPU_RELEASE.json', {'released': True, 'basis': 'momentary_idle', 'evidence': {'idle': True}})
    with pytest.raises(RuntimeError, match='parent-authenticated'):
        with render.gpu_admission(tmp_path):
            pytest.fail('idle poll admitted')


def test_dual_lock_preserves_peer_file_and_rejects_foreign_process(tmp_path, monkeypatch):
    write_json(tmp_path / 'records/GPU_RELEASE.json', {'released': True,
        'basis': 'verified_owner_queue_complete', 'evidence': {'receipt': 'owner-completion.json'}})
    host, legacy = tmp_path / 'host.lock', tmp_path / 'legacy.lock'
    host.write_text('peer-owner-evidence')
    monkeypatch.setattr(render, 'HOST_LOCK', host)
    monkeypatch.setattr(render, 'LEGACY_LOCK', legacy)
    monkeypatch.setattr(render.subprocess, 'check_output', lambda *a, **k: '123456, foreign_training\n')
    with pytest.raises(RuntimeError, match='Foreign GPU'):
        with render.gpu_admission(tmp_path):
            pytest.fail('admitted foreign workload')
    assert host.read_text() == 'peer-owner-evidence'
    monkeypatch.setattr(render.subprocess, 'check_output', lambda *a, **k: '')
    with render.gpu_admission(tmp_path) as receipt:
        assert receipt['basis'] == 'verified_owner_queue_complete'
    assert host.read_text() == 'peer-owner-evidence'


def test_schematic_frozen_controls_are_exact_and_bulbs_are_visible():
    patch = [{'kind': 'beam', 'position': [0., 6., -1.]},
             {'kind': 'bulb', 'position': [6., 5., 3.]}]
    renderer = render.SchematicRenderer(patch)
    directions = np.array([[0., -1., 0.], [0., 0., 0.]])
    colors = np.ones((2, 3))
    levels = np.array([.65, .65])
    first = renderer.render_controls(directions, colors, levels)
    assert np.array_equal(first, renderer.render_controls(directions, colors, levels))
    dark = renderer.render_controls(directions, colors, np.array([.65, 0.]))
    assert np.any(first != dark)
    bad = directions.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        renderer.render_controls(bad, colors, levels)


def test_pair_order_and_filter_errors():
    plan = {'algorithms': [{'id': '01'}, {'id': '02'}], 'tracks': [{'id': 'shpongle'}, {'id': 'cure_for_me'}]}
    assert [(a['id'], t['id']) for a, t in render.pairs(plan)] == [
        ('01', 'shpongle'), ('01', 'cure_for_me'), ('02', 'shpongle'), ('02', 'cure_for_me')]
    with pytest.raises(ValueError, match='Unknown algorithm'):
        list(render.pairs(plan, algorithm='missing'))


def test_frozen_source_hash_drift_is_rejected(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('original')
    plan = {'files': {str(source): render.sha(source)}}
    render.verify_files(plan)
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='Frozen input changed'):
        render.verify_files(plan)
