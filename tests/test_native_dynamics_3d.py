"""CPU-only preview contract and inherited lit-body geometry checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.rendering import render_native_dynamics_3d as preview


def test_native_body_clock_and_actual_dmx_clock_include_bad_burst():
    body, ticks = preview.frame_indices()
    np.testing.assert_array_equal(body, np.arange(2700, 4500))
    assert len(ticks) == 1800 and ticks[0] == 3600 and ticks[-1] == 5999
    assert body[1350] / 30 == 135 and ticks[1350] / 40 == 135
    assert np.max(np.abs(body / 30 - ticks / 40)) <= 1/80


def test_local_pose_preserves_articulation_height_and_facing():
    source = np.arange(132, dtype=float).reshape(2, 22, 3) / 100
    before = source.copy()
    local = preview.display_pose(source)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_allclose(local[:, :, 1], source[:, :, 1])
    np.testing.assert_allclose(local[:, 1:] - local[:, :1], source[:, 1:] - source[:, :1])
    np.testing.assert_allclose(local[:, 0, [0, 2]], 0)
    modified = source.copy(); modified[0, 20, 0] += .5
    changed = preview.display_pose(modified) - local
    assert np.count_nonzero(changed) == 1 and changed[0, 20, 0] == pytest.approx(.5)


def test_unguarded_render_fails_before_gpu_import():
    with pytest.raises(ValueError, match='sequential curator'):
        with preview.renderers(Path('/tmp'), [], False):
            pytest.fail('Must not create a renderer')


def test_smoke_requires_matching_inspection_and_pixel_hashes(tmp_path):
    out = tmp_path / 'smoke'; out.mkdir()
    image = out / 'actual.png'; image.write_bytes(b'pixels')
    pins = {'test': 'pinned'}
    record = out / 'SMOKE.json'
    record.write_text(json.dumps({'inputs': pins, 'files': {str(image): preview.sha(image)}}))
    (out / 'INSPECTED.json').write_text(json.dumps({'inspected': True, 'smoke_sha256': preview.sha(record)}))
    assert preview.require_smoke(tmp_path, pins) == preview.sha(record)
    image.write_bytes(b'changed')
    with pytest.raises(ValueError, match='pixels changed'):
        preview.require_smoke(tmp_path, pins)


def test_encoder_original_speed_crop_budget():
    command = preview.encode_command(Path('/tmp/review.mp4'), Path('/tmp/audio.wav'))
    assert '-ss' not in command
    assert command[command.index('-c:a') + 1] == 'copy'
    assert command[command.index('-frames:v') + 1] == '1800'
    assert command[command.index('-t') + 1] == '60'
    assert command[command.index('-b:v') + 1] == '1100k'
    assert not any('atempo' in x or 'setpts' in x for x in command)


def test_compositor_native_size_and_reject_bad_view():
    views = [np.zeros((600, 640, 3), dtype=np.uint8)] * 2
    out = preview.compose(views, 'C4', {'title': 'AURORA — Cure For Me'}, 135.)
    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8
    with pytest.raises(ValueError, match='dimensions'):
        preview.compose([np.zeros((1, 1, 3), dtype=np.uint8)] * 2, 'C4', {'title': 'x'}, 135.)


@pytest.mark.parametrize('problem', ['pts', 'pcm', None])
def test_audit_requires_uniform_pts_and_exact_prepared_audio(tmp_path, monkeypatch, problem):
    """Exercise real audit branching while replacing only expensive decoders."""
    out = tmp_path / 'pair'; out.mkdir()
    movie = out / 'preview.mp4'; movie.write_bytes(b'movie')
    pins = {'test': 'inputs'}
    (out / 'RENDER.json').write_text(json.dumps({'inputs': pins, 'sha256': preview.sha(movie)}))
    reference = out / 'reference.m4a'; reference.write_bytes(b'reference')
    record = out / 'REFERENCE.json'
    record.write_text(json.dumps({'decoded_pcm_sha256': preview.hashlib.sha256(b'pcm').hexdigest()}))
    monkeypatch.setattr(preview, 'load', lambda *args: ({}, {}, None, None, None, pins))
    monkeypatch.setattr(preview, 'folder', lambda *args: out)
    monkeypatch.setattr(preview, 'require_smoke', lambda *args: 'smoke')
    monkeypatch.setattr(preview, 'audio_paths', lambda *args: (reference, record))
    monkeypatch.setattr(preview.native.base, 'decoded_audio', lambda path: b'wrong' if problem == 'pcm' and path == movie else b'pcm')
    monkeypatch.setattr(preview.subprocess, 'run', lambda *args, **kwargs: None)
    probe = {'streams': [dict(codec_type='video', width=1280, height=720, r_frame_rate='30/1', nb_read_frames='1800', duration='60'), dict(codec_type='audio', duration='60', start_time='0')]}
    times = np.arange(1800) / 30
    if problem == 'pts': times[100] += .01
    frames = {'frames': [{'best_effort_timestamp_time': str(t)} for t in times]}
    monkeypatch.setattr(preview.subprocess, 'check_output', lambda command, **kwargs: json.dumps(frames if '-show_frames' in command else probe))
    if problem:
        with pytest.raises(ValueError, match='timestamps' if problem == 'pts' else 'PCM'):
            preview.audit(tmp_path, 'C4', 'shpongle')
        assert not (out / 'AUDIT.json').exists()
    else:
        result = preview.audit(tmp_path, 'C4', 'shpongle')
        assert result['pts_uniform'] and result['audio_matches_prepared_reference']


def test_prepare_reference_is_explicit_pinned_and_reused(tmp_path, monkeypatch):
    original = tmp_path / 'original.wav'; original.write_bytes(b'original')
    track = {'id': 'shpongle', 'audio': str(original)}
    monkeypatch.setattr(preview, 'campaign', lambda *args: ({}, track))
    monkeypatch.setattr(preview.native, 'verify_source', lambda *args: preview.sha(original))
    monkeypatch.setattr(preview.native, 'resource_guard', lambda *args: None)
    commands = []
    def encode(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b'AAC90to150')
    monkeypatch.setattr(preview.subprocess, 'run', encode)
    monkeypatch.setattr(preview.native.base, 'decoded_audio', lambda path: b'PCM90to150')
    first = preview.prepare_audio(tmp_path, 'shpongle')
    second = preview.prepare_audio(tmp_path, 'shpongle')
    assert first == second and len(commands) == 1
    assert commands[0][commands[0].index('-af') + 1] == 'atrim=start=90:end=150,asetpts=PTS-STARTPTS'
    assert first['source_start_s'] == 90 and first['duration_s'] == 60 and first['playback_rate'] == 1
    reference, _ = preview.audio_paths(tmp_path, track)
    reference.write_bytes(b'changed')
    with pytest.raises(ValueError, match='binding changed'):
        preview.audio_pins(tmp_path, {}, track)


def test_viewer_fill_revision_is_separate_by_camera_and_preserves_v1_namespace(tmp_path, monkeypatch):
    from lightsem.sim import lit_body_renderer
    from scripts.motion import native_dynamics_guard
    instances = []
    class FakeRenderer:
        def __init__(self, patch, cfg, **kwargs):
            self.cfg, self.options = cfg, kwargs
            instances.append(self)
        def presentation_config(self):
            return {'neutral_fill_radiance': self.options['neutral_fill']}
        def close(self):
            pass
    monkeypatch.setattr(native_dynamics_guard, 'authenticate_guarded_job',
        lambda *args: {'identity': {'command': ['--root', str(tmp_path)]}})
    monkeypatch.setattr(lit_body_renderer, 'LitBodyStageRenderer', FakeRenderer)
    with preview.renderers(tmp_path, [], True) as (renderers, metadata):
        assert [r.options['neutral_fill'] for r in renderers] == [.50, .12]
        assert metadata['fill_by_view'] == {'body': .50, 'stage': .12}
        assert metadata['presentation_by_view']['body']['neutral_fill_radiance'] == .50
        assert metadata['presentation_by_view']['stage']['neutral_fill_radiance'] == .12
        assert renderers[0].cfg.cam_pos == (2.8, 2.15, 3.8)
        assert renderers[1].cfg.cam_pos == (0., 6., 11.)
        assert all(r.options['body_offset'] == (0., 0., -3.5) for r in renderers)
    assert 'preview_3d_v2' in preview.folder(tmp_path, 'C4', 'shpongle').parts
    assert 'preview_3d_v2' in preview.audio_paths(tmp_path, {'id': 'shpongle'})[0].parts
