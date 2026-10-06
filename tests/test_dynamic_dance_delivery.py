"""Independent delivery verifier: auth drift, decoded pixels, honest labels, HTTP."""
from copy import deepcopy
from dataclasses import replace
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
import wave

import numpy as np
from PIL import Image, ImageDraw
import pytest

from scripts.rendering import verify_dynamic_dance_delivery as verify
from scripts.rendering import render_dynamic_dance as render
from test_dynamic_dance_render import campaign  # shared CPU-only pair fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def full_plan(root):
    tracks, pins = [], {}
    for song in verify.SONGS:
        track = {'id': song}
        for key in ('body', 'audio', 'audio_reference'):
            path = root/(song+'_'+key)
            path.write_bytes((song+key).encode())
            track[key] = str(path)
            if key != 'audio_reference':
                pins[str(path)] = verify.sha(path)
        tracks.append(track)
    plan = {'algorithms': [{'id': method} for method in verify.METHODS], 'tracks': tracks, 'files': pins,
            'prefix_s': 180., 'review_start_s': 60., 'review_duration_s': 120.,
            'body_fps': 30., 'fps': 40., 'render_fps': 30, 'render_size': [1920, 1080],
            'mapper_id': '01', 'counts': {'beams': 8, 'lasers': 5, 'bulbs': 50},
            'display_training_seed': 0, 'generation_seed': 7001}
    write(root/'PLAN.json', plan)
    return plan


def binding_fixture(root):
    plan = full_plan(root)
    legacy = deepcopy(plan)
    for track in plan['tracks']:
        legacy['files'][track['audio_reference']] = verify.sha(track['audio_reference'])
    path = root/'legacy.json'
    write(path, legacy)
    binding = {'schema_version': 1, 'plan_sha256': verify.sha(root/'PLAN.json'),
               'review_start_s': 60., 'review_duration_s': 120., 'tracks': {}}
    for track in plan['tracks']:
        binding['tracks'][track['id']] = {**{key: track[key] for key in ('audio', 'audio_reference')},
            **{key+'_sha256': verify.sha(track[key]) for key in ('audio', 'audio_reference')},
            'verification': {'basis': 'legacy_plan_pin', 'evidence_path': str(path), 'evidence_sha256': verify.sha(path)}}
    write(root/'RENDER_BINDINGS.json', binding)
    preserved = root/'records/original.npz'
    preserved.parent.mkdir(exist_ok=True)
    preserved.write_bytes(b'original retained')
    evidence = root/'records/D4_ORIGINAL_ARTIFACTS_MANIFEST.json'
    write(evidence, {'files': {str(preserved): verify.sha(preserved)}})
    variants = {'schema_version': 1, 'plan_sha256': verify.sha(root/'PLAN.json'),
                'selection_policy': 'implementation_correction_only', 'variants': {'D4': {
                    'artifact_id': 'D4_floor_v2', 'reason': 'Preserve floor height',
                    'references': {str(evidence): verify.sha(evidence)}}}}
    path = root/'records/VARIANT_BINDINGS.json'
    write(path, variants)
    write(path.with_name(path.name+'.sha256.json'), {'sha256': verify.sha(path)})
    return plan


def test_plan_requires_external_identity_and_two_actual_song_bodies(tmp_path):
    plan = full_plan(tmp_path)
    digest = verify.sha(tmp_path/'PLAN.json')
    assert verify.authenticate_plan(tmp_path, digest) == plan
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify.authenticate_plan(tmp_path, '0'*64)
    Path(plan['tracks'][0]['body']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify.authenticate_plan(tmp_path, digest)


def test_bindings_check_legacy_audio_and_preserved_original_artifacts(tmp_path):
    plan = binding_fixture(tmp_path)
    resolved, pins, _ = verify.authenticate_bindings(tmp_path, plan)
    assert resolved['D4'] == 'D4_floor_v2' and resolved['D1'] == 'D1'
    assert str(tmp_path/'records/VARIANT_BINDINGS.json') in pins
    (tmp_path/'records/original.npz').write_bytes(b'overwritten')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify.authenticate_bindings(tmp_path, plan)


def test_binding_rejects_new_song_soundtrack_and_outcome_selection(tmp_path):
    plan = binding_fixture(tmp_path)
    binding = verify.read(tmp_path/'RENDER_BINDINGS.json')
    binding['tracks']['shpongle']['audio_reference'] = plan['tracks'][1]['audio_reference']
    write(tmp_path/'RENDER_BINDINGS.json', binding)
    with pytest.raises(ValueError, match='Audio path mismatch'):
        verify.authenticate_bindings(tmp_path, plan)
    binding['tracks']['shpongle']['audio_reference'] = plan['tracks'][0]['audio_reference']
    write(tmp_path/'RENDER_BINDINGS.json', binding)
    path = tmp_path/'records/VARIANT_BINDINGS.json'
    variants = verify.read(path)
    variants['selection_policy'] = 'highest_score'
    write(path, variants)
    write(path.with_name(path.name+'.sha256.json'), {'sha256': verify.sha(path)})
    with pytest.raises(ValueError, match='Outcome-based'):
        verify.authenticate_bindings(tmp_path, plan)


def test_audit_requires_full_uniform_decode_and_unchanged_audio():
    audit = {'decoded_frames': 3600, 'pts_uniform': True, 'full_software_AV_decode': True, 'pcm_sha256': 'pcm',
             'video': {'width': 1920, 'height': 1080, 'r_frame_rate': '30/1', 'codec_name': 'av1'},
             'audio': {'duration': '120.0'}}
    receipt = {'frames': 3600, 'duration_s': 120., 'start_s': 60., 'codec': 'av1_nvenc'}
    verified = {'audio_matches_source_crop': True}
    verify.validate_media_audit(audit, receipt, verified, 'pcm')
    for key, value in [('decoded_frames', 3599), ('pts_uniform', False), ('full_software_AV_decode', False), ('pcm_sha256', 'different')]:
        with pytest.raises(ValueError):
            verify.validate_media_audit({**audit, key: value}, receipt, verified, 'pcm')


def test_composition_catches_substituted_body_without_light_renderer(campaign, monkeypatch):
    root, plan, algorithm, track = campaign
    baseline, candidate, music, _ = render.load_comparison(*campaign)
    strip = render.TimelineStrip(baseline, candidate, music, plan)
    dark = np.zeros((540, 960, 3), np.uint8)
    original = render.compose(dark, dark, baseline, candidate, 0, algorithm, track, strip)
    monkeypatch.setattr(render, 'renderer_for', lambda *a, **k: pytest.fail('Verifier invoked light renderer'))
    verify.verify_composition(original, baseline, candidate, 0, algorithm, track, strip)
    body = candidate.body.copy()
    body[0, 20] += [.7, .2, 0]
    with pytest.raises(ValueError, match='actual saved body'):
        verify.verify_composition(original, baseline, replace(candidate, body=body), 0, algorithm, track, strip)


def test_pixel_tolerance_reports_loss_and_rejects_missing_thin_body_or_dark_stage():
    image = Image.new('RGB', (1920, 1080), (12, 18, 25))
    draw = ImageDraw.Draw(image)
    for x in (0, 480, 960, 1440):
        draw.rectangle((x, 86, x+479, 322), fill=(238, 241, 243))
        draw.line((x+200, 290, x+280, 130), fill=(39, 114, 156), width=3)
    draw.line((1100, 420, 1550, 810), fill=(80, 150, 220), width=7)
    expected = np.asarray(image)
    noisy = np.clip(expected.astype(int)+2, 0, 255).astype(np.uint8)
    measured = verify.compare_pixels(expected, noisy)
    assert measured['passed'] and measured['regions']['candidate_front']['active_mean_abs'] == 2.
    missing = expected.copy()
    missing[86:323, 960:1440] = [238, 241, 243]
    assert not verify.compare_pixels(expected, missing)['passed']
    missing = expected.copy()
    missing[365:836, 962:1918] = [12, 18, 25]
    assert not verify.compare_pixels(expected, missing)['passed']


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg unavailable')
def test_actual_software_seek_and_lossy_decode_use_requested_frame(campaign):
    root, plan, algorithm, track = campaign
    baseline, candidate, music, _ = render.load_comparison(*campaign)
    strip = render.TimelineStrip(baseline, candidate, music, plan)
    dark = np.zeros((540, 960, 3), np.uint8)
    frames = []
    for index, tick in enumerate((0, 20, 40, 60)):
        pixels = render.compose(dark, dark, baseline, candidate, tick, algorithm, track, strip).copy()
        # Large deterministic frame IDs expose inaccurate seeking independently
        # of small neighboring-body displacement; these are test files only.
        pixels[450:650, 100:300] = (40+index*45, 110, 170)
        frames.append(pixels)
        Image.fromarray(pixels).save(root/f'frame_{index:02}.png')
    movie = root/'seek_fixture.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '1', '-framerate', '30',
        '-i', str(root/'frame_%02d.png'), '-c:v', 'libx264', '-threads', '1', '-crf', '18',
        '-pix_fmt', 'yuv420p', str(movie)], check=True)
    for index in range(4):
        decoded = verify.extract_frame(movie, index)
        assert abs(float(decoded[500:600, 150:250, 0].mean()) - (40+index*45)) < 5
        assert verify.compare_pixels(frames[index], decoded)['passed']


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg unavailable')
def test_audio_fingerprint_measures_pcm_and_detects_changed_waveform(tmp_path):
    samples = np.zeros((4410, 2), np.int16)
    def retain(name, values):
        path = tmp_path/name
        with wave.open(str(path), 'wb') as writer:
            writer.setnchannels(2)
            writer.setsampwidth(2)
            writer.setframerate(44100)
            writer.writeframes(values.tobytes())
        return path
    original = verify.audio_digest(retain('original.wav', samples))
    assert original['samples_per_channel'] == 4410
    assert verify.audio_digest(retain('identical.wav', samples)) == original
    samples[100, 0] = 512
    assert verify.audio_digest(retain('changed.wav', samples))['sha256'] != original['sha256']


def test_duplicate_labels_distinguish_body_and_lighting_identity():
    baseline = {'shpongle': {'visual': 'original', 'lighting': 'original_light'}}
    fingerprints = {('D1', 'shpongle'): {'visual': 'v1', 'lighting': 'l1'},
                    ('D2', 'shpongle'): {'visual': 'v1', 'lighting': 'l1'},
                    ('D3', 'shpongle'): {'visual': 'v3', 'lighting': 'l1'}}
    rows = []
    for index, method in enumerate(('D1', 'D2', 'D3'), 1):
        fp = fingerprints[method, 'shpongle']
        rows.append({'id': f'{index:02}', 'algorithm_id': method, 'track_id': 'shpongle',
            'visual_sha256': fp['visual'], 'light_stream_sha256': fp['lighting'],
            'duplicate_of': '01' if index == 2 else None, 'light_duplicate_of': '01' if index > 1 else None,
            'full_visual_matches_baseline': False, 'light_only_duplicate': index == 3})
    verify.verify_duplicate_labels(rows, fingerprints, baseline)
    rows[1]['duplicate_of'] = None
    with pytest.raises(ValueError, match='duplicate representative'):
        verify.verify_duplicate_labels(rows, fingerprints, baseline)


def test_http_all_sixteen_routes_ranges_downloads_and_allowlist(tmp_path, monkeypatch):
    from scripts.rendering import serve_body_portfolio as serve
    review = tmp_path/'review'
    review.mkdir()
    (review/'index.html').write_text('<!doctype html><title>test gallery</title>')
    rows = []
    for index in range(1, 17):
        path = tmp_path/'media'/f'{index:02}.mp4'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes(range(256))*8)
        rows.append({'id': f'{index:02}', 'status': 'complete', 'path': str(path), 'download_name': f'D{index}_song_dark.mp4'})
    manifest = {'variants': rows}
    write(review/'MANIFEST.json', manifest)
    monkeypatch.setattr(serve.base, 'ROOT', review)
    server = ThreadingHTTPServer(('127.0.0.1', 0), serve.PortfolioHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = verify.verify_http(f'http://127.0.0.1:{server.server_port}', tmp_path, manifest)
        assert len(result['media']) == 16 and len(result['denied_routes']) == 7
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize('url', ['https://example.com', 'http://127.0.0.1/path', 'http://user@127.0.0.1', 'http://127.0.0.1?path=/etc/passwd'])
def test_http_verifier_never_follows_external_or_arbitrary_origins(tmp_path, url):
    with pytest.raises(ValueError, match='plain local HTTP'):
        verify.verify_http(url, tmp_path, {'variants': []})


def test_verification_preserves_existing_reports_and_records_partial_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'disk_guard', lambda root: None)
    (tmp_path/'PLAN.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify.verify_delivery(tmp_path, '0'*64, 'delivery_v1')
    path = tmp_path/'delivery_verification/delivery_v1/REPORT.json'
    original = path.read_bytes()
    assert json.loads(original)['status'] == 'failed'
    with pytest.raises(FileExistsError):
        verify.verify_delivery(tmp_path, '0'*64, 'delivery_v1')
    assert path.read_bytes() == original


def test_floor_checked_before_any_artifact_write(tmp_path, monkeypatch):
    class Usage:
        free = verify.MIN_FREE_BYTES + 1
    monkeypatch.setattr(verify.shutil, 'disk_usage', lambda root: Usage())
    with pytest.raises(ValueError, match='20 GiB'):
        verify.verify_delivery(tmp_path, '0'*64, 'delivery_v1')
    assert not (tmp_path/'delivery_verification').exists()


@pytest.fixture
def pixel_collection_harness(tmp_path, monkeypatch):
    """Exercise the full verification loop with synthetic I/O seams only.

    Identity, decoder and HTTP primitives have independent tests above. This
    fixture tests coverage/control flow; its dummy bytes are never real media
    or used for a campaign delivery verdict.
    """
    digest = 'a'*64
    plan = {'algorithms': [{'id': method} for method in verify.METHODS],
            'tracks': [{'id': song, 'audio_reference': str(tmp_path/(song+'.m4a'))} for song in verify.SONGS],
            'review_start_s': 60., 'review_duration_s': 120.}
    resolved = {method: ('D4_floor_v2' if method == 'D4' else method) for method in verify.METHODS}
    documents, rows = {}, []
    indices = np.rint(np.arange(1800, 5400)*40/30).astype(int)
    candidate = SimpleNamespace(indices=indices, times=np.arange(7200)/40)
    pins = {'audio_reference_sha256': digest}
    checks = {'body': [], 'probe': [], 'pcm': [], 'pixels': 0, 'duplicates': [], 'http': []}
    for method in verify.METHODS:
        for song in verify.SONGS:
            movie = tmp_path/'media'/method/song/'dark.mp4'
            movie.parent.mkdir(parents=True)
            movie.write_bytes(b'test control flow only')
            folder = movie.parent
            comparison = {'method': method, 'song': song}
            documents[tmp_path/'timelines'/resolved[method]/song/'comparison.json'] = comparison
            rows.append({'id': f'{len(rows)+1:02}', 'algorithm_id': method, 'track_id': song,
                         'status': 'complete', 'artifact_revision': resolved[method], 'comparison': comparison,
                         'comparison_sha256': digest, 'banner': 'diagnostic', 'path': str(movie),
                         'sha256': digest, 'bytes': movie.stat().st_size})
            documents[folder/'dark_render.json'] = {'path': str(movie), 'inputs': pins, 'sha256': digest,
                'bytes': movie.stat().st_size, 'frames': 3600, 'duration_s': 120., 'start_s': 60.,
                'codec': 'av1_nvenc', 'mode': 'dark', 'smoke_sha256': digest, 'inspection_sha256': digest}
            documents[folder/'dark_verified.json'] = {'inputs': pins, 'video_sha256': digest,
                'audit_sha256': digest, 'source_reference_sha256': digest, 'audio_matches_source_crop': True}
            documents[folder/'dark_audit.json'] = {'path': str(movie), 'decoded_frames': 3600,
                'pts_uniform': True, 'full_software_AV_decode': True, 'pcm_sha256': digest,
                'video': {'width': 1920, 'height': 1080, 'r_frame_rate': '30/1', 'codec_name': 'av1'},
                'audio': {'duration': '120.0'}}
            smoke = tmp_path/'smoke'/method/song/'dark/SMOKE.json'
            documents[smoke] = {'inputs': pins, 'optical_controls_excluded_from_production': True,
                'files': {str(smoke.parent/f'{int(indices[frame]):06d}_actual.png'): digest for frame in verify.FRAME_INDICES},
                'samples': [{'tick': int(indices[frame]), 'frozen_pixels_exact': True} for frame in verify.FRAME_INDICES]}
            documents[smoke.with_name('INSPECTED.json')] = {'smoke_sha256': digest, 'inspected': True}
    documents[tmp_path/'review/MANIFEST.json'] = {'plan_sha256': digest, 'variant_binding_files': {},
        'mode': 'dark', 'tracks': plan['tracks'], 'algorithms': plan['algorithms'],
        'review_start_s': 60., 'review_duration_s': 120., 'variants': rows}
    monkeypatch.setattr(verify, 'disk_guard', lambda root: None)
    monkeypatch.setattr(verify, 'read', lambda path: documents[Path(path)])
    monkeypatch.setattr(verify, 'sha', lambda path: digest)
    monkeypatch.setattr(verify, 'pin', lambda *args: None)
    monkeypatch.setattr(verify, 'verify_pins', lambda *args: None)
    monkeypatch.setattr(verify, 'authenticate_plan', lambda *args: plan)
    monkeypatch.setattr(verify, 'authenticate_bindings', lambda *args: (resolved, {}, {}))
    def body(root, plan, artifact, track):
        checks['body'].append((artifact, track['id']))
        return {'visual': artifact+track['id'], 'lighting': artifact+track['id']}
    monkeypatch.setattr(verify, 'authenticate_body', body)
    def pcm(path):
        checks['pcm'].append(str(path))
        return {'sha256': digest, 'samples_per_channel': 120*44100}
    monkeypatch.setattr(verify, 'audio_digest', pcm)
    def probe(movie, audit):
        checks['probe'].append(str(movie))
        return {'video': audit['video'], 'audio': audit['audio']}
    monkeypatch.setattr(verify, 'probe_media', probe)
    monkeypatch.setattr(render, 'load_comparison', lambda *args: (candidate, candidate, {}, pins))
    monkeypatch.setattr(render, 'variant_metadata', lambda root, method: {'artifact_revision': resolved[method]})
    monkeypatch.setattr(render, 'evidence_summary', lambda comparison: {'banner': 'diagnostic'})
    monkeypatch.setattr(render, 'TimelineStrip', lambda *args: None)
    monkeypatch.setattr(verify, 'verify_composition', lambda *args: None)
    reference = Image.new('RGB', (1920, 1080))
    monkeypatch.setattr(verify.Image, 'open', lambda *args: reference)
    monkeypatch.setattr(verify, 'extract_frame', lambda *args: np.asarray(reference))
    def pixels(reference, decoded):
        index = checks['pixels']
        checks['pixels'] += 1
        passed = index not in (23, 63)  # D3/Cure and D8/Cure, final frame.
        return {'passed': passed, 'regions': {'traces': {'passed': passed,
            'mean_abs': 4.7865, 'p95_abs': 22., 'active_mean_abs': 17. if passed else 18.0074524}},
            'limits_8bit_RGB': verify.PIXEL_LIMITS.copy()}
    monkeypatch.setattr(verify, 'compare_pixels', pixels)
    def duplicates(rows, fingerprints, baseline):
        checks['duplicates'].append(len(fingerprints))
    monkeypatch.setattr(verify, 'verify_duplicate_labels', duplicates)
    def http(url, root, manifest):
        checks['http'].append(len(manifest['variants']))
        return {'checked_media': len(manifest['variants'])}
    monkeypatch.setattr(verify, 'verify_http', http)
    return tmp_path, digest, checks


def test_collect_pixel_failures_covers_all_pairs_and_retains_exact_failed_samples(pixel_collection_harness):
    root, digest, checks = pixel_collection_harness
    thresholds = verify.PIXEL_LIMITS.copy()
    report, path = verify.verify_delivery(root, digest, 'collected', url='http://127.0.0.1:8775', collect_pixel_failures=True)
    assert report['status'] == 'completed_with_pixel_failures'
    assert report['pixel_comparison_passed'] is False and report['pixel_comparison_completed'] is True
    assert len(report['pairs']) == 16 and checks['pixels'] == 64
    assert len(checks['body']) == 18 and len(checks['probe']) == 16 and len(checks['pcm']) == 18
    assert checks['duplicates'] == [16] and checks['http'] == [16]
    assert [(s['method'], s['song'], s['frame']) for s in report['pixel_failures']] == [
        ('D3', 'cure_for_me', 3599), ('D8', 'cure_for_me', 3599)]
    for sample in report['pixel_failures']:
        assert sample['passed'] is False
        assert sample['regions']['traces']['active_mean_abs'] == 18.0074524
        assert sample['limits_8bit_RGB'] == thresholds
    assert verify.PIXEL_LIMITS == thresholds
    assert json.loads(path.read_text())['pixel_failures'] == report['pixel_failures']


def test_default_strict_pixel_mode_stops_after_first_failed_pair(pixel_collection_harness):
    root, digest, checks = pixel_collection_harness
    with pytest.raises(ValueError, match='Decoded smoke pixel comparison failed: D3/cure_for_me'):
        verify.verify_delivery(root, digest, 'strict')
    report = json.loads((root/'delivery_verification/strict/REPORT.json').read_text())
    assert report['status'] == 'failed' and report['collect_pixel_failures'] is False
    assert report['pixel_comparison_passed'] is False and report['pixel_comparison_completed'] is False
    assert len(report['pairs']) == 6 and len(report['pixel_failures']) == 1 and checks['pixels'] == 24
    assert checks['duplicates'] == [] and checks['http'] == []


def test_collection_never_swallows_non_pixel_http_failure(pixel_collection_harness, monkeypatch):
    root, digest, checks = pixel_collection_harness
    def fail_http(*args):
        raise ValueError('HTTP identity mismatch')
    monkeypatch.setattr(verify, 'verify_http', fail_http)
    with pytest.raises(ValueError, match='HTTP identity mismatch'):
        verify.verify_delivery(root, digest, 'non_pixel_failure', url='http://127.0.0.1:8775', collect_pixel_failures=True)
    report = json.loads((root/'delivery_verification/non_pixel_failure/REPORT.json').read_text())
    assert report['status'] == 'failed' and report['pixel_comparison_passed'] is False
    assert len(report['pairs']) == 16 and len(report['pixel_failures']) == 2
    assert checks['duplicates'] == [16]


def test_pixel_collection_cli_is_explicit_warning_and_exit_two(monkeypatch, capsys, tmp_path):
    calls = []
    def completed(*args, **kwargs):
        calls.append(kwargs)
        return {'status': 'completed_with_pixel_failures', 'pairs': [None]*16}, tmp_path/'REPORT.json'
    monkeypatch.setattr(verify, 'verify_delivery', completed)
    monkeypatch.setattr(sys, 'argv', ['verify', '--root', str(tmp_path), '--plan-sha256', 'a'*64,
        '--revision', 'warning', '--collect-pixel-failures'])
    assert verify.main() == 2
    assert calls == [{'collect_pixel_failures': True}]
    output = capsys.readouterr().out
    assert 'DELIVERY_COMPLETED_WITH_PIXEL_FAILURES 16' in output and 'DELIVERY_VERIFIED' not in output
