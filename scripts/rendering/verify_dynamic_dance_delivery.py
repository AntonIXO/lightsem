"""CPU-only independent delivery checks; never render or alter campaign results.

Run after all sixteen render/audit/verified receipts and review/MANIFEST.json exist::

    python -m scripts.rendering.verify_dynamic_dance_delivery --root ROOT \
        --plan-sha256 FROZEN_PLAN_DIGEST --revision delivery_v1 [--url http://127.0.0.1:8775]

The external PLAN digest is mandatory. This verifier authenticates the existing
full software-decode audit instead of decoding every video again. Four fixed
frames and each delivered audio stream are independently software-decoded.
Lossy RGB comparison thresholds are implementation constants, not fit to results.
Successful delivery verification says nothing about artistic/scientific success.
Optional --collect-pixel-failures completes coverage with unchanged thresholds;
any failed sample remains a failed pixel comparison, with CLI exit status 2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image, ImageDraw

METHODS = tuple(f'D{i}' for i in range(1, 9))
SONGS = ('shpongle', 'cure_for_me')
FRAME_INDICES = (0, 1200, 2399, 3599)
WIDTH, HEIGHT, FPS, FRAMES = 1920, 1080, 30, 3600
MIN_FREE_BYTES = 20 * 2**30
# RGB 8-bit absolute error; thin colored skeletons incur 4:2:0 chroma loss.
PIXEL_LIMITS = {'mean_abs': 6., 'p95_abs': 24., 'active_mean_abs': 18.}
REGIONS = {'baseline_front': (0, 86, 958//2, 323),
           'baseline_oblique': (480, 86, 958, 323),
           'candidate_front': (962, 86, 1440, 323),
           'candidate_oblique': (1440, 86, 1920, 323),
           'baseline_stage': (2, 365, 958, 836),
           'candidate_stage': (962, 365, 1918, 836),
           'traces': (0, 864, 1920, 1080)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def pin(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Missing SHA256: ' + str(path))
    require(Path(path).is_file() and sha(path) == expected, 'SHA256 mismatch: ' + str(path))


def verify_pins(values):
    require(isinstance(values, dict) and bool(values), 'Missing file pins')
    for path, expected in values.items():
        pin(path, expected)


def inside(path, root):
    path = Path(path)
    require(path.is_absolute() and path.resolve().is_relative_to(root.resolve()), 'Path escapes campaign: ' + str(path))
    return path


def authenticate_plan(root, expected_sha):
    pin(root / 'PLAN.json', expected_sha)
    plan = read(root / 'PLAN.json')
    require([a['id'] for a in plan['algorithms']] == list(METHODS), 'Expected ordered D1–D8 methods')
    require([t['id'] for t in plan['tracks']] == list(SONGS), 'Expected independent Shpongle and Cure For Me tracks')
    expected = {'prefix_s': 180., 'review_start_s': 60., 'review_duration_s': 120.,
                'body_fps': 30., 'fps': 40., 'render_fps': 30, 'render_size': [1920, 1080],
                'mapper_id': '01', 'counts': {'beams': 8, 'lasers': 5, 'bulbs': 50},
                'display_training_seed': 0, 'generation_seed': 7001}
    require(all(plan.get(k) == v for k, v in expected.items()), 'PLAN differs from approved delivery contract')
    verify_pins(plan['files'])
    require(len({t['body'] for t in plan['tracks']}) == 2, 'Both songs reuse the same saved body')
    require(len({plan['files'][t['body']] for t in plan['tracks']}) == 2, 'Both songs have identical saved body bytes')
    return plan


def authenticate_bindings(root, plan):
    """Check bindings without trusting renderer admission helpers."""
    binding_path = root / 'RENDER_BINDINGS.json'
    binding = read(binding_path)
    require(binding.get('schema_version') == 1 and binding.get('plan_sha256') == sha(root / 'PLAN.json'),
            'Render binding PLAN/schema mismatch')
    require(all(binding.get(k) == plan[k] for k in ('review_start_s', 'review_duration_s')), 'Render crop mismatch')
    require(set(binding.get('tracks', {})) == set(SONGS), 'Render binding song coverage mismatch')
    pins = {str(binding_path): sha(binding_path)}
    for track in plan['tracks']:
        row = binding['tracks'][track['id']]
        for key in ('audio', 'audio_reference'):
            require(row.get(key) == track[key] and Path(track[key]).is_absolute(), 'Audio path mismatch')
            pin(track[key], row.get(key + '_sha256'))
        require(plan['files'].get(track['audio']) == row['audio_sha256'], 'Source audio lacks original PLAN pin')
        evidence = row.get('verification', {})
        require(evidence.get('basis') == 'legacy_plan_pin', 'Unknown audio reference proof')
        path = Path(evidence.get('evidence_path', ''))
        require(path.is_absolute(), 'Audio legacy evidence must be absolute')
        pin(path, evidence.get('evidence_sha256'))
        legacy = read(path)
        require(all(legacy.get(k) == plan[k] for k in ('review_start_s', 'review_duration_s')), 'Legacy audio crop mismatch')
        matches = [t for t in legacy.get('tracks', []) if t.get('id') == track['id']]
        require(len(matches) == 1 and all(matches[0].get(k) == track[k] for k in ('audio', 'audio_reference')),
                'Legacy evidence identifies another song')
        require(all(legacy.get('files', {}).get(track[k]) == row[k + '_sha256'] for k in ('audio', 'audio_reference')),
                'Legacy evidence lacks exact audio pins')
        pins[str(path)] = sha(path)
    variant_path = root / 'records/VARIANT_BINDINGS.json'
    seal = variant_path.with_name(variant_path.name + '.sha256.json')
    pin(variant_path, read(seal).get('sha256'))
    variants = read(variant_path)
    require(variants.get('schema_version') == 1 and variants.get('plan_sha256') == sha(root / 'PLAN.json'),
            'Variant binding PLAN/schema mismatch')
    require(variants.get('selection_policy') == 'implementation_correction_only', 'Outcome-based revision selection')
    require(set(variants.get('variants', {})) <= set(METHODS), 'Unknown variant method')
    resolved = {method: method for method in METHODS}
    variant_pins = {str(variant_path): sha(variant_path), str(seal): sha(seal)}
    for method, row in variants['variants'].items():
        require(re.fullmatch('[A-Za-z0-9_-]+', row.get('artifact_id', '')) and row.get('reason'), 'Invalid variant identity/reason')
        require(bool(row.get('references')), 'Variant lacks preserved evidence')
        for path, digest in row['references'].items():
            inside(path, root)
            pin(path, digest)
            # Authenticated preservation manifests must still describe present files.
            value = read(path)
            if 'ORIGINAL_ARTIFACTS_MANIFEST' in Path(path).name:
                verify_pins(value['files'])
        variant_pins.update(row['references'])
        resolved[method] = row['artifact_id']
    require(resolved['D4'] == 'D4_floor_v2', 'Expected explicit D4 fixed-floor revision')
    return resolved, variant_pins, pins


def array_fingerprint(arrays, keys):
    result = hashlib.sha256()
    for key in keys:
        value = np.ascontiguousarray(arrays[key])
        result.update(str((key, value.shape, value.dtype.str)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def authenticate_body(root, plan, artifact, track):
    folder = root / 'timelines' / artifact / track['id']
    report = read(folder / 'report.json')
    body_path = root / 'bodies' / artifact / track['id'] / 'body.npz'
    inputs = report.get('inputs', {})
    pin(body_path, inputs.get('body_sha256'))
    pin(body_path.with_suffix('.json'), inputs.get('body_metadata_sha256'))
    metadata = read(body_path.with_suffix('.json'))
    pin(body_path, metadata.get('npz_sha256'))
    require(inputs.get('plan_sha256') == sha(root / 'PLAN.json'), 'Mapped body PLAN mismatch')
    require(report.get('body_sha256') == sha(body_path), 'Body identity differs within mapping receipt')
    require(inputs.get('P2_timeline_sha256') == plan['files'][track['baseline_timeline']] and
            inputs.get('P2_patch_sha256') == plan['files'][track['baseline_patch']], 'P2 source identity mismatch')
    verify_pins(inputs['implementation'])
    if inputs.get('dance_plan'):
        verify_pins(inputs['dance_plan'])
    for name, key in (('timeline.npz', 'timeline_sha256'), ('patch.json', 'patch_sha256')):
        pin(folder / name, report.get(key))
    with np.load(folder / 'timeline.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ('body', 'universes', 'time')}
    require(arrays['body'].shape == (7200, 22, 3) and arrays['time'].shape == (7200,), 'Mapped body shape mismatch')
    require(np.array_equal(arrays['time'], np.arange(7200)/40) and np.isfinite(arrays['body']).all(), 'Mapped body clock/nonfinite')
    require(arrays['universes'].dtype == np.uint8 and arrays['universes'].shape == (7200, 1, 512), 'Unexpected emitted DMX shape/dtype')
    with np.load(body_path, allow_pickle=False) as data:
        world, time = data['world'], data['time']
        require(world.shape == (5400, 22, 3) and np.isfinite(world).all(), 'Invalid saved native body')
        require(np.allclose(time, np.arange(5400)/30, atol=1e-7, rtol=0), 'Native body clock mismatch')
        require(np.allclose(world[::3], arrays['body'][::4], atol=1e-6, rtol=0),
                'Mapped body does not match saved body at coincident 30/40 Hz timestamps')
        if artifact == 'baseline':
            with np.load(track['body'], allow_pickle=False) as original:
                require(np.array_equal(original['world'], world), 'Baseline body is not original same-song body')
    provenance = metadata.get('metadata', {})
    if artifact != 'baseline':
        if artifact in ('D6', 'D7', 'D8'):
            require(provenance.get('seed') == 0 and provenance.get('draw_seed') == 7001, 'Display model/noise seed changed')
            pin(provenance['checkpoint'], provenance.get('checkpoint_sha256'))
            require(provenance.get('timeline_path') == track['music_timeline'] and
                    provenance.get('timeline_sha256') == sha(track['music_timeline']) and
                    provenance.get('timeline_metadata_sha256') == sha(Path(track['music_timeline']).with_suffix('.json')),
                    'Learned body used another song/counterfactual MusicTimeline')
            require(provenance.get('audio_intervention') is None, 'Counterfactual body substituted into displayed result')
            if artifact == 'D6':
                require(provenance.get('source_body') == track['body'] and
                        provenance.get('source_body_sha256') == plan['files'][track['body']], 'D6 uses another song body')
        else:
            require(provenance.get('song') == track['id'] and provenance.get('campaign_inputs', {}).get('source_body_sha256')
                    == plan['files'][track['body']], 'Candidate uses a different song source body')
            require(provenance.get('campaign_inputs', {}).get('music_npz_sha256') == sha(track['music_timeline']) and
                    provenance.get('campaign_inputs', {}).get('control') is None, 'Counterfactual music substituted into candidate')
    # Independent raw uint8 pin, rather than accepting a report status as proof.
    dmx = arrays['universes']
    expected = hashlib.sha256(str(dmx.dtype).encode() + str(dmx.shape).encode() + dmx.tobytes()).hexdigest()
    require(report.get('emitted_DMX_sha256') == expected, 'Emitted DMX array hash mismatch')
    return {'visual': array_fingerprint(arrays, ('body', 'universes', 'time')),
            'lighting': array_fingerprint(arrays, ('universes', 'time')),
            'body_path': str(body_path), 'body_sha256': sha(body_path), 'report_sha256': sha(folder/'report.json')}


def audio_digest(path):
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-i', str(path),
               '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '44100', '-f', 'f32le', '-']
    result, length = hashlib.sha256(), 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for chunk in iter(lambda: proc.stdout.read(1 << 20), b''):
            result.update(chunk)
            length += len(chunk)
        error = proc.stderr.read().decode(errors='replace')
        require(proc.wait() == 0, 'Audio software decode failed: ' + error[:1000])
    return {'sha256': result.hexdigest(), 'samples_per_channel': length // 8}


def validate_media_audit(audit, render, verified, expected_pcm):
    require(render.get('frames') == FRAMES and render.get('duration_s') == 120. and render.get('start_s') == 60.,
            'Render frame count/duration/crop mismatch')
    require(audit.get('decoded_frames') == FRAMES and audit.get('pts_uniform') is True and
            audit.get('full_software_AV_decode') is True, 'No complete uniform 3600-frame software audit')
    require(audit.get('pcm_sha256') == expected_pcm and verified.get('audio_matches_source_crop') is True,
            'Audit audio differs from unchanged source crop')
    video, audio = audit['video'], audit['audio']
    require((video.get('width'), video.get('height')) == (WIDTH, HEIGHT) and
            Fraction(video['r_frame_rate']) == FPS, 'Audit video size/rate mismatch')
    require(abs(float(audio['duration']) - 120.) <= .06, 'Audit audio duration mismatch')
    require(video.get('codec_name') == {'libx264': 'h264', 'av1_nvenc': 'av1'}.get(render.get('codec')),
            'Unexpected delivery codec')


def probe_media(movie, audit):
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(movie)], text=True))
    videos = [v for v in result['streams'] if v['codec_type'] == 'video']
    audios = [v for v in result['streams'] if v['codec_type'] == 'audio']
    require(len(videos) == len(audios) == 1, 'Expected one video and one audio stream')
    video, audio = videos[0], audios[0]
    require(all(video.get(k) == audit['video'].get(k) for k in ('codec_name', 'width', 'height', 'r_frame_rate')),
            'Independent ffprobe differs from audited format')
    require(Fraction(video['avg_frame_rate']) == FPS and abs(float(video.get('start_time', 0))) < 1e-5,
            'Video average rate/start timestamp mismatch')
    if 'nb_frames' in video:
        require(int(video['nb_frames']) == FRAMES, 'Independent container frame count mismatch')
    require(abs(float(video['duration']) - 120.) < .001 and abs(float(audio['duration']) - 120.) <= .06,
            'Independent probe duration mismatch')
    return {'video': video, 'audio': audio}


def extract_frame(movie, frame):
    """Accurate input seeking decodes from the preceding keyframe, never uses CUDA."""
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-hwaccel', 'none', '-threads', '2',
        '-ss', f'{frame/FPS:.9f}', '-i', str(movie), '-map', '0:v:0', '-frames:v', '1', '-an',
        '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', '-'])
    require(len(raw) == WIDTH * HEIGHT * 3, 'Sampled frame decode size mismatch')
    return np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 3)


def compare_pixels(reference, decoded):
    require(reference.shape == decoded.shape == (HEIGHT, WIDTH, 3), 'Pixel comparison requires 1080p RGB')
    metrics = {}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        expected, actual = reference[y0:y1, x0:x1], decoded[y0:y1, x0:x1]
        delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        # Exclude flat background so black stages cannot hide missing beams and
        # pale body panels cannot hide missing thin skeletons.
        background = np.median(expected.reshape(-1, 3), axis=0)
        active = np.max(np.abs(expected.astype(float) - background), axis=-1) >= 20
        values = {'mean_abs': float(delta.mean()), 'p95_abs': float(np.quantile(delta, .95)),
                  'active_mean_abs': float(delta[active].mean()) if active.any() else 0.,
                  'active_pixels': int(active.sum())}
        values['passed'] = all(values[k] <= bound for k, bound in PIXEL_LIMITS.items())
        metrics[name] = values
    return {'passed': all(v['passed'] for v in metrics.values()), 'regions': metrics,
            'limits_8bit_RGB': PIXEL_LIMITS.copy(),
            'scope': 'Four sampled lossy decoded frames; no claim of pixel identity for every frame.'}


def verify_composition(reference, baseline, candidate, tick, algorithm, track, strip):
    """Recompose only saved body/trace panels; never invoke the light renderer."""
    from scripts.rendering import render_dynamic_dance as render
    require(reference.shape == (HEIGHT, WIDTH, 3), 'Smoke frame dimensions mismatch')
    stages = reference[324:864, :960], reference[324:864, 960:]
    recomposed = render.compose(*stages, baseline, candidate, int(tick), algorithm, track, strip)
    require(np.array_equal(reference[:324], recomposed[:324]) and
            np.array_equal(reference[864:], recomposed[864:]), 'Smoke does not show actual saved body/trace composition')


def verify_duplicate_labels(rows, fingerprints, baseline):
    visual_seen, light_seen = {}, {}
    for row in rows:
        key = row['algorithm_id'], row['track_id']
        current, original = fingerprints[key], baseline[row['track_id']]
        visual_key, light_key = (key[1], current['visual']), (key[1], current['lighting'])
        duplicate = visual_seen.get(visual_key)
        light_duplicate = 'baseline' if current['lighting'] == original['lighting'] else light_seen.get(light_key)
        exact_baseline = current['visual'] == original['visual']
        expected = {'visual_sha256': current['visual'], 'light_stream_sha256': current['lighting'],
                    'duplicate_of': duplicate, 'light_duplicate_of': light_duplicate,
                    'full_visual_matches_baseline': exact_baseline,
                    'light_only_duplicate': light_duplicate is not None and duplicate is None and not exact_baseline}
        require(all(row.get(k) == v for k, v in expected.items()), 'Incorrect duplicate representative labels: ' + row['id'])
        visual_seen.setdefault(visual_key, row['id'])
        light_seen.setdefault(light_key, row['id'])


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, route, method='GET', headers=None, maximum=1 << 20):
    opener = build_opener(NoRedirect)
    req = Request(url + route, method=method, headers=headers or {})
    try:
        response = opener.open(req, timeout=20)
    except HTTPError as error:
        response = error
    with response:
        body = response.read(maximum + 1)
        require(len(body) <= maximum, 'HTTP response exceeds bounded verification read')
        return response.status, dict(response.headers.items()), body


def verify_http(url, root, manifest):
    parsed = urlsplit(url)
    require(parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost', '::1') and
            not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and
            parsed.path in ('', '/'), 'Review verification requires a plain local HTTP origin')
    url = url.rstrip('/')
    for route, path in (('/', root/'review/index.html'), ('/index.html', root/'review/index.html'),
                        ('/manifest.json', root/'review/MANIFEST.json')):
        status, headers, body = request(url, route, maximum=8 << 20)
        require(status == 200 and body == path.read_bytes(), 'HTTP serves stale/different review: ' + route)
        require(int(headers['Content-Length']) == len(body), 'Review Content-Length mismatch')
    media = []
    for row in manifest['variants']:
        path = inside(row['path'], root/'media')
        size, route = path.stat().st_size, '/media/' + row['id'] + '.mp4'
        status, headers, body = request(url, route, method='HEAD')
        require(status == 200 and not body and int(headers['Content-Length']) == size and
                headers.get('Content-Type') == 'video/mp4' and headers.get('Accept-Ranges') == 'bytes', 'Media HEAD mismatch: ' + route)
        samples = [(0, min(63, size-1)), (size//2, min(size//2+63, size-1)), (max(0, size-64), size-1)]
        for index, (start, end) in enumerate(samples):
            range_value = 'bytes=-64' if index == 2 else f'bytes={start}-{end}'
            status, headers, body = request(url, route, headers={'Range': range_value})
            with path.open('rb') as stream:
                stream.seek(start)
                expected = stream.read(end-start+1)
            require(status == 206 and body == expected and headers.get('Content-Range') == f'bytes {start}-{end}/{size}'
                    and int(headers['Content-Length']) == len(expected), 'Media Range bytes mismatch: ' + route)
        status, headers, _ = request(url, route+'?download=1', method='HEAD')
        expected_name = "attachment; filename*=UTF-8''" + quote(row['download_name'], safe='')
        require(status == 200 and headers.get('Content-Disposition') == expected_name, 'Incorrect download filename/header: ' + route)
        status, headers, _ = request(url, route, headers={'Range': f'bytes={size}-'})
        require(status == 416 and headers.get('Content-Range') == 'bytes */'+str(size), 'Out-of-bounds Range accepted')
        media.append({'id': row['id'], 'path': route, 'range_checks': len(samples), 'download_header_verified': True})
    denied = ['/PLAN.json', '/../../etc/passwd', '/%2e%2e/%2e%2e/etc/passwd', '/media/../PLAN.json',
              '/media/99.mp4', '/media/01.mp4/extra', '/root/lightsem/AGENTS.md']
    for route in denied:
        status, _, _ = request(url, route)
        require(status in (400, 403, 404), 'Server exposed an unallowlisted route: ' + route)
    return {'url': url, 'media': media, 'denied_routes': denied,
            'scope': 'HTTP routes, bytes and headers; browser playback and visual review are separate.'}


def disk_guard(root):
    # Reserve an additional 128 MiB for all reports/thumbnails before creating any.
    require(shutil.disk_usage(root).free >= MIN_FREE_BYTES + 128 * 2**20, '20 GiB free-space floor plus verifier write allowance')


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_delivery(root, expected_plan_sha, revision, url=None, mode='dark', collect_pixel_failures=False):
    from scripts.rendering import render_dynamic_dance as render
    root = Path(root).resolve()
    require(re.fullmatch('[A-Za-z0-9_-]+', revision), 'Unsafe verification revision')
    disk_guard(root)
    destination = root / 'delivery_verification' / revision
    destination.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'created_at': datetime.now(timezone.utc).isoformat(),
              'verifier_sha256': sha(__file__), 'plan_sha256': expected_plan_sha,
              'mode': mode, 'frames_sampled_per_movie': list(FRAME_INDICES), 'pairs': [],
              'collect_pixel_failures': collect_pixel_failures, 'pixel_failures': [],
              'pixel_comparison_passed': None, 'pixel_comparison_completed': False,
              'limitations': ['Authenticated prior full software-decode audit reused; no second full video decode.',
                             'Four decoded frames cannot establish whole-video artistic or musical quality.',
                             'CPU recomposition verifies body/trace panels only; stages use pinned smoke pixels.',
                             'Contact sheets require independent human visual review.']}
    try:
        plan = authenticate_plan(root, expected_plan_sha)
        resolved, variant_pins, binding_pins = authenticate_bindings(root, plan)
        report.update(variant_binding_files=variant_pins, audio_binding_files=binding_pins)
        manifest_path = root/'review/MANIFEST.json'
        manifest = read(manifest_path)
        report['manifest_sha256'] = sha(manifest_path)
        require(manifest.get('plan_sha256') == expected_plan_sha and manifest.get('variant_binding_files') == variant_pins,
                'Review manifest plan/variant identity mismatch')
        require(manifest.get('mode') == mode and manifest.get('tracks') == plan['tracks'] and
                manifest.get('algorithms') == plan['algorithms'] and manifest.get('review_start_s') == 60. and
                manifest.get('review_duration_s') == 120., 'Review manifest contract mismatch')
        require(sha(root/'review/index.html') == sha(render.TEMPLATE), 'Review HTML is not frozen renderer template')
        report['html_sha256'] = sha(root/'review/index.html')
        rows = manifest['variants']
        expected_pairs = [(method, song) for method in METHODS for song in SONGS]
        require([(row['algorithm_id'], row['track_id']) for row in rows] == expected_pairs and
                [row['id'] for row in rows] == [f'{i:02}' for i in range(1, 17)], 'Review must list exactly 16 ordered comparisons')
        baseline_fingerprints = {track['id']: authenticate_body(root, plan, 'baseline', track) for track in plan['tracks']}
        reference_pcm = {track['id']: audio_digest(track['audio_reference']) for track in plan['tracks']}
        fingerprints = {}
        for row in rows:
            disk_guard(root)
            method, song = row['algorithm_id'], row['track_id']
            print('VERIFY', method, song, flush=True)
            algorithm = next(a for a in plan['algorithms'] if a['id'] == method)
            track = next(t for t in plan['tracks'] if t['id'] == song)
            require(row.get('status') == 'complete' and row.get('artifact_revision') == resolved[method], 'Incomplete/wrong revision: '+row['id'])
            fingerprints[method, song] = authenticate_body(root, plan, resolved[method], track)
            baseline, candidate, music, pins = render.load_comparison(root, plan, algorithm, track)
            comparison_path = root/'timelines'/resolved[method]/song/'comparison.json'
            comparison = read(comparison_path)
            require(row.get('comparison') == comparison and row.get('comparison_sha256') == sha(comparison_path), 'Gallery comparison evidence mismatch')
            expected_summary = render.evidence_summary(comparison)
            require(all(row.get(k) == v for k, v in expected_summary.items()), 'Gallery failure/evidence labels mismatch')
            metadata = render.variant_metadata(root, method)
            require(all(row.get(k) == v for k, v in metadata.items()), 'Gallery correctness-revision disclosure mismatch')
            folder = root/'media'/method/song
            movie = folder/(mode+'.mp4')
            render_path, audit_path, verified_path = [folder/(mode+suffix) for suffix in ('_render.json', '_audit.json', '_verified.json')]
            receipt, audit, verified = read(render_path), read(audit_path), read(verified_path)
            require(row.get('path') == str(movie) == receipt.get('path'), 'Media path is not exact allowlisted pair')
            inside(movie, root/'media')
            pin(movie, receipt.get('sha256'))
            require(row.get('sha256') == verified.get('video_sha256') == receipt['sha256'] and
                    row.get('bytes') == receipt.get('bytes') == movie.stat().st_size, 'Movie identity/size mismatch')
            pin(audit_path, verified.get('audit_sha256'))
            require(all(r.get('inputs') == pins for r in (receipt, verified)), 'Render/verified inputs changed')
            require(verified.get('source_reference_sha256') == pins['audio_reference_sha256'], 'Verified audio-reference pin mismatch')
            require(audit.get('path') == str(movie) and receipt.get('mode') == mode, 'Audit identifies another movie')
            validate_media_audit(audit, receipt, verified, reference_pcm[song]['sha256'])
            probe = probe_media(movie, audit)
            delivered_pcm = audio_digest(movie)
            require(delivered_pcm == reference_pcm[song], 'Independent delivered PCM differs from unchanged reference')
            smoke_path = root/'smoke'/method/song/mode/'SMOKE.json'
            inspection_path = smoke_path.with_name('INSPECTED.json')
            pin(smoke_path, receipt.get('smoke_sha256'))
            pin(inspection_path, receipt.get('inspection_sha256'))
            smoke, inspection = read(smoke_path), read(inspection_path)
            require(smoke.get('inputs') == pins and inspection.get('smoke_sha256') == sha(smoke_path) and
                    inspection.get('inspected') is True, 'Smoke/visual inspection inputs mismatch')
            verify_pins(smoke['files'])
            require(smoke.get('optical_controls_excluded_from_production') is True, 'Smoke optical controls unclear')
            ticks = candidate.indices[list(FRAME_INDICES)]
            require([v.get('tick') for v in smoke['samples']] == ticks.tolist() and
                    all(v.get('frozen_pixels_exact') is True for v in smoke['samples']), 'Smoke samples are not fixed production frames')
            strip = render.TimelineStrip(baseline, candidate, music, plan)
            sheet = Image.new('RGB', (1440, 4*294), (12, 18, 25))
            samples = []
            for slot, (frame, tick) in enumerate(zip(FRAME_INDICES, ticks)):
                smoke_frame_path = smoke_path.parent/f'{int(tick):06d}_actual.png'
                pin(smoke_frame_path, smoke['files'].get(str(smoke_frame_path)))
                reference = np.asarray(Image.open(smoke_frame_path).convert('RGB'))
                verify_composition(reference, baseline, candidate, tick, algorithm, track, strip)
                decoded = extract_frame(movie, frame)
                metric = compare_pixels(reference, decoded)
                samples.append({'frame': frame, 'source_tick': int(tick), 'source_time_s': float(candidate.times[tick]),
                                'smoke_sha256': sha(smoke_frame_path), **metric})
                difference = np.minimum(np.abs(reference.astype(np.int16)-decoded.astype(np.int16))*4, 255).astype(np.uint8)
                for column, (pixels, label) in enumerate(((reference, 'Pinned smoke'), (decoded, 'Decoded delivery'), (difference, 'Absolute RGB error x4'))):
                    sheet.paste(Image.fromarray(pixels).resize((480, 270)), (column*480, slot*294+24))
                    ImageDraw.Draw(sheet).text((column*480+5, slot*294+4), f'{method}/{song} frame {frame}: {label}', fill='white')
            sheet_path = destination/(method+'_'+song+'_samples.jpg')
            with sheet_path.open('xb') as stream:
                sheet.save(stream, format='JPEG', quality=93)
            pair_result = {'method': method, 'song': song, 'artifact_revision': resolved[method],
                           'movie_sha256': receipt['sha256'], 'render_sha256': sha(render_path),
                           'audit_sha256': sha(audit_path), 'verified_sha256': sha(verified_path),
                           'inputs': pins, 'body_authentication': fingerprints[method, song],
                           'audio': delivered_pcm, 'independent_probe': probe, 'samples': samples,
                           'contact_sheet': str(sheet_path), 'contact_sheet_sha256': sha(sheet_path),
                           'pixel_comparison_passed': all(sample['passed'] for sample in samples),
                           'evidence_banner': expected_summary['banner']}
            report['pairs'].append(pair_result)
            failed_samples = [{'method': method, 'song': song, **sample} for sample in samples if not sample['passed']]
            report['pixel_failures'].extend(failed_samples)
            if failed_samples:
                report['pixel_comparison_passed'] = False
                if not collect_pixel_failures:
                    raise ValueError('Decoded smoke pixel comparison failed: '+method+'/'+song)
        report['pixel_comparison_completed'] = True
        report['pixel_comparison_passed'] = not report['pixel_failures']
        verify_duplicate_labels(rows, fingerprints, baseline_fingerprints)
        report['http'] = verify_http(url, root, manifest) if url else {'status': 'not_requested'}
        # Catch concurrent gallery/source mutation rather than silently verifying a mixed revision.
        pin(manifest_path, report['manifest_sha256'])
        pin(root/'review/index.html', report['html_sha256'])
        pin(root/'PLAN.json', expected_plan_sha)
        verify_pins(variant_pins)
        verify_pins(binding_pins)
        report['status'] = 'passed' if report['pixel_comparison_passed'] else 'completed_with_pixel_failures'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_new(destination/'REPORT.json', report)
        raise
    write_new(destination/'REPORT.json', report)
    write_new(destination/'REPORT.json.sha256.json', {'sha256': sha(destination/'REPORT.json')})
    return report, destination/'REPORT.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True, help='Externally frozen original PLAN hash; never infer it from current files')
    parser.add_argument('--revision', required=True, help='New exclusive verification directory name')
    parser.add_argument('--mode', choices=('dark', 'schematic'), default='dark')
    parser.add_argument('--url', help='Optional running loopback gallery origin')
    parser.add_argument('--collect-pixel-failures', action='store_true',
                        help='Keep fixed pixel thresholds, retain failures and check every remaining pair; warning completion exits 2')
    args = parser.parse_args()
    report, path = verify_delivery(args.root, args.plan_sha256, args.revision, args.url, args.mode,
                                   collect_pixel_failures=args.collect_pixel_failures)
    label = 'DELIVERY_VERIFIED' if report['status'] == 'passed' else 'DELIVERY_COMPLETED_WITH_PIXEL_FAILURES'
    print(label, len(report['pairs']), path, flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
