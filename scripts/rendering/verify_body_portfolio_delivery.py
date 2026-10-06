"""Independent, CPU-only verification of the completed V3 portfolio.

Authenticates saved outputs without importing the generation/render pipeline or
rerendering. Existing full-decode audit receipts are checked, not repeated.
Optional pixel sampling software-decodes four frames per movie and reports
encoding differences; it applies no artistic or encoding acceptance threshold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.message import Message
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FRAMES = (0, 1200, 2399, 3599)
TICKS = (2400, 4000, 5599, 7199)
MAX_COMBINED_BYTES = 2 * 2 ** 30
MIN_FREE_BYTES = 20 * 2 ** 30
WIDTH, HEIGHT = 1920, 1080


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read(path):
    path = Path(path)
    require(path.is_file(), 'Required file missing: ' + str(path))
    return json.loads(path.read_text())


def digest(path):
    path = Path(path)
    require(path.is_file(), 'Required file missing: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_hash(path, expected):
    actual = digest(path)
    require(actual == expected, 'File hash mismatch: ' + str(path))
    return actual


def check_sources(root):
    plan = read(root / 'PLAN.json')
    source_root = Path(plan['source_root']).resolve()
    require(source_root == root / 'source', 'Frozen source is outside the requested campaign')
    manifest_path = Path(plan['source_manifest']).resolve()
    require(manifest_path == root / 'source_manifest.json', 'Unexpected source manifest path')
    pins = plan.get('files', {})
    require(str(manifest_path) in pins, 'PLAN must pin source_manifest.json')
    check_hash(manifest_path, pins[str(manifest_path)])
    source_manifest = read(manifest_path)
    require(source_manifest['source_root'] == str(source_root), 'Source root binding mismatch')
    source_files = source_manifest['files']
    require(set(source_files) == set(source_manifest['frozen_paths']), 'Source manifest membership mismatch')
    require(bool(source_files), 'Empty source manifest')
    for filename, expected in source_files.items():
        require(Path(filename).resolve().is_relative_to(source_root), 'Frozen source escapes source root')
        require(pins.get(filename) == expected, 'PLAN/source manifest hash disagreement: ' + filename)
    for filename, expected in pins.items():
        check_hash(filename, expected)
    return plan, {'plan_sha256': digest(root / 'PLAN.json'), 'source_manifest_sha256': digest(manifest_path),
        'frozen_source_files': len(source_files), 'all_plan_input_files': len(pins),
        'all_frozen_source_files_unchanged': True, 'all_plan_inputs_unchanged': True}


def presentation_binding(root, plan_hash, source_hash):
    path = root / 'records/RENDER_PRESENTATION_BINDING.json'
    if not path.exists():
        return {}, None, {}
    binding = read(path)
    require(binding['plan_sha256'] == plan_hash, 'Presentation binding belongs to a different PLAN')
    require(binding['base_source_manifest_sha256'] == source_hash, 'Presentation source binding mismatch')
    check_hash(binding['wrapper'], binding['wrapper_sha256'])
    status_path = root / 'records/PRESENTATION_STATUS.json'
    check_hash(status_path, binding['status_sha256'])
    statuses = read(status_path)['pairs']
    pins = {'presentation_status_sha256': binding['status_sha256'],
            'presentation_wrapper_sha256': binding['wrapper_sha256']}
    return pins, digest(path), statuses


def reference_pcm(track, plan):
    """Bind the pinned reference file to the prior independently checked crop."""
    reference = Path(track['audio_reference'])
    require(str(reference) in plan['files'], 'Unpinned audio reference: ' + str(reference))
    reference_sha = check_hash(reference, plan['files'][str(reference)])
    primary_path = Path('/root/lightsem/experiments/2026-09-30-body-transfer/records/DELIVERY_PRIMARY_AUDIT_V1.json')
    primary = read(primary_path)
    rows = [row for row in primary['media'] if row['id'] == track['id']]
    require(len(rows) == 1, 'Historical reference receipt missing/ambiguous')
    row = rows[0]
    old_audit_path = reference.parent / (track['id'] + '_body_to_63_P2_1080p_av1_audit.json')
    check_hash(old_audit_path, row['audit_sha256'])
    old_audit = read(old_audit_path)
    require(old_audit['pcm_sha256'] == row['reference_pcm_sha256'], 'Historical source PCM receipt disagreement')
    require(old_audit['decoded_frames'] == 3600 and old_audit['pts_uniform'] is True
            and old_audit['full_software_AV_decode'] is True, 'Historical source audit is incomplete')
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(reference),
               '-map', '0:a:0', '-vn', '-ac', '2', '-ar', '44100', '-f', 'f32le', '-']
    decoded = subprocess.run(command, check=True, capture_output=True, timeout=60)
    actual_pcm_sha = hashlib.sha256(decoded.stdout).hexdigest()
    require(actual_pcm_sha == row['reference_pcm_sha256'], 'Actual decoded reference PCM differs from historical receipt')
    return {'reference_path': str(reference), 'reference_sha256': reference_sha,
        'pcm_sha256': row['reference_pcm_sha256'], 'historical_primary_receipt': str(primary_path),
        'historical_primary_sha256': digest(primary_path), 'historical_media_audit': str(old_audit_path),
        'historical_media_audit_sha256': row['audit_sha256'],
        'actual_reference_decoded_once': True, 'reference_decode_command': command,
        'method': 'Actual source-reference PCM decoded once per song and checked against historical crop receipt; no repeated full video decode'}


def validate_audit(audit, pcm):
    video = audit['video']
    require((video['codec_name'], video['width'], video['height'], video['r_frame_rate'])
            == ('av1', WIDTH, HEIGHT, '30/1'), 'Unexpected audited video format')
    require(audit['decoded_frames'] == 3600 and audit.get('expected_frames', 3600) == 3600,
            'Unexpected audited frame count')
    require(audit['pts_uniform'] is True and audit['full_software_AV_decode'] is True,
            'Incomplete decode/PTS audit')
    require(audit['pcm_sha256'] == pcm['pcm_sha256'], 'PCM hash differs from the selected song reference')
    require(abs(float(audit['audio']['duration']) - 120) <= .06, 'Audited audio duration mismatch')


def pixel_metrics(actual, reference):
    import numpy as np
    x, y = np.asarray(actual, dtype=np.float64).reshape(-1), np.asarray(reference, dtype=np.float64).reshape(-1)
    require(x.shape == y.shape and len(x) > 0, 'Invalid image comparison shape')
    std_x, std_y = float(x.std()), float(y.std())
    correlation = float(np.corrcoef(x, y)[0, 1]) if min(std_x, std_y) > 0 else None
    return {'mae_8bit': float(np.abs(x - y).mean()), 'rmse_8bit': float(np.sqrt(np.mean((x - y) ** 2))),
            'pearson_correlation': correlation, 'max_abs_error_8bit': float(np.max(np.abs(x - y))),
            'value_count': int(len(x))}


def decode_sample(movie, frame):
    import numpy as np
    # Seek to an integer second, then select an integer frame offset. This
    # avoids floating-point -ss rounding into the following 30 Hz frame.
    seconds, offset = divmod(frame, 30)
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-hwaccel', 'none', '-threads', '1',
        '-ss', str(seconds), '-i', str(movie), '-map', '0:v:0', '-an',
        '-vf', f"select='eq(n,{offset})'", '-fps_mode', 'passthrough', '-frames:v', '1',
        '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', '-']
    completed = subprocess.run(command, check=True, capture_output=True, timeout=120)
    require(len(completed.stdout) == WIDTH * HEIGHT * 3, 'Missing/truncated decoded sample')
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3), command


def sample_pixels(movie, smoke_path, smoke):
    import numpy as np
    from PIL import Image
    results = []
    for frame, tick in zip(FRAMES, TICKS):
        filename = smoke_path.parent / f'{tick:06d}_actual.png'
        require(str(filename) in smoke['files'], 'Expected actual smoke PNG is not pinned')
        check_hash(filename, smoke['files'][str(filename)])
        reference = np.asarray(Image.open(filename).convert('RGB'))
        require(reference.shape == (HEIGHT, WIDTH, 3), 'Unexpected smoke image dimensions')
        decoded, command = decode_sample(movie, frame)
        regions = {'full_image': (slice(None), slice(None)),
                   'body_inset': (slice(480, 1058), slice(20, 414)),
                   'stage': (slice(112, 1080), slice(414, 1920))}
        results.append({'video_frame': frame, 'source_tick_40hz': tick,
            'reference_png': str(filename), 'reference_sha256': smoke['files'][str(filename)],
            'decode_command': command, 'regions': {name: pixel_metrics(decoded[region], reference[region])
                                                  for name, region in regions.items()}})
    return {'samples': results, 'sample_frames': list(FRAMES),
        'regions_xyxy': {'full_image': [0, 0, 1920, 1080], 'body_inset': [20, 480, 414, 1058],
                        'stage': [414, 112, 1920, 1080]},
        'acceptance_threshold_applied': False,
        'scope': 'Encoding similarity only. Artistic quality and encoding acceptance require primary review.'}


def verify_pair(root, plan, algorithm, track, source_check, presentation_pins, statuses, pcm, sample):
    key = algorithm['id'] + '/' + track['id']
    timeline = root / 'timelines' / algorithm['id'] / track['id']
    media = root / 'media' / algorithm['id'] / track['id']
    report = read(timeline / 'report.json')
    pins = {'timeline_sha256': digest(timeline / 'timeline.npz'), 'patch_sha256': digest(timeline / 'patch.json'),
        'numeric_report_sha256': digest(timeline / 'report.json'), 'audio_sha256': digest(track['audio']),
        'plan_sha256': source_check['plan_sha256'], **presentation_pins}
    require(report['timeline_sha256'] == pins['timeline_sha256'] and report['patch_sha256'] == pins['patch_sha256'],
            'Numeric report/DMX pin disagreement: ' + key)
    expected_status = 'diagnostic' if statuses.get(key, {}).get('diagnostic') else report['status']
    require(expected_status in ('diagnostic', 'engineering_valid'), 'Invalid numeric/presentation status')
    if algorithm['id'] == '03':
        require(expected_status == 'diagnostic', 'Algorithm 03 must remain a diagnostic')
    smoke_path = root / 'smoke' / algorithm['id'] / track['id'] / 'dark/SMOKE.json'
    smoke = read(smoke_path)
    require(smoke['inputs'] == pins and smoke['mode'] == 'dark', 'Smoke input binding disagreement: ' + key)
    require(smoke['finite_emitted_controls'] is True, 'Smoke finite-values check missing')
    require([s['tick'] for s in smoke['samples']] == list(TICKS), 'Smoke sample timing differs')
    require(all(s['static_pixels_exact'] is True for s in smoke['samples']), 'Frozen-control pixel check failed')
    for filename, expected in smoke['files'].items():
        require(Path(filename).resolve().is_relative_to(smoke_path.parent), 'Smoke image outside pair directory')
        check_hash(filename, expected)
    require(len(smoke['files']) == 8, 'Expected four actual and four fixed-brightness smoke PNGs')
    movie = media / 'dark.mp4'
    render_path, audit_path, verified_path = (media / ('dark_' + suffix + '.json') for suffix in ('render', 'audit', 'verified'))
    render, audit, verified = read(render_path), read(audit_path), read(verified_path)
    video_sha = check_hash(movie, render['sha256'])
    require(Path(render['path']).resolve() == movie and Path(audit['path']).resolve() == movie,
            'Media receipts point to another movie: ' + key)
    require(render['bytes'] == movie.stat().st_size <= 50_000_000, 'Video byte count/size cap mismatch')
    require(render['inputs'] == pins == verified['inputs'], 'Render/verified input pins disagree: ' + key)
    require(render['mode'] == 'dark' and render['codec'] == 'av1_nvenc', 'Wrong renderer/codec route')
    require(render['frames'] == 3600 and render['duration_s'] == 120 and render['start_s'] == 60,
            'Unexpected rendered source crop')
    require(render['numeric_status'] == expected_status, 'Video diagnostic caption status mismatch')
    require(render['renderer_config'] == smoke['renderer_config'], 'Smoke/full renderer configuration drift')
    require(render['gpu_release'] == smoke['gpu_release'], 'GPU release binding drift')
    release = render['gpu_release']
    check_hash(release['path'], release['sha256'])
    require(verified['video_sha256'] == video_sha and verified['audit_sha256'] == digest(audit_path),
            'Media verification receipt/hash disagreement')
    require(verified['source_reference_sha256'] == pcm['reference_sha256']
            and verified['audio_matches_source_crop'] is True, 'Source crop verification missing')
    validate_audit(audit, pcm)
    result = {'algorithm_id': algorithm['id'], 'track_id': track['id'], 'path': str(movie),
        'sha256': video_sha, 'bytes': movie.stat().st_size, 'numeric_status': expected_status,
        'render_receipt_sha256': digest(render_path), 'audit_receipt_sha256': digest(audit_path),
        'verified_receipt_sha256': digest(verified_path), 'smoke_receipt_sha256': digest(smoke_path),
        'decoded_frames_from_receipt': 3600, 'uniform_pts_from_receipt': True,
        'full_software_decode_from_receipt': True, 'audio_matches_reference_pcm_from_receipts': True}
    if sample:
        result['pixel_samples'] = sample_pixels(movie, smoke_path, smoke)
    return result


def verify_manifest(root, plan, completed, plan_hash, presentation_hash):
    path = root / 'review/MANIFEST.json'
    manifest = read(path)
    require(manifest['plan_sha256'] == plan_hash and manifest['mode'] == 'dark', 'Wrong review PLAN/render mode')
    require(manifest['tracks'] == plan['tracks'] and manifest['algorithms'] == plan['algorithms'], 'Review campaign definitions differ')
    if presentation_hash is not None:
        require(manifest.get('presentation_binding_sha256') == presentation_hash, 'Review presentation binding missing')
    variants = manifest['variants']
    require(len(variants) == 32 and {r['id'] for r in variants} == {f'{i:02d}' for i in range(1, 33)},
            'Review must contain exactly 30 candidates plus two baselines')
    require(all(r['status'] == 'complete' for r in variants), 'Review still contains incomplete entries')
    by_id = {row['id']: row for row in variants}
    for index, expected in enumerate(completed, 1):
        row = by_id[f'{index:02d}']
        for key in ('algorithm_id', 'track_id', 'path', 'sha256', 'bytes', 'numeric_status'):
            require(row.get(key) == expected[key], 'Review media field mismatch: ' + row['id'] + '/' + key)
        require(row['mode'] == 'dark', 'Candidate review mode mismatch')
        require(row['download_name'] == expected['algorithm_id'] + '_' + expected['track_id'] + '_dark.mp4',
                'Candidate download filename mismatch')
        if row['algorithm_id'] == '03':
            require(row['numeric_status'] == 'diagnostic' and row.get('diagnostic_note'), 'Algorithm 03 diagnostic disclosure missing')
    for index, track in enumerate(plan['tracks'], 31):
        row = by_id[f'{index:02d}']
        expected = Path('/root/lightsem/runs/body_transfer_dark_stage_20260930_v1') / track['id'] / (track['id'] + '_body_P2_dark_stage_1080p_av1.mp4')
        require(row['algorithm_id'] == 'baseline' and row['track_id'] == track['id'] and row['mode'] == 'dark', 'Wrong baseline pairing')
        require(Path(row['path']).resolve() == expected and row['download_name'] == expected.name, 'Baseline source/download mismatch')
        check_hash(expected, row['sha256'])
        require(expected.stat().st_size == row['bytes'], 'Baseline file size mismatch')
        historical = read('/root/lightsem/experiments/2026-09-30-body-transfer/records/DARK_STAGE_DELIVERY_PRIMARY_V1.json')
        historical_rows = [item for item in historical['media'] if item['id'] == track['id']]
        require(len(historical_rows) == 1 and historical_rows[0]['sha256'] == row['sha256'],
                'Baseline movie differs from its historical independent receipt')
    return manifest, {'path': str(path), 'sha256': digest(path), 'complete_candidates': 30, 'complete_baselines': 2,
                       'algorithm_03_disclosed_as_diagnostic': True}


def check_publisher(root, plan_hash, manifest_sha):
    candidates = sorted((root / 'records').glob('REVIEW_PRESENTATION_DARK_V*.json'))
    if not candidates:
        return {'publisher_binding_present': False, 'deployed_html_sha256': digest(root / 'review/index.html')}
    path = candidates[-1]
    receipt = read(path)
    require(receipt['mode'] == 'dark' and receipt['plan_sha256'] == plan_hash
            and receipt['manifest_sha256'] == manifest_sha, 'Publisher binding campaign/manifest mismatch')
    check_hash(receipt['source_template'], receipt['source_template_sha256'])
    check_hash(root / 'review/index.html', receipt['deployed_html_sha256'])
    require(receipt['source_template_sha256'] == receipt['deployed_html_sha256'], 'Published HTML differs from bound template')
    publisher = Path(receipt.get('publisher_source', Path(__file__).with_name('publish_body_portfolio_review.py')))
    check_hash(publisher, receipt['publisher_sha256'])
    return {'publisher_binding_present': True, 'receipt': str(path), 'receipt_sha256': digest(path),
            'deployed_html_sha256': receipt['deployed_html_sha256']}


def verify_http(base_url, root, manifest):
    base_url = base_url.rstrip('/')
    results = []
    for route, local in (('/manifest.json', root / 'review/MANIFEST.json'), ('/', root / 'review/index.html')):
        with urlopen(base_url + route, timeout=15) as response:
            require(response.status == 200, 'HTTP document status mismatch')
            payload = response.read()
        require(hashlib.sha256(payload).hexdigest() == digest(local), 'HTTP document differs from local artifact')
    for row in manifest['variants']:
        route = base_url + '/media/' + row['id'] + '.mp4?' + urlencode({'v': row['sha256'], 'download': '1'})
        request = Request(route, headers={'Range': 'bytes=0-63'})
        with urlopen(request, timeout=15) as response:
            require(response.status == 206, 'HTTP Range did not return 206: ' + row['id'])
            require(response.headers.get('Content-Range') == f"bytes 0-63/{row['bytes']}", 'HTTP Content-Range mismatch')
            require(response.headers.get('Content-Length') == '64', 'HTTP Range byte count mismatch')
            disposition = Message()
            disposition['Content-Disposition'] = response.headers.get('Content-Disposition', '')
            require(disposition.get_filename() == row['download_name'], 'HTTP download filename mismatch: ' + row['id'])
            payload = response.read()
        with Path(row['path']).open('rb') as stream:
            require(payload == stream.read(64), 'HTTP video prefix differs from authenticated movie')
        results.append({'id': row['id'], 'range_206': True, 'first_64_bytes_match': True,
                        'download_name': row['download_name']})
    return {'base_url': base_url, 'document_bytes_match': True, 'media': results,
            'browser_playback_test_performed': False, 'scope': 'HTTP bytes/Range/download headers only'}


def storage_accounting(roots):
    records = []
    for root in roots:
        require(root.is_dir(), 'Budget root missing: ' + str(root))
        files = [p for p in root.rglob('*') if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        free = shutil.disk_usage(root).free
        require(free >= MIN_FREE_BYTES, 'Less than 20 GiB free at ' + str(root))
        records.append({'root': str(root), 'file_count': len(files), 'file_bytes': total, 'free_bytes': free})
    total = sum(row['file_bytes'] for row in records)
    require(total <= MAX_COMBINED_BYTES, 'Combined V2+V3 size exceeds 2 GiB')
    return {'roots': records, 'combined_file_bytes': total, 'maximum_combined_bytes': MAX_COMBINED_BYTES,
        'minimum_free_bytes': MIN_FREE_BYTES, 'scope': 'Sum of actual logical file sizes at verification time, before writing this new receipt'}


def summarize_pixels(completed):
    summary = {}
    for region in ('full_image', 'body_inset', 'stage'):
        samples = [{'algorithm_id': pair['algorithm_id'], 'track_id': pair['track_id'],
                    'video_frame': sample['video_frame'], **sample['regions'][region]}
                   for pair in completed for sample in pair['pixel_samples']['samples']]
        correlations = [sample for sample in samples if sample['pearson_correlation'] is not None]
        summary[region] = {'largest_mae_sample': max(samples, key=lambda sample: sample['mae_8bit']),
            'lowest_correlation_sample': min(correlations, key=lambda sample: sample['pearson_correlation']) if correlations else None}
    return {'regions': summary, 'acceptance_threshold_applied': False,
            'next_review': 'Primary reviewer inspects largest-error candidates; no artistic ranking follows from codec errors.'}


def verify(root, base_url=None, sample=False):
    root = root.resolve()
    require(root.name == 'body_transfer_20261004_v3', 'This verifier checks the completed V3 delivery')
    roots = [root.with_name('body_transfer_20261004_v2'), root]
    plans, checks = {}, {}
    for campaign in roots:
        plans[campaign.name], checks[campaign.name] = check_sources(campaign)
    plan, source_check = plans[root.name], checks[root.name]
    require([a['id'] for a in plan['algorithms']] == [f'{i:02d}' for i in range(1, 16)], 'Expected algorithms 01..15 in fixed order')
    require([t['id'] for t in plan['tracks']] == ['shpongle', 'cure_for_me'], 'Expected the two fixed song identities/order')
    require(plan['review_start_s'] == 60 and plan['review_duration_s'] == 120 and plan['prefix_s'] == 180, 'Source window drift')
    extra_pins, binding_hash, statuses = presentation_binding(root, source_check['plan_sha256'], source_check['source_manifest_sha256'])
    references = {t['id']: reference_pcm(t, plan) for t in plan['tracks']}
    completed = []
    for algorithm in plan['algorithms']:
        for track in plan['tracks']:
            completed.append(verify_pair(root, plan, algorithm, track, source_check, extra_pins, statuses,
                                         references[track['id']], sample))
            print('VERIFIED', algorithm['id'], track['id'], file=sys.stderr, flush=True)
    expected_movies = {Path(row['path']) for row in completed}
    require(set(root.glob('media/*/*/dark.mp4')) == expected_movies, 'Unexpected/missing candidate dark movies')
    manifest, manifest_check = verify_manifest(root, plan, completed, source_check['plan_sha256'], binding_hash)
    publisher = check_publisher(root, source_check['plan_sha256'], manifest_check['sha256'])
    http = verify_http(base_url, root, manifest) if base_url else {'performed': False, 'browser_playback_test_performed': False}
    # Recheck frozen source/input hashes after all media operations too.
    for campaign in roots:
        _, after = check_sources(campaign)
        require(after == checks[campaign.name], 'Source pins changed during verification')
    return {'status': 'verified', 'at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
        'verifier_path': str(Path(__file__).resolve()), 'verifier_sha256': digest(__file__),
        'source_checks': checks, 'audio_reference_receipts': references, 'media': completed,
        'manifest': manifest_check, 'publisher': publisher, 'http': http, 'storage': storage_accounting(roots),
        'pixel_summary': summarize_pixels(completed) if sample else None,
        'sample_pixels_performed': sample, 'artistic_acceptance_claimed': False,
        'browser_playback_test_performed': False, 'full_decode_repeated': False,
        'scope': 'Independent delivery/provenance verification. Pixel differences are descriptive; no artistic thresholds.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base-url', help='Optional existing local review server')
    parser.add_argument('--sample-pixels', action='store_true')
    parser.add_argument('--output', type=Path, help='Write a new immutable JSON receipt; fail if it already exists')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new verification receipt path')
    try:
        result = verify(args.root, args.base_url, args.sample_pixels)
    except (VerificationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        result = {'status': 'failed', 'root': str(args.root.resolve()), 'at_utc': datetime.now(timezone.utc).isoformat(),
                  'error': str(error), 'error_type': type(error).__name__, 'browser_playback_test_performed': False,
                  'verifier_sha256': digest(__file__)}
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(text + '\n')
        print(str(args.output))
    else:
        print(text)
    if result['status'] != 'verified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
