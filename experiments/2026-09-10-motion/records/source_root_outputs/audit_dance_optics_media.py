"""Create-only full software media audit for the registered 13+2 videos.
No GPU, renderer imports, vault, raw RGB cache, or perceptual acceptance.
Use: CUDA_VISIBLE_DEVICES="" bash run.sh outputs/audit_dance_optics_media.py --limit 1
"""
from __future__ import annotations
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

DEFAULT = Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01')
PAIR = ('closer_native_legacy', 'closer_zero_angular_flux')
SCOPE = ('Full software decoding and exact media timing only. Generated frame count is not encoded '
         'or decoded count. No decoded clipping measurement, visual, motion, or music acceptance.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    path.chmod(0o444)
    side = Path(str(path) + '.sha256')
    with side.open('x') as f:
        f.write(digest(path) + '\n'); f.flush(); os.fsync(f.fileno())
    side.chmod(0o444)


def authenticated(path):
    path = Path(path)
    if digest(path) != Path(str(path) + '.sha256').read_text().strip():
        raise ValueError('Receipt hash mismatch: ' + str(path))
    return json.loads(path.read_text())


def safe_child(root, name):
    child = (root / name).resolve()
    if not child.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes root: ' + name)
    return child


def verify_artifacts(folder, hashes):
    for name, sha in hashes.items():
        if digest(safe_child(folder, name)) != sha:
            raise ValueError('Artifact hash mismatch: ' + str(folder / name))


def validate_frozen(out):
    freeze = authenticated(out / 'freeze.json')
    verify_artifacts(out / 'frozen', freeze['files'])
    for name in ('design.json', 'registration.json'):
        if digest(out / name) != freeze['files'][name]:
            raise ValueError('Current registration changed: ' + name)
    if digest(freeze['source_path']) != freeze['files']['candidate.npz']:
        raise ValueError('Original candidate changed')
    if digest(freeze['audio_path']) != freeze['audio_sha256']:
        raise ValueError('Source audio changed')
    design = json.loads((out / 'frozen/design.json').read_text())
    registration = json.loads((out / 'frozen/registration.json').read_text())
    if json.loads(registration['hypothesis']['plan']) != design or json.loads(registration['experiment']['config_note']) != design:
        raise ValueError('Design/registration mismatch')
    expected = {f'{c}_{z}_{e}' for c in ('legacy', 'closer') for z in ('native', 'zero')
                for e in ('legacy', 'angular_peak', 'angular_flux')}
    expected.add('closer_zero_static_bright_foil')
    if set(design['arms']) != expected or design['intervals']['full'] != [0., 277.3]:
        raise ValueError('Not the registered 13-arm, 277.3-second design')
    for arm in design['arms'].values():
        r = arm['renderer']
        if (r['width'], r['height'], r['fps']) != (640, 360, 30.):
            raise ValueError('Unexpected registered video geometry')
    return freeze, design


def members(design):
    return [('renders', arm) for arm in design['arms']] + [('matched', arm) for arm in PAIR]


def complete(path):
    return path.is_file() and Path(str(path) + '.sha256').is_file()


def render_provenance(out, group, arm, freeze, design):
    folder = out / group / arm
    r = authenticated(folder / 'render.json')
    verify_artifacts(folder, r['artifacts'])
    if 'video.mp4' not in r['artifacts'] or 'started.json' not in r['artifacts']:
        raise ValueError('Render receipt omits video/start binding')
    start = authenticated(folder / 'started.json')
    if start['freeze_sha256'] != digest(out / 'freeze.json'):
        raise ValueError('Render not bound to registered freeze')
    if r['arm'] != arm or start['arm'] != arm or start['matched'] != (group == 'matched'):
        raise ValueError('Wrong render identity')
    if r['renderer_registered'] != design['arms'][arm]['renderer']:
        raise ValueError('Render settings differ from registration')
    if r['duration_s'] != 277.3 or r['generated_frames'] != 8319:
        raise ValueError('Unexpected registered generated count/duration')
    if group == 'matched':
        match = r['brightness_match']
        if match['pair'] != list(PAIR) or match['raw_rerender_byte_identical'] is not True:
            raise ValueError('Unregistered or failed matched rerender')
        if set(match['original_receipts']) != set(design['arms']):
            raise ValueError('Matched receipt does not bind all originals')
        for name, sha in match['original_receipts'].items():
            original_path = out / 'renders' / name / 'render.json'
            authenticated(original_path)
            if digest(original_path) != sha:
                raise ValueError('Matched original receipt changed')
    return r, {'freeze_sha256': digest(out / 'freeze.json'),
               'render_receipt_sha256': digest(folder / 'render.json'),
               'source_audio_sha256': freeze['audio_sha256'],
               'video_sha256': r['artifacts']['video.mp4'],
               'verified_render_artifacts': r['artifacts']}


def q(value):
    value = Fraction(value)
    return {'rational_seconds': str(value), 'seconds': float(value)}


def stream_end(stream):
    tb = Fraction(stream['time_base'])
    start = Fraction(int(stream.get('start_pts', 0))) * tb
    duration = Fraction(int(stream['duration_ts'])) * tb if 'duration_ts' in stream else None
    return {'time_base': str(tb), 'start': q(start),
            'duration': q(duration) if duration is not None else None,
            'end': q(start + duration) if duration is not None else None,
            'duration_source': 'integer duration_ts * time_base, NOT rounded duration string'}


def timeline(frames, stream, kind):
    tb = Fraction(stream['time_base'])
    rate = int(stream['sample_rate']) if kind == 'audio' else None
    gaps, bad_duration, missing = [], [], []
    pts, ends, samples = [], [], 0
    for i, f in enumerate(frames):
        if 'pts' not in f:
            missing.append(i)
            continue
        t = int(f['pts']) * tb
        if kind == 'audio':
            n = int(f['nb_samples']); duration = Fraction(n, rate); samples += n
        else:
            ticks = f.get('duration', f.get('pkt_duration'))
            duration = int(ticks) * tb if ticks is not None else None
            if duration != Fraction(1, 30):
                bad_duration.append({'frame': i, 'duration': q(duration) if duration is not None else None})
        if pts:
            expected = ends[-1] if kind == 'audio' else pts[-1] + Fraction(1, 30)
            if expected is not None and t != expected:
                gaps.append({'frame': i, 'previous_pts': q(pts[-1]), 'expected_pts': q(expected),
                             'actual_pts': q(t), 'delta': q(t - expected)})
        pts.append(t); ends.append(t + duration if duration is not None else None)
    return {'decoded_frame_count': len(frames), 'pts_count': len(pts), 'missing_pts_frames': missing,
            'first_pts': q(pts[0]) if pts else None, 'last_pts': q(pts[-1]) if pts else None,
            'decoded_end': q(ends[-1]) if ends and ends[-1] is not None else None,
            'end_rule': 'last PTS + nb_samples/sample_rate' if kind == 'audio' else 'last PTS + frame duration',
            'gap_or_overlap_count': len(gaps), 'gap_or_overlap_details': gaps,
            'non_30fps_frame_duration_count': len(bad_duration), 'bad_duration_details': bad_duration,
            'sample_count_per_channel': samples if kind == 'audio' else None,
            'regular_30fps' if kind == 'video' else 'pts_nb_samples_contiguous':
                bool(frames) and not missing and not gaps and not bad_duration}


def parse_framemd5(path):
    count = 0; pts = []; durations = []; tb = None
    for line in Path(path).read_text().splitlines():
        if line.startswith('#tb 0:'): tb = Fraction(line.split(':', 1)[1].strip())
        if not line or line.startswith('#'): continue
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 6 or int(fields[0]) != 0:
            raise ValueError('Unexpected framemd5 row')
        count += 1; pts.append(int(fields[2])); durations.append(int(fields[3]))
    if tb is None: raise ValueError('Missing framemd5 time base')
    return {'actual_decoded_frames': count, 'time_base': str(tb),
            'first_pts': q(pts[0] * tb) if pts else None,
            'last_pts': q(pts[-1] * tb) if pts else None,
            'decoded_end': q((pts[-1] + durations[-1]) * tb) if pts else None,
            'regular_30fps': bool(pts) and all((b-a)*tb == Fraction(1,30) for a,b in zip(pts,pts[1:]))
                              and all(d*tb == Fraction(1,30) for d in durations)}


def file_command(cmd, target, stem):
    # No shells, hardware decoding, or unbounded in-memory subprocess output.
    with (target / (stem + '.stdout')).open('xb') as stdout, (target / (stem + '.stderr')).open('xb') as stderr:
        proc = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
    return {'command': cmd, 'returncode': proc.returncode,
            'stdout': stem + '.stdout', 'stderr': stem + '.stderr',
            'stderr_nonempty': (target / (stem + '.stderr')).stat().st_size != 0}


def pcm_command(cmd, target):
    h = hashlib.sha256(); size = 0
    with (target / 'audio_pcm.stderr').open('xb') as stderr:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr)
        try:
            for block in iter(lambda: proc.stdout.read(1024 * 1024), b''):
                h.update(block); size += len(block)
            rc = proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            raise
        finally: proc.stdout.close()
    return {'command': cmd, 'returncode': rc, 'stderr': 'audio_pcm.stderr',
            'stderr_nonempty': (target / 'audio_pcm.stderr').stat().st_size != 0,
            'sha256': h.hexdigest(), 'bytes': size,
            'representation': 'interleaved pcm_f32le, native sample rate/channels; no resampling, normalization or trimming; decoded padding retained'}


def audit_one(out, group, arm, freeze, design):
    target = out / 'media_audit' / (group + '__' + arm)
    target.mkdir(parents=True, exist_ok=False)
    result = {'schema': 1, 'group': group, 'arm': arm, 'scope': SCOPE,
              'started_unix': time.time(), 'audit_script_sha256': digest(__file__),
              'status': 'failed', 'commands': {}, 'perceptual_acceptance': None,
              'decoded_clipping_measured': False}
    # All exceptions retain a receipt and partial stdout/stderr, never retry in place.
    try:
        render, provenance = render_provenance(out, group, arm, freeze, design)
        result['provenance'] = provenance
        write_json(target / 'started.json', dict(result, status='started'))
        video = out / group / arm / 'video.mp4'
        result['commands']['ffmpeg_version'] = file_command(['ffmpeg', '-version'], target, 'ffmpeg_version')
        result['commands']['ffprobe_version'] = file_command(['ffprobe', '-version'], target, 'ffprobe_version')
        # codec whitelist excludes all hardware decoders. libdav1d is the native AV1 default.
        base = ['ffprobe', '-v', 'error', '-threads', '2', '-codec_whitelist', 'av1,libdav1d,aac']
        cmd = base + ['-show_error', '-show_streams', '-show_format', '-show_frames', '-count_frames', '-of', 'json', str(video)]
        record = file_command(cmd, target, 'probe_frames_streams')
        result['commands']['probe_frames_streams'] = record
        if record['returncode']:
            raise RuntimeError('ffprobe frame decode failed; retained stderr')
        data = json.loads((target / record['stdout']).read_text())
        streams = data['streams']
        vs = [s for s in streams if s['codec_type'] == 'video']
        aus = [s for s in streams if s['codec_type'] == 'audio']
        if len(streams) != 2 or len(vs) != 1 or len(aus) != 1:
            raise ValueError('Expected exactly one video and one audio stream')
        v, a = vs[0], aus[0]
        if v['codec_name'] != 'av1' or a['codec_name'] != 'aac':
            raise ValueError('Expected AV1/AAC')
        result['streams'] = {'video': v, 'audio': a, 'format': data['format']}
        result['video_timeline'] = timeline([f for f in data['frames'] if f['media_type'] == 'video'], v, 'video')
        result['audio_timeline'] = timeline([f for f in data['frames'] if f['media_type'] == 'audio'], a, 'audio')
        result['stream_timing'] = {'video': stream_end(v), 'audio': stream_end(a)}
        cmd = base + ['-show_error', '-show_packets', '-of', 'json', str(video)]
        result['commands']['probe_packets'] = file_command(cmd, target, 'probe_packets')
        # Independent complete video decode; per-frame MD5 holds no RGB cache.
        cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-hwaccel', 'none', '-threads', '2',
               '-c:v', 'libdav1d', '-i', str(video), '-map', '0:v:0', '-an',
               '-fps_mode', 'passthrough', '-c:v', 'rawvideo', '-threads:v', '2',
               '-f', 'framemd5', str(target / 'video.framemd5')]
        result['commands']['video_decode'] = file_command(cmd, target, 'video_decode')
        if result['commands']['video_decode']['returncode']:
            raise RuntimeError('Full ffmpeg video decode failed')
        result['video_decode'] = parse_framemd5(target / 'video.framemd5')
        cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-hwaccel', 'none', '-threads', '2',
               '-c:a', 'aac', '-i', str(video), '-map', '0:a:0', '-vn',
               '-c:a', 'pcm_f32le', '-threads:a', '2', '-f', 'f32le', 'pipe:1']
        result['pcm'] = pcm_command(cmd, target)
        result['commands']['audio_decode'] = {k:result['pcm'][k] for k in ('command','returncode','stderr','stderr_nonempty')}
        vt, at = result['video_timeline'], result['audio_timeline']
        expected_end = q(Fraction(2773, 10))
        count = result['video_decode']['actual_decoded_frames']
        result['frame_counts'] = {'generated_by_renderer': render['generated_frames'],
            'container_declared_nb_frames': int(v['nb_frames']) if 'nb_frames' in v else None,
            'ffprobe_decoded': vt['decoded_frame_count'], 'ffprobe_nb_read_frames': int(v['nb_read_frames']),
            'ffmpeg_actual_decoded': count,
            'scope': 'Actual decoded presentation frames, not a count of AV1 coded subframes or packet count'}
        result['timing_differences'] = {}
        for key, end in [('video_stream', result['stream_timing']['video']['end']),
                         ('audio_stream', result['stream_timing']['audio']['end']),
                         ('video_decoded', vt['decoded_end']), ('audio_decoded', at['decoded_end'])]:
            result['timing_differences'][key + '_minus_registered_277_3_s'] = q(Fraction(end['rational_seconds']) - Fraction(2773, 10)) if end else None
        result['checks'] = {
            'all_commands_exit_zero': all(c['returncode'] == 0 for c in result['commands'].values()),
            'no_error_level_decoder_messages': all(not c['stderr_nonempty'] for c in result['commands'].values()),
            'registered_640x360': (v['width'], v['height']) == (640,360),
            'declared_rates_30fps': Fraction(v['avg_frame_rate']) == Fraction(v['r_frame_rate']) == 30,
            'both_decoders_count_8319': vt['decoded_frame_count'] == count == 8319,
            'container_and_probe_counts_agree': result['frame_counts']['container_declared_nb_frames'] == int(v['nb_read_frames']) == count,
            'both_decoders_regular_30fps': vt['regular_30fps'] and result['video_decode']['regular_30fps'],
            'video_first_pts_zero': vt['first_pts'] == q(0),
            'video_decoded_end_exact_277_3_s': vt['decoded_end'] == expected_end,
            'video_stream_end_exact_277_3_s': result['stream_timing']['video']['end'] == expected_end,
            'audio_first_pts_zero': at['first_pts'] == q(0),
            'audio_pts_nb_samples_contiguous': at['pts_nb_samples_contiguous'],
            'audio_stream_end_exact_277_3_s': result['stream_timing']['audio']['end'] == expected_end,
            'pcm_bytes_match_probe_samples': result['pcm']['bytes'] == at['sample_count_per_channel'] * int(a['channels']) * 4,
        }
        result['pcm'].update(sample_rate=int(a['sample_rate']), channels=int(a['channels']),
                             channel_layout=a.get('channel_layout'),
                             sample_count_per_channel=result['pcm']['bytes'] // (4 * int(a['channels'])))
        # Rehash every bound render artifact after decode, not just a stat check.
        _, after = render_provenance(out, group, arm, freeze, design)
        if after != provenance: raise RuntimeError('Provenance changed during decode')
        result['checks']['source_artifacts_unchanged_after_decode'] = True
        result['status'] = 'complete'
        result['all_exact_media_checks_pass'] = all(result['checks'].values())
        result['audio_decoded_tail_note'] = ('AAC priming/edit lists and final codec frame padding may make decoded PCM end differ from stream end. '
            'Exact PTS, nb_samples, packet side data and both ends are retained; no silence/trim/acceptance inference.')
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['completed_unix'] = time.time()
    result['artifacts'] = {x.name: digest(x) for x in sorted(target.iterdir()) if x.is_file()}
    for path in target.iterdir():
        if path.is_file(): path.chmod(0o444)
    write_json(target / 'audit.json', result)
    print(json.dumps({'member': group + '__' + arm, 'status': result['status'],
                      'all_exact_media_checks_pass': result.get('all_exact_media_checks_pass'),
                      'error': result.get('error')}), flush=True)
    return result


def coverage(out, freeze, design):
    rows = []; audited = []
    for group, arm in members(design):
        source = out / group / arm / 'render.json'
        target = out / 'media_audit' / (group + '__' + arm)
        row = {'group': group, 'arm': arm}
        try:
            if complete(target / 'audit.json'):
                audit = authenticated(target / 'audit.json')
                verify_artifacts(target, audit['artifacts'])
                if audit['status'] == 'complete':
                    _, prov = render_provenance(out, group, arm, freeze, design)
                    if prov != audit['provenance']: raise ValueError('Audited source provenance changed')
                    audited.append(audit)
                    row.update(status='audited_complete', exact_checks_pass=audit['all_exact_media_checks_pass'],
                               audit_receipt_sha256=digest(target / 'audit.json'))
                else: row.update(status='audit_failed', error=audit.get('error'))
            elif target.exists(): row['status'] = 'audit_partial_blocked_no_overwrite'
            elif complete(source): row['status'] = 'render_receipt_present_audit_pending'
            else: row['status'] = 'render_not_completed_skipped'
        except Exception as exc: row.update(status='integrity_failed', error=str(exc))
        rows.append(row)
    all15 = len(audited) == 15
    decoded_all = all15 and all(a['checks']['all_commands_exit_zero'] for a in audited)
    pcm_comparison = None
    if decoded_all:
        signatures = {(a['pcm']['sha256'], a['pcm']['bytes'], a['pcm']['sample_rate'],
                       a['pcm']['channels'], a['pcm']['channel_layout']) for a in audited}
        pcm_comparison = {'all_15_same_decoded_pcm': len(signatures) == 1,
            'unique_signature_count': len(signatures),
            'members': {a['group'] + '__' + a['arm']: a['pcm'] for a in audited},
            'expectation': 'All 15 mux the same source audio; decoded equality was tested, not assumed.',
            'scope': 'Equality of canonical decoded AAC PCM among 15 outputs, not identity with lossy source FLAC.'}
    return {'schema':1, 'created_unix':time.time(), 'scope':SCOPE, 'expected_members':15,
            'completed_audits':len(audited), 'all_15_audited':all15,
            'all_15_full_decode_commands_succeeded':decoded_all,
            'all_15_exact_media_checks_pass': all(a['all_exact_media_checks_pass'] for a in audited) if all15 else None,
            'audio_comparison':pcm_comparison, 'members':rows,
            'perceptual_acceptance':None, 'decoded_clipping_measured':False,
            'audit_script_sha256':digest(__file__), 'freeze_sha256':digest(out / 'freeze.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--limit', type=int, default=1, help='Maximum next missing completed members to audit; 0 = coverage only')
    args = parser.parse_args()
    if args.limit < 0: parser.error('--limit must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    out = args.out.resolve(); freeze, design = validate_frozen(out)
    run = 0
    for group, arm in members(design):
        target = out / 'media_audit' / (group + '__' + arm)
        if target.exists() or not complete(out / group / arm / 'render.json'): continue
        if run >= args.limit: break
        audit_one(out, group, arm, freeze, design); run += 1
    report = coverage(out, freeze, design)
    reports = out / 'media_audit' / 'coverage'; reports.mkdir(parents=True, exist_ok=True)
    path = reports / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '_' + uuid.uuid4().hex + '.json')
    write_json(path, report)
    print(json.dumps({'coverage_receipt':str(path), 'completed_audits':report['completed_audits'],
                      'all_15_audited':report['all_15_audited'],
                      'all_15_exact_media_checks_pass':report['all_15_exact_media_checks_pass']}), flush=True)
    return 1 if any(row['status'] in ('integrity_failed','audit_failed','audit_partial_blocked_no_overwrite') for row in report['members']) else 0


if __name__ == '__main__':
    sys.exit(main())
