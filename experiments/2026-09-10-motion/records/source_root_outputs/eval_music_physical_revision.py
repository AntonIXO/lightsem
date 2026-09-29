"""Frozen physical-objective evaluation. No training, TEST, vault or selection.

Run with bash run_torch.sh outputs/eval_music_physical_revision.py --help.
`freeze` writes metadata only; all inference needs BOTH completed 30000-step runs.
Scheduling budgets consume the next missing items, never select windows or draws.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

EXPERIMENT = Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')
NATIVE = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
PROTOCOL_SHA256 = 'c5ea616d907510b92f06383abc7b4e7cb7d6862bc367b5769e838aebf0ed6405'
ARMS = ('masked_native', 'physical')
CHECKPOINTS = ('initial', 'final')
DRAW_COUNT, EULER_STEPS = 6, 32
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10000, 20260910


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            sha.update(block)
    return sha.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def json_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value, label='value'):
    # Tensor checks remain on the tensor's native device. No sample filtering.
    if hasattr(value, 'detach'):
        ok = bool(value.isfinite().all())
    else:
        ok = bool(np.isfinite(value).all())
    if not ok:
        raise FloatingPointError('Nonfinite '+label+'; no sample may be discarded')


def immutable_bytes(path, content):
    """Atomic create, never overwrite. Link commits a complete fsynced payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == content, 'Immutable artifact differs: '+str(path))
        return
    fd, name = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, 0o444)
        try:
            os.link(name, path)
        except FileExistsError:
            require(path.read_bytes() == content, 'Concurrent immutable write differs')
    finally:
        os.unlink(name)


def save_json(path, value):
    """Self-authenticating single-file envelope; avoids JSON/sidecar crash gap."""
    immutable_bytes(path, json_bytes({'payload': value, 'sha256': json_hash(value)})+b'\n')


def load_json(path):
    envelope = read_json(path)
    require(json_hash(envelope['payload']) == envelope['sha256'], 'JSON integrity: '+str(path))
    return envelope['payload']


def save_npz(path, arrays):
    import io
    for name, value in arrays.items():
        finite(value, name)
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            require(set(old.files) == set(arrays), 'Interrupted NPZ keys differ')
            for name, value in arrays.items():
                require(old[name].dtype == value.dtype and np.array_equal(old[name], value),
                        'Interrupted NPZ content differs: '+name)
        return
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    immutable_bytes(path, buffer.getvalue())


@contextmanager
def output_lock(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.evaluation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_training_freeze(experiment=EXPERIMENT):
    require(digest(experiment/'protocol.json') == PROTOCOL_SHA256, 'Wrong physical protocol')
    p = read_json(experiment/'protocol.json')
    s = read_json(experiment/'frozen_sources.json')
    require(json_hash(s['files']) == s['source_sha256'] == p['source_sha256'], 'Source manifest identity')
    require(Path(s['snapshot']).resolve() == (experiment/'source').resolve(), 'Wrong source location')
    for relative, sha in s['files'].items():
        path = experiment/'source'/relative
        require(path.resolve().is_relative_to((experiment/'source').resolve()), 'Source path escape')
        require(digest(path) == sha, 'Frozen source drift: '+relative)
    for name, sha in p['inputs'].items():
        require(digest(name) == sha, 'Frozen input drift: '+name)
    require(p['design']['arms'] == list(ARMS) and p['design']['training']['steps'] == 30000,
            'Unsupported design')
    require(not p['design']['test_access'] and not p['design']['evaluation']['test_access'], 'TEST forbidden')
    return p


def frozen_modules(experiment=EXPERIMENT):
    """Fail on mixed live/frozen package imports; never invoke old context/load_model."""
    source = (experiment/'source').resolve()
    for name, module in list(sys.modules.items()):
        if name == 'lightsem' or name.startswith('lightsem.'):
            location = getattr(module, '__file__', None)
            require(location and Path(location).resolve().is_relative_to(source), 'Live module preloaded: '+name)
    sys.path.insert(0, str(source))
    names = ['music_model', 'music_revision_model', 'music_data', 'music_evaluation', 'geometry']
    result = {n: importlib.import_module('lightsem.experiments.motion_study.'+n) for n in names}
    for n, module in result.items():
        require(Path(module.__file__).resolve().is_relative_to(source), 'Wrong imported source: '+n)
    return result


def short_windows(entries):
    """Metadata-only deterministic controls. A full recording cannot shift in itself."""
    entries = sorted((e for e in entries if e['partition'] == 'val'), key=lambda e: e['key'])
    result = []
    for index, entry in enumerate(entries):
        frames = entry['n_frames']
        if frames >= 960:
            continue
        token = int(hashlib.sha256(f'short:{entry["key"]}:0'.encode()).hexdigest()[:12], 16)
        donor = next((entries[(index+j) % len(entries)] for j in range(1, len(entries))
                      if entries[(index+j) % len(entries)]['canonical_id'] != entry['canonical_id']
                      and entries[(index+j) % len(entries)]['n_frames'] >= frames), None)
        result.append({'key': entry['key'], 'canonical_id': entry['canonical_id'], 'start': 0,
                       'frames': frames, 'donor_key': None if donor is None else donor['key'],
                       'donor_start': None if donor is None else token % (donor['n_frames']-frames+1),
                       'shift_start': None,
                       'unsupported': {'shift': 'Full aligned recording leaves zero contiguous shift support; no wrap/pad/repeat',
                                       **({'donor': 'No canonical-distinct VAL recording of sufficient length'} if donor is None else {})},
                       'draw_seeds': [int((token+7001+i*104729) % 2**31) for i in range(6)]})
    return result


def validate_windows(windows, entries, primary=False):
    by_key = {e['key']: e for e in entries if e['partition'] == 'val'}
    seen = set()
    for w in windows:
        require((w['key'], w['start']) not in seen, 'Duplicate window')
        seen.add((w['key'], w['start']))
        e = by_key[w['key']]
        require(e['canonical_id'] == w['canonical_id'], 'Canonical mismatch')
        require(w['frames'] >= 15 and 0 <= w['start'] <= e['n_frames']-w['frames'], 'Invalid target support')
        require(len(w['draw_seeds']) == 6 and len(set(w['draw_seeds'])) == 6, 'Six distinct fixed draws required')
        if w['donor_key'] is not None:
            d = by_key[w['donor_key']]
            require(d['canonical_id'] != e['canonical_id'], 'Same-song donor')
            require(0 <= w['donor_start'] <= d['n_frames']-w['frames'], 'Invalid donor support')
        if w['shift_start'] is not None:
            require(0 <= w['shift_start'] <= e['n_frames']-w['frames'] and
                    abs(w['shift_start']-w['start']) >= 150, 'Invalid shift support')
        if primary:
            require(w['frames'] == 960 and w['donor_key'] is not None and w['shift_start'] is not None,
                    'Primary needs both wrong-audio controls')


def freeze(output):
    p = verify_training_freeze()
    manifest = read_json(NATIVE/'data/audio_manifest.json')
    require(all(e['partition'] in ('train', 'val') for e in manifest['entries']), 'Unexpected partition')
    old = read_json(NATIVE/'music/protocol.json')
    # The historic review protocol carries the fixed clips. No TEST manifest is opened.
    review_path = NATIVE/'music/review/protocol.json'
    review = read_json(review_path)
    review_sidecar = read_json(str(review_path)+'.sha256.json')
    require(digest(review_path) == review_sidecar['sha256'], 'Review protocol authentication')
    require(review['clips'] and len(review['clips']) == 12, 'Expected fixed12 clips')
    source_manifest = '/mnt/lightsem-data/experiments/lightsem/motion_representation_20260910/data/manifest.json'
    require(review['inputs'][source_manifest] == manifest['source_manifest_sha256'], 'Review source identity')
    val = {e['key']: e for e in manifest['entries'] if e['partition'] == 'val'}
    for clip in review['clips']:
        require(clip['key'] in val and clip['canonical_id'] == val[clip['key']]['canonical_id'], 'Non-VAL review')
        require(0 <= clip['start'] <= val[clip['key']]['n_frames']-clip['frames'], 'Review clip support')
    main = old['evaluation']['windows']
    short = short_windows(manifest['entries'])
    require(len(main) == 119 and len(short) == 3, 'Unexpected frozen cohort sizes')
    validate_windows(main, manifest['entries'], primary=True)
    validate_windows(short, manifest['entries'])
    demo_path = NATIVE/'data/shpongle/input.json'
    demo = read_json(demo_path)
    require(digest(demo_path) == read_json(str(demo_path)+'.sha256.json')['sha256'], 'Shpongle metadata integrity')
    files = {str(review_path): digest(review_path), str(demo_path): digest(demo_path),
             demo['path']: demo['sha256'], demo['audio_path']: demo['source']['source_audio_sha256']}
    for e in val.values():
        for path_field, hash_field in [('path', 'sha256'), ('audio_path', 'audio_sha256'), ('wav_path', 'wav_sha256')]:
            files[e[path_field]] = e[hash_field]
    # Matched historical review arrays already exist. Freeze references, not new inference
    # or old model loading. Rendering must verify these hashes before use.
    review_assets = {}
    for label, path in [('incumbent', NATIVE/'music/review/incumbent/report.json'),
                        ('previous_native_flow', NATIVE/'music/full_sequences/flow/seed_0/report.json'),
                        ('shpongle_previous_delivery', NATIVE/'delivery/final_diagnostic/inference/shpongle.json')]:
        meta = read_json(path)
        require(digest(path) == read_json(str(path)+'.sha256.json')['sha256'], 'Historical review metadata integrity')
        rows = meta.get('rows', [meta])
        review_assets[label] = {'report_path': str(path), 'report_sha256': digest(path),
                                'arrays': [{k: r[k] for k in ('key', 'path', 'sha256')} for r in rows]}
    # Freeze hashes from authenticated metadata, not new measurements. Verify at consumption.
    value = {'schema': 1, 'protocol_sha256': digest(EXPERIMENT/'protocol.json'),
             'source_sha256': p['source_sha256'], 'evaluator_sha256': digest(__file__),
             'inputs': files, 'main_windows': main, 'short_windows': short,
             'review_clips': review['clips'], 'full_keys': list(dict.fromkeys(c['key'] for c in review['clips'])),
             'full_seed': 7001, 'shpongle': demo, 'matched_review_assets': review_assets,
             'shpongle_seeds': [7001+i*104729 for i in range(6)],
             'draws': 6, 'steps': 32, 'bootstrap': {'draws': 10000, 'seed': 20260910},
             'short_summary': 'Same TRAIN-normalized18 channels; full recordings; half-second means omit only final <15-frame ES tail. Physical metrics and saved arrays use EVERY real frame.',
             'aggregation': 'Mean draws per window, equal windows per recording, equal recordings per canonical song, equal songs',
             'ratio_definition': 'Ratios of canonical-weighted generated/reference metric means, as frozen native gate helper; NOT mean window ratios',
             'excess_definition': 'max(0, canonical generated mean - canonical source mean); sliding baseline comparator is matched_native final',
             'precision': 'FP32 neural/Euler/FK; no autocast, no TF32; six draws,32Euler; no discarded samples',
             'direct_audio_lighting_enabled': False, 'test_access': False, 'selection': 'none',
             'remaining_external': ['Matched rendering with incumbent and both new arms', 'All12 synced clip reviews',
                                    'Full Shpongle matched review', 'Human music-fit acceptance and independent full-pipeline gates']}
    save_json(output/'evaluation_protocol.json', value)
    immutable_bytes(output/'evaluator.py', Path(__file__).read_bytes())
    print('Frozen metadata only. No model or VAL arrays opened.', flush=True)
    return value


def context(output):
    p = verify_training_freeze()
    q = load_json(output/'evaluation_protocol.json')
    require(q['protocol_sha256'] == digest(EXPERIMENT/'protocol.json') and q['source_sha256'] == p['source_sha256'], 'Evaluation protocol mismatch')
    require(digest(__file__) == q['evaluator_sha256'] == digest(output/'evaluator.py'), 'Evaluator changed after freeze')
    for path in (NATIVE/'music/review/protocol.json', NATIVE/'data/shpongle/input.json'):
        require(digest(path) == q['inputs'][str(path)], 'Supplemental metadata drift')
    return p, q


def training_identity(arm, p):
    return {'arm': arm, 'seed': 0, 'protocol_sha256': digest(EXPERIMENT/'protocol.json'), 'source_sha256': p['source_sha256']}


def completed_training(p):
    """No initial or final VAL until BOTH scheduled training runs have completed."""
    reports, initial = {}, []
    for arm in ARMS:
        run = EXPERIMENT/'runs'/arm
        report = read_json(run/'report.json')
        require(report['identity'] == training_identity(arm, p) and report['steps'] == 30000, 'Incomplete/wrong training run')
        for checkpoint in CHECKPOINTS:
            require(digest(run/(checkpoint+'.pt')) == report[checkpoint+'_sha256'], 'Training checkpoint drift')
        initial_id = read_json(run/'initial_identity.json')
        require(initial_id['identity'] == report['identity'], 'Initial identity mismatch')
        initial.append(initial_id['model_sha256'])
        logs = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
        require(logs and logs[-1]['step'] == 30000, 'Missing final training log')
        require(all(r['arm'] == arm and 1 <= r['step'] <= 30000 for r in logs), 'Training log scope')
        for row in logs:
            finite([v for v in row.values() if isinstance(v, (int, float))], 'logged loss/gradient')
        reports[arm] = {**report, 'report_sha256': digest(run/'report.json'),
                        'training_log_sha256': digest(run/'training.jsonl'), 'logged_rows': len(logs),
                        'finite_evidence': 'Frozen backward_checked enforces all loss terms and gradient norm at EVERY optimizer update; sparse saved log also checked. No per-update raw tensors retained.'}
    require(initial[0] == initial[1], 'Arms do not share initialization')
    return reports


def load_model(arm, checkpoint, p, training, modules, device):
    import torch
    require(arm in ARMS and checkpoint in CHECKPOINTS, 'Only both frozen arms initial/final supported')
    path = EXPERIMENT/'runs'/arm/(checkpoint+'.pt')
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state['identity'] == training_identity(arm, p), 'Checkpoint identity mismatch')
    require(state['step'] == (0 if checkpoint == 'initial' else 30000), 'Intermediate evaluation forbidden')
    norms = read_json(NATIVE/'data/motion_normalizers.json')
    audio_norms = read_json(NATIVE/'data/audio_normalizers.json')
    require(state['normalizers'] == norms and state['audio_normalizers'] == audio_norms, 'Checkpoint normalizers drift')
    revision = modules['music_revision_model']
    weights = revision.PhysicalWeights(**p['design']['weights']) if arm == 'physical' else revision.PhysicalWeights()
    model = revision.MusicRevisionModel(modules['music_model'].MusicConfig(), norms, arm=arm, weights=weights)
    require(json_hash(state['model_config']) == json_hash(model.config_dict()), 'Checkpoint model config drift')
    for name, value in state['model'].items():
        finite(value, 'checkpoint '+name)
    model.load_state_dict(state['model'], strict=True)
    require(np.array_equal(model.mean.numpy(), np.asarray(norms['mean'], np.float32)) and
            np.array_equal(model.std.numpy(), np.asarray(norms['std'], np.float32)) and
            np.array_equal(model.delta_scale.numpy(), np.asarray(norms['delta_scale'], np.float32)), 'State buffer normalizers drift')
    return model.float().to(device).eval(), norms, audio_norms


def execution_identity(arm, checkpoint, stage, q, training, device):
    import torch
    return {'arm': arm, 'seed': 0, 'checkpoint': checkpoint, 'stage': stage,
            'checkpoint_sha256': training[arm][checkpoint+'_sha256'],
            'protocol_sha256': q['protocol_sha256'], 'evaluation_sha256': json_hash(q),
            'evaluator_sha256': q['evaluator_sha256'], 'source_sha256': q['source_sha256'],
            'torch_version': str(torch.__version__), 'device': device,
            'precision': 'float32', 'tf32': False, 'draws': 6, 'steps': 32}


def load_val(key, manifest, q, audio_norms):
    e = next(e for e in manifest['entries'] if e['key'] == key and e['partition'] == 'val')
    for field in ('audio_path', 'path'):
        require(digest(e[field]) == q['inputs'][e[field]], 'VAL input drift: '+key)
    with np.load(e['audio_path'], allow_pickle=False) as z:
        raw = z['x']
    with np.load(e['path'], allow_pickle=False) as z:
        target = z['features']
    require(raw.shape == (e['n_frames'], 92) and target.shape == (e['n_frames'], 139), 'Aligned shape mismatch')
    finite(raw, 'audio'); finite(target, 'reference motion')
    audio = ((raw-np.asarray(audio_norms['mean'], np.float32))/np.asarray(audio_norms['std'], np.float32)).astype(np.float32)
    finite(audio, 'normalized audio')
    return audio, target.astype(np.float32)


def control_arrays(w, loader):
    audio, motion = loader(w['key'])
    start, frames = w['start'], w['frames']
    controls = {'correct': audio[start:start+frames]}
    if w['donor_key'] is not None:
        donor, _ = loader(w['donor_key'])
        controls['donor'] = donor[w['donor_start']:w['donor_start']+frames]
    if w['shift_start'] is not None:
        controls['shift'] = audio[w['shift_start']:w['shift_start']+frames]
    require(all(x.shape == (frames, 92) for x in controls.values()), 'Unsupported control slice')
    return controls, motion[start:start+frames]


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def evaluate_window(model, w, controls, target, norms, modules, save_motion=False):
    """Reusable CPU synthetic test seam. Native geometry/score, identical six noise fields."""
    import torch
    device = model.mean.device
    finite(target, 'window reference')
    sm = torch.tensor(norms['summary']['mean'], device=device, dtype=torch.float32)
    ss = torch.tensor(norms['summary']['std'], device=device, dtype=torch.float32)
    require(bool((ss > 0).all()), 'Nonpositive summary normalization')
    physical = modules['music_evaluation'].physical
    summary_fn = modules['music_data'].motion_summary
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        y = torch.from_numpy(target[None]).to(device)
        quality, joints = physical(y)
        finite(joints, 'reference FK')
        target_summary = (summary_fn(y, joints)[0]-sm)/ss
        noise = torch.cat([model.noise(1, w['frames'], generator=torch.Generator(device=device).manual_seed(s))
                           for s in w['draw_seeds']], dim=0)
        finite(noise, 'six noise draws')
        arrays = {'reference': target_summary.cpu().numpy()}
        if save_motion:
            arrays.update(reference_features=target, reference_joints=joints[0].cpu().numpy())
        metrics = {'source_'+k: v for k, v in quality.items()}
        hashes = {}
        for condition, audio in controls.items():
            finite(audio, condition+' audio')
            x = torch.from_numpy(audio[None]).to(device).expand(6, -1, -1)
            prediction = model.generate(x, steps=32, initial_noise=noise.clone(), chunked=True)
            require(prediction.shape == (6, w['frames'], 139), 'Generated shape/draw mismatch')
            finite(prediction, condition+' all generated draws')
            if condition == 'correct':
                generated_quality, generated_joints = physical(prediction)
                metrics.update({'prediction_'+k: v for k, v in generated_quality.items()})
            else:
                generated_joints = modules['geometry'].joints_from_features(prediction)
            finite(generated_joints, condition+' FK')
            summary = (summary_fn(prediction, generated_joints)-sm)/ss
            finite(summary, condition+' summary')
            metrics[condition+'_es'] = modules['music_data'].energy_score(summary, target_summary)
            arrays[condition] = summary.cpu().numpy()
            native = prediction.cpu().numpy()
            hashes[condition] = array_hash(native)
            if save_motion:
                arrays[condition+'_features'] = native
                arrays[condition+'_joints'] = generated_joints.cpu().numpy()
        finite(list(metrics.values()), 'physical/ES metrics')
    return metrics, arrays, {'prediction_content_sha256': hashes,
                            'noise_sha256': array_hash(noise.cpu().numpy()),
                            'audio_content_sha256': {c: array_hash(x) for c, x in controls.items()},
                            'finite_all_draws': True, 'effective_draws': 6,
                            'summary_frames_scored': w['frames']//15*15,
                            'physical_frames_scored': w['frames']}


def aggregate(rows):
    require(bool(rows), 'Cannot aggregate empty cohort')
    records = defaultdict(list)
    metric_keys = set(rows[0]['metrics'])
    seen = set()
    for row in rows:
        marker = (row['key'], row['start'])
        require(marker not in seen, 'Duplicate evaluation row')
        seen.add(marker)
        require(set(row['metrics']) == metric_keys, 'Unequal metric support: report separately')
        finite(list(row['metrics'].values()), 'row metrics')
        records[(row['canonical_id'], row['key'])].append(row['metrics'])
    songs = defaultdict(list)
    for (song, key), values in records.items():
        songs[song].append({k: float(np.mean([v[k] for v in values])) for k in sorted(metric_keys)})
    per_song = {song: {k: float(np.mean([v[k] for v in values])) for k in sorted(metric_keys)}
                for song, values in sorted(songs.items())}
    return {'metrics': {k: float(np.mean([v[k] for v in per_song.values()])) for k in sorted(metric_keys)},
            'per_song': per_song, 'canonical_songs': len(songs), 'recordings': len(records), 'windows': len(rows)}


def comparison(per_song, control):
    values = np.asarray([[v[control+'_es'], v['correct_es']] for _, v in sorted(per_song.items())], np.float64)
    finite(values, 'bootstrap input')
    require(len(values) > 0, 'Empty bootstrap')
    before, after = values.mean(0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(1)
    relative = (means[:, 0]-means[:, 1])/np.maximum(means[:, 0], 1e-12)
    return {'before': float(before), 'after': float(after),
            'relative_improvement': float((before-after)/max(before, 1e-12)),
            'relative_ci95': np.quantile(relative, [.025, .975]).tolist(), 'canonical_songs': len(values),
            'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
            'method': 'paired canonical-song percentile bootstrap after equal recording/window reduction'}


def quality(summary):
    m = summary['metrics']
    return {**{k+'_ratio': m['prediction_'+k]/max(m['source_'+k], 1e-8)
               for k in ('kinetic', 'travel', 'contact_coverage')},
            **{k+'_excess': max(0., m['prediction_'+k]-m['source_'+k])
               for k in ('sliding', 'penetration', 'high_frequency')}}


def numerical_gates(physical_summary, baseline_summary, chunk_max_abs):
    q, b = quality(physical_summary), quality(baseline_summary)
    c = {control: comparison(physical_summary['per_song'], control) for control in ('donor', 'shift')}
    gates = {'conditioning_magnitude': all(x['relative_improvement'] >= .05 for x in c.values()),
             'conditioning_ci': all(x['relative_ci95'][0] > 0 for x in c.values()),
             'energy': .5 <= q['kinetic_ratio'] <= 2.,
             'travel': .25 <= q['travel_ratio'] <= 2.,
             'contacts': .5 <= q['contact_coverage_ratio'] <= 1.5,
             'floor': q['penetration_excess'] <= .05,
             'high_frequency': q['high_frequency_excess'] <= .05,
             'sliding_absolute': q['sliding_excess'] <= .20,
             'sliding_reduction_vs_matched_native': q['sliding_excess'] <= .8*b['sliding_excess'],
             'continuity': chunk_max_abs <= 5e-5}
    return {'gates': gates, 'physical_quality': q, 'matched_native_quality': b, 'conditioning': c,
            'sliding_zero_baseline_note': 'Literal declared <=.8*baseline gate; zero/zero meets inequality but does not establish a nonzero relative reduction.'}


def verify_row(path, identity, spec):
    row = load_json(path)
    require(row['identity'] == identity and row['spec'] == spec, 'Partial evaluation identity/spec differs')
    require(digest(row['arrays_path']) == row['arrays_sha256'], 'Partial arrays changed')
    require(row['finite_all_draws'], 'Nonfinite prior result')
    return row


def stage_inference(stage, arm, checkpoint, device, budget, output, p, q):
    import torch
    training = completed_training(p)
    modules = frozen_modules()
    model, norms, audio_norms = load_model(arm, checkpoint, p, training, modules, device)
    identity = execution_identity(arm, checkpoint, stage, q, training, device)
    out = output/arm/checkpoint/stage
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'failure.json').exists(), 'Prior nonfinite failure retained; stop, no replacement samples')
    save_json(out/'identity.json', identity)
    manifest = read_json(NATIVE/'data/audio_manifest.json')
    cache = {}
    def loader(key):
        if key not in cache:
            cache[key] = load_val(key, manifest, q, audio_norms)
        return cache[key]
    if stage in ('primary', 'short'):
        specs = q['main_windows' if stage == 'primary' else 'short_windows']
    elif stage == 'full':
        specs = [{'key': key, 'draw_seeds': [q['full_seed']]} for key in q['full_keys']]
    else:
        # One draw per restartable item. Full6 Shpongle, no draw selection.
        specs = [{'key': 'shpongle', 'draw_seeds': [s]} for s in q['shpongle_seeds']]
    rows, new = [], 0
    for index, spec in enumerate(specs):
        path = out/f'{index:03d}.json'
        if path.exists():
            rows.append(verify_row(path, identity, spec))
            continue
        if new >= budget:
            continue
        try:
            if stage in ('primary', 'short'):
                controls, target = control_arrays(spec, loader)
                metrics, arrays, extra = evaluate_window(model, spec, controls, target, norms, modules, save_motion=stage == 'short')
                row = {**spec, 'metrics': metrics, **extra}
            else:
                if stage == 'full':
                    audio, target = loader(spec['key'])
                else:
                    demo = q['shpongle']
                    require(digest(demo['path']) == q['inputs'][demo['path']], 'Shpongle input drift')
                    require(digest(demo['audio_path']) == q['inputs'][demo['audio_path']], 'Shpongle waveform drift')
                    with np.load(demo['path'], allow_pickle=False) as z:
                        audio = ((z['x']-np.asarray(audio_norms['mean'], np.float32))/np.asarray(audio_norms['std'], np.float32)).astype(np.float32)
                    require(audio.shape == (demo['frames'], 92), 'Shpongle full duration mismatch')
                    target = None
                arrays, row = evaluate_full(model, audio, target, spec['draw_seeds'][0], modules,
                                             compare_chunk=(stage == 'full'))
                row.update(key=spec['key'])
            arrays_path = path.with_suffix('.npz')
            save_npz(arrays_path, arrays)
            row.update(identity=identity, spec=spec, arrays_path=str(arrays_path), arrays_sha256=digest(arrays_path))
            save_json(path, row)
            rows.append(row)
            new += 1
            print(json.dumps({'stage': stage, 'arm': arm, 'checkpoint': checkpoint, 'item': index,
                              'completed': len(rows), 'total': len(specs)}), flush=True)
        except FloatingPointError as error:
            save_json(out/'failure.json', {'identity': identity, 'spec': spec, 'error': str(error),
                                           'gate': 'finite failed', 'samples_discarded': 0})
            raise
    if len(rows) != len(specs):
        print(f'Budget pause: {len(rows)}/{len(specs)} complete. No partial aggregate/gates.', flush=True)
        return
    result = {'identity': identity, 'rows': rows, 'complete': True, 'expected_items': len(specs),
              'finite_all_draws': True, 'test_access': False, 'direct_audio_lighting_enabled': False}
    if stage in ('primary', 'short'):
        result['summary'] = aggregate(rows)
    elif stage == 'full':
        result['chunk_max_abs'] = max(r['chunk_max_abs'] for r in rows)
        result['continuity_passed'] = result['chunk_max_abs'] <= 5e-5
    save_json(out/'report.json', result)


def evaluate_full(model, audio, target, seed, modules, compare_chunk=True):
    import torch
    finite(audio, 'full audio')
    device = model.mean.device
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        x = torch.from_numpy(audio[None]).to(device)
        noise = model.noise(1, len(audio), generator=torch.Generator(device=device).manual_seed(seed))
        finite(noise, 'full noise')
        chunk = model.generate(x, steps=32, initial_noise=noise.clone(), chunked=True)
        finite(chunk, 'full chunk native')
        error = None
        if compare_chunk:
            direct = model.generate(x, steps=32, initial_noise=noise.clone(), chunked=False)
            finite(direct, 'full direct native')
            error = float((direct-chunk).abs().max())
            finite(error, 'full/chunk max difference')
        require(chunk.shape == (1, len(audio), 139), 'Full generation shape mismatch')
        # physical() integrates the complete native root ONCE. Save these world joints;
        # consumers slice them, never integrate sliced native velocity again.
        metrics, joints = modules['music_evaluation'].physical(chunk)
        finite(joints, 'full world joints')
        features, world = chunk[0].cpu().numpy(), joints[0].cpu().numpy()
        arrays = {'features': features, 'joints': world, 'root': world[:, 0],
                  'contacts': features[:, 135:], 'times': np.arange(len(audio), dtype=np.float64)/30}
        row = {'draw_seed': seed, 'effective_draws': 1, 'frames': len(audio), 'chunk_max_abs': error,
               'quality': metrics, 'root_displacement_m': float(np.linalg.norm(world[-1, 0, [0, 2]]-world[0, 0, [0, 2]])),
               'root_max_excursion_m': float(np.linalg.norm(world[:, 0][:, [0, 2]]-world[0, 0, [0, 2]], axis=1).max()),
               'noise_sha256': array_hash(noise.cpu().numpy()), 'finite_all_draws': True,
               'audio_content_sha256': array_hash(audio),
               'root_integration': 'Exactly once over full native sequence in physical()->joints_from_features; saved world joints/root must be sliced directly'}
        if target is not None:
            require(target.shape == features.shape, 'Full reference shape mismatch')
            finite(target, 'full target')
            source_metrics, source_joints = modules['music_evaluation'].physical(torch.from_numpy(target[None]).to(device))
            finite(source_joints, 'full reference joints')
            reference = source_joints[0].cpu().numpy()
            arrays.update(reference_features=target, reference_joints=reference, reference_root=reference[:, 0], reference_contacts=target[:, 135:])
            row['source_quality'] = source_metrics
            row['source_root_displacement_m'] = float(np.linalg.norm(reference[-1, 0, [0, 2]]-reference[0, 0, [0, 2]]))
        finite(list(metrics.values()), 'full physical metrics')
    return arrays, row


def report(output, p, q):
    training = completed_training(p)
    artifacts = {}
    all_reports = {}
    for arm in ARMS:
        for checkpoint in CHECKPOINTS:
            stages = ('primary', 'short', 'full', 'shpongle') if checkpoint == 'final' else ('primary', 'short')
            for stage in stages:
                path = output/arm/checkpoint/stage/'report.json'
                r = load_json(path)
                require(r['complete'] and r['identity']['evaluation_sha256'] == json_hash(q), 'Incomplete/stale report')
                expected = execution_identity(arm, checkpoint, stage, q, training, r['identity']['device'])
                require(r['identity'] == expected, 'Report identity mismatch')
                specs = q['main_windows'] if stage == 'primary' else q['short_windows'] if stage == 'short' else (
                    [{'key': key, 'draw_seeds': [q['full_seed']]} for key in q['full_keys']] if stage == 'full' else
                    [{'key': 'shpongle', 'draw_seeds': [seed]} for seed in q['shpongle_seeds']])
                require(len(r['rows']) == r['expected_items'] == len(specs), 'Missing report items')
                for index, spec in enumerate(specs):
                    require(verify_row(path.parent/f'{index:03d}.json', expected, spec) == r['rows'][index], 'Row/report mismatch')
                if stage in ('primary', 'short'):
                    require(aggregate(r['rows']) == r['summary'], 'Aggregate mismatch')
                all_reports[(arm, checkpoint, stage)] = r
                artifacts[str(path)] = digest(path)
    # Noise and controls MUST match across all four primary/short cells and both full cells.
    for stage in ('primary', 'short', 'full', 'shpongle'):
        cells = [r for (arm, checkpoint, s), r in all_reports.items() if s == stage]
        first = cells[0]
        for cell in cells[1:]:
            require(cell['identity']['device'] == first['identity']['device'], 'Mixed RNG devices')
            for a, b in zip(first['rows'], cell['rows']):
                require(a['noise_sha256'] == b['noise_sha256'] and a['audio_content_sha256'] == b['audio_content_sha256'], 'Unpaired noise/audio cells')
    final = {arm: all_reports[(arm, 'final', 'primary')] for arm in ARMS}
    chunk = max(all_reports[(arm, 'final', 'full')]['chunk_max_abs'] for arm in ARMS)
    numeric = numerical_gates(final['physical']['summary'], final['masked_native']['summary'], chunk)
    coverage = read_json(EXPERIMENT/'coverage.json')
    short_train = [r for r in coverage['rows'] if r['real_frames'] < 960]
    masks_ok = len(short_train) == coverage['short_recordings'] == 3 and all(r['real_frames'] == r['valid_frames_per_crop'] for r in short_train)
    numeric['gates'].update(finite=True, short_holds=masks_ok, visual=None, human_music_fit_acceptance=None)
    value = {'evaluation_sha256': json_hash(q), 'training': training, 'artifacts': artifacts, **numeric,
             'primary': {arm: {cp: all_reports[(arm, cp, 'primary')]['summary'] for cp in CHECKPOINTS} for arm in ARMS},
             'short_supplemental': {arm: {cp: all_reports[(arm, cp, 'short')]['summary'] for cp in CHECKPOINTS} for arm in ARMS},
             'short_control_support': q['short_windows'], 'source_short_hold_coverage': coverage,
             'short_masks_evidence': 'Frozen coverage per-record valid lengths, tested sampler/loss mask source and preregistered engineering; not inferred from generated contacts',
             'full_chunk_max_abs': chunk, 'continuity_scope': 'Full12 fixed review recordings, both final arms, FP32 native max absolute difference; world root integrated once per full sequence',
             'remaining_external': q['remaining_external'], 'matched_review_assets': q['matched_review_assets'],
             'review_clips': q['review_clips'], 'test_access': False, 'seed1_authorized': False,
             'pipeline_qualified': False, 'selection': None,
             'note': 'Numerical diagnostics are not visual or human music-fit acceptance. No defaults/production promotion.'}
    save_json(output/'numerical_report.json', value)
    print(json.dumps({'gates': value['gates'], 'pipeline_qualified': False}), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'verify', 'primary', 'short', 'full', 'shpongle', 'report'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--checkpoint', choices=CHECKPOINTS, default='final')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Explicit GPU opt-in; parent schedules under shared gpu_lock')
    parser.add_argument('--budget', type=int, default=100000, help='Next missing windows/recordings/draws; no cohort selection')
    parser.add_argument('--output', type=Path, default=EXPERIMENT/'evaluation')
    args = parser.parse_args()
    require(args.budget > 0, 'Budget must be positive')
    require(args.output.resolve().is_relative_to((EXPERIMENT/'evaluation').resolve()), 'Output must be experiment/evaluation or a child')
    with output_lock(args.output):
        if args.stage == 'freeze':
            freeze(args.output)
            return
        p, q = context(args.output)
        if args.stage == 'verify':
            print('Frozen sources, input metadata and evaluator identity verified. No models or VAL arrays opened.')
            return
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if args.stage == 'report':
            report(args.output, p, q)
            return
        require(args.arm is not None, '--arm is required for inference')
        if args.device == 'cuda':
            # Training uses this exact frozen lock. Never overlap training/render GPU work.
            frozen_modules()
            from lightsem.experiments.stage11_common import gpu_lock
            with gpu_lock(NATIVE):
                stage_inference(args.stage, args.arm, args.checkpoint, args.device, args.budget, args.output, p, q)
        else:
            stage_inference(args.stage, args.arm, args.checkpoint, args.device, args.budget, args.output, p, q)


if __name__ == '__main__':
    main()
