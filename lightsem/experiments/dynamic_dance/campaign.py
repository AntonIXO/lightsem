"""Immutable campaign identities and artifact I/O for dynamic musical dance."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
WORKSPACE = Path('/root/lightsem')
METHODS = {
    'D1': 'Musical-event retiming',
    'D2': 'Phrase preparation and release',
    'D3': 'Coordinated source-specific timing',
    'D4': 'Recurring motifs and development',
    'D5': 'Expressive trajectory refinement',
    'D6': 'Learned effort and duration planner',
    'D7': 'Stronger source-conditioned generation',
    'D8': 'Hierarchical choreography generation',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x') as handle:
        handle.write(payload)
    return sha(path)


def verify_files(files):
    for path, expected in files.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('Frozen input drift: ' + str(path))


def disk_guard(root=ROOT):
    root = Path(root)
    if shutil.disk_usage(root if root.exists() else root.parent).free < 20 * 2**30:
        raise RuntimeError('20 GiB free-space reserve would be violated')


def _variant_bindings(root):
    """Explicit correctness revisions; never choose a better-scoring result."""
    root=Path(root).resolve()
    path=root/'records/VARIANT_BINDINGS.json'
    if not path.exists():
        return None
    value=read(path)
    seal=path.with_name(path.name+'.sha256.json')
    if not seal.exists() or read(seal).get('sha256')!=sha(path):
        raise ValueError('Unauthenticated variant bindings')
    if value.get('schema_version')!=1 or value.get('plan_sha256')!=sha(root/'PLAN.json'):
        raise ValueError('Variant binding uses another original PLAN')
    if value.get('selection_policy')!='implementation_correction_only':
        raise ValueError('Outcome-based variant selection is forbidden')
    for method,entry in value.get('variants',{}).items():
        if method not in METHODS or not re.fullmatch(r'[A-Za-z0-9_-]+',entry.get('artifact_id','')):
            raise ValueError('Invalid logical method/artifact identifier')
        if not entry.get('reason') or not entry.get('references'):
            raise ValueError('Correctness revision requires reason and preserved evidence')
        for filename,digest in entry['references'].items():
            reference=Path(filename)
            if not reference.is_absolute() or not reference.resolve().is_relative_to(root):
                raise ValueError('Variant evidence must stay inside this campaign')
            if sha(reference)!=digest:
                raise ValueError('Changed variant correction evidence')
    return value


def variant_metadata(root,method):
    value=_variant_bindings(root)
    logical,separator,control=method.partition('__')
    entry={} if value is None else value.get('variants',{}).get(logical,{})
    artifact=entry.get('artifact_id',logical)+(separator+control if separator else '')
    return {**entry,'logical_method':method,'artifact_id':artifact,
            'reason':entry.get('reason'),'original_artifact_id':method}


def resolve_variant(root,method):
    return variant_metadata(root,method)['artifact_id']


def variant_binding_pins(root):
    value=_variant_bindings(root)
    if value is None:
        return {}
    path=Path(root).resolve()/'records/VARIANT_BINDINGS.json'
    pins={str(path):sha(path),str(path.with_name(path.name+'.sha256.json')):sha(path.with_name(path.name+'.sha256.json'))}
    for entry in value['variants'].values():
        pins.update(entry['references'])
    return pins


def tracks():
    songs = [('shpongle', 'Shpongle', '/root/shpongle.flac'),
             ('cure_for_me', 'AURORA — Cure For Me',
              '/root/lightsem/experiments/legacy/stages/outputs/stage8/demo_songs/Cure_For_Me.wav')]
    result = []
    for key, title, audio in songs:
        folder = WORKSPACE / 'runs/body_transfer_20260930_v1' / key
        result.append({'id': key, 'title': title, 'audio': audio,
                       'audio_reference': str(folder / 'audio_reference.m4a'),
                       'body': str(folder / 'body.npz'),
                       'body_provenance': str(folder / 'BODY_PROVENANCE.json'),
                       'baseline_timeline': str(folder / 'timeline.npz'),
                       'baseline_patch': str(folder / 'patch.json'),
                       'music_timeline': str(ROOT / 'features/timelines' / (key + '.npz'))})
    return result


def initialize(root=ROOT):
    from lightsem.conductor.body_transfer import TransferConfig
    root = Path(root)
    disk_guard(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'PLAN.json'
    if path.exists():
        plan = read(path)
        verify_files(plan['files'])
        return plan
    song_rows = tracks()
    for song in song_rows:
        song['music_timeline'] = str(root / 'features/timelines' / (song['id'] + '.npz'))
    paths = {song[key] for song in song_rows for key in
             ('audio', 'body', 'body_provenance', 'baseline_timeline', 'baseline_patch')}
    paths.add(str(WORKSPACE / 'configs/light_layers/baseline.json'))
    paths.add('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/data/audio_manifest.json')
    paths.add('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/music/runs/flow/seed_0/final.pt')
    plan = {
        'schema_version': 1, 'id': root.name,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'algorithms': [{'id': key, 'name': name} for key, name in METHODS.items()],
        'tracks': song_rows, 'prefix_s': 180., 'review_start_s': 60.,
        'review_duration_s': 120., 'body_fps': 30., 'fps': 40.,
        'render_fps': 30, 'render_size': [1920, 1080],
        'counts': {'beams': 8, 'lasers': 5, 'bulbs': 50},
        'mapper_id': '01', 'transfer': asdict(TransferConfig()),
        'generation_seed': 7001, 'training_seeds': [0, 1], 'display_training_seed': 0,
        'target_updates': 30000,
        'training_manifest': '/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/data/audio_manifest.json',
        'minimum_free_bytes': 20 * 2**30,
        'gpu_budget_hours': 24,
        'stage_budget_hours': {'features': 6, 'training': 12, 'evaluation': 4, 'rendering': 2},
        'pure_retiming': {'source_order_preserved': True, 'fixed_endpoints': True,
                          'rate_min': 0., 'rate_max': 1.5,
                          'infeasible_anchors': 'report, never silently force or truncate'},
        'D4_scope': 'Recomposition of own per-song generated fragments; not monotone retiming or new independent dance information.',
        'D5_scope': 'Fixed-time upper-body trajectory refinement; pelvis/leg chains preserved, per-joint displacement at most 0.12 m.',
        'data_policy': 'TRAIN-only fitting/normalization; VAL is exposed development; no TEST motion or MUSDB TEST access; showcase songs are exposed.',
        'comparison_policy': 'Same weights within each cross-song/counterfactual arm, same initial noise/non-audio context, fixed P2 RGB and mapper. Learned arms have explicitly different registered weights.',
        'failure_policy': 'Finite failures may be rendered as diagnostics. No fallback, seed substitution, fabricated event, lowered gate or silent rerun.',
        'artistic_acceptance': 'User review pending; engineering support never establishes musical or artistic superiority.',
        'files': {p: sha(p) for p in sorted(paths)},
        'source_freeze_policy': 'Each execution phase references an immutable source snapshot and exact command queue; PLAN contains protected original inputs.',
    }
    write_new(path, plan)
    return plan


def freeze_source(root=ROOT, revision='implementation_v1', extra_files=()):
    """Snapshot current new implementation and its in-repo dependencies once."""
    root = Path(root)
    plan = read(root / 'PLAN.json')
    verify_files(plan['files'])
    dest = root / 'sources' / revision
    if dest.exists():
        raise FileExistsError('Preserve existing snapshot: ' + str(dest))
    source_paths = []
    for folder in ('lightsem', 'scripts'):
        source_paths.extend(p for p in (WORKSPACE / folder).rglob('*')
                            if p.is_file() and p.suffix in ('.py', '.html')
                            and '__pycache__' not in p.parts)
    source_paths.extend((WORKSPACE / 'tests').glob('test_dynamic_dance*.py'))
    source_paths.extend(Path(p) for p in extra_files)
    hashes, origins = {}, {}
    for original in sorted(set(source_paths)):
        relative = original.relative_to(WORKSPACE)
        target = dest / relative
        expected = sha(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha(original) != expected or sha(target) != expected:
            raise RuntimeError('Source changed while freezing: ' + str(original))
        hashes[str(target)] = expected
        origins[str(target)] = str(original)
    receipt = {'source_root': str(dest), 'files': hashes, 'origins': origins,
               'plan_sha256': sha(root / 'PLAN.json')}
    write_new(root / 'records' / (revision + '_manifest.json'), receipt)
    return receipt
