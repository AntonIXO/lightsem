"""Leakage-aware source preparation. No low-motion exclusion, no TEST tensors."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import DATA, ROOT
from .geometry import FPS, FEET, BODY_HEIGHT, OFFSETS, prepare_native
from ..stage11_common import authenticated_json, read_authenticated, digest, json_digest

# Official FineDance@Genre identifiers as published in the LODGE data loader.
# Preserve known author exclusions, but do NOT copy its variance or intro trims.
TEST = set('063 132 143 036 098 198 130 012 211 193 179 065 137 161 092 120 037 109 204 144'.split())
AUTHOR_IGNORE = set('116 117 118 119 120 121 122 123 202'.split())
UPSTREAM = 'https://github.com/li-ronghui/LODGE/blob/main/dld/data/FineDance_dataset.py'


class Groups:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def fingerprint(path, destination):
    if destination.exists():
        result = read_authenticated(destination)
        if result['wav_sha256'] != digest(path):
            raise ValueError('Fingerprint audio changed')
        return result
    process = subprocess.run(['fpcalc', '-json', '-raw', '-length', '0', str(path)],
                             text=True, capture_output=True, check=False)
    try:
        result = json.loads(process.stdout)
    except ValueError as exc:
        raise RuntimeError(f'Fingerprint failed for {path}: {process.stderr}') from exc
    if process.returncode or len(result['fingerprint']) < 100:
        raise RuntimeError(f'Invalid fingerprint: {path}: {process.stderr}')
    result.update(wav_sha256=digest(path), path=str(path), algorithm='chromaprint2-full-file',
                  command=['fpcalc', '-json', '-raw', '-length', '0'])
    authenticated_json(destination, result)
    return result


def fingerprint_similarity(a, b):
    """Candidate-offset search followed by exact bit agreement over >=16 seconds.

    Candidate hashes are for recall only. Decisions use complete 32-bit codes,
    at least128 aligned codes, and >=.85 agreement (random expectation .5).
    """
    a, b = np.asarray(a, np.uint32), np.asarray(b, np.uint32)
    reverse = defaultdict(list)
    for j in range(len(b)):
        reverse[int(b[j]) & 65535].append(j)
    offsets = Counter()
    for i in range(0, len(a), 3):
        js = reverse.get(int(a[i]) & 65535, [])
        if len(js) <= 20:
            for j in js:
                offsets[j - i] += 1
    best = (0., 0, 0)
    for offset, votes in offsets.most_common(5):
        if votes < 3:
            continue
        lo, hi = max(0, -offset), min(len(a), len(b)-offset)
        if hi-lo < 128:
            continue
        xor = np.bitwise_xor(a[lo:hi], b[lo+offset:hi+offset])
        bit_errors = np.unpackbits(xor.view(np.uint8)).reshape(-1, 32).sum(1)
        # Match a contiguous >=16-second region: recordings may use different excerpts.
        running = np.convolve(bit_errors, np.ones(128)/128, mode='valid')
        agreement = 1 - float(running.min())/32
        if agreement > best[0]:
            best = agreement, offset, hi-lo
    return best


def build_split(data=DATA, root=ROOT):
    data, root = Path(data) / 'finedance', Path(root)
    path = root / 'data/split.json'
    if path.exists():
        return read_authenticated(path)
    acquisition = read_authenticated(Path(data).parent / 'acquisition.json')
    motions = sorted((data/'motion').glob('*.npy'))
    ids = [p.stem for p in motions]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Empty or ambiguous motion identities')
    canonical_ids = sorted({key.removeprefix('M') for key in ids})
    labels = {key: json.loads((data/'label_json'/f'{key}.json').read_text()) for key in canonical_ids}
    wavs = {key: data/'music_wav'/f'{key}.wav' for key in canonical_ids}
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda key: fingerprint(wavs[key], root/'data/fingerprints'/f'{key}.json'), canonical_ids))
    prints = dict(zip(canonical_ids, values))
    groups = Groups(canonical_ids)
    reasons = []
    aliases = defaultdict(list)
    exact = defaultdict(list)
    for key in canonical_ids:
        name = re.sub(r'[^\w]+', '', str(labels[key]['name']).casefold())
        if name:
            aliases[name].append(key)
        exact[prints[key]['wav_sha256']].append(key)
    for kind, mapping in [('song_name', aliases), ('exact_audio', exact)]:
        for matches in mapping.values():
            for key in matches[1:]:
                groups.join(matches[0], key)
                reasons.append({'a': matches[0], 'b': key, 'reason': kind})
    for index, a in enumerate(canonical_ids):
        for b in canonical_ids[index+1:]:
            if groups.find(a) == groups.find(b):
                continue
            score, offset, overlap = fingerprint_similarity(prints[a]['fingerprint'], prints[b]['fingerprint'])
            if score >= .85:
                groups.join(a, b)
                reasons.append({'a': a, 'b': b, 'reason': 'chromaprint_region',
                                'bit_agreement': score, 'offset_codes': offset, 'overlap_codes': overlap})
    components = defaultdict(list)
    for key in canonical_ids:
        components[groups.find(key)].append(key)
    group_info = {}
    for members in components.values():
        group = 'fd-' + hashlib.sha256(','.join(sorted(members)).encode()).hexdigest()[:16]
        # Test membership propagates across duplicate aliases, preventing leakage.
        partition = 'test' if TEST.intersection(members) else (
            'val' if int(hashlib.sha256(('motion-split-v1:'+group).encode()).hexdigest(), 16) % 5 == 0 else 'train')
        for key in members:
            group_info[key] = group, partition
    entries = []
    for key, motion in zip(ids, motions):
        source = key.removeprefix('M')
        group, partition = group_info[source]
        info = sf.info(wavs[source])
        entries.append({'key': key, 'canonical_id': group, 'partition': partition,
                        'source_id': source, 'motion_path': str(motion), 'motion_sha256': digest(motion),
                        'wav_path': str(wavs[source]), 'wav_sha256': prints[source]['wav_sha256'],
                        'audio_duration_s': info.duration, 'label': labels[source],
                        'excluded': source in AUTHOR_IGNORE,
                        'exclusion_reason': 'published_author_ignore_list' if source in AUTHOR_IGNORE else None})
    report = {'entries': entries, 'duplicate_edges': reasons, 'official_test_ids': sorted(TEST),
              'author_ignore': sorted(AUTHOR_IGNORE), 'upstream_split_source': UPSTREAM,
              'acquisition_sha256': digest(Path(data).parent/'acquisition.json'),
              'archive_sha256': acquisition['archive_sha256'], 'source_sha256': digest(__file__),
              'split_rule': 'official TEST + duplicate closure; SHA256 group modulo5==0 VAL, else TRAIN',
              'test_audio_read_for_identity_only': True, 'test_motion_tensors_read': False,
              'counts': {p: {'recordings': sum(e['partition']==p and not e['excluded'] for e in entries),
                            'song_groups': len({e['canonical_id'] for e in entries if e['partition']==p and not e['excluded']})}
                         for p in ('train', 'val', 'test')}}
    if report['counts']['val']['song_groups'] < 12:
        raise RuntimeError('Insufficient independent validation songs')
    authenticated_json(path, report)
    print(json.dumps({'stage': 'split', 'counts': report['counts'], 'duplicate_edges': len(reasons)}), flush=True)
    return report


def source_metrics(value):
    joints = value['joints']
    velocity = np.diff(joints, axis=0) * FPS
    acceleration = np.diff(velocity, axis=0) * FPS
    feet = joints[:, FEET]
    contact = value['contacts'][:-1] > .5
    speed = np.linalg.norm(np.diff(feet, axis=0) * FPS, axis=-1)
    return {'frames': len(joints), 'duration_s': len(joints)/FPS,
            'velocity_square_mean': float(np.square(velocity).sum(-1).mean()),
            'acceleration_square_mean': float(np.square(acceleration).sum(-1).mean()),
            'root_speed_mean': float(np.linalg.norm(value['features'][:, :2], axis=1).mean()),
            'foot_penetration_p95_m': float(np.quantile(np.maximum(-feet[..., 1], 0), .95)),
            'contact_speed_mean_m_s': float(speed[contact].mean()) if contact.any() else 0.,
            'contact_prevalence': value['contacts'].mean(0).tolist(),
            'rotation_step_p99': float(np.quantile(np.linalg.norm(np.diff(value['rotations_6d'], axis=0), axis=-1), .99))}


def select_reviews(rows):
    candidates = []
    frames = 20*FPS
    for entry in rows:
        if entry['partition'] != 'val' or not entry['audio_eligible']:
            continue
        with np.load(entry['path']) as value:
            joints, features = value['joints'], value['features']
        for start in range(0, len(joints)-frames+1, frames):
            part = joints[start:start+frames]
            speed = float(np.sqrt(np.square(np.diff(part, axis=0)*FPS).sum(-1).mean()))
            travel = float(np.linalg.norm(features[start:start+frames, :2], axis=-1).mean())
            candidates.append({'key': entry['key'], 'canonical_id': entry['canonical_id'], 'start': start,
                               'frames': frames, 'body_speed': speed, 'root_speed': travel})
    chosen, used = [], set()
    for category, order in [('holds', lambda x: x['body_speed']),
                            ('travel', lambda x: -x['root_speed']), ('coordination', lambda x: -x['body_speed'])]:
        count = 0
        for candidate in sorted(candidates, key=lambda x: (order(x), x['key'], x['start'])):
            if candidate['canonical_id'] in used:
                continue
            chosen.append(dict(candidate, category=category))
            used.add(candidate['canonical_id'])
            count += 1
            if count == 4:
                break
        if count != 4:
            raise RuntimeError('Not enough independent source review clips')
    return chosen


def prepare(data=DATA, root=ROOT):
    torch.set_num_threads(4)
    root = Path(root)
    path = root / 'data/manifest.json'
    if path.exists():
        return read_authenticated(path)
    split = build_split(data, root)
    rows = []
    for source in split['entries']:
        if source['partition'] == 'test' or source['excluded']:
            continue
        if digest(source['motion_path']) != source['motion_sha256']:
            raise ValueError('Source motion changed')
        raw = np.load(source['motion_path'], allow_pickle=False)
        value = prepare_native(raw)
        difference = len(raw)/FPS-source['audio_duration_s']
        # Motion reconstruction does not require audio. Quarantine unresolved
        # timing from the paired stage instead of inventing an alignment or
        # discarding useful motion (including quiet frames) from the codec.
        audio_eligible = abs(difference) <= .15
        destination = root/'data/motion'/f'{source["key"]}.npz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            np.savez_compressed(destination, **value)
        else:
            with np.load(destination) as prior:
                if not all(np.array_equal(prior[key], array) for key, array in value.items()):
                    raise ValueError('Partial prepared motion differs')
        row = dict(source, path=str(destination), sha256=digest(destination),
                   metrics=source_metrics(value), n_frames=len(raw), fps=FPS, audio_offset_s=0.,
                   audio_eligible=audio_eligible, duration_difference_s=difference,
                   audio_quarantine_reason=None if audio_eligible else 'unresolved_released_duration_mismatch')
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps({'stage': 'prepare', 'records': len(rows)}), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        if row['partition'] == 'train':
            grouped[row['canonical_id']].append(row)
    mean, square = np.zeros(139), np.zeros(139)
    v2 = a2 = 0.
    for members in grouped.values():
        for row in members:
            with np.load(row['path']) as value:
                features = value['features'].astype(np.float64)
            weight = 1/len(grouped)/len(members)
            mean += weight*features.mean(0)
            square += weight*np.square(features).mean(0)
            v2 += weight*row['metrics']['velocity_square_mean']
            a2 += weight*row['metrics']['acceleration_square_mean']
    norms = {'mean': mean.tolist(), 'std': np.sqrt(np.maximum(square-mean*mean, .05**2)).tolist(),
             'velocity_rms': float(np.sqrt(v2)), 'acceleration_rms': float(np.sqrt(a2)),
             'body_height_m': BODY_HEIGHT, 'fit_partition': 'train', 'weighting': 'equal song group / recording / frame'}
    authenticated_json(root/'data/normalizers.json', norms)
    reviews = select_reviews(rows)
    report = {'entries': rows, 'counts': split['counts'], 'review_clips': reviews,
              'audio_quarantine': [{'key': row['key'], 'difference_s': row['duration_difference_s']}
                                   for row in rows if not row['audio_eligible']],
              'split_sha256': digest(root/'data/split.json'), 'normalizers_sha256': digest(root/'data/normalizers.json'),
              'geometry_source_sha256': digest(Path(__file__).with_name('geometry.py')),
              'prepare_source_sha256': digest(__file__), 'offsets_sha256': json_digest(OFFSETS.tolist()),
              'native_fps': FPS, 'test_motion_tensors_read': False, 'low_motion_clips_removed': False,
              'time_alignment': 'released frame0 to waveform t0; no fitted music-motion phase offsets'}
    authenticated_json(path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    result = prepare(args.data, args.root)
    print(json.dumps({'counts': result['counts'], 'reviews': result['review_clips']}), flush=True)
