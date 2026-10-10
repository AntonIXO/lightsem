"""Build an all-fit native-dynamics manifest from fresh audio and pose caches.

All Stage11 TRAIN recordings are candidates.  Every valid coach remains a
separate track; canonical groups are retained for the sampler.  There is no
validation partition in this manifest, and TEST payloads are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.motion.audit_justdance_admission_20261007 import _audit_pose_coach
from scripts.motion.build_justdance_admission_package_20261007 import _mask_pose, COCO
from lightsem.experiments.native_dynamics.data import Quality


FPS = 30.0
WINDOW = 480


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage-manifest', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/data/manifest.json'))
    p.add_argument('--audio-manifest', type=Path, required=True)
    p.add_argument('--ids-file', type=Path, required=True)
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--extra-ids-file', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    root = args.output_root
    stage = json.loads(args.stage_manifest.read_text())
    audio = json.loads(args.audio_manifest.read_text())
    audio_rows = {r['recording_id']: r for r in audio['rows']}
    ids = {line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()}
    records = {r['recording_id']: r for r in stage['recordings'] if r.get('partition') == 'train' and r['recording_id'] in ids}
    if args.extra_ids_file:
        catalog = {r['yt_id']: r for r in json.loads(Path('/mnt/lightsem-data/cache/justdance_manifest.json').read_text()) if r.get('yt_id')}
        songs = {r['yt_id']: r for r in json.loads(Path('/mnt/lightsem-data/cache/justdance_songs.json').read_text()).get('rows', []) if r.get('yt_id')}
        for rid in {line.strip() for line in args.extra_ids_file.read_text().splitlines() if line.strip()}:
            if rid in records or rid not in catalog:
                continue
            row = catalog[rid]; song = songs.get(rid, {})
            source = Path('/mnt/lightsem-data/justdance')/row['file']
            records[rid] = {'recording_id': rid, 'canonical_id': f'legacy-{song.get("song_key", rid)}',
                            'partition': 'train', 'source_path': str(source),
                            'source_sha256': sha256(source), 'title_normalized': song.get('title_norm', row.get('file', '')),
                            'coaches': [], 'legacy_extra': True}
    pose_out = root/'data/poses'; pose_out.mkdir(parents=True, exist_ok=True)
    entries, exclusions = [], []
    for rid in sorted(records):
        rec = records[rid]; pose = args.pose_dir/f'{rid}.npz'; ar = audio_rows.get(rid)
        if ar is None or not pose.exists():
            exclusions.append({'recording_id': rid, 'reason': 'missing fresh audio or pose cache'}); continue
        if ar['classification'] == 'silent':
            exclusions.append({'recording_id': rid, 'reason': 'fresh waveform is silent'}); continue
        with np.load(pose, allow_pickle=False) as z:
            coaches = int(z['K'].shape[1]) if z['K'].ndim == 4 else 1
        for ci in range(coaches):
            audit = _audit_pose_coach(pose, ci, ar['duration_s'], [])
            gate = audit['fixed_gate']
            if abs(audit['audio_pose_diff_s']) > Quality().maximum_av_duration_difference_s:
                exclusions.append({'recording_id': rid, 'coach_index': ci,
                                   'reason': 'pose/audio duration binding mismatch',
                                   'audio_pose_diff_s': audit['audio_pose_diff_s'],
                                   'fixed_gate': gate})
                continue
            if gate.get('status') != 'ok' or gate.get('eligible_windows', 0) <= 0:
                exclusions.append({'recording_id': rid, 'coach_index': ci, 'reason': 'pose window gate failed', 'fixed_gate': gate}); continue
            value, mask, target, mask_meta = _mask_pose(pose, ci)
            good = (mask.sum(1) >= 10).astype(float)
            window = np.convolve(good, np.ones(WINDOW)/WINDOW, mode='valid')
            starts = np.flatnonzero(window >= .80)[::15].astype(np.int32)
            if not len(starts):
                exclusions.append({'recording_id': rid, 'coach_index': ci, 'reason': 'no valid masked window after cache write'}); continue
            key = f'{rid}_c{ci}'
            cache = pose_out/f'{key}.npz'
            with cache.open('wb') as stream:
                np.savez_compressed(stream, time=target, pose=value, mask=mask, starts=starts)
            stage_coach = next((c for c in rec.get('coaches', []) if c.get('coach_index') == ci), {})
            entries.append({'key': key, 'feature_key': f'jd_{key}', 'partition': 'train', 'domain': 'jd',
                            'recording_id': rid, 'coach_index': ci, 'coach_id': stage_coach.get('coach_id', f'{rid}_c{ci}'),
                            'group': rec['canonical_id'], 'canonical_id': rec['canonical_id'],
                            'title': rec.get('title_normalized', ''), 'pose_cache': str(cache), 'pose_cache_sha256': sha256(cache),
                            'wav_path': ar['wav_path'], 'wav_sha256': ar['wav_sha256'], 'duration_s': ar['duration_s'],
                            'frames': len(target), 'quality': gate, 'fresh_audio_classification': ar['classification'],
                            'source_path': rec['source_path'], 'source_sha256': rec['source_sha256'],
                            'mask_contract': mask_meta['mask_contract'], 'preserves_native_timestamps_and_masks': True,
                            'timeline_path': str(root/'features/timelines'/f'jd_{key}.npz')})
    entries.sort(key=lambda r: (r['group'], r['recording_id'], r['coach_index']))
    report = {'schema_version': 1, 'status': 'full_fit_manifest_unqualified',
              'selection': 'all Stage11 TRAIN recordings with fresh non-silent audio and at least one finite-to-finite masked 16-second window; all coaches retained separately',
              'partition_policy': 'all entries are fit/train; historical Stage11 VAL/TEST manifests are preserved as metadata and never opened',
              'entries': entries, 'exclusions': exclusions,
              'counts': {'tracks': len(entries), 'recordings': len({r['recording_id'] for r in entries}), 'groups': len({r['group'] for r in entries}),
                         'whole_record_qualified_tracks': sum(r['quality'].get('coverage', 0) >= .80 for r in entries),
                         'window_only_tracks': sum(r['quality'].get('coverage', 0) < .80 for r in entries)},
              'minutes': sum(r['duration_s'] for r in entries)/60,
              'stage_manifest_sha256': sha256(args.stage_manifest), 'audio_manifest_sha256': sha256(args.audio_manifest),
              'ids_file_sha256': sha256(args.ids_file), 'pose_model': 'yolo11m-pose.pt@960',
              'pose_extractor_sha256': sha256(Path('/root/lightsem/scripts/data/extract_justdance.py')),
              'manifest_builder_sha256': sha256(Path(__file__))}
    out = root/'data/MANIFEST.json'; out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], **report['counts'], 'minutes': report['minutes']}, sort_keys=True))


if __name__ == '__main__': main()
