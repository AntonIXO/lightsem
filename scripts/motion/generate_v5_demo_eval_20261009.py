"""Generate two own-song v2-audio bodies and DMX for machine evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from lightsem.experiments.dynamic_dance.audio import (
    AudioAnalyzer, extract_base92, load_waveform, waveform_counterfactual,
)
from lightsem.experiments.native_dynamics.data import sha


COMBO = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1')


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def extract_demo_audio(root: Path) -> Path:
    plan = json.loads((COMBO / 'PLAN.json').read_text())
    analyzer = AudioAnalyzer('cuda', 'cpu')
    index = []
    custom = json.loads(json.dumps(plan))
    for track in custom['tracks']:
        song = track['id']
        wav = Path(track['audio'])
        wave = load_waveform(wav)
        duration = wave.shape[-1] / 44100.
        target = np.arange(int(np.ceil(duration * 30)), dtype=np.float64) / 30.
        for condition, intervention in (
            ('correct', None), ('gain', {'kind': 'gain', 'db': 6.0}),
        ):
            key = song if condition == 'correct' else f'{song}__gain'
            work_wave = wave
            meta = None
            if intervention is not None:
                work_wave, meta = waveform_counterfactual(wave, 44100, intervention)
            wave_id = hashlib.sha256(np.ascontiguousarray(work_wave).tobytes()).hexdigest()
            entry = {
                'key': key, 'partition': 'showcase' if condition == 'correct' else 'counterfactual',
                'wav_path': str(wav), 'wav_sha256': sha(wav),
                'timeline_path': str(root / 'features/timelines' / f'{key}.npz'),
                'base92_receipt': None,
            }
            base = extract_base92(work_wave, target, entry, root, wave_id, meta)
            entry.update(base92_receipt=base, base92_path=base['path'], base92_sha256=base['npz_sha256'])
            receipt = analyzer.extract(entry, root, meta)
            index.append({
                'song': song, 'condition': condition, 'key': key,
                'wav_path': str(wav), 'wav_sha256': sha(wav),
                'analyzed_waveform_sha256': wave_id,
                'base92_path': base['path'], 'base92_sha256': base['npz_sha256'],
                'timeline_path': receipt['timeline_path'], 'timeline_sha256': receipt['timeline_sha256'],
                'timeline_receipt_sha256': sha(root / 'features/receipts' / f'{key}.json'),
                'intervention': meta,
            })
            if condition == 'correct':
                track['base92'] = base['path']; track['music_timeline'] = receipt['timeline_path']
            else:
                track['waveform_controls']['gain'] = {'base92': base['path'], 'music_timeline': receipt['timeline_path']}
    combo = root / 'combo'
    combo.mkdir(parents=True, exist_ok=True)
    (combo / 'PLAN.json').write_text(json.dumps(custom, indent=2, sort_keys=True) + '\n')
    shutil.copytree(COMBO / 'colors', combo / 'colors', dirs_exist_ok=True)
    records = combo / 'records'; records.mkdir(parents=True, exist_ok=True)
    donor = COMBO / 'records/DONOR_DURATION_BINDINGS_v1.json'
    shutil.copy2(donor, records / donor.name)
    write_new(root / 'DEMO_AUDIO_FEATURES.json', {'schema_version': 1, 'recipe_version': 2,
        'rows': index, 'model': analyzer.muq().metadata,
        'policy': 'Own-song audio is freshly analyzed with v2 MuQ/Base92; gain is a declared intervention.'})
    return combo


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root; root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v5/learning/runs/joint/seed_0_retry_cublas2/final.pt')
    training_run = checkpoint.parent
    run = root / 'learning/runs/joint/seed_0'; run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, run / 'final.pt')
    report = json.loads((training_run / 'REPORT.json').read_text())
    report['artifacts'] = {str(run / 'final.pt'): sha(run / 'final.pt')}
    (run / 'REPORT.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    shutil.copy2(training_run / 'IDENTITY.json', run / 'IDENTITY.json')
    write_new(root / 'STUDY_PLAN.json', {'source': 'native-full-fit-v5', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha(checkpoint), 'audio_recipe_version': 2})
    shutil.copy2(checkpoint.parents[4] / 'TRAINING_PROTOCOL.json', root / 'TRAINING_PROTOCOL.json')
    combo = extract_demo_audio(root)
    from lightsem.experiments.native_dynamics import showcase
    showcase.COMBO = combo
    showcase.CONDITIONS = ('correct', 'gain')
    showcase.prepare(root)
    # The showcase helper uses a read-only symlink for historical colors;
    # materialize it so the guarded demo artifact contains no symlinks.
    colors = root / 'colors'
    if colors.is_symlink():
        target = colors.resolve(); colors.unlink(); shutil.copytree(target, colors)
    for song in ('shpongle', 'cure_for_me'):
        report_path = colors / song / 'REPORT.json'
        report = json.loads(report_path.read_text())
        artifacts = {}
        for artifact_path, digest in report.get('artifacts', {}).items():
            old_path = Path(artifact_path)
            if old_path.is_absolute() and old_path.is_relative_to(COMBO / 'colors' / song):
                local = colors / song / old_path.relative_to(COMBO / 'colors' / song)
                artifacts[str(local)] = digest
            else:
                artifacts[artifact_path] = digest
        report['artifacts'] = artifacts
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for song in ('shpongle', 'cure_for_me'):
        showcase.generate(root, 'joint', 0, song)
    write_new(root / 'DEMO_COMPLETE.json', {'status': 'raw_bodies_and_dmx_complete',
        'songs': ['shpongle', 'cure_for_me'], 'conditions': ['correct', 'gain', 'plan_shift'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'audio_features': str(root / 'DEMO_AUDIO_FEATURES.json')})


if __name__ == '__main__':
    main()
