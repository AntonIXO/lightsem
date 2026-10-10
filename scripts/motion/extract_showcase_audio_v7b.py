"""Extract fresh showcase audio features for the current authenticated sources."""
from __future__ import annotations
import hashlib
from pathlib import Path
from lightsem.experiments.dynamic_dance.audio import AudioAnalyzer, load_waveform


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('/root/lightsem/experiments/native_bf16_v7b/demo_features')
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        'shpongle': Path('/root/shpongle.flac'),
        'cure_for_me': Path('/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav'),
    }
    analyzer = AudioAnalyzer('cuda', 'cpu')
    rows = []
    for key, path in sources.items():
        wave = load_waveform(path)
        entry = {
            'key': key, 'partition': 'showcase', 'canonical_id': key,
            'wav_path': str(path), 'wav_sha256': digest(path),
            'timeline_path': str(root / 'features' / 'timelines' / f'{key}.npz'),
        }
        receipt = analyzer.extract(entry, root)
        rows.append({'key': key, 'source_path': str(path), 'source_sha256': digest(path),
                     'timeline_path': receipt['timeline_path'],
                     'timeline_sha256': receipt['timeline_sha256'],
                     'input_sha256': receipt['input_sha256'],
                     'analyzed_waveform_sha256': receipt['base92']['analyzed_waveform_sha256'],
                     'muq': receipt['timings'], 'elapsed_s': receipt['elapsed_s']})
        print(key, rows[-1], flush=True)
    (root / 'SHOWCASE_AUDIO_MANIFEST.json').write_text(__import__('json').dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
