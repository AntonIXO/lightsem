"""Isolated pinned MuQ/source/rhythm/structure extraction for native CLI audio."""
from pathlib import Path
import argparse
from .baseline import sha
from .pipeline import accelerator_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    from contextlib import nullcontext
    from lightsem.experiments.dynamic_dance.audio import AudioAnalyzer
    entry = {'key': 'input', 'partition': 'showcase', 'wav_path': str(args.wav.resolve()),
             'wav_sha256': sha(args.wav), 'timeline_path': str(args.root / 'features/timelines/input.npz')}
    with accelerator_lock() if args.device == 'cuda' else nullcontext():
        AudioAnalyzer(device=args.device, structure_device='cpu').extract(entry, args.root)


if __name__ == '__main__':
    main()
