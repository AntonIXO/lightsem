#!/usr/bin/env python3
"""Curator-invoked extraction; caller must hold the campaign and global GPU locks."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lightsem.experiments.dynamic_dance.audio import (
    AudioAnalyzer, CAMPAIGN_ROOT, prepare_manifest, read_receipt, build_index, extraction_preflight, prepare_dependency_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('manifest', 'extract', 'index', 'status', 'preflight'))
    parser.add_argument('--root', type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument('--keys', nargs='+', help='Explicit frozen record keys; no implicit broad extraction')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--structure-device', default='cpu')
    parser.add_argument('--intervention', type=json.loads, help='JSON waveform intervention; output key gets supplied --counterfactual-key')
    parser.add_argument('--counterfactual-key')
    args = parser.parse_args()
    if args.stage == 'manifest':
        prepare_dependency_manifest(args.root)
        result = prepare_manifest(args.root)
        print(json.dumps({'stage': 'manifest', 'entries': len(result['entries']), 'path': str(args.root / 'features/input_manifest.json')}), flush=True)
    elif args.stage == 'preflight':
        if not args.keys:
            parser.error('preflight requires explicit --keys')
        print(json.dumps(extraction_preflight(args.keys, args.root, device=args.device)), flush=True)
    elif args.stage == 'extract':
        if not args.keys:
            parser.error('extract requires explicit --keys')
        if args.intervention and (len(args.keys) != 1 or not args.counterfactual_key):
            parser.error('counterfactual requires exactly one input key and --counterfactual-key')
        manifest = read_receipt(args.root / 'features/input_manifest.json')
        lookup = {e['key']: e for e in manifest['entries']}
        analyzer = AudioAnalyzer(device=args.device, structure_device=args.structure_device)
        for key in args.keys:
            entry = dict(lookup[key])
            if args.intervention:
                entry.update(key=args.counterfactual_key, partition='counterfactual',
                             timeline_path=str(args.root / 'features/timelines' / (args.counterfactual_key + '.npz')))
            result = analyzer.extract(entry, args.root, args.intervention)
            print(json.dumps({'stage': 'extracted', **result}), flush=True)
    else:
        result = build_index(args.root, require_complete=args.stage == 'index')
        print(json.dumps({'stage': args.stage, 'complete': result['complete'], 'completed': len(result['entries']), 'missing': result['missing']}), flush=True)


if __name__ == '__main__':
    main()
