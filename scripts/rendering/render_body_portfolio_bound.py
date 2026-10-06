"""Bind post-evaluation presentation labels to otherwise frozen rendering.

This wrapper changes captions and review annotations only. The base renderer,
optics, body, DMX, audio and numerical reports remain immutable.
"""
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if '--root' not in sys.argv:
        raise ValueError('Explicit --root is required')
    root = Path(sys.argv[sys.argv.index('--root') + 1]).resolve()
    plan = json.loads((root / 'PLAN.json').read_text())
    binding = json.loads((root / 'records/RENDER_PRESENTATION_BINDING.json').read_text())
    if sha(__file__) != binding['wrapper_sha256'] or sha(root / 'PLAN.json') != binding['plan_sha256']:
        raise RuntimeError('Presentation implementation/plan drift')
    status_path = root / 'records/PRESENTATION_STATUS.json'
    if sha(status_path) != binding['status_sha256']:
        raise RuntimeError('Presentation labels changed')
    statuses = json.loads(status_path.read_text())['pairs']
    sys.path.insert(0, plan['source_root'])
    from scripts.rendering import render_body_portfolio as base
    if not Path(base.__file__).resolve().is_relative_to(Path(plan['source_root'])):
        raise RuntimeError('Base renderer must come from the frozen source')
    original_load = base.load_pair
    original_review = base.review

    def load_pair(root_arg, plan_arg, algorithm, track):
        controls, body, patch, indices, report, pins = original_load(root_arg, plan_arg, algorithm, track)
        key = algorithm['id'] + '/' + track['id']
        label = statuses[key]
        report = dict(report)
        if label['diagnostic']:
            report['status'] = 'diagnostic'
        report['presentation_note'] = label['note']
        pins = {**pins, 'presentation_status_sha256': binding['status_sha256'],
                'presentation_wrapper_sha256': binding['wrapper_sha256']}
        return controls, body, patch, indices, report, pins

    def review(root_arg, plan_arg, mode):
        original_review(root_arg, plan_arg, mode)
        path = root_arg / 'review/MANIFEST.json'
        manifest = json.loads(path.read_text())
        for row in manifest['variants']:
            label = statuses.get(row['algorithm_id'] + '/' + row['track_id'])
            if label:
                if label['diagnostic']:
                    row['numeric_status'] = 'diagnostic'
                row['diagnostic_note'] = label['note']
        manifest['presentation_binding_sha256'] = sha(root_arg / 'records/RENDER_PRESENTATION_BINDING.json')
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False))

    base.load_pair = load_pair
    base.review = review
    base.main()


if __name__ == '__main__':
    main()
