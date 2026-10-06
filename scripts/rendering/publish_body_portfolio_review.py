"""Bind a presentation-only review revision without changing frozen generation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(root):
    root = Path(root)
    manifest_path = root / 'review/MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['plan_sha256'] != sha(root / 'PLAN.json'):
        raise RuntimeError('Review belongs to a different frozen campaign')
    template = Path(__file__).resolve().parents[1] / 'light_layers/body_portfolio_review.html'
    target = root / 'review/index.html'
    template_hash, publisher_hash = sha(template), sha(__file__)
    archive = root / 'presentation'
    template_archive = archive / ('review_template_' + template_hash[:16] + '.html')
    publisher_archive = archive / ('publisher_' + publisher_hash[:16] + '.py')
    mode = manifest['mode']
    receipt = {'revision': 3, 'mode': mode, 'source_template': str(template_archive),
        'authoring_template': str(template),
        'source_template_sha256': template_hash, 'deployed_html_sha256': template_hash,
        'manifest_sha256': sha(manifest_path), 'plan_sha256': manifest['plan_sha256'],
        'publisher_source': str(publisher_archive), 'publisher_sha256': publisher_hash,
        'changes': ['Human-readable method summaries', 'Media URLs include the video hash to prevent stale schematic playback after dark delivery',
                    'Downloads use manifest filenames and explicit download route', 'Preview/render mode is visible', 'Notes are isolated by frozen campaign hash', 'Post-evaluation diagnostic notes remain visible',
                    'Separate brightness limitations and hand-encounter inspection times', 'Presentation source is archived by content hash'],
        'scope': 'Presentation only; frozen mappers, DMX, videos and scientific outcomes are unchanged.'}
    path = root / 'records' / ('REVIEW_PRESENTATION_' + mode.upper() + '_V3.json')
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise RuntimeError('Presentation receipt already exists for different content')
    archive.mkdir(exist_ok=True)
    for original, archived in ((template, template_archive), (Path(__file__), publisher_archive)):
        content = original.read_bytes()
        if archived.exists() and archived.read_bytes() != content:
            raise RuntimeError('Archived presentation source changed')
        if not archived.exists():
            with archived.open('xb') as stream:
                stream.write(content)
    target.write_bytes(template_archive.read_bytes())
    if not path.exists():
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
    print('PUBLISHED', target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root)
