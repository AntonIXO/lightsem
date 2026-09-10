"""Acquire the official public subset with capacity checks and resumable transport."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from . import DATA
from ..stage11_common import authenticated_json, digest, read_authenticated

URL = ('https://drive.usercontent.google.com/download?'
       'id=1zQvWG9I0H4U3Zrm8d_QD_ehenZvqfQfS&export=download&confirm=t&download=1')
EXPECTED_BYTES = 4447681161  # Official public release, verified HTTP Content-Range.


def acquire(root=DATA):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'acquisition.json').exists():
        return read_authenticated(root / 'acquisition.json')
    archive = root / 'finedance.rar'
    current = archive.stat().st_size if archive.exists() else 0
    if current > EXPECTED_BYTES:
        raise ValueError('Unexpected archive length; preserve for inspection')
    if shutil.disk_usage(root).free < EXPECTED_BYTES - current + 4 * 2**30:
        raise RuntimeError('Insufficient space for transport and 4 GiB reserve')
    if current != EXPECTED_BYTES:
        subprocess.run(['curl', '-fL', '--retry', '3', '--connect-timeout', '30',
                        '-C', '-', '-o', str(archive), URL], check=True)
    if archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError('Download length differs from official release')
    if archive.open('rb').read(8) != b'Rar!\x1a\x07\x01\x00':
        raise ValueError('Expected RAR5 archive')
    listing = subprocess.check_output(['unrar', 'lt', '-c-', str(archive)], text=True)
    (root / 'archive_listing.txt').write_text(listing)
    members = []
    for block in listing.split('\n\n'):
        fields = dict(line.strip().split(': ', 1) for line in block.splitlines() if ': ' in line)
        if 'Name' not in fields or fields.get('Type') != 'File':
            continue
        path = Path(fields['Name'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe archive path')
        members.append({'path': str(path), 'bytes': int(fields['Size'])})
    if not members:
        raise ValueError('No recognized archive members')
    total = sum(m['bytes'] for m in members)
    if shutil.disk_usage(root).free < total + 4 * 2**30:
        raise RuntimeError(f'Extraction needs {total / 2**30:.2f} GiB plus 4 GiB reserve')
    subprocess.run(['unrar', 't', '-idq', str(archive)], check=True)
    subprocess.run(['unrar', 'x', '-o-', '-idq', str(archive), str(root) + '/'], check=True)
    for member in members:
        path = root / member['path']
        if path.stat().st_size != member['bytes']:
            raise ValueError(f'Incomplete extraction: {path}')
    report = {'source': URL, 'archive_bytes': EXPECTED_BYTES,
              'archive_sha256': digest(archive), 'unpacked_bytes': total, 'members': members,
              'license': 'FineDance public research release; retain upstream attribution and terms',
              'archive_retained': False}
    authenticated_json(root / 'acquisition.json', report)
    # Only the newly downloaded, CRC-verified transport archive is disposable.
    archive.unlink()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DATA)
    print(json.dumps(acquire(parser.parse_args().root)), flush=True)
