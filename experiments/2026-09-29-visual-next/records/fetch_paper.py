"""Fetch a public primary-source paper and retain exact bytes and metadata."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("key")
ap.add_argument("pdf_url")
ap.add_argument("abstract_url")
args = ap.parse_args()
if not re.fullmatch(r"[a-z0-9_-]+", args.key):
    raise ValueError("Invalid paper key")
root = Path('/root/lightsem/runs/visual_next_20260929/papers')
root.mkdir(parents=True, exist_ok=True)
meta = root / (args.key + '.json')
if meta.exists():
    print(meta.read_text())
    raise SystemExit(0)
if shutil.disk_usage(root).free < 2**30 + 70*2**20:
    raise RuntimeError("Disk floor")

def fetch(url, limit):
    if not url.startswith('https://'):
        raise ValueError('HTTPS source required')
    req = urllib.request.Request(url, headers={'User-Agent': 'LightSem-research/1.0 (paper verification)'})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError('Download exceeds bound')
        return data, response.geturl()

html_path = root / (args.key + '.html')
if html_path.exists():
    page, abstract_resolved = html_path.read_bytes(), args.abstract_url
else:
    page, abstract_resolved = fetch(args.abstract_url, 4*2**20)
    with html_path.open('xb') as f:
        f.write(page)
raw = page.decode('utf-8', errors='replace')
metadata = {}
for tag in re.findall(r'<meta\b[^>]*>', raw, re.I):
    attrs = dict((k.lower(), html.unescape(v)) for k, _, v in
                 re.findall(r'([\w:-]+)\s*=\s*([\"\'])(.*?)\2', tag))
    name = attrs.get('name', '')
    if name.startswith('citation_'):
        metadata.setdefault(name, []).append(attrs.get('content', ''))
pdf, pdf_resolved = fetch(args.pdf_url, 64*2**20)
if not pdf.startswith(b'%PDF-'):
    raise RuntimeError('Not a PDF')
path = root / (args.key + '.pdf')
with path.open('xb') as f:
    f.write(pdf)
text_path = path.with_suffix('.txt')
subprocess.run(['pdftotext', '-layout', str(path), str(text_path)], check=True)
info = subprocess.check_output(['pdfinfo', str(path)], text=True)
record = {'key': args.key, 'pdf_url': args.pdf_url, 'pdf_resolved_url': pdf_resolved,
          'abstract_url': args.abstract_url, 'abstract_resolved_url': abstract_resolved,
          'pdf_path': str(path), 'text_path': str(text_path), 'bytes': len(pdf),
          'pdf_sha256': hashlib.sha256(pdf).hexdigest(),
          'text_sha256': hashlib.sha256(text_path.read_bytes()).hexdigest(),
          'metadata': metadata, 'pdfinfo': info, 'fetched': True, 'read_status': 'not_yet_read'}
with meta.open('x') as f:
    json.dump(record, f, indent=2, ensure_ascii=False)
print(json.dumps(record, ensure_ascii=False))
