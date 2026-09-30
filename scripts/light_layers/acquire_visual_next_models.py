#!/usr/bin/env python3
"""Fetch pinned primary-source code/weights, bounded and without inference."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import requests

from lightsem.model.light_layers.visual import sha256, write_json

ROOT = Path('/root/lightsem/runs/visual_next_20260929')
RECORDS = Path('/root/lightsem/experiments/2026-09-29-visual-next/records/execution')


def fetch(url, path, cap, expected_md5=None, expected_sha256=None):
    if path.exists():
        raise RuntimeError('Refuse to replace existing acquisition: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=(30,120), stream=True) as r:
        r.raise_for_status()
        if int(r.headers.get('content-length',0)) > cap:
            raise RuntimeError('Declared file exceeds acquisition cap')
        count=0
        with path.with_suffix(path.suffix+'.partial').open('xb') as f:
            for block in r.iter_content(1024**2):
                count += len(block)
                if count > cap or shutil.disk_usage(ROOT).free < 2*1024**3:
                    raise RuntimeError('Acquisition size/disk floor')
                f.write(block)
    temp=path.with_suffix(path.suffix+'.partial')
    if expected_md5 and hashlib.md5(temp.read_bytes()).hexdigest() != expected_md5:
        raise RuntimeError('Released MD5 mismatch')
    if expected_sha256 and sha256(temp) != expected_sha256:
        raise RuntimeError('Released LFS SHA256 mismatch')
    temp.rename(path)
    print('FETCHED',path.name,count,flush=True)
    return {'path':str(path),'url':url,'bytes':count,'sha256':sha256(path)}


def main():
    revisions=json.loads((RECORDS/'UPSTREAM_REVISIONS.json').read_text())
    acquired={}
    for name,row in revisions.items():
        dest=ROOT/'vendor'/name
        if dest.exists():raise RuntimeError('Existing vendor directory '+str(dest))
        url=f'https://api.github.com/repos/{row["repo"]}/tarball/{row["revision"]}'
        archive=ROOT/'vendor'/(name+'.tar.gz')
        archive_receipt=fetch(url,archive,64*1024**2)
        dest.mkdir()
        extracted=0
        with tarfile.open(archive) as tf:
            for member in tf:
                rel=Path(*Path(member.name).parts[1:])
                if not member.isfile() or '..' in rel.parts or rel.is_absolute():continue
                if rel.suffix not in ('.py','.yaml','.yml','.json','.md','.txt') and rel.name != 'LICENSE':continue
                extracted+=member.size
                if extracted>64*1024**2:raise RuntimeError('Vendor source size cap')
                out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
                with out.open('xb') as f:shutil.copyfileobj(tf.extractfile(member),f)
        acquired[name]={'upstream':row,'archive':archive_receipt,
            'source_files':{str(p.relative_to(dest)):sha256(p) for p in dest.rglob('*') if p.is_file()}}
    weights=ROOT/'weights'
    acquired['cutie']['weights']=fetch(
        'https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth',
        weights/'cutie-base-mega.pth',150*1024**2,
        expected_md5='a6071de6136982e396851903ab4c083a')
    acquired['dinov2']['weights']=fetch(
        'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        weights/'dinov2_vits14_pretrain.pth',100*1024**2)
    # Original released TF SavedModel, converted with the released converter later.
    # No unverifiable third-party PyTorch weight substitution.
    row=revisions['transnetv2'];acquired['transnetv2']['tensorflow_weights']=[]
    for f in ('saved_model.pb','variables/variables.data-00000-of-00001','variables/variables.index'):
        relative='inference/transnetv2-weights/'+f
        pointer=requests.get(f'https://raw.githubusercontent.com/{row["repo"]}/{row["revision"]}/{relative}',timeout=30)
        pointer.raise_for_status()
        digest=next(line.split('sha256:',1)[1] for line in pointer.text.splitlines() if line.startswith('oid sha256:'))
        acquired['transnetv2']['tensorflow_weights'].append(fetch(
            f'https://media.githubusercontent.com/media/{row["repo"]}/{row["revision"]}/{relative}',
            weights/'transnetv2-tensorflow'/f,32*1024**2,expected_sha256=digest))
    write_json(RECORDS/'MODEL_ACQUISITION.json',{'status':'primary_source_weights_and_code_acquired',
        'inference_performed':False,'models':acquired,
        'pending':'Official TransNet TensorFlow-to-PyTorch conversion and source-bound smoke.'})


if __name__=='__main__':main()
