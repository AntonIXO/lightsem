"""Artifact and protocol primitives for the Stage 11 campaign."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import time

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908')
WORKSPACE = Path('/root/lightsem')
SCHEMA = 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def authenticated_json(path, value):
    atomic_json(path, value)
    atomic_json(str(path) + '.sha256.json', {'sha256': digest(path)})


def read_authenticated(path):
    expected = read_json(str(path) + '.sha256.json')['sha256']
    if digest(path) != expected:
        raise RuntimeError(f'Artifact digest mismatch: {path}')
    return read_json(path)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state(rng):
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy_legacy': np.random.get_state(),
            'numpy_batch': rng.bit_generator.state, 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, rng):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy_legacy'])
    rng.bit_generator.state = state['numpy_batch']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(value, tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require_space(root, gb=15):
    free = shutil.disk_usage(root).free / 2**30
    if free < gb:
        raise RuntimeError(f'Insufficient disk: {free:.2f} GiB available; need {gb}')
    return free


@contextmanager
def gpu_lock(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'gpu.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Another Stage11 GPU process owns the accelerator') from e
        out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        others = [line for line in out.splitlines() if line.strip() and line.split(',')[0].strip() != str(os.getpid())]
        if others:
            raise RuntimeError('Unexpected accelerator process: ' + '; '.join(others))
        yield
        fcntl.flock(f, fcntl.LOCK_UN)


def source_manifest(workspace=WORKSPACE):
    workspace = Path(workspace)
    paths = list((workspace / 'lightsem').rglob('*.py'))
    paths += list((workspace / 'lightsem').rglob('*.json'))
    paths += list(workspace.glob('*.py')) + list(workspace.glob('*.sh'))
    paths += list((workspace / 'tests').glob('test_stage11*.py'))
    return {str(p.relative_to(workspace)): digest(p) for p in sorted(set(paths)) if p.is_file()}


def freeze_sources(root=ROOT, workspace=WORKSPACE):
    root, workspace = Path(root), Path(workspace)
    freeze_path = root / 'frozen_sources.json'
    current = source_manifest(workspace)
    if freeze_path.exists():
        if read_authenticated(freeze_path)['files'] != current:
            raise RuntimeError('A different source freeze already exists')
        return read_authenticated(freeze_path)
    snapshot = root / 'source'
    for rel in current:
        dst = snapshot / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / rel, dst)
        dst.chmod(0o444)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip()
    subprocess.check_call(['git', 'cat-file', '-e', sha+'^{commit}'], cwd=workspace)
    info = {'schema_version': SCHEMA, 'git_head': sha, 'files': current,
            'source_sha256': json_digest(current), 'snapshot': str(snapshot), 'frozen_at': time.time(),
            'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=workspace, text=True)}
    authenticated_json(freeze_path, info)
    return info


def verify_sources(root=ROOT, check_live=False):
    freeze = read_authenticated(Path(root) / 'frozen_sources.json')
    snapshot = Path(freeze['snapshot'])
    for rel, sha in freeze['files'].items():
        if digest(snapshot / rel) != sha:
            raise RuntimeError('Frozen source drift: '+rel)
        if check_live and digest(WORKSPACE / rel) != sha:
            raise RuntimeError('Live source drift: '+rel)
    return freeze


def artifact_manifest(paths):
    return {str(Path(p).resolve()): {'sha256': digest(p), 'bytes': Path(p).stat().st_size} for p in paths}


def verify_artifacts(manifest):
    if not manifest:
        raise RuntimeError('Artifact checker received zero artifacts')
    for path, expected in manifest.items():
        if Path(path).stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise RuntimeError('Artifact drift: '+path)
    return True
