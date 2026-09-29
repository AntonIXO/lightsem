"""Versioned inputs and immutable identities for the September overnight study."""
from pathlib import Path
import json
import subprocess
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest, json_digest

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
BASE = Path('/mnt/lightsem-data/experiments/lightsem/beat_conditioned_20260909')
STAGE = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908')
MUQ_REVISION = '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'
FAMILIES = ('base', 'fourier', 'muq')
ARCHITECTURES = ('once', 'film')
ARMS = tuple(f'{f}_{a}' for f in FAMILIES for a in ARCHITECTURES)
SEEDS = (0, 1)

def base_context():
    m = read_authenticated(BASE/'data_manifest.json')
    n = read_authenticated(BASE/'normalizers.json')
    return m, n

def initial_setup(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ('features', 'diagnostics', 'calibration', 'runs', 'evaluation', 'delivery', 'research'):
        (root/name).mkdir(exist_ok=True)
    m,n = base_context()
    pin = {'baseline_commit': '36ed994', 'base_manifest_sha256': digest(BASE/'data_manifest.json'),
           'base_normalizers_sha256': digest(BASE/'normalizers.json'),
           'base_checkpoint_sha256': digest(BASE/'run/final.pt'),
           'base_root': str(BASE), 'counts': m['counts'], 'test_access': False,
           'arms': list(ARMS), 'seeds': list(SEEDS), 'muq_revision': MUQ_REVISION}
    path = root/'input_pins.json'
    if path.exists():
        if read_authenticated(path) != pin: raise RuntimeError('Initial input identity changed')
    else: authenticated_json(path, pin)
    return m,n,pin

def verify_entry(e):
    if e['partition'] not in ('train','val'): raise ValueError('TEST input prohibited')
    if digest(e['path']) != e['sha256']: raise RuntimeError('Input tensor changed: '+e['key'])

def feature_path(family, key, root=ROOT):
    return Path(root)/'features'/family/(key+'.npy')

def dump_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    np.savez_compressed(path, **arrays)

def git_revision():
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd='/root/lightsem', text=True).strip()

def freeze_night_sources(root):
    import shutil
    from ..stage11.stage11_common import freeze_sources,verify_sources
    root=Path(root)
    if (root/'frozen_sources.json').exists():return verify_sources(root)
    info=freeze_sources(root)
    workspace=Path('/root/lightsem')
    for path in sorted((workspace/'tests').glob('test_night*.py')):
        rel=str(path.relative_to(workspace));dest=root/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);dest.chmod(0o444);info['files'][rel]=digest(path)
    info['source_sha256']=json_digest(info['files'])
    authenticated_json(root/'frozen_sources.json',info);return info
