"""Freeze a guarded GPU queue for the new full-fit seed0 run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v5')
GUARD = Path('/root/lightsem/scripts/motion/native_full_fit_guard_rtmpose_v5_20261009.py')
PYTHON = Path('/root/lightsem/.venv/night-access/bin/python')


def sha(path: Path) -> str:
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def freeze_sources(stage: str) -> tuple[dict[str, str], dict[str, str]]:
    common = {str(GUARD): sha(GUARD)}
    imports = {}
    if stage == 'features':
        files = [Path('/root/lightsem/scripts/motion/native_dynamics_features.py'),
                 Path('/root/lightsem/lightsem/experiments/dynamic_dance/audio.py'),
                 Path('/root/lightsem/lightsem/experiments/dynamic_dance/contracts.py'),
                 Path('/root/lightsem/lightsem/experiments/dynamic_dance/structure.py'),
                 Path('/root/lightsem/lightsem/audio/online.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/full_fit_admission.py')]
        imports = {f'lightsem.experiments.dynamic_dance.{name}': str(Path('/root/lightsem/lightsem/experiments/dynamic_dance')/f'{name}.py') for name in ('audio','contracts','structure')}
        common.update({str(p): sha(p) for p in files})
    elif stage == 'training':
        files = [Path('/root/lightsem/scripts/motion/train_native_dynamics.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/training.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/data.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/dataset.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/model.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/kinematics.py'),
                 Path('/root/lightsem/lightsem/experiments/native_repair/training.py'),
                 Path('/root/lightsem/lightsem/experiments/native_repair/losses.py'),
                 Path('/root/lightsem/lightsem/experiments/native_dynamics/full_fit_admission.py'),
                 Path('/root/lightsem/scripts/motion/prepare_finedance_full_fit_20261008.py')]
        imports = {'lightsem.experiments.native_dynamics.training': str(files[1]),
                   'lightsem.experiments.native_dynamics.data': str(files[2]),
                   'lightsem.experiments.native_dynamics.dataset': str(files[3]),
                   'lightsem.experiments.native_dynamics.model': str(files[4]),
                   'lightsem.experiments.native_dynamics.kinematics': str(files[5]),
                   'lightsem.experiments.native_repair.training': str(files[6]),
                   'lightsem.experiments.native_repair.losses': str(files[7]),
                   'lightsem.experiments.native_dynamics.full_fit_admission': str(files[8]),
                   'scripts.motion.prepare_finedance_full_fit_20261008': str(files[9])}
        common.update({str(p): sha(p) for p in files})
    else:
        raise ValueError(stage)
    return common, imports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('features','training'))
    p.add_argument('--queue', type=Path, required=True)
    args = p.parse_args()
    root = ROOT
    sources, imports = freeze_sources(args.stage)
    inputs = {str(root/'data/MANIFEST.json'): sha(root/'data/MANIFEST.json'),
              str(root/'raw/audio/MANIFEST.json'): sha(root/'raw/audio/MANIFEST.json'),
              str(root/'TRAINING_PROTOCOL.json'):sha(root/'TRAINING_PROTOCOL.json'),
              str(Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/learning/cache/manifest.json')):
              sha(Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/learning/cache/manifest.json'))}
    for extra in ('FD_MOTION_OVERLAP_MANIFEST.json','FD_MOTION_MANIFEST.json','FD_MANIFEST.json','SUPERVISION.json'):
        path=root/'data'/extra
        if path.exists(): inputs[str(path)]=sha(path)
    complete=root/'features/COMPLETE.json'
    if complete.exists(): inputs[str(complete)]=sha(complete)
    if args.stage=='training':
        required=[root/'data/FD_MANIFEST.json',root/'data/SUPERVISION.json',complete,
                  root/'features/MUQ_IDENTITY_AUDIT.json']
        for path in required:inputs[str(path)]=sha(path)
    if args.stage == 'features':
        command = [str(PYTHON), '/root/lightsem/scripts/motion/native_dynamics_features.py', '--root', str(root), '--guarded', '--guard-script', str(GUARD)]
        outputs = [str(root/'features')]
        timeout = 24 * 3600
        budget = 'features'
        job_id = 'full-fit-features-20261008'
    else:
        command = [str(PYTHON), '/root/lightsem/scripts/motion/train_native_dynamics.py', 'train', '--root', str(root), '--arm', 'joint', '--seed', '0', '--guarded', '--guard-script', str(GUARD)]
        outputs = [str(root/'learning/runs/joint/seed_0')]
        timeout = 12 * 3600
        budget = 'training'
        job_id = 'full-fit-training-joint-seed0-rtmpose-v5-20261009'
    queue = {'schema_version':1,'id':f'native-full-fit-rtmpose-v5-{args.stage}-20261009','root':str(root),'source':'/root/lightsem',
             'guard_sha256':sha(GUARD),'limits':{'min_free_bytes':20*2**30,'min_available_memory_bytes':2**30,'poll_seconds':2.0},
             'jobs':[{'id':job_id,'command':command,'cwd':'/root/lightsem','python':str(PYTHON),
                      'env':{'PYTHONPATH':'/root/lightsem','TORCH_HOME':'/mnt/lightsem-data/torch-home','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
                      'gpu':True,'budget_group':budget,'timeout_seconds':timeout,'source_files':sources,
                      'imports':imports,'inputs':inputs,'outputs':outputs}]}
    args.queue.parent.mkdir(parents=True,exist_ok=True)
    args.queue.write_text(json.dumps(queue,indent=2,sort_keys=True)+'\n')
    digest=sha(args.queue); args.queue.with_name(args.queue.name+'.sha256').write_text(digest+'\n')
    print(json.dumps({'queue':str(args.queue),'sha256':digest,'stage':args.stage,'job_id':job_id},sort_keys=True))


if __name__ == '__main__': main()
