"""Freeze one exact feature job and one seed-0 training job for v7b."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path('/root/lightsem/experiments/native_bf16_v7b')
GUARD = Path('/root/lightsem/scripts/motion/native_full_fit_guard_bf16_v7b.py')
PYTHON = Path('/root/lightsem/.venv/night-access/bin/python')

def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()

def frozen(stage):
    common={str(GUARD):sha(GUARD)}
    imports={}
    if stage=='features':
        files=[Path('/root/lightsem/scripts/motion/native_dynamics_features_reuse.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/feature_cache.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/full_fit_admission.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/data.py'),
               Path('/root/lightsem/lightsem/experiments/dynamic_dance/audio.py'),
               Path('/root/lightsem/lightsem/experiments/dynamic_dance/contracts.py'),
               Path('/root/lightsem/lightsem/experiments/dynamic_dance/structure.py'),
               Path('/root/lightsem/lightsem/audio/online.py')]
        imports={'lightsem.experiments.native_dynamics.feature_cache':str(files[1]),
                 'lightsem.experiments.native_dynamics.full_fit_admission':str(files[2]),
                 'lightsem.experiments.native_dynamics.data':str(files[3]),
                 'lightsem.experiments.dynamic_dance.audio':str(files[4]),
                 'lightsem.experiments.dynamic_dance.contracts':str(files[5]),
                 'lightsem.experiments.dynamic_dance.structure':str(files[6])}
    elif stage in ('preflight','training'):
        files=[Path('/root/lightsem/scripts/motion/train_native_dynamics.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/training.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/data.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/dataset.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/model.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/kinematics.py'),
               Path('/root/lightsem/lightsem/experiments/native_repair/training.py'),
               Path('/root/lightsem/lightsem/experiments/native_repair/losses.py'),
               Path('/root/lightsem/lightsem/experiments/native_dynamics/full_fit_admission.py'),
               Path('/root/lightsem/scripts/motion/prepare_finedance_full_fit_20261008.py')]
        imports={'lightsem.experiments.native_dynamics.training':str(files[1]),
                 'lightsem.experiments.native_dynamics.data':str(files[2]),
                 'lightsem.experiments.native_dynamics.dataset':str(files[3]),
                 'lightsem.experiments.native_dynamics.model':str(files[4]),
                 'lightsem.experiments.native_dynamics.kinematics':str(files[5]),
                 'lightsem.experiments.native_repair.training':str(files[6]),
                 'lightsem.experiments.native_repair.losses':str(files[7]),
                 'lightsem.experiments.native_dynamics.full_fit_admission':str(files[8]),
                 'scripts.motion.prepare_finedance_full_fit_20261008':str(files[9])}
    else: raise ValueError(stage)
    common.update({str(p):sha(p) for p in files})
    return common,imports

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('features','preflight','training'));p.add_argument('--queue',type=Path,required=True);a=p.parse_args()
    sources,imports=frozen(a.stage)
    inputs={}
    for rel in ('data/MANIFEST.json','data/FD_MANIFEST.json','data/FD_MOTION_OVERLAP_MANIFEST.json','raw/audio/MANIFEST.json','TRAINING_PROTOCOL.json'):
        path=ROOT/rel
        if path.exists(): inputs[str(path)]=sha(path)
    reuse=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v5')
    if a.stage=='features':
        for rel in ('features/COMPLETE.json','features/MUQ_IDENTITY_AUDIT.json','features/BASE92_MUQ_BINDING.json'):
            path=reuse/rel;inputs[str(path)]=sha(path)
        command=[str(PYTHON),'/root/lightsem/scripts/motion/native_dynamics_features_reuse.py','--root',str(ROOT),'--reuse-root',str(reuse),'--guarded','--guard-script',str(GUARD)]
        outputs=[str(ROOT/'features')];job='full-fit-features-bf16-v7b-20261009';timeout=24*3600;budget='features'
    elif a.stage=='preflight':
        for rel in ('data/SUPERVISION.json','features/COMPLETE.json','features/MUQ_IDENTITY_AUDIT.json'):
            path=ROOT/rel;inputs[str(path)]=sha(path)
        command=[str(PYTHON),'/root/lightsem/scripts/motion/train_native_dynamics.py','preflight','--root',str(ROOT),'--guarded','--guard-script',str(GUARD)]
        outputs=[str(ROOT/'learning/PREFLIGHT.json')];job='full-fit-preflight-bf16-v7b-20261009';timeout=12*3600;budget='training'
    else:
        for rel in ('data/SUPERVISION.json','features/COMPLETE.json','features/MUQ_IDENTITY_AUDIT.json','learning/PREFLIGHT.json'):
            path=ROOT/rel;inputs[str(path)]=sha(path)
        command=[str(PYTHON),'/root/lightsem/scripts/motion/train_native_dynamics.py','train','--root',str(ROOT),'--arm','joint','--seed','0','--guarded','--guard-script',str(GUARD)]
        outputs=[str(ROOT/'learning/runs/joint/seed_0')];job='full-fit-training-joint-seed0-bf16-v7b-20261009';timeout=12*3600;budget='training'
    queue={'schema_version':1,'id':f'native-bf16-v7b-{a.stage}-20261009','root':str(ROOT),'source':'/root/lightsem','guard_sha256':sha(GUARD),'limits':{'min_free_bytes':20*2**30,'min_available_memory_bytes':1*2**30,'poll_seconds':2.0,'max_rss_bytes':None},'jobs':[{'id':job,'command':command,'cwd':'/root/lightsem','python':str(PYTHON),'env':{'PYTHONPATH':'/root/lightsem','TORCH_HOME':'/mnt/lightsem-data/torch-home','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','RTMPOSE_POSE_PROFILE':'bf16'},'gpu':True,'budget_group':budget,'timeout_seconds':timeout,'source_files':sources,'imports':imports,'inputs':inputs,'outputs':outputs}]}
    a.queue.parent.mkdir(parents=True,exist_ok=True);a.queue.write_text(json.dumps(queue,indent=2,sort_keys=True)+'\n');a.queue.with_name(a.queue.name+'.sha256').write_text(sha(a.queue)+'\n');print(json.dumps({'queue':str(a.queue),'sha256':sha(a.queue),'stage':a.stage,'job_id':job}))
if __name__=='__main__':main()
