"""Continue the v7b raw cache into admission, exact features, smoke, training."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT=Path('/root/lightsem/experiments/native_bf16_v7b')
IDS=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1.all_ids.txt')
POSE=ROOT/'raw/pose_rtmpose_bf16_prod'
GUARD=Path('/root/lightsem/scripts/motion/native_full_fit_guard_bf16_v7b.py')

def pose_processes():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv=[x for x in proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0') if x]
        except (FileNotFoundError,PermissionError): continue
        names={Path(x).name for x in argv}
        if 'extract_rtmpose_native.py' in names or 'run_rtmpose_sharded_bf16_v7.py' in names: found.append(argv)
    return found

def status(stage,**extra):
    path=ROOT/'PIPELINE_STATUS.json';old=json.loads(path.read_text()) if path.exists() else {}
    old.update(schema_version=1,stage=stage,updated_unix=time.time(),**extra)
    path.write_text(json.dumps(old,indent=2)+'\n')

def run(cmd,torch=False):
    env=os.environ.copy();env['PYTHONPATH']='/root/lightsem'
    if torch:
        env['TORCH_ENV']='/root/lightsem/.venv/torch-access';argv=['bash','run_torch.sh']+cmd
    else: argv=['bash','run.sh']+cmd
    print('RUN',argv,flush=True);subprocess.run(argv,cwd='/root/lightsem',env=env,check=True)

def guarded(stage):
    queue=ROOT/'records/queues'/f'{stage}_bf16_v7b.json';queue.parent.mkdir(parents=True,exist_ok=True)
    run(['scripts/motion/make_native_bf16_v7b_queue.py',stage,'--queue',str(queue)])
    run(['scripts/motion/dynamic_dance_guard.py','--queue',str(queue),'--queue-sha256',(queue.with_name(queue.name+'.sha256')).read_text().strip()],torch=True)

def main():
    requested=len([x for x in IDS.read_text().splitlines() if x.strip()])
    status('waiting_for_rtmpose',requested_recordings=requested)
    while True:
        count=len(list(POSE.glob('*.npz')));active=pose_processes()
        status('waiting_for_rtmpose',requested_recordings=requested,completed_pose_files=count,active_processes=len(active))
        if not active:
            if count != requested: raise RuntimeError(f'RTMPose terminal incomplete: {count}/{requested}')
            break
        time.sleep(60)
    status('building_manifest',completed_pose_files=count)
    run(['scripts/motion/build_native_full_fit_manifest_rtmpose_20261008.py','--stage-manifest','/mnt/lightsem-data/experiments/lightsem/stage11_20260908/data/manifest.json','--audio-manifest',str(ROOT/'raw/audio/MANIFEST.json'),'--ids-file',str(IDS),'--pose-dir',str(POSE),'--output-root',str(ROOT)])
    status('preparing_supervision')
    run(['-m','lightsem.experiments.native_dynamics.dataset','--root',str(ROOT)],torch=True)
    status('extracting_exact_muq')
    guarded('features')
    status('running_small_training_smoke')
    guarded('preflight')
    status('training_all_data_seed0')
    guarded('training')
    status('complete',checkpoint=str(ROOT/'learning/runs/joint/seed_0/final.pt'))

if __name__=='__main__':main()
