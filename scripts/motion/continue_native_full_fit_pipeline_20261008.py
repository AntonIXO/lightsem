"""Continue the explicitly authorized all-fit run after RTMPose cache completion.

This watcher never starts a second pose/GPU job. It waits for the locked
RTMPose sharded process, then creates a new manifest, exact MuQ features, and
one guarded seed-0 joint training run. The selected baseline is untouched.
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1')
IDS=ROOT.with_name(ROOT.name+'.all_ids.txt')
POSE=ROOT/'raw/pose_rtmpose_native_prod'
STATUS=ROOT/'PIPELINE_STATUS.json'

def run(cmd, *, torch=False, module=False):
    env=os.environ.copy(); env['PYTHONPATH']='/root/lightsem';
    if torch:
        full=['bash','run_torch.sh']+(['-m'] if module else [])+cmd; env['TORCH_ENV']='/root/lightsem/.venv/torch-access'
    else:
        full=['/usr/bin/python3']+cmd
    print('RUN',full,flush=True); subprocess.run(full,cwd='/root/lightsem',env=env,check=True)

def write(stage, **extra):
    value={'schema_version':1,'stage':stage,'updated_unix':time.time(),**extra}; STATUS.write_text(json.dumps(value,indent=2)+'\n')

def pose_processes():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv=[x for x in proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0') if x]
        except (FileNotFoundError,PermissionError):
            continue
        names={Path(x).name for x in argv}
        if 'extract_rtmpose_native.py' in names or 'run_rtmpose_sharded.py' in names:
            found.append({'pid':int(proc.name),'argv':argv})
    return found

def main():
    write('waiting_for_rtmpose',requested_ids=len(IDS.read_text().splitlines()))
    while True:
        count=len(list(POSE.glob('*.npz')))
        active=pose_processes()
        write('waiting_for_rtmpose',completed_pose_files=count,active_processes=len(active),active_pids=[x['pid'] for x in active])
        if not active:
            if count != len([x for x in IDS.read_text().splitlines() if x.strip()]):
                raise SystemExit(f'RTMPose stopped incomplete: {count}')
            break
        time.sleep(60)
    write('building_manifest',completed_pose_files=count)
    run(['scripts/motion/build_native_full_fit_manifest_rtmpose_20261008.py','--stage-manifest','/mnt/lightsem-data/experiments/lightsem/stage11_20260908/data/manifest.json','--audio-manifest',str(ROOT/'raw/audio/MANIFEST.json'),'--ids-file',str(IDS),'--pose-dir',str(POSE),'--output-root',str(ROOT)])
    write('preparing_supervision')
    run(['lightsem.experiments.native_dynamics.dataset','--root',str(ROOT)],torch=True,module=True)
    write('queueing_features')
    queue=ROOT/'records/queues/features_rtmpose_all_20261008.json'; queue.parent.mkdir(parents=True,exist_ok=True)
    run(['scripts/motion/make_native_full_fit_queue_20261007.py','features','--queue',str(queue)])
    run(['scripts/motion/native_full_fit_guard_20261007.py','--queue',str(queue),'--queue-sha256',(queue.with_name(queue.name+'.sha256')).read_text().strip()],torch=True)
    write('features_complete')
    queue=ROOT/'records/queues/training_rtmpose_all_20261008.json';
    run(['scripts/motion/make_native_full_fit_queue_20261007.py','training','--queue',str(queue)])
    run(['scripts/motion/native_full_fit_guard_20261007.py','--queue',str(queue),'--queue-sha256',(queue.with_name(queue.name+'.sha256')).read_text().strip()],torch=True)
    write('training_complete',checkpoint=str(ROOT/'learning/runs/joint/seed_0/final.pt'))

if __name__=='__main__': main()
