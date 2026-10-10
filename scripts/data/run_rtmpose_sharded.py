"""Run one locked RTMPose job with several independent CUDA workers."""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess
from pathlib import Path
from contextlib import ExitStack
from lightsem.experiments.native_dynamics.rtmpose_cache import DETECTOR, POSE_MODEL, sha

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v1')
LOCKS = [ROOT/'gpu.lock', Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/gpu.lock'),
         Path('/tmp/lightsem-audio-understanding-gpu.lock')]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ids-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=3)
    a=p.parse_args(); ids=sorted({x.strip() for x in a.ids_file.read_text().splitlines() if x.strip()})
    workers=max(1,min(a.workers,len(ids))); shards=[[] for _ in range(workers)]
    for i,rid in enumerate(ids): shards[i%workers].append(rid)
    shard_dir=a.output/'.shards'; shard_dir.mkdir(parents=True,exist_ok=True)
    commands=[]
    for i,shard in enumerate(shards):
        path=shard_dir/f'ids_{i:02d}.txt'; path.write_text('\n'.join(shard)+'\n')
        commands.append(['bash','scripts/data/run_rtmpose_system.sh','scripts/data/extract_rtmpose_native.py',
                         '--ids-file',str(path),'--output',str(a.output),'--no-lock'])
    identity={'extractor_sha256':sha(Path('/root/lightsem/scripts/data/extract_rtmpose_native.py')),
              'tracking_sha256':sha(Path('/root/lightsem/lightsem/experiments/native_dynamics/rtmpose_cache.py')),
              'detector_sha256':sha(DETECTOR),'pose_sha256':sha(POSE_MODEL),
              'ids_sha256':sha(a.ids_file),'start_s':0.,'max_frames':None}
    (a.output/'RUN.json').write_text(json.dumps(identity,indent=2)+'\n')
    with ExitStack() as stack:
        for path in LOCKS:
            handle=stack.enter_context(path.open('a+')); fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU already occupied')
        print(json.dumps({'workers':workers,'records':len(ids),'output':str(a.output)}),flush=True)
        procs=[subprocess.Popen(cmd, cwd='/root/lightsem', env={**os.environ,'RTMPOSE_LOCK_ALREADY_HELD':'1','RTMPOSE_SKIP_RECEIPT':'1'}) for cmd in commands]
        codes=[proc.wait() for proc in procs]
        print(json.dumps({'worker_exit_codes':codes}),flush=True)
        if any(codes): raise SystemExit(1)

if __name__=='__main__': main()
