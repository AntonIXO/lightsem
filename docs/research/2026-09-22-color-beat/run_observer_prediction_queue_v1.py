"""Orchestrate only the frozen observer CLI; no new model or scoring methods."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from lightsem.experiments.light_layers import training as t, evaluation as e

p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['bind-checkpoints','predict'],required=True)
p.add_argument('--queue-root',type=Path,required=True)
p.add_argument('--registration',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--manifest',type=Path)
p.add_argument('--transforms',type=Path)
p.add_argument('--baselines',type=Path)
p.add_argument('--torch-env',type=Path,default=Path('/root/lightsem/.venv/torch-access'))
p.add_argument('--gpu-authorized',action='store_true')
a=p.parse_args()
if a.output.exists():
    raise RuntimeError('Output exists; preserve artifacts, no silent retry')
completion=json.loads((a.queue_root/'queue_complete.json').read_text())
queue=json.loads((a.queue_root/'queue_manifest.json').read_text())['plan']
if completion.get('plan_sha256')!=queue['plan_sha256'] or len(queue['jobs'])!=15:
    raise RuntimeError('Full15-fit queue completion is required')
registration=json.loads(a.registration.read_text())
def ref(path):
    path=Path(path)
    return {'path':str(path.resolve()),'sha256':t.sha256(path)}
if a.phase=='bind-checkpoints':
    checkpoints={}
    for job in queue['jobs']:
        directory=Path(job['output_dir'])
        run_path=directory/'complete.json'
        run=json.loads(run_path.read_text())
        if run.get('status')!='complete' or run.get('smoke',True) or run['config']['steps']!=6000:
            raise RuntimeError('Missing complete full fit')
        for role,name in [('selected','best.pt'),('final','final.pt')]:
            checkpoint={**ref(directory/name),'recipe':job['recipe'],'seed':job['seed'],
                'role':role,'bindings':run['bindings'],'training_run_receipt':ref(run_path)}
            e.validate_training_receipt(checkpoint,registration)
            key=f"{job['index']:02d}_seed{job['seed']}_{job['recipe']}_{role}"
            checkpoints[key]=checkpoint
    registration.update(checkpoints=checkpoints,
        original_definition_data_registration=ref(a.registration),
        checkpoint_binding_stage='After all prescribed fits, before prediction; no selection beyond fixed best/final roles')
    t.write_json(a.output,registration)
    print(json.dumps({'status':'checkpoint_binding_candidate','checkpoints':len(checkpoints),**ref(a.output)}))
    raise SystemExit(0)
if not a.gpu_authorized or any(x is None for x in (a.manifest,a.transforms,a.baselines)):
    raise RuntimeError('Prediction needs explicit GPU authorization and frozen input paths')
# This parent-owned driver runs only after all training workers have exited.
active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
if active.stdout.strip():
    raise RuntimeError('Another GPU compute process is active')
expected={(r,s,role) for r in t.OBSERVER_RECIPES for s in (0,1,2) for role in ('selected','final')}
actual={(c['recipe'],c['seed'],c['role']) for c in registration['checkpoints'].values()}
if actual!=expected or len(registration['checkpoints'])!=30:
    raise RuntimeError('Need exactly30 fixed selected/final checkpoints')
a.output.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,TORCH_ENV=str(a.torch_env))
indices={'selected':[],'final':[]}
for role in ('selected','final'):
    for key,checkpoint in registration['checkpoints'].items():
        if checkpoint['role']!=role:
            continue
        destination=a.output/role/key
        destination.parent.mkdir(parents=True,exist_ok=True)
        log=a.output/(key+'.log')
        t.campaign_space(queue['campaign_root'],16*1024**2)
        command=['bash','run_torch.sh','-u','run_light_layers_evaluate.py','predict',
                 '--manifest',str(a.manifest),'--transforms',str(a.transforms),
                 '--registration',str(a.registration),'--baselines',str(a.baselines),
                 '--checkpoint',key,'--output',str(destination),'--device','cuda','--gpu-authorized']
        print(json.dumps({'event':'predict_start','key':key}),flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,cwd='/root/lightsem',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode:
            raise RuntimeError(f'Prediction failed for {key}; inspect {log}; artifacts retained')
        receipt=json.loads((destination/'complete.json').read_text())
        e.validate_prediction_receipt(receipt,registration,registration['baselines'])
        indices[role].append(ref(destination/'complete.json'))
        print(json.dumps({'event':'predict_complete','key':key}),flush=True)
    t.write_json(a.output/(role+'_index.json'),indices[role])
t.write_json(a.output/'complete.json',{'status':'complete','checkpoints':30,
    'registration':ref(a.registration),'indices':{role:ref(a.output/(role+'_index.json')) for role in indices},
    'scoring':'Not performed; parent must bind each index before separate selected/final scoring'})
