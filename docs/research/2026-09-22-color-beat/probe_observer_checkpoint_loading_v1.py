"""Checkpoint integrity only. No audio/video input, inference, or scoring."""
import json
import sys
from pathlib import Path
from lightsem.experiments.light_layers import training as t, evaluation as e
queue_root, registration_path, transforms_path, receipt_path = map(Path,sys.argv[1:])
registration=json.loads(registration_path.read_text())
transforms=json.loads(transforms_path.read_text())
plan=json.loads((queue_root/'queue_manifest.json').read_text())['plan']
checked=[]
for job in plan['jobs']:
    if job['seed']!=0:
        continue
    directory=Path(job['output_dir'])
    run_path=directory/'complete.json'
    run=json.loads(run_path.read_text())
    for role,name in [('selected','best.pt'),('final','final.pt')]:
        path=directory/name
        checkpoint={'path':str(path),'sha256':t.sha256(path),'recipe':job['recipe'],
            'seed':job['seed'],'role':role,'bindings':run['bindings'],
            'training_run_receipt':{'path':str(run_path),'sha256':t.sha256(run_path)}}
        e.validate_training_receipt(checkpoint,registration)
        model,metadata=e.load_bound_model(checkpoint,transforms,
            transforms_sha256=t.sha256(transforms_path),expected_bindings=run['bindings'],device='cpu')
        checked.append({'recipe':job['recipe'],'seed':job['seed'],'role':role,
            'step':metadata['step'],'parameters':sum(p.numel() for p in model.parameters()),
            'checkpoint_sha256':checkpoint['sha256']})
        del model
receipt={'status':'passed','scope':'checkpoint_integrity_only','cuda_used':False,
         'source_data_access':False,'prediction_or_metric_scoring':False,'checked':checked}
t.write_json(receipt_path,receipt)
print(json.dumps({'status':'passed','checkpoints':len(checked)}))
