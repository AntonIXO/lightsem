import json
import sys
from pathlib import Path
import numpy as np
from lightsem.experiments.light_layers import training as t, evaluation as e
manifest_path, transform_path, exclusions_path, output_path, experiment_id = sys.argv[1:]
m=json.loads(Path(manifest_path).read_text())
transforms=json.loads(Path(transform_path).read_text())
scope=t.validate_run_scope(m,t.TrainConfig(),observer_proxy=True)
e.validate_transforms(transforms)
store=t.SourceStore(m)
tempos={}
for rid in store.rows:
    period=store.audio(rid)[:,64+t.AUDIO_SCALARS.index('beat_period_s')]
    if not np.isfinite(period).all() or np.any(period<=0):
        raise RuntimeError('Invalid donor tempo source')
    tempos[rid]=60/float(np.median(period))
donors=e.donor_map(m['records'],tempos)
targets={}
for row in m['records']:
    if row['partition'] in ('val','dev'):
        labels=e.freeze_labels(store.visual(row['recording_id']),transforms)
        targets[row['recording_id']]=labels.content_hash()
registration={'protocol_id':e.PROTOCOL_ID,'scope':e.SCOPE,'registered_before_fit':False,
    'authorization_status':'pending_parent_vault_binding','vault_experiment_id':experiment_id,
    'authorized_phases':['prepare-baselines','predict','score'],
    'scope_sha256':scope['scope_sha256'],
    'bindings':{'definition_sha256':t.json_hash(e.evaluation_definition()),
                'implementation_sha256':t.json_hash(e.implementation_hashes()),
                'manifest_sha256':t.sha256(manifest_path),'transforms_sha256':t.sha256(transform_path)},
    'donor_tempos_bpm':tempos,'donor_map':donors,'donor_map_sha256':t.json_hash(donors),
    'donor_audit':e.donor_audit(m['records'],donors),'target_hashes':targets,
    'original_exclusions':json.loads(Path(exclusions_path).read_text())['original_exclusions'],
    'original_exclusion_ledger':{'path':exclusions_path,'sha256':t.sha256(exclusions_path)},
    'checkpoints':{}}
t.write_json(output_path,registration)
print(json.dumps({'status':'candidate_binding_ready','target_groups':len(targets),
                  'scope_sha256':scope['scope_sha256'],'path':output_path,'sha256':t.sha256(output_path)}))
