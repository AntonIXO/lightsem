"""Assemble capture shards without copying features; no fitting or GPU work."""
import argparse
import json
from pathlib import Path
from dataclasses import asdict
from lightsem.experiments.light_layers import training as t

p=argparse.ArgumentParser()
p.add_argument('--plan',type=Path,required=True)
p.add_argument('--protocol',type=Path,required=True)
p.add_argument('--identity',type=Path,required=True)
p.add_argument('--split',type=Path,required=True)
p.add_argument('--definition',type=Path,required=True)
p.add_argument('--implementation',type=Path,required=True)
p.add_argument('--cpu-receipt',type=Path,required=True)
p.add_argument('--experiment-id',required=True)
p.add_argument('--manifest-output',type=Path,required=True)
p.add_argument('--binding-output',type=Path,required=True)
p.add_argument('--capture-exclusions',type=Path)
a=p.parse_args()
if a.manifest_output.exists() or a.binding_output.exists():
    raise RuntimeError('No overwrite of manifest or binding artifacts')
plan=json.loads(a.plan.read_text())
protocol=json.loads(a.protocol.read_text())
identity=json.loads(a.identity.read_text())
implementation=json.loads(a.implementation.read_text())
original_path=Path(protocol['selection']['manifest'])
if t.sha256(original_path)!=protocol['selection']['sha256']:
    raise RuntimeError('Original capture manifest drift')
original=json.loads(original_path.read_text())
inventory={r['recording_id']:r for r in original['inputs']}
locations=plan['record_locations']
if set(locations)!=set(inventory):
    raise RuntimeError('Aggregate locations must cover the exact original inventory')
for worker in plan['workers']:
    marker=Path(worker['output'])/'complete.json'
    value=json.loads(marker.read_text())
    result_path=marker.with_name('results.json')
    if value['results_sha256']!=t.sha256(result_path):
        raise RuntimeError('Shard completion hash mismatch')
additional=[] if a.capture_exclusions is None else json.loads(a.capture_exclusions.read_text())
excluded={rid:row['reason'] for rid,row in identity['exclusions'].items()}
for row in additional:
    if row['recording_id'] in excluded:
        raise RuntimeError('Duplicate exclusion')
    excluded[row['recording_id']]=row['reason']
rows=[]
assembled=None
for parent in sorted({str(Path(path).parent) for path in locations.values()}):
    group={rid for rid,path in locations.items() if str(Path(path).parent)==parent}
    skip={rid:excluded.get(rid,'Located in a different immutable capture shard')
          for rid in inventory if rid not in group or rid in excluded}
    part=t.build_training_manifest(original_path,parent,a.split,mode='sam2',exclusions=skip)
    if assembled is None:
        assembled=part
    rows.extend(part['records'])
order={rid:i for i,rid in enumerate(inventory)}
assembled['records']=sorted(rows,key=lambda row:order[row['recording_id']])
def ref(path):
    return {'path':str(path.resolve()),'sha256':t.sha256(path)}
assembled.update(study_kind=t.OBSERVER_STUDY_KIND,
    unknown_cuts_policy=t.OBSERVER_UNKNOWN_CUTS_POLICY,
    observer_protocol=ref(a.protocol),identity_exclusions=ref(a.identity),
    capture_exclusions=additional,capture_location_plan=ref(a.plan),
    fit_definition=ref(a.definition),
    smoke_selection={'original_recording_ids':list(inventory),'exclusions':excluded,
                     'replacement_records':False,'scope':'immutable_capture_inventory_not_a_smoke'})
scope=t.validate_run_scope(assembled,t.TrainConfig(),observer_proxy=True)
store=t.SourceStore(assembled)
if t.json_hash(t.implementation_hashes())!=implementation['training_source_sha256']:
    raise RuntimeError('Training code differs from frozen implementation')
t.write_json(a.manifest_output,assembled)
bindings=t.provenance(a.manifest_output,store,t.TrainConfig(),scope=scope)
bindings.pop('config_sha256')
bindings.update(campaign_queue_source_sha256=implementation['queue_source_sha256'],
    cpu_test_receipt_sha256=t.sha256(a.cpu_receipt),
    fit_definition_sha256=t.sha256(a.definition),
    evaluation_definition_sha256=implementation['evaluation_definition_sha256'])
registration={'registered_before_fit':False,'authorization_status':'pending_parent_vault_data_binding',
    'vault_experiment_id':a.experiment_id,'study_kind':t.OBSERVER_STUDY_KIND,
    'unknown_cuts_policy':t.OBSERVER_UNKNOWN_CUTS_POLICY,'bindings':bindings,
    'recipes':{r:{'config_sha256':[t.json_hash(asdict(t.TrainConfig(recipe=r,seed=s)))
                                for s in (0,1,2)]} for r in t.OBSERVER_RECIPES}}
t.write_json(a.binding_output,registration)
print(json.dumps({'manifest':ref(a.manifest_output),'candidate_binding':ref(a.binding_output),
                  'usable_records':len(rows),'scope':scope},indent=2))
