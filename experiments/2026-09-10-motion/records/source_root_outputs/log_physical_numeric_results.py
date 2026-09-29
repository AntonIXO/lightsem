import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
path=OUT/'evaluation/numerical_report.json';env=json.loads(path.read_text());r=env['payload']
assert hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==env['sha256']
G=r['gates'];gates={'sliding':G['sliding_absolute'] and G['sliding_reduction_vs_matched_native'],'conditioning':G['conditioning_magnitude'] and G['conditioning_ci'],**{k:G[k] for k in ('energy','travel','contacts','floor','high_frequency','continuity','short_holds','finite')}}
eid='exp-dance-physical-objective-20260911';before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 effects=call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':'motion-music-generated-sliding-excess','before':r['matched_native_quality']['sliding_excess'],'after':r['physical_quality']['sliding_excess']}]})
 gate_result=call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':v,'evidence':str(path)} for k,v in gates.items()]})
 assert hashes()==before
 (OUT/'numeric_vault_receipt.json').write_text(json.dumps({'effects':effects,'gates':gate_result,'unrelated_vault_unchanged':True,'unrelated_hashes':before,'scope':'Completed frozen numerical evaluation only. Visual remains pending; experiment NOT closed and no pipeline/default qualification.','numeric_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2));print('Numerical results recorded; visual pending; experiment left open')
finally:c.close()
