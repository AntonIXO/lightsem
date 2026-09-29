import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01');DEST=OUT/'vibescience_final_result.json'
assert not DEST.exists()
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 eid='exp-dance-physical-objective-20260911'
 leaves=call(c,'leaves',{'unevaluated_only':True});assert any(x['experiment_id']==eid and not x['closed'] for x in leaves['leaves'])
 report=OUT/'visual_review_support_revision_01/review_report.json';r=json.loads(report.read_text())['payload'];assert r['all13_rendered'] and r['all13_observer_receipts'] and not r['pipeline_qualified']
 note=Path('/root/lightsem/outputs/dance_physical_final_outcome.md').read_text()
 gate=call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':'visual','passed':False,'evidence':str(report)+'; all13 scheduled renders/sample reviews complete, but matched incumbent lighting unavailable; no overall visual/music-fit acceptance established. See outputs/dance_physical_final_outcome.md for instrument scope.'}]})
 result=call(c,'close_experiment',{'experiment_id':eid,'notes':note})
 assert hashes()==before
 DEST.write_text(json.dumps({'result':result,'visual_gate':gate,'scope':'This fixed recipe only: physical sliding improvement; conditioning and absolute sliding fail. No pipeline qualification.','unrelated_vault_unchanged':True,'unrelated_hashes':before,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()},indent=2))
 print(eid,result['verdict'])
finally:c.close()
