import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
BASE=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911');OUT=BASE/'optics_full_controls_01';DEST=OUT/'vibescience_final_result.json'
assert not DEST.exists()
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
r=json.loads((OUT/'report.json').read_text());note=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_optics_final_outcome.md').read_text()
# Independent review files and all15 coverage receipt must exist; frozen report not rewritten.
evidence=[Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_optics_visual_'+k+'.md') for k in ('legacy_camera','closer_camera','matched_foil')]
assert all(p.exists() for p in evidence)
coverage=[json.loads(p.read_text()) for p in (OUT/'media_audit/coverage').glob('*.json')]
assert any((x.get('payload',x)).get('all_15_exact_media_checks_pass') is True for x in coverage)
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 eid='exp-dance-optics-full-20260911';leaves=call(c,'leaves',{'unevaluated_only':True});assert any(x['experiment_id']==eid and not x['closed'] for x in leaves['leaves'])
 arms=r['geometry']['arms']
 diag=call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':'motion-light-projected-motion-radius','before':arms['legacy_native_legacy']['active']['primary'],'after':arms['closer_zero_angular_flux']['active']['primary']}]})
 gate=call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':True,'evidence':'experiments/2026-09-10-motion/outputs/loose/dance_optics_final_outcome.md; '+str(OUT/'report.json')+'; independent optics_visual_audits_01. Music gate is explicitly a no-music-claim/held-source control, NOT music acceptance.'} for k in ('proxy','pixels','parity','coverage','brightness_control','music')]})
 result=call(c,'close_experiment',{'experiment_id':eid,'notes':note})
 lineage=call(c,'lineage',{'experiment_id':'exp-dance-stage-lean-20260911'})
 (BASE/'stage_lineage_final.json').write_text(json.dumps(lineage,indent=2))
 assert hashes()==before
 DEST.write_text(json.dumps({'result':result,'diagnostics':diag,'gates':gate,'scope':'Registered geometry/optics controls only; no music, preference, hardware or pipeline acceptance.','unrelated_vault_unchanged':True,'unrelated_hashes':before,'review_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}},indent=2))
 print(eid,result['verdict'])
finally:c.close()
