import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
BASE=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911');OUT=BASE/'stage_geometry_02';DEST=OUT/'vibescience_final_result.json';assert not DEST.exists()
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
m=json.loads((OUT/'measurements.json').read_text());audit=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_stage_anchors_visual_audit.md');assert audit.exists()
note='Instrument versus claim: only static role-anchor changes on the saved Shpongle body, same simulator90/360 and photometry. Active wall fraction increases from '+str(m['baseline']['segments']['active']['wall_head_fraction'])+' to '+str(m['candidate']['segments']['active']['wall_head_fraction'])+'. Floor, raw/first-hit geometry, changing trajectories and all registered numeric gates pass. Full paired stage01 versus stage02 movie decoded8319frames; all278 chronological1Hz paired samples and10 exact stills inspected. This fulfills the explicitly sampled visual-coverage procedure, NOT human music-fit or aesthetic acceptance. Persistent fan/row remains. Stage02 redistributes already-separated targets; do not credit it with the earlier stage01 contraction rescue. Geometric wall fraction is not decoded coverage: ROI luma3.570->3.484 and Y>=16 area6.999->6.658percent in that audit. No hardware calibration, defaults, seed1, TEST, or pipeline promotion. See experiments/2026-09-10-motion/outputs/loose/dance_stage_anchors_visual_audit.md.'
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 eid='exp-dance-stage-anchors-20260911';lin=call(c,'lineage',{'experiment_id':'exp-dance-stage-lean-20260911'});assert any(x['experiment_id']==eid and not x['closed'] for x in lin['path'])
 call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':'motion-light-active-wall-fraction','before':m['baseline']['segments']['active']['wall_head_fraction'],'after':m['candidate']['segments']['active']['wall_head_fraction']}]})
 gates=dict(m['gates']);gates['visual']=True
 call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':v,'evidence':str(audit if k=='visual' else OUT/'measurements.json')+'; visual means planned sampled coverage only, not music/overall acceptance.'} for k,v in gates.items()]})
 result=call(c,'close_experiment',{'experiment_id':eid,'notes':note});assert hashes()==before
 DEST.write_text(json.dumps({'result':result,'scope':note,'unrelated_vault_unchanged':True,'unrelated_hashes':before,'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest()},indent=2));print(eid,result['verdict'])
finally:c.close()
