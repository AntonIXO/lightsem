import json,hashlib
from pathlib import Path
from log_to_vault import Client
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z');r=json.loads((out/'numeric_report.json').read_text());reg=json.loads((out/'registration.json').read_text());eid=reg['experiment']['id']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 v=c.call(n,a)
 if not isinstance(v,dict) or v.get('ok') is False:raise RuntimeError(str(v))
 return v
before=hashes();c=Client();receipts={}
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 receipts['diagnostics']=call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':'companion-active-angular-speed-ratio','before':1.,'after':r['active_speed_ratio']}]})
 evidence={
 'mechanics':'All4arms full277.3s decoded speed<=90 and acceleration<=360 simulatorcaps. Candidate83.5749/354.9845. Not hardware certification.',
 'source':'Saved full world unchanged; incumbent pose_stage baseline recomputation byte-exact. No inference, training, retiming or source edits.',
 'photometry':'All non-angle DMX bytes equal across baseline,companion,shiftedclock,staticstyle. Actual pixel brightness differs and will be matched by one global downward scalar for comparison.',
 'independence':'13synthetic tests pass including stationary body emitted90/150bpm period changes; no-confidence and quiet targets have no autonomous sweeps. FullShpongle shiftedclock angleRMS5.40005deg, staticstyle targetRMS.253585m.',
 'quiet':'0-20s median baseline2.816060 vs candidate2.510120deg/s <=max(3,baseline*1.2).',
 'timing':'Synthetic90/120/150bpm and100->140ramp median clock errors8.31/8.74/8.31/9.23ms <=60ms. Real detector self-evidence.439758 atclock vs.392291 halfbeatshift is diagnostic only, not independent beat annotations or human musicfit.'}
 receipts['gates']=call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':True,'evidence':v} for k,v in evidence.items()]})
 receipts['update']=call(c,'update_problem',{'id':'dancing-light-generative-20260903','add_update':'2026-09-12 rhythm companion: user explicitly prioritizes LIGHT; dance diagnostic stays unchanged. New procedural2/4/8beat light phrases use motion spatialstyle and direct audio clock. Preregistered active speed>=1.25ratio FAILED:20.783433->23.757422deg/s (1.143094ratio). This failure is retained; direction-only record_diagnostics is NOT proof of committed magnitude. The magnitude was in the original frozen plan/prediction note but not a separately machine-enforced gate; do not auto-close as fully supported. Current source/mechanics/quiet/photometry/synthetictiming checks pass within scope. Six1080pAV1 previews fully decode with uniformPTS, exactframes and pairidenticalPCM. Sampled preview shows wider independent light gestures, no humanAVacceptance. Full277.3s native1080pAV1 delivery running; media gate pending. Artifacts: '+str(out)})
 assert hashes()==before;receipts['unrelated_vault_unchanged']=True
 (out/'numeric_vault_receipt.json').write_text(json.dumps(receipts,indent=2));print('RECORDED',eid)
finally:c.close()
