import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/timing_revision_01')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
m=json.loads((OUT/'measurements.json').read_text());before=hashes();c=Client();eid='exp-dance-tracking-20260911'
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':'motion-light-requested-emitted-lag','before':.45,'after':m['candidate']['segments']['active']['lag']['lag_s']}]})
 gs=m['gates'];gs['visual']=False
 call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':v,'evidence':str(OUT/('review.json' if k=='visual' else 'measurements.json'))} for k,v in gs.items()]})
 r=call(c,'close_experiment',{'experiment_id':eid,'notes':'Scope: requested-motion-to-emitted-DMX timing on saved Shpongle only. Lag.450->.225s improves direction but fails<=.20s;1-2Hz amplitude.243->.301 fails>=.35. Hard mechanics/quiet/photometry/spatial pass. Full synchronized comparison rendered and decoded; all278 chronological1Hz samples inspected, not continuous audiovisual viewing; floor fan and local crown remain, visual acceptance false. No native generator qualification or product fix.'})
 assert hashes()==before
 (OUT/'vibescience_results.json').write_text(json.dumps({'result':r,'unrelated_unchanged':True,'unrelated_hashes':before},indent=2));print(r)
finally:c.close()
