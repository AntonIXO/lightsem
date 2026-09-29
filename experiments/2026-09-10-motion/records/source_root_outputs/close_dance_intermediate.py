import json,hashlib
from pathlib import Path
from log_to_vault import Client
BASE=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911')
AUDIT=BASE/'visual_audit'
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
cases=[('timing_revision_02','exp-dance-projection-20260911','motion-light-requested-emitted-lag','lag'),('actuator_budget_01','exp-dance-actuator-budget-20260911','motion-light-angular-band-amplitude','band'),('stage_geometry_01','exp-dance-stage-geometry-20260911','motion-light-turn-interval-span','span')]
notes={
 'lag':'Scope requested-motion-to-DMX on saved Shpongle, not audio. Signed lag0 versus.45s; active1-2Hz amplitude.243->.221 fails. Exact original software mechanics preserved. Visual audit retains crown/floor row. No overall qualification.',
 'band':'Scope simulated actuator-budget counterfactual, not hardware safety or music learning. Same targets/convex solver: active1-2Hz amplitude.221->.790; zero signed lag; declares90/360 rather than30/80. Old speed/accel caps exceeded on28.29/71.82percent of samples. Numerical gates pass; crown/row visual acceptance fails. No default promotion.',
 'span':'Scope turn-safe stage geometry on same saved body and90/360 simulator, not music or hardware. Recorded contraction span rises to10.610m, motion/validity/quiet/photometry pass. Wall fraction.00958 fails.10; raw1.125percent locates surface-family shortfall before actuator. Visually wider/separated194-201s but row-family remains. No full acceptance.'}
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 for name,eid,diag,kind in cases:
  root=BASE/name;m=json.loads((root/'measurements.json').read_text())
  def val(arm):
   r=m[arm]['segments']['contraction' if kind=='span' else 'active']
   return r['lag']['lag_s'] if kind=='lag' else (r['band_amplitude_ratio']['1-2'] if kind=='band' else r['spatial']['hit_x_span']['median'])
  call(c,'record_diagnostics',{'experiment_id':eid,'measurements':[{'diagnostic_id':diag,'before':val('baseline'),'after':val('candidate')}]})
  gs=m['gates'];gs['visual']=False
  call(c,'record_gate_results',{'experiment_id':eid,'results':[{'gate_id':k,'passed':v,'evidence':str(AUDIT if k=='visual' else root/'measurements.json')} for k,v in gs.items()]})
  result=call(c,'close_experiment',{'experiment_id':eid,'notes':notes[kind]+' Independent audit reviewed all278chronological1Hz samples for this movie plus exact stills; all8319encoded frames decode,8320was generated receipt count. No continuous human audiovisual viewing/music-fit score. See outputs/dance_visual_audit.md.'})
  (root/'vibescience_results.json').write_text(json.dumps({'result':result,'scope':notes[kind]},indent=2));print(eid,result['verdict'],flush=True)
 assert hashes()==before
 (BASE/'intermediate_close_integrity.json').write_text(json.dumps({'unrelated_unchanged':True,'hashes':before},indent=2))
finally:c.close()
