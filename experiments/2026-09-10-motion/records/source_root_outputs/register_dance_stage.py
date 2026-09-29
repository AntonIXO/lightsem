import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_01')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();design=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-light-turn-interval-span','name':'First-surface span during recorded turn contraction','direction':'higher_better','unit':'metres','description':'Median first finite floor/back-wall hit x span over194-201s, all40Hz frames. Canonical rig; truss occlusion not represented. Not image support or choreography quality.'})
 h=call(c,'propose_hypothesis',{'id':'h-dance-stage-geometry-20260911','problem_id':'dancing-light-generative-20260903','statement':'Turn-separated bounded regional geometry and a floor/back-wall chart prevent the recorded contraction while retaining changing surface trajectories.','rationale':'Independent audit isolated amplified (Q-I)*neutral contraction and floor-only projection. Fixed synthetic geometry passes8 tests; root-only yaw retains9.5m span with explicit facing cue, raised hands reach wall, finite first-hit DMX roundtrip within2mm on tested rays. Two compose integration tests pass; no real stage candidate measured.','predicted_effects':[{'diagnostic_id':'motion-light-turn-interval-span','direction':'up','magnitude_note':'Recorded contraction median first-hit x span increases to>=6m versus fixed actuator-budget comparator. Independent motion, surface, quiet, mechanics and visual gates.'}],'gates':[{'id':k,'description':v} for k,v in design['gates'].items()],'plan':json.dumps(design),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-stage-geometry-20260911','hypothesis_id':'h-dance-stage-geometry-20260911','parent_experiment_id':'exp-dance-actuator-budget-20260911','external_run':str(OUT),'config_note':json.dumps(design)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2))
 print(json.dumps(e,indent=2))
finally:c.close()
