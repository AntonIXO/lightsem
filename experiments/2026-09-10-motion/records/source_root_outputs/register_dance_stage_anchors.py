import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_02')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();design=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-light-active-wall-fraction','name':'Active back-wall first-hit head-frame fraction','direction':'higher_better','unit':'fraction','description':'Fraction of6 moving-head first finite hits that reach back wall on70-90s at40Hz. Geometric first-hit diagnostic, not pixel lighting or musical fit.'})
 h=call(c,'propose_hypothesis',{'id':'h-dance-stage-anchors-20260911','problem_id':'dancing-light-generative-20260903','statement':'Repositioning only static hand/torso role anchors lets the same generated gestures use both floor and wall while retaining separated changing trajectories.','rationale':'Stage01 fixes contraction10.61m and dynamic checks but wall fraction.009583 fails>=.10. Raw wall1.125% places failure in chart anchors, not actuator. New static bases7.7/3.2/8.0 are a declared geometry recipe, not temporal invention. Original fixed anchors/results preserved; no candidate mapped yet.','predicted_effects':[{'diagnostic_id':'motion-light-active-wall-fraction','direction':'up','magnitude_note':'Wall head-frame fraction .009583->>=.10, retaining floor>=.10, span, dynamic, quiet and simulator mechanical gates.'}],'gates':[{'id':k,'description':v} for k,v in design['gates'].items()],'plan':json.dumps(design),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-stage-anchors-20260911','hypothesis_id':'h-dance-stage-anchors-20260911','parent_experiment_id':'exp-dance-stage-geometry-20260911','external_run':str(OUT),'config_note':json.dumps(design)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2))
 print(json.dumps(e,indent=2))
finally:c.close()
