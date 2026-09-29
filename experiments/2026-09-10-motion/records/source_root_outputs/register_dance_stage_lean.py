import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_03_lean')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();design=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-light-torso-projected-band','name':'Torso midshaft projected1-2Hz RMS motion','direction':'higher_better','unit':'pixels','description':'Mean two torso-head geometric midshaft projected1-2Hz RMS on70-90s; fixed640x360 legacycamera .85scale. Welch40Hz,400Hann,linear detrend,integrate1<=f<2. Not rendered pixel tracking/music fit.'})
 h=call(c,'propose_hypothesis',{'id':'h-dance-stage-lean-20260911','problem_id':'dancing-light-generative-20260903','statement':'Explicit signed root-up lean restores projected torso movement omitted by turn-separated regional mapping while retaining separate bounded lanes.','rationale':'Stage02 numeric gates pass, but full rotated-neutral subtraction erases rigid-body roll and pitch direction: opposite30deg controls match within2e-15m. New optional2m root-up mapping distinguishes opposite leans by>1.9m in synthetic tests, keeps static input static and span>=8.2m. No real candidate yet.','predicted_effects':[{'diagnostic_id':'motion-light-torso-projected-band','direction':'up','magnitude_note':'Mean torso midshaft1-2Hz geometric motion >=2x matched baseline and>=1px at unchanged camera; independent spatial/surface/mechanics/quiet/visual gates.'}],'gates':[{'id':k,'description':v} for k,v in design['gates'].items()],'plan':json.dumps(design),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-stage-lean-20260911','hypothesis_id':'h-dance-stage-lean-20260911','parent_experiment_id':'exp-dance-stage-anchors-20260911','external_run':str(OUT),'config_note':json.dumps(design)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2))
 print(json.dumps(e,indent=2))
finally:c.close()
