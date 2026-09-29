import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_full_controls_01')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();design=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-light-projected-motion-radius','name':'Projected motion to cone radius','direction':'higher_better','unit':'ratio','description':design['primary']})
 h=call(c,'propose_hypothesis',{'id':'h-dance-optics-full-20260911','problem_id':'dancing-light-generative-20260903','statement':'A fixed closer camera and angularly narrow, integral-normalized head cone increase observability of saved emitted movement, separate from generated music quality.','rationale':design['preview_provenance'],'predicted_effects':[{'diagnostic_id':'motion-light-projected-motion-radius','direction':'up','magnitude_note':'Combined closer_zero_angular_flux>=2x legacy_native_legacy under frozen geometric instrument; independent pixel/clipping/brightness and coverage controls.'}],'gates':[{'id':k,'description':v} for k,v in design['gates'].items()],'plan':json.dumps(design),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-optics-full-20260911','hypothesis_id':h['id'],'parent_experiment_id':'exp-dance-stage-anchors-20260911','external_run':str(OUT),'config_note':json.dumps(design)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2));print(e['id'])
finally:c.close()
