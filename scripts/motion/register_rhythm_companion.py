import json, hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
out=Path('/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z'); p=json.loads((out/'plan.json').read_text())
def hashes(): return {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('/root/vibescience-vault').rglob('*') if f.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False: raise RuntimeError(str(r))
 return r
before=hashes(); c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 d=call(c,'register_diagnostic',{'id':'companion-active-angular-speed-ratio','name':'Companion to pose-stage active coordinate speed ratio','unit':'ratio','direction':'higher_better','description':'Median emitted six-head pan/tilt coordinate speed at40Hz in fixedShpongle70-90s. Ratio against unchanged-body pose_stage baseline; movement amount, NOT synchronization or aesthetic preference.'})
 h=call(c,'propose_hypothesis',{'id':'h-'+p['id'],'problem_id':'dancing-light-generative-20260903','statement':'Separating slow motion-derived spatial style from direct musical timing yields more active emitted lighting than literal pose mapping without accelerating the diagnostic body.','predicted_effects':[{'diagnostic_id':d['id'],'direction':'up','magnitude_note':'Baseline ratio1, candidate>=1.25; independent timing, quiet, mechanics, source and media gates.'}],'rationale':'User accepts dance as a diagnostic and asks to fix light rhythm/static appearance. This is explicit procedural light composition using audio and saved internal motion, not a learned motion qualification or source retiming.','plan':json.dumps(p,sort_keys=True),'gates':[{'id':k,'description':v,'blocking':True} for k,v in p['gates'].items()],'ambition_class':'mechanism','gap_closure_estimate':.15})
 e=call(c,'start_experiment',{'id':'exp-'+p['id'],'hypothesis_id':h['id'],'git_ref':'01a14fe plus separately hashed new implementation','external_run':str(out),'config_note':json.dumps(p,sort_keys=True)})
 assert hashes()==before
 (out/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'diagnostic':d,'unrelated_vault_unchanged':True,'before_real_execution':True},indent=2))
 print(e['id'])
finally: c.close()
