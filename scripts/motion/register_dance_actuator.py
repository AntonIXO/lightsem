import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/actuator_budget_01')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();design=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-light-angular-band-amplitude','name':'Active1-2Hz emitted to requested angular RMS amplitude','direction':'higher_better','unit':'ratio','description':'Square-root Welch1-2Hz summed12channel angular power ratio emitted/requested on70-90s,40Hz,10s Hann windows. Descriptive nonlinear trajectory ratio, not audio/music alignment.'})
 h=call(c,'propose_hypothesis',{'id':'h-dance-actuator-budget-20260911','problem_id':'dancing-light-generative-20260903','statement':'Higher declared simulated actuator limits restore more fast motion under the same full-track convex target-fitting objective.','rationale':'Full-track convex fit has0 signed lag but only.221143 active1-2Hz amplitude under30/80 software caps. Rig definitions provide pan/tilt ranges, no measured speed/acceleration hardware caps. New90/360 simulation-only profile tests the actuator budget, not physical safety. No new candidate measured.','predicted_effects':[{'diagnostic_id':'motion-light-angular-band-amplitude','direction':'up','magnitude_note':'Active1-2Hz ratio .221143 -> >=.45; same raw requests; independent timing, quiet, spatial and simulator mechanics gates.'}],'gates':[{'id':k,'description':v} for k,v in design['gates'].items()],'plan':json.dumps(design),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-actuator-budget-20260911','hypothesis_id':'h-dance-actuator-budget-20260911','parent_experiment_id':'exp-dance-projection-20260911','external_run':str(OUT),'config_note':json.dumps(design)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2))
 print(json.dumps(e,indent=2))
finally:c.close()
