import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/timing_revision_02')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();design=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-light-requested-emitted-lag','name':'Requested to emitted active angular lag','direction':'lower_better','unit':'seconds','description':'Pooled demeaned12channel correlation argmax nonnegative lags0..2s,40Hz active70-90s. Target-to-DMX timing, not audio alignment.'})
 h=call(c,'propose_hypothesis',{'id':'h-dance-projection-20260911','problem_id':'dancing-light-generative-20260903','statement':'Full-track convex angular fitting reduces requested-to-emitted delay while preserving hard quantized mechanics.','rationale':'Rev02 .45s lag; timing01 zero-phase/feedforward reduces it to.225s but fails. Synthetic feasible/infeasible/static tests pass for quantization-safe convex fit. OSQP preflight failed convergence; Clarabel0.11.1 passed before any real projection output. No candidate mapping yet.','predicted_effects':[{'diagnostic_id':'motion-light-requested-emitted-lag','direction':'down','magnitude_note':'Active0.450s -> <=0.10s; independent bandwidth and spatial/visual gates retained.'}],'gates':[{'id':k,'description':v} for k,v in design['gates'].items()],'plan':json.dumps(design),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-projection-20260911','hypothesis_id':'h-dance-projection-20260911','parent_experiment_id':'exp-dance-tracking-20260911','external_run':str(OUT),'config_note':json.dumps(design)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_vault_unchanged':True,'unrelated_hashes':before},indent=2))
 print(json.dumps(e,indent=2))
finally:c.close()
