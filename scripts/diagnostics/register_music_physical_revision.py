import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,n,a):
 r=c.call(n,a)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(str(r))
 return r
before=hashes();c=Client();d=json.loads((OUT/'design.json').read_text())
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 call(c,'register_diagnostic',{'id':'motion-music-generated-sliding-excess','name':'Generated-contact foot sliding excess','direction':'lower_better','unit':'metres/second','description':'Generated feet speed selected by generated contacts on adjacent frames minus reference-label source foot speed; six draws, fixed119VALwindows, canonical/record/window weighting. Report coverage separately; no music-fit implication.'})
 h=call(c,'propose_hypothesis',{'id':'h-dance-physical-objective-20260911','problem_id':'dancing-light-generative-20260903','statement':'Explicit clean-estimate FK/support/root objectives reduce generated sliding on held-out native motion versus a matched masked-native flow baseline without losing independent quality or conditioning gates.','rationale':'Existing native target objective cheaply admits root drift and is not invariant to geometry-equivalent6D scales. Audio is responsive but useful VAL generalization weak. Fifteen physical and15revision CPU tests pass; discarded32step/arm TRAIN-only GPU preflight finite with identical initialization/draws and exact prefetch parity. All128TRAIN/109songs,413quiet2s source windows inventoried. Architecture is held fixed to test physical objective, not assume stronger conditioning.','predicted_effects':[{'diagnostic_id':'motion-music-generated-sliding-excess','direction':'down','magnitude_note':'At least20percent reduction versus matched masked-native final arm; absolute excess<=.20m/s and all independent gates remain required.'}],'gates':[{'id':k,'description':v} for k,v in d['gates'].items()],'plan':json.dumps(d),'ambition_class':'mechanism'})
 e=call(c,'start_experiment',{'id':'exp-dance-physical-objective-20260911','hypothesis_id':'h-dance-physical-objective-20260911','external_run':str(OUT),'config_note':json.dumps(d)})
 assert hashes()==before
 (OUT/'registration.json').write_text(json.dumps({'hypothesis':h,'experiment':e,'unrelated_unchanged':True,'unrelated_hashes':before},indent=2));print(e['id'])
finally:c.close()
