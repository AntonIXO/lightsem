import json,hashlib
from pathlib import Path
from log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 result=c.call('update_problem',{'id':'dancing-light-generative-20260903','add_update':'User continuation after spatial correction: collapse is fixed but lights still do not dance; diagnose and make the pipeline work. Original rev02 visual failure and failed native-music qualification remain unchanged. Unpreregistered post-hoc analysis of all119 saved native VAL window summaries (28 canonical songs; no new inference/TEST) separates each of18 physical channels. Flow correct/shift per-channel energy score for hand speed is .940/.942(left), .923/.931(right); hand-spread .730/.732. These nearly unchanged scores are consistent with the frozen weak local-conditioning finding, not proof of zero information or a new qualification test. Regression constant-in-time generated-summary controls score better than original sequences in every channel; flow constant controls score worse in every channel. Thus this instrument does distinguish flow temporal variation from constancy, but this is not evidence of musical choreography. Exact values and scope: '+str(OUT/'summary_channel_diagnosis.json')+'. Independent mapper and generator mechanism audits are running. No training, defaults, original evidence or thresholds changed.'})
 assert isinstance(result,dict) and result.get('ok') is not False,result
 after=hashes();assert before==after,'Unrelated vault changed'
 (OUT/'diagnosis_vault_update.json').write_text(json.dumps({'result':result,'unrelated_unchanged':True,'unrelated_hashes':before},indent=2))
 print('Unpreregistered diagnosis recorded; unrelated vault unchanged')
finally:c.close()
