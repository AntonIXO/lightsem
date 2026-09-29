from log_to_vault import Client
from pathlib import Path
import json
p=json.loads(Path('/root/lightsem/docs/research/2026-09-28-end-to-end/ROW65_NATIVE_SCOPE_SUPPLEMENT_DEFINITION_V1.json').read_text());c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 request=p['payload'];r=c.call('interpret_experiment',request)
 with Path('/root/lightsem/docs/research/2026-09-28-end-to-end/ROW65_NATIVE_SCOPE_SUPPLEMENT_RECEIPT_V1.json').open('x') as f:json.dump(r,f,indent=2)
 assert isinstance(r,dict) and r.get('experiment_id')==request['experiment_id'],repr(r)
 assert r.get('lesson')==request['lesson'] and r.get('failure_mode')==request['failure_mode'],repr(r)
 assert r.get('transfers_to')==request['transfers_to'] and r.get('verdict')==p['expected_verdict'],repr(r)
 print('Exact supplementary native interpretation verified',flush=True)
finally:c.close()
