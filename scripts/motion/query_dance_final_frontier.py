import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
before=hashes();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 r=c.call('leaves',{'unevaluated_only':True})
 assert isinstance(r,(list,dict)) and not (isinstance(r,dict) and r.get('ok') is False),repr(r)
 Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/vault_final_frontier_before.json').write_text(json.dumps(r,indent=2))
 assert hashes()==before
 print(json.dumps(r,indent=2))
finally:c.close()
