import json
from pathlib import Path
from log_to_vault import Client
c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 r=c.call('leaves',{})
 assert not isinstance(r,str),r
 Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/leaves.json').write_text(json.dumps(r,indent=2))
 print(json.dumps(r,indent=2)[-14000:])
finally:c.close()
