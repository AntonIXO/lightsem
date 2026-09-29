import json
from pathlib import Path
from log_to_vault import Client
c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 data={}
 for name,args in [('recall',{'query':'surface dancing motion'}),('get_problem',{'id':'dancing-light-generative-20260903'}),('leaves',{'problem_id':'dancing-light-generative-20260903'})]:
  data[name]=c.call(name,args)
 Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/vault_before.json').write_text(json.dumps(data,indent=2))
 print(json.dumps(data,indent=2)[-28000:])
finally:c.close()
