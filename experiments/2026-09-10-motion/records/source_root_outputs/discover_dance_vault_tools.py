import json
from pathlib import Path
from log_to_vault import Client
c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 c._id+=1;c._send({'jsonrpc':'2.0','id':c._id,'method':'tools/list','params':{}});r=c._read()
 tools={t['name']:t for t in r['result']['tools']}
 Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/vault_tools_final.json').write_text(json.dumps(tools,indent=2))
 print(json.dumps({k:tools[k] for k in ('leaves','get_experiment','get_problem')},indent=2))
finally:c.close()
