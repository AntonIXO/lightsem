"""Post-result compatibility repair; original study, maps and failure retained."""
import importlib.util,sys,time,copy,shutil,json
from pathlib import Path
PATH=Path('/root/lightsem/outputs/review_music_physical_support_revision.py')
s=importlib.util.spec_from_file_location('support_review',PATH);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
OLD=m.EXPERIMENT/'visual_review';NEW=m.DEFAULT;assert not NEW.exists()
p=m.loadjson(OLD/'protocol.json');f=m.loadjson(OLD/'freeze.json')
for rel,sha in f['source_files'].items():m.checked(OLD/'source'/rel,sha);m.writebytes(NEW/'source'/rel,(OLD/'source'/rel).read_bytes())
for path,sha in f['inputs'].items():m.checked(path,sha)
m.writebytes(NEW/'reviewer.py',PATH.read_bytes())
revision={'type':'post-result engineering compatibility repair','original_protocol_sha256':m.digest(OLD/'protocol.json'),'original_freeze_sha256':m.digest(OLD/'freeze.json'),'original_reviewer_sha256':m.digest(OLD/'reviewer.py'),'original_mapping_failure':'Legacy record027 last175.275s vs native last175.333333s; full-record guard failed although requested140-159.9667s is entirely supported. Original partial directory and log retained.','policy':'Interpolate old2D only inside strict archived timestamp hull. Save validity mask and zero storage outside. Render explicit unavailable marker outside support, never an invented pose. All new-model/native-world/DMX/clip/map/optics settings unchanged.','concurrency':'Serialize map/render operations sharing this visual output lock. Prior renderer failed before any render directory existed.','new_validation_results_already_exist':True,'model_checkpoint_selection':None}
p2=copy.deepcopy(p);p2['engineering_revision']=revision
m.savejson(NEW/'protocol.json',p2)
f2=copy.deepcopy(f);f2.update(created_unix=time.time(),script_sha256=m.digest(NEW/'reviewer.py'),protocol_sha256=m.digest(NEW/'protocol.json'),pre_real_evaluation=False,original_plan_pre_real_evaluation=True,engineering_revision=revision)
f2['inputs'].update({str(OLD/'protocol.json'):m.digest(OLD/'protocol.json'),str(OLD/'freeze.json'):m.digest(OLD/'freeze.json'),str(OLD/'reviewer.py'):m.digest(OLD/'reviewer.py'),str(m.EXPERIMENT/'legacy_support_audit.json'):m.digest(m.EXPERIMENT/'legacy_support_audit.json')})
m.savejson(NEW/'freeze.json',f2)
adopted=[]
for item in p['items']:
 source=OLD/'mapped'/item['id'];path=source/'map.json'
 if not path.exists():continue
 r=m.loadjson(path);assert r['item']==item and r['freeze_sha256']==m.digest(OLD/'freeze.json')
 dest=NEW/'mapped'/item['id'];dest.mkdir(parents=True)
 for rel,sha in r['files'].items():m.checked(source/rel,sha);m.writebytes(dest/rel,(source/rel).read_bytes())
 rr=copy.deepcopy(r);rr['freeze_sha256']=m.digest(NEW/'freeze.json');rr['adopted_from']={'path':str(path),'sha256':m.digest(path),'original_freeze_sha256':m.digest(OLD/'freeze.json'),'scope':'Core arrays reused byte-for-byte; old2D temporal support is independently masked at render under revised policy.'}
 m.savejson(dest/'map.json',rr);adopted.append(item['id'])
m.savejson(NEW/'adoption.json',{'adopted_completed_maps':adopted,'original_failures_retained':True,'source_files_unchanged':True,'numeric_evaluation_unchanged':True,'revision':revision})
print('SUPPORT REVISION PREPARED; adopted',len(adopted),flush=True)
