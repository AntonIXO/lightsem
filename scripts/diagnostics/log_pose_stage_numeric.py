import json,hashlib
from pathlib import Path
from scripts.vault.log_to_vault import Client
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01');DEST=OUT/'numeric_vault_receipt.json';assert not DEST.exists()
def auth(path):
 x=json.loads(Path(path).read_text());h=hashlib.sha256(json.dumps(x['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert h==x['sha256'];return x['payload']
def unrelated():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/root/vibescience-vault').rglob('*') if p.is_file()}
def call(c,name,args):
 r=c.call(name,args)
 if not isinstance(r,dict) or r.get('ok') is False:raise RuntimeError(repr(r))
 return r
report=auth(OUT/'numeric_report.json');assert report['all_maps_complete'] and report['all_numeric_gates_pass']
shpo=auth(OUT/'mapped/shpongle_prior_flow0/map.json');refs=auth(OUT/'references/stage.json');assert refs['pass'] and refs['wins']==8
b=shpo['reports']['baseline']['primary'];a=shpo['reports']['candidate']['primary']
reg=json.loads((OUT/'registration.json').read_text());plan=json.loads(reg['hypothesis']['plan']);assert reg['experiment']['id']=='exp-dance-pose-stage-20260911'
keys=['primary','mechanics','valid','surfaces','quiet','nonangle','collapse','oracle_transfer'];assert all(k in plan['gates'] for k in keys)
before=unrelated();c=Client()
try:
 assert c.initialize()['name']=='vibescience-lightsem'
 diagnostics=call(c,'record_diagnostics',{'experiment_id':reg['experiment']['id'],'measurements':[{'diagnostic_id':'motion-light-broadband-motion-radius','before':b,'after':a}]})
 gates=call(c,'record_gate_results',{'experiment_id':reg['experiment']['id'],'results':[{'gate_id':k,'passed':True,'evidence':str(OUT/'numeric_report.json')+'; frozen exact-edge .5-4Hz per-head geometric ratio, fixed saved bodies; no music/visual acceptance.'} for k in keys]})
 note=f'Pose-stage frozen numerical stages completed on saved priorflow0 Shpongle and12fixed reference windows, no new inference. NEW exact-edge .5-4Hz emitted projected midshaft motion/radius {b}->{a} ({a/b}x) vs same-body stage02 in70-90s; all registered numeric gates pass.8/8travel/coord windows improve versus required6/8; holds reported, not rewarded for more motion. Same quiet median2.816059829deg/s; simulator90/360 constraints, finite surfaces, nonangle parity and selected all-head-cloud checks pass. Pair overlaps are not all-head collapse. Fullmedia rendered but independent visual/PTS/audio audits stillpending; pixels/coverage/brightness gates NOTlogged by this numeric step. Experiment remains OPEN, pipelineNOTqualified, no humanAV/music acceptance. Physical-objective conditioning and absolute sliding failures are unchanged.'
 update=call(c,'update_problem',{'id':'dancing-light-generative-20260903','add_update':note})
 assert unrelated()==before
 DEST.write_text(json.dumps({'diagnostics':diagnostics,'gates':gates,'problem_update':update,'scope':note,'unrelated_vault_unchanged':True,'numeric_report_sha256':hashlib.sha256((OUT/'numeric_report.json').read_bytes()).hexdigest()},indent=2));print('Logged numeric gates only; experiment remains open.')
finally:c.close()
