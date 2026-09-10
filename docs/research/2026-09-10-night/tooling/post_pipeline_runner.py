"""Persistent serial follow-through after the immutable main pipeline finishes."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import authenticated_json,read_authenticated,digest,artifact_manifest,verify_artifacts
TOOLS=ROOT/'tooling';PY='/mnt/lightsem-data/venv-torch/bin/python';ENC='/mnt/lightsem-data/venv-night-encoders/bin/python'
STAGES=[
 ('beat_reference',ENC,'beat_this_reference.py',['run'],[ROOT/'rhythm/beat_this_reference/report.json'],True),
 ('muq_local_controls',PY,'muq_local_controls.py',['run'],[ROOT/'diagnostics/muq_local_controls/report.json'],True),
 ('pose_videos',PY,'pose_videos.py',['run'],[ROOT/'delivery/pose/report.json'],True),
 ('comparison',PY,'whole_track_delivery.py',['composite'],[ROOT/'delivery/shpongle/comparison.json'],True),
 ('whole_track_observe',PY,'whole_track_delivery.py',['observe'],[ROOT/'delivery/shpongle/whole_track_observation.json'],False),
 ('audit_inputs',PY,'independent_audit.py',['inputs'],[ROOT/'diagnostics/independent_input_audit.json'],False),
 ('audit_training',PY,'independent_audit.py',['training','--complete'],[ROOT/'diagnostics/independent_training_audit.json'],False),
 ('audit_pose',PY,'independent_audit.py',['poses'],[ROOT/'diagnostics/independent_pose_audit.json'],False),
 ('audit_rhythm',PY,'independent_audit.py',['rhythm'],[ROOT/'diagnostics/independent_rhythm_audit.json'],False),
 ('audit_native',PY,'independent_audit.py',['evaluation'],[ROOT/'diagnostics/independent_evaluation_audit.json'],False),
 ('audit_videos',PY,'video_audit.py',[],[ROOT/'diagnostics/independent_video_audit.json'],False),
 ('scientific_outcomes',PY,'make_outcomes.py',[],[ROOT/'delivery/scientific_outcomes.json'],False),
]

def freeze():
    paths={TOOLS/name for _,_,name,_,_,_ in STAGES}|{Path(__file__),ROOT/'protocol.json'}
    paths|={ROOT/'rhythm/beat_this_reference/protocol.json',ROOT/'diagnostics/muq_local_controls/protocol.json',ROOT/'diagnostics/muq_local_controls/analysis_decision.json',ROOT/'delivery/pose/protocol.json'}
    p={'files':{str(path):digest(path) for path in sorted(paths)},'stages':[{'name':name,'python':python,'tool':tool,'args':args,'outputs':[str(v) for v in outputs],'uses_gpu':gpu} for name,python,tool,args,outputs,gpu in STAGES],
        'main_source_sha256':read_authenticated(ROOT/'protocol.json')['source_sha256'],
        'main_outcomes_available_at_freeze':bool(list((ROOT/'evaluation').glob('*/seed_*/inference_report.json'))),
        'interpretation':'Required supplemental checks and delivery, using existing final weights. No additional training, main criterion changes, seed changes or parallel GPU work.'}
    path=ROOT/'post_pipeline_protocol.json'
    if path.exists():
        if read_authenticated(path)!=p:raise RuntimeError('Post-pipeline protocol drift')
    else:authenticated_json(path,p)

def main():
    protocol=read_authenticated(ROOT/'post_pipeline_protocol.json')
    for path,sha in protocol['files'].items():
        if digest(path)!=sha:raise RuntimeError('Post-pipeline source drift: '+path)
    state_path=ROOT/'post_pipeline_state.json'
    state=read_authenticated(state_path) if state_path.exists() else {'stages':[],'protocol_sha256':digest(ROOT/'post_pipeline_protocol.json')}
    if state['protocol_sha256']!=digest(ROOT/'post_pipeline_protocol.json'):raise RuntimeError('Post-pipeline state drift')
    state['waiting_for_main']=True;authenticated_json(state_path,state)
    while True:
        try:main_state=read_authenticated(ROOT/'pipeline_state.json')
        except (RuntimeError,FileNotFoundError):time.sleep(1);continue
        if main_state.get('failure'):raise RuntimeError('Main pipeline failed; preserve and review before recovery')
        if main_state.get('gpu_pipeline_complete'):break
        time.sleep(30)
    state['waiting_for_main']=False;authenticated_json(state_path,state)
    completed={s['name']:s for s in state['stages']};env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for stage in protocol['stages']:
        name=stage['name']
        if name in completed:verify_artifacts(completed[name]['artifacts']);continue
        for path,sha in protocol['files'].items():
            if digest(path)!=sha:raise RuntimeError('Post-pipeline source changed before execution: '+path)
        state['active_stage']=name;authenticated_json(state_path,state);begin=time.monotonic()
        argv=[stage['python'],str(TOOLS/stage['tool']),*stage['args']]
        print(json.dumps({'stage_start':name,'argv':argv}),flush=True)
        with (ROOT/('post_'+name+'.log')).open('a') as log:result=subprocess.run(argv,cwd=ROOT/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['failure']={'stage':name,'returncode':result.returncode};authenticated_json(state_path,state);raise RuntimeError('Post-pipeline failure: '+name)
        if not stage['outputs'] or any(not Path(path).is_file() for path in stage['outputs']):raise RuntimeError('Missing post-stage output '+name)
        record={'name':name,'wall_seconds':time.monotonic()-begin,'uses_gpu':stage['uses_gpu'],'artifacts':artifact_manifest(stage['outputs'])}
        state['stages'].append(record);state['active_stage']=None;state.pop('failure',None);authenticated_json(state_path,state)
        print(json.dumps({'stage_complete':name,'wall_seconds':record['wall_seconds']}),flush=True)
    state['post_pipeline_complete']=True;state['requires_report_and_vault_close']=True;authenticated_json(state_path,state)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:main()
