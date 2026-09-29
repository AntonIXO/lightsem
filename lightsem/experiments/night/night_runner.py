"""Persistent serial GPU pipeline; no agent-dependent timer starts the next job."""
from pathlib import Path
import os,subprocess,sys,time,json
from .night_common import ROOT,ARMS,SEEDS
from ..stage11.stage11_common import authenticated_json,read_authenticated,artifact_manifest,verify_artifacts,digest

def main(root=ROOT):
    root=Path(root);p=read_authenticated(root/'protocol.json');state_path=root/'pipeline_state.json'
    state=read_authenticated(state_path) if state_path.exists() else {'protocol_sha256':digest(root/'protocol.json'),'stages':[]}
    if state['protocol_sha256']!=digest(root/'protocol.json'):raise RuntimeError('Pipeline protocol drift')
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    completed={r['name']:r for r in state['stages']}
    def run(name,module,args,outputs,python=sys.executable):
        if name in completed:
            verify_artifacts(completed[name]['artifacts']);return
        now=time.monotonic();state['active_stage']=name;authenticated_json(state_path,state)
        command=[python,'-m',module,*args]
        print(json.dumps({'stage_start':name,'argv':command}),flush=True)
        with (root/(name+'.log')).open('a') as log:
            result=subprocess.run(command,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['failure']={'stage':name,'returncode':result.returncode};authenticated_json(state_path,state)
            raise RuntimeError('Pipeline stage failed: '+name)
        if not outputs or any(not Path(path).is_file() for path in outputs):raise RuntimeError('Stage produced incomplete artifacts: '+name)
        record={'name':name,'wall_seconds':time.monotonic()-now,'artifacts':artifact_manifest(outputs)}
        state['stages'].append(record);state['active_stage']=None;state.pop('failure',None);authenticated_json(state_path,state)
        print(json.dumps({'stage_completed':name,'wall_seconds':record['wall_seconds']}),flush=True)
    resume=['--resume'] if any((root/'runs').glob('*/seed_*/initial.pt')) else []
    run('main_training','lightsem.experiments.night.night_training',['campaign',*resume],[root/'training_report.json'])
    infer_reports=[root/'evaluation'/a/f'seed_{s}'/'inference_report.json' for s in SEEDS for a in ARMS]
    video_reports=[root/'evaluation'/a/f'seed_{s}'/'video_report.json' for s in SEEDS for a in ARMS]
    run('main_inference','lightsem.experiments.night.night_evaluation',['infer'],infer_reports)
    run('main_renders','lightsem.experiments.night.night_evaluation',['render'],video_reports)
    run('incumbent_adapter','lightsem.experiments.night.night_evaluation',['incumbent'],[root/'evaluation/incumbent_adapter/report.json'])
    run('main_summary','lightsem.experiments.night.night_evaluation',['summarize'],[root/'evaluation/report.json'])
    choice=read_authenticated(root/'evaluation/report.json')['chosen_arm']
    if choice and choice.startswith('muq'):
        run('demo_embeddings','lightsem.experiments.night.night_demo',['prepare_muq'],[root/'features/shpongle_muq.json'],python='/mnt/lightsem-data/venv-night-encoders/bin/python')
    run('demo_inference','lightsem.experiments.night.night_demo',['infer'],[root/'delivery/shpongle/inference_report.json'])
    run('demo_render','lightsem.experiments.night.night_demo',['render'],[root/'delivery/shpongle/render_report.json'])
    run('demo_observe','lightsem.experiments.night.night_demo',['observe'],[root/'delivery/shpongle/observation_report.json'])
    state['gpu_pipeline_complete']=True;state['requires_independent_audit_and_scientific_close']=True;authenticated_json(state_path,state)

if __name__=='__main__':main()
