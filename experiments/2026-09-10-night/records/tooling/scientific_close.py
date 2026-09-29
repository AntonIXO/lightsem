"""Record computed measurements/gates in the correct vault; let Store compute verdicts."""
from pathlib import Path
import argparse,json,sys
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest

def store_instance():
    sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    return Store('/root/lightsem-vault')

def component_ids():
    result={}
    for family in ('fourier','muq'):
        result[family]=[(f'night-{family}-{arch}-seed{s}-energy',arch,s,'lower_better','standardized_distance') for arch in ('once','film') for s in (0,1)]
    result['conditioning']=[(f'night-film-{family}-seed{s}-controlled',family,s,'higher_better','fraction') for family in ('base','fourier','muq') for s in (0,1)]
    return result

def register_components():
    store=store_instance();rows=[]
    started=bool(list((ROOT/'evaluation').glob('*/seed_*/inference_report.json')))
    for study,components in component_ids().items():
        for id,axis,seed,direction,unit in components:
            if not store.vault.exists('diagnostics',id):
                if started:raise RuntimeError('Cannot add component basis after main outcomes')
                store.register_diagnostic(id=id,name=id,direction=direction,unit=unit,
                    description=f'Complete paired component of the already registered night {study} study: {axis},training seed{seed}. Reported alongside the fixed primary aggregate to retain regressions; not a new outcome-selected hypothesis.')
            rows.append({'study':study,'diagnostic_id':id,'axis':axis,'seed':seed})
    authenticated_json(ROOT/'research/component_diagnostic_registration.json',{'rows':rows,'registered_before_main_inference':not started,'source_sha256':digest(__file__)})
    print(json.dumps({'registered_components':len(rows)}))

def costs():
    rows=[]
    def add(name,seconds,scope,path):rows.append({'name':name,'seconds':float(seconds),'scope':scope,'source':str(path)})
    for name,path,key in [
        ('MuQ extraction',ROOT/'features/muq_native/report.json','gpu_seconds'),
        ('MuQ TRAIN layer selection',ROOT/'features/muq/layer_selection.json','gpu_seconds'),
        ('MERT extraction control',ROOT/'diagnostics/embedding_controls/input_manifest.json','gpu_task_seconds'),
        ('Embedding ridge controls',ROOT/'diagnostics/embedding_controls/report.json','gpu_task_seconds'),
        ('TRAIN rig calibration',ROOT/'calibration/report.json','elapsed_s')]:
        r=read_authenticated(path);add(name,r[key],'Elapsed GPU task, including required input/output work',path)
    pose=read_authenticated(ROOT/'diagnostics/pose_probe_report.json')
    for r in pose['rows']:add('Pose '+r['name']+'/seed'+str(r['seed']),r['wall_seconds'],'Training plus GPU evaluation task',ROOT/'diagnostics/pose_probes'/r['name']/f"seed_{r['seed']}"/'report.json')
    pipeline=read_authenticated(ROOT/'pipeline_state.json');cpu=[]
    for r in pipeline['stages']:
        if r['name'] in ('main_summary','demo_observe'):cpu.append({'name':r['name'],'seconds':r['wall_seconds']})
        else:add(r['name'],r['wall_seconds'],'Elapsed GPU stage, including necessary loading, serialization and within-stage observation',ROOT/'pipeline_state.json')
    post=read_authenticated(ROOT/'post_pipeline_state.json')
    if not post.get('post_pipeline_complete'):raise RuntimeError('Supplementary follow-through incomplete')
    for r in post['stages']:
        if r['uses_gpu']:add(r['name'],r['wall_seconds'],'Elapsed GPU stage',ROOT/'post_pipeline_state.json')
        else:cpu.append({'name':r['name'],'seconds':r['wall_seconds']})
    training=read_authenticated(ROOT/'training_report.json')
    result={'gpu_stage_rows':rows,'gpu_stage_elapsed_seconds':sum(r['seconds'] for r in rows),'main_optimizer_seconds':training['training_seconds'],
        'cpu_post_and_summary_stages':cpu,'cpu_post_and_summary_seconds':sum(r['seconds'] for r in cpu),
        'scope':'Recorded elapsed times of GPU tasks, not CUDA kernel-only utilization. Includes required CPU preparation/output inside those tasks; excludes idle waiting and standalone CPU work. Small engineering preflights and the interrupted invalid Fourier/MERT work are not counted as valid scientific work. One shared GPU was used serially.'}
    authenticated_json(ROOT/'delivery/compute_ledger.json',result);return result

def close():
    store=store_instance();outcomes=read_authenticated(ROOT/'delivery/scientific_outcomes.json');registration=read_authenticated(ROOT/'vibescience_registration.json')
    ledger=costs();results=[]
    byname={r['experiment_id'].removeprefix('exp-night-').removesuffix('-20260909'):r for r in registration['rows']}
    for name,value in outcomes['studies'].items():
        reg=byname[name];experiment=store.vault.read_experiment(reg['experiment_id'])
        measurements=[{'diagnostic_id':reg['diagnostic_id'],'before':value['before'],'after':value['after']}]
        if name in component_ids():
            comparisons=value['details']['comparisons']
            for id,axis,seed,_,_ in component_ids()[name]:
                match=next(r for r in comparisons if r.get('architecture',r.get('family'))==axis and r['seed']==seed)
                measurements.append({'diagnostic_id':id,'before':match['before'],'after':match['after']})
        failed=[k for k,v in value['gates'].items() if not v]
        evidence=json.dumps({'before':value['before'],'after':value['after'],'details':value['details']},sort_keys=True)
        gate_results=[{'gate_id':k,'passed':v,'evidence':evidence+'; full outcome artifact '+str(ROOT/'delivery/scientific_outcomes.json')+' sha256='+digest(ROOT/'delivery/scientific_outcomes.json')} for k,v in value['gates'].items()]
        if {r['gate_id'] for r in gate_results}!={r['id'] for r in reg['gates']}:raise ValueError('Gate membership differs from registration')
        if not experiment.closed:
            store.record_diagnostics(reg['experiment_id'],measurements)
            store.record_gate_results(reg['experiment_id'],gate_results)
            notes=(value['lesson']+' Failed magnitude/absolute gates: '+(', '.join(failed) or 'none')+'. '
                'All matched architecture/seed components are recorded as separate diagnostics; no flattering subset replaces the primary aggregate. '
                'Two training seeds, development VAL, no TEST access. Main sourcebc9eab5e730df1c8794ca55e9aa6e66799f1e9ca7f6ddb066bda6b17f8145c77. '
                'The initial Fourier column error and unscored MERT weight-normalization mapping error were preserved and corrected before main training; unaffected completed pose probes were adopted by exact input/code audit. '
                'Compute is shared across these six analyses; do not sum a reused model as six independent training campaigns. Human aesthetic preference and real fixture acceleration feasibility were not measured.')
            result=store.close_experiment(reg['experiment_id'],notes=notes,
                artifacts=[str(ROOT/'delivery/scientific_outcomes.json'),str(ROOT/'protocol.json'),str(ROOT/'evaluation/report.json'),str(ROOT/'delivery/compute_ledger.json'),str(ROOT/'diagnostics/independent_video_audit.json')],
                key_numbers={'primary_before':value['before'],'primary_after':value['after'],'all_study_gates_pass':float(value['all_gates_pass']),'main_candidate_selected':float(outcomes['chosen_arm'] is not None)},
                do_not_retry=['Do not repeat the identical recipe with a different seed to select a better outcome. A follow-up requires a new fixed protocol and a changed, justified mechanism.'] if failed else [])
        else:
            actual={m.diagnostic_id:m for m in experiment.diagnostics_measured}
            for m in measurements:
                if m['diagnostic_id'] not in actual or actual[m['diagnostic_id']].before!=m['before'] or actual[m['diagnostic_id']].after!=m['after']:raise RuntimeError('Closed experiment measurement drift')
            if {r.gate_id:r.passed for r in experiment.gate_results}!=value['gates']:raise RuntimeError('Closed experiment gate drift')
            result={'experiment_id':experiment.id,'verdict':experiment.verdict.value,'already_closed':True}
        experiment=store.vault.read_experiment(reg['experiment_id']);verdict=experiment.verdict.value
        negative=verdict in ('refutes','inconclusive','directional_only','null','weak_signal')
        store.interpret_experiment(reg['experiment_id'],lesson=value['lesson'],failure_mode='recipe_conditional' if negative else None)
        results.append({'study':name,'experiment_id':experiment.id,'hypothesis_id':reg['hypothesis_id'],'computed_verdict':verdict,'all_study_gates_pass':value['all_gates_pass'],'result':result})
    problem=store.vault.read_problem('dancing-light-generative-20260903')
    update_text='Overnight expression study 2026-09-09: '+(
        f"All12main cells and12pose probes completed; main candidate={outcomes['chosen_arm']!r}. Recorded GPU-stage elapsed time={ledger['gpu_stage_elapsed_seconds']/3600:.3f}h; optimizer work={ledger['main_optimizer_seconds']/3600:.3f}h. "
        'Six new studies closed with computed verdicts and independently audited magnitude/output gates. Reports and videos: '+str(ROOT/'delivery')+'. '
        'Movement acceleration constraints and new fixture types remain deferred. Conditional response, reconstruction, group geometry and human lighting preference remain distinct claims.')
    if not any(u.text==update_text for u in problem.updates):store.update_problem(problem.id,add_update=update_text)
    calibration={r['diagnostic_id']:store.calibration(diagnostic_id=r['diagnostic_id']) for r in registration['rows']}
    causal=store.causal_map(problem_id=problem.id)
    authenticated_json(ROOT/'delivery/vibescience_close.json',{'rows':results,'calibration':calibration,'causal_map':causal,'source_sha256':digest(__file__),'vault':'/root/lightsem-vault'})
    print(json.dumps({'closed':[{'study':r['study'],'verdict':r['computed_verdict']} for r in results]}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['register','close']);a=ap.parse_args()
    if a.stage=='register':register_components()
    else:close()
