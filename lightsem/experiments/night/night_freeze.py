"""Freeze effective inputs, tests, implementation and a performance-only compute budget."""
from pathlib import Path
import json,sys
import numpy as np
from .night_common import ROOT,BASE,ARMS,SEEDS,base_context,freeze_night_sources
from .night_features import FOURIER_CHANNELS,FOURIER_NAMES,fourier_features
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest
from ..training_runtime import TrainingRuntime

def training_config(steps,preflight):
    runtime=TrainingRuntime.from_training(preflight)
    return {'steps':steps,'batch':32,'frames':640,'lr':.0003,'warmup':500,
            'beat_dropout':.25,'precision':runtime.precision,'runtime':runtime.as_dict()}

def freeze(root=ROOT):
    root=Path(root);m,n=base_context()
    if (root/'protocol.json').exists():raise RuntimeError('Main study already frozen')
    if tuple(n['feature_names'][i] for i in FOURIER_CHANNELS)!=FOURIER_NAMES:raise ValueError('Fourier schema does not name flux channels')
    for e in m['entries']:
        if e['partition'] not in ('train','val') or digest(e['path'])!=e['sha256']:raise RuntimeError('Source data drift')
    checks=[]
    for partition in ('train','val'):
        entries=[e for e in m['entries'] if e['partition']==partition]
        for i in np.linspace(0,len(entries)-1,5,dtype=int):
            e=entries[i]
            with np.load(e['path']) as z:expected=fourier_features(z['x'])
            actual=np.load(root/'features/fourier'/(e['key']+'.npy'))
            if not np.array_equal(expected,actual):raise RuntimeError('Recomputed named Fourier data differs')
            checks.append(e['key'])
    preflight=read_authenticated(root/'engineering_preflight.json')
    if preflight['model_source_sha256']!=digest(Path(__file__).with_name('night_model.py')):raise RuntimeError('Engineering model drift')
    if preflight['source_sha256']!=digest(Path(__file__).with_name('night_training.py')):raise RuntimeError('Engineering training implementation drift')
    if preflight.get('runtime_source_sha256')!=digest(Path(__file__).with_name('training_runtime.py')):raise RuntimeError('Engineering runtime drift; rerun preflight')
    if {r['arm'] for r in preflight['rows']}!=set(ARMS):raise ValueError('Incomplete engineering matrix')
    pose=read_authenticated(root/'diagnostics/pose_probe_report.json')
    if {(r['name'],r['seed']) for r in pose['rows']}!={(k,s) for k in ('descriptor128','descriptor256','pose_code24','audio_base','audio_fourier','audio_muq') for s in SEEDS}:
        raise ValueError('Incomplete reconstruction matrix')
    previous_gpu=read_authenticated(root/'features/muq_native/report.json')['gpu_seconds']
    previous_gpu+=read_authenticated(root/'features/muq/layer_selection.json')['gpu_seconds']
    previous_gpu+=sum(r['wall_seconds'] for r in pose['rows'])
    control=root/'diagnostics/embedding_controls/input_manifest.json'
    if control.exists():previous_gpu+=read_authenticated(control)['gpu_task_seconds']
    control_eval=root/'diagnostics/embedding_controls/report.json'
    if control_eval.exists():previous_gpu+=read_authenticated(control_eval)['gpu_task_seconds']
    reserve=3600.;training_budget=36000.-previous_gpu-reserve
    if training_budget<3600:raise ValueError('Insufficient remaining budget for planned main matrix')
    step_sum=sum(r['step_seconds_median'] for r in preflight['rows'])*len(SEEDS)
    steps=100*int((training_budget/step_sum)//100)
    pins={}
    for path in (BASE/'data_manifest.json',BASE/'normalizers.json',BASE/'evaluation/input_manifest.json',
                 root/'input_pins.json',root/'expression_config.json',root/'engineering_preflight.json',root/'evaluation/input_manifest.json',
                 root/'features/fourier/report.json',root/'features/muq/layer_selection.json',root/'features/muq_weight_audit.json',root/'engineering_evaluation.json'):
        pins[str(path)]=digest(path)
    for e in m['entries']:pins[e['path']]=e['sha256']
    for family in ('fourier','muq'):
        path=root/'features'/family/'training_array.json';r=read_authenticated(path);pins[str(path)]=digest(path);pins[r['path']]=r['sha256']
    for row in read_authenticated(root/'evaluation/input_manifest.json')['rows'].values():
        for entry in row['files'].values():pins[entry['path']]=entry['sha256']
    for row in read_authenticated(BASE/'evaluation/input_manifest.json')['rows']:pins[row['path']]=row['sha256']
    source=freeze_night_sources(root)
    protocol={'frozen':True,'schema_version':1,'source_sha256':source['source_sha256'],'source_snapshot':source['snapshot'],
        'input_artifacts':pins,'arms':list(ARMS),'seeds':list(SEEDS),'main_model_count':12,'test_access':False,
        'validation_is_development':True,'training':training_config(steps,preflight),
        'compute_budget':{'target_gpu_task_seconds':36000,'prior_valid_gpu_task_seconds':previous_gpu,'reserved_final_gpu_seconds':reserve,
                          'main_training_budget_seconds':training_budget,'estimated_main_seconds':step_sum*steps,
                          'selection_rule':'common steps floor_to100 of remaining budget divided by summed engineering step costs; no outcomes enter rule'},
        'evaluation':{'songs':40,'frames':2400,'draw_seeds':[101,102,103,104,105,106],'flow_steps':8,'chunk_size':640,
                      'render_entries':read_authenticated(BASE/'protocol.json')['evaluation']['render_entries'],
                      'modes':['full','noise2','zero','beat_only6','donor_full','donor_clock_fixed','shift_four_bars','aux_only_donor','core_only_donor']},
        'thresholds':{'controlled_song_fraction_min':.75,'audio_response_over_span_min':.15,'audio_response_over_noise_min':.5,
          'alignment_advantage_min':.05,'local_alignment_advantage_min':.01,'event_ap_advantage_min':.01,'positive_ci_lower_required':True,
          'lit_motion_min':.5,'span_min_deg':8.,'upward_max':.01,'hf_max':.15,'visible_motion_min':.5,'screen_brightness_min':.002,'screen_support_min':.01,
          'valid_clip_fraction_min':.75,'head_dimmer_span_min':.4,'heads_with_dimmer_span_min':4,'off_interval_min_frames':6,'off_heads_min':2,
          'unrelated_velocity_abs_correlation_max':.85,'mirror_target_error_max_m':1e-8,'mirror_ray_error_max_deg':.05,
          'ending_border_luma_reduction_min':.3,'native_time_std_ratio_range':[.25,2.],'native_supported_fraction_min':.8,'native_draw_std_min':.05,
          'feature_energy_score_improvement_min':.05,'film_controlled_fraction_improvement_min':.25,'both_training_seeds_required':True},
        'checks':{'recomputed_fourier_entries':checks,'named_flux_columns':list(FOURIER_CHANNELS),'minimum_free_disk_gib':15},
        'claims':'Native audio-conditioned motion and separate group/brightness functionality; human aesthetic superiority unmeasured',
        'controls_caveat':'Explicit clock-held hybrids are not rhythm-free waveform interventions; MuQ contains rhythm.',
        'selection':'Final scheduled weights only. All12cells evaluated. Candidate must pass mandatory output/conditioning gates in both seeds; else old weights plus new adapter.'}
    authenticated_json(root/'protocol.json',protocol)
    print(json.dumps({'steps':steps,'main_training_hours':step_sum*steps/3600,'prior_gpu_hours':previous_gpu/3600,'source_sha256':source['source_sha256']}),flush=True)
    return protocol

if __name__=='__main__':freeze()
