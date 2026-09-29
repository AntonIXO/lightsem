"""Independent verification of the focused efficiency pilot (run after completion)."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import torch

from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest,verify_sources,verify_artifacts
from lightsem.experiments.night_common import BASE

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910')
REFERENCE=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
ARMS=('base_once','muq_film')
MODES=('full','noise2','zero','beat_only6','donor_full','donor_clock_fixed','shift_four_bars','aux_only_donor','core_only_donor')


def check(condition,message):
    if not condition:raise AssertionError(message)


def native_energy(prediction,truth):
    prediction=np.asarray(prediction,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    count=prediction.shape[0];scores=[]
    for start in range(0,len(truth)-79,80):
        draws=prediction[:,start:start+80].reshape(count,-1)
        target=truth[start:start+80].reshape(1,-1)
        scale=np.sqrt(draws.shape[1])
        fit=cdist(draws,target).mean()/scale
        diversity=cdist(draws,draws).sum()/(2*count*(count-1)*scale)
        scores.append(fit-diversity)
    return float(np.mean(scores))


def paired(before,after,lower):
    before=np.array(before);after=np.array(after);sign=1 if lower else -1
    ix=np.random.default_rng(20260910).integers(0,len(before),(10000,len(before)))
    delta=sign*(np.mean(after)/np.mean(before)-1)
    values=sign*(np.mean(after[ix],axis=1)/np.mean(before[ix],axis=1)-1)
    interval=np.quantile(values,[.025,.975])
    return float(delta),interval,bool(delta<=.03 and interval[1]<=.03)


def output_gates(rows,videos):
    metrics=[r['metrics'] for r in rows]
    def positive(field,threshold):
        a=np.array([r[field] for r in metrics if r[field] is not None])
        if not len(a):return False
        ix=np.random.default_rng(991).integers(0,len(a),(10000,len(a)))
        return bool(a.mean()>=threshold and np.quantile(a[ix].mean(1),.025)>0)
    physical=[];brightness=[]
    for v in videos:
        b=v['beam'];s=v['screen'];c=v['composition']
        physical.append(b['lit_motion_fraction']>=.5 and b['span_2s_median_deg']>=8 and
                        b['upward_light_fraction'] is not None and b['upward_light_fraction']<=.01 and
                        b['direction_hf_above4hz_fraction']<=.15 and s['visible_motion_frame_fraction']>=.5 and
                        s['mean_brightness']>=.002 and s['support_fraction']>=.01)
        brightness.append(c['heads_with_span_ge_point4']>=4 and v['off_intervals']['heads_with_off_at_least_150ms']>=2)
    return {'conditioning':bool(np.mean([r['controlled'] for r in metrics])>=.75),
            'native_alignment':positive('alignment_advantage',.05),'local_alignment':positive('local_alignment_advantage',.01),
            'event_selectivity':positive('event_ap_advantage',.01),'visible_motion':bool(np.mean(physical)>=.75),
            'brightness':bool(np.mean(brightness)>=.75),
            'no_fast_jitter':bool(np.mean([r['beam']['direction_hf_above4hz_fraction'] for r in metrics])<=.15),
            'native_temporal_variation':bool(np.mean([r['native_dispersion']['fraction_ratio_in_point25_to2'] or 0 for r in metrics])>=.8),
            'native_draw_variation':bool(np.mean([r['native_dispersion']['mean_within_audio_draw_std'] for r in metrics])>=.05),
            'group_independence':bool(np.mean([v['independent']['unrelated_velocity_abs_correlation'] is not None and
                                             v['independent']['unrelated_velocity_abs_correlation']<.85 for v in videos])>=.75),
            'mirror_geometry':all(v['mirror_target_error_m']<1e-8 and v['mirror_emitted_ray_error_deg']<.05 for v in videos)}


def main():
    p=read_authenticated(ROOT/'protocol.json');summary=read_authenticated(ROOT/'summary.json');verify_sources(ROOT,check_live=True)
    check(summary['protocol_sha256']==digest(ROOT/'protocol.json'),'summary protocol')
    check(p['test_access'] is False and summary['test_access'] is False,'TEST access')
    for path,sha in p['input_artifacts'].items():check(digest(path)==sha,'input drift: '+path)
    norms=read_authenticated(BASE/'normalizers.json')
    inputs={r['entry']['key']:r for r in read_authenticated(BASE/'evaluation/input_manifest.json')['rows']}
    check(len(inputs)==40 and len({r['entry']['canonical_id'] for r in inputs.values()})==40,'canonical cohort')
    audit_keys=set(p['evaluation']['render_entries']);check(len(audit_keys)==8,'audit cohort')
    indexed={(c['arm'],c['seed']):c for c in summary['cells']}
    check(set(indexed)=={(a,s) for a in ARMS for s in (0,1)},'complete matrix')
    cells=[];max_energy_error=0.;array_count=0
    for seed in (0,1):
        for arm in ARMS:
            run=ROOT/'runs'/arm/f'seed_{seed}';ref_run=REFERENCE/'runs'/arm/f'seed_{seed}'
            r=read_authenticated(run/'report.json');complete=read_authenticated(run/'complete_report.json')
            verify_artifacts(r['artifacts'])
            check(r['steps']==15200 and r['samples']==486400,'exposure')
            check(r['runtime']==p['training']['runtime'] and r['precision']=='bf16','runtime')
            check(r['identity']['protocol_sha256']==digest(ROOT/'protocol.json'),'checkpoint protocol')
            check(r['identity']['source_sha256']==p['source_sha256'],'checkpoint source')
            check(complete['training_report_sha256']==digest(run/'report.json'),'complete timing binding')
            initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
            reference_initial=torch.load(ref_run/'initial.pt',map_location='cpu',weights_only=False)
            check(initial['model'].keys()==reference_initial['model'].keys(),'initial keys')
            check(all(torch.equal(v,reference_initial['model'][k]) for k,v in initial['model'].items()),'initial tensors')
            final=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
            old=torch.load(ref_run/'final.pt',map_location='cpu',weights_only=False)
            check(final['step']==15200 and final['scheduler']['last_epoch']==15200,'final step')
            check(final['rng']['numpy_batch']==old['rng']['numpy_batch'],'sampler final cursor')
            check(torch.equal(final['flow_rng'],old['flow_rng']),'flow final RNG')
            check(all(v.dtype==torch.float32 and v.isfinite().all() for v in final['model'].values()),'FP32 finite weights')
            check(all(state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32 and
                      int(state['step'])==15200 for state in final['optimizer']['state'].values()),'optimizer states')
            folder=ROOT/'evaluation'/arm/f'seed_{seed}';reference_folder=REFERENCE/'evaluation'/arm/f'seed_{seed}'
            obs=read_authenticated(folder/'inference_report.json');ref=read_authenticated(reference_folder/'inference_report.json')
            videos=read_authenticated(folder/'video_report.json')['rows'];ref_videos=read_authenticated(reference_folder/'video_report.json')['rows']
            bykey={r['key']:r for r in obs['rows']};oldkey={r['key']:r for r in ref['rows']}
            check(set(bykey)==set(inputs)==set(oldkey),'evaluation membership')
            check(len(videos)==8 and {v['key'] for v in videos}==audit_keys,'video membership')
            check({k for k,v in bykey.items() if v['retained_arrays']}==audit_keys,'compact membership')
            before={k:[] for k in ('energy','a_mse','event_ap')};after={k:[] for k in before};no_collapse=True
            for key in sorted(inputs):
                row=bykey[key];prior=oldkey[key];verify_artifacts(row['artifacts'])
                check(row['checkpoint_sha256']==digest(run/'final.pt'),'row checkpoint')
                hashes=read_authenticated(folder/key/'output_hashes.json');check(set(hashes)==set(MODES),'control membership')
                with np.load(inputs[key]['path']) as z:
                    truth=(z['y'].astype(np.float32)-np.asarray(norms['y_center'],np.float32))/np.asarray(norms['y_scale'],np.float32)
                    event_truth=z['b']
                with np.load(reference_folder/key/'full.npz') as z:old_a=z['standardized'][0,:,:4].astype(float)
                before['a_mse'].append(float(np.mean((old_a-truth[:,:4])**2)));after['a_mse'].append(row['metrics']['a_mse'])
                before['energy'].append(prior['metrics']['energy_score']['full']);after['energy'].append(row['metrics']['energy_score']['full'])
                valid=[j for j,v in enumerate(prior['metrics']['event_ap']['full']) if v is not None]
                if valid:
                    before['event_ap'].append(float(np.mean([prior['metrics']['event_ap']['full'][j] for j in valid])))
                    after['event_ap'].append(float(np.mean([row['metrics']['event_ap']['full'][j] for j in valid])))
                old_constant=np.asarray(prior['metrics']['native_dispersion']['exact_constant_channels'])
                new_constant=np.asarray(row['metrics']['native_dispersion']['exact_constant_channels'])
                no_collapse &= not bool(np.any(new_constant & ~old_constant))
                if key not in audit_keys:
                    check(not list((folder/key).glob('*.npz')),'unexpected bulk retention')
                    continue
                for mode in MODES:
                    with np.load(folder/key/(mode+'.npz')) as z:
                        for name in ('standardized','event_scores','dmx'):
                            a=np.ascontiguousarray(z[name]);record=hashes[mode][name]
                            check(hashlib.sha256(a.tobytes()).hexdigest()==record['sha256'],'array hash')
                            check(list(a.shape)==record['shape'] and str(a.dtype)==record['dtype'],'array schema')
                        actual=native_energy(z['standardized'][:,:,4:],truth[:,4:])
                        error=abs(actual-row['metrics']['energy_score'][mode]);max_energy_error=max(max_energy_error,error)
                        check(error<1e-10,'independent energy score')
                        if mode=='full':
                            mse=float(np.mean((z['standardized'][0,:,:4].astype(float)-truth[:,:4])**2))
                            check(abs(mse-row['metrics']['a_mse'])<1e-12,'independent A MSE')
                        for j in range(3):
                            if 0<event_truth[:,j].sum()<len(event_truth):
                                ap=average_precision_score(event_truth[:,j],z['event_scores'][:,j])
                                check(abs(ap-row['metrics']['event_ap'][mode][j])<1e-12,'independent AP')
                    array_count+=1
            c=indexed[arm,seed]
            passes=[]
            for metric in before:
                delta,ci,passed=paired(before[metric],after[metric],metric!='event_ap');observed=c['comparisons'][metric]
                check(abs(delta-observed['relative_degradation'])<1e-12 and np.max(abs(ci-observed['ci95']))<1e-12,'paired CI')
                check(passed==observed['passed'],'noninferiority gate');passes.append(passed)
            old_gates=output_gates(ref['rows'],ref_videos);new_gates=output_gates(obs['rows'],videos)
            check(old_gates==c['baseline_gates'] and new_gates==c['candidate_gates'],'independent output gates')
            baseline=read_authenticated(ref_run/'report.json');speedup=baseline['wall_seconds']/complete['complete_seconds']
            expected={'speed':speedup>=2.,'quality':all(passes),'preserved_gates':all(not v or new_gates[k] for k,v in old_gates.items()),
                      'no_new_collapse':no_collapse,'integrity':True}
            check(expected==c['gates'] and all(expected.values())==c['all_gates_pass'],'final cell gates')
            for v in videos:verify_artifacts(v['artifacts'])
            cells.append({'arm':arm,'seed':seed,'authenticated':True,'sampler_matches_reference':True,'flow_rng_matches_reference':True,
                          'gates':expected,'report_sha256':digest(run/'report.json')})
    eligible=[a for a in ARMS if all(all(c['gates'].values()) for c in cells if c['arm']==a)]
    check(eligible==summary['eligible_variants'],'eligibility')
    result={'passed':True,'cells':cells,'retained_control_arrays_recomputed':array_count,'maximum_energy_error':max_energy_error,
            'input_hashes_checked':len(p['input_artifacts']),'eligible_variants':eligible,
            'summary_sha256':digest(ROOT/'summary.json'),'audit_source_sha256':digest(__file__)}
    authenticated_json(ROOT/'independent_audit.json',result);print(json.dumps(result))


def regenerate_unsaved():
    """Reproduce every control for one non-retained song per cell, without saves."""
    import gc
    from lightsem.experiments.stage11_common import gpu_lock
    from lightsem.experiments.night_evaluation import load_cell,normalized_input,generate_groups,DRAW_SEEDS
    from lightsem.experiments.night_render import compose_scene,ExpressionConfig
    from lightsem.conductor.rig import canonical_rig
    p=read_authenticated(ROOT/'protocol.json');cfg=ExpressionConfig(**read_authenticated(ROOT/'expression_config.json'))
    prepared=read_authenticated(REFERENCE/'evaluation/input_manifest.json');rig=canonical_rig()
    inputs={r['entry']['key']:r for r in read_authenticated(BASE/'evaluation/input_manifest.json')['rows']}
    key=next(k for k in sorted(inputs) if k not in p['evaluation']['render_entries']);results=[]
    with gpu_lock(ROOT):
        for seed in (0,1):
            for arm in ARMS:
                model,checkpoint,_,norms=load_cell(arm,seed,ROOT);family=arm.split('_')[0]
                auxnorm=read_authenticated(REFERENCE/'features/muq/training_array.json') if family=='muq' else None
                source=prepared['rows'][family+'/'+key]
                expected=read_authenticated(ROOT/'evaluation'/arm/f'seed_{seed}'/key/'output_hashes.json')
                encoded=None
                for mode in MODES:
                    info=source['files'][mode]
                    with np.load(info['path']) as z:x=normalized_input(z['raw_audio'],z['auxiliary'],norms,auxnorm,info['aux_missing'])
                    if mode=='full':
                        with torch.no_grad():encoded=model.encode_audio_chunked(x.cuda(),640,output_device='cuda')
                    seeds=tuple(s+1000 for s in DRAW_SEEDS) if mode=='noise2' else DRAW_SEEDS
                    y,event=generate_groups(model,x,seeds=seeds,steps=8,encoded=encoded if mode in ('full','noise2') else None)
                    frames,_,_=compose_scene(y,norms['target_names'],x.shape[1]/40.,event_scores=event,rig=rig,config=cfg)
                    for name,array in [('standardized',y),('event_scores',event),('dmx',frames)]:
                        actual=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                        check(actual==expected[mode][name]['sha256'],f'unsaved replay {arm} {seed} {mode} {name}')
                results.append({'arm':arm,'seed':seed,'key':key,'all_nine_controls_exact':True,'checkpoint_sha256':checkpoint})
                del model,encoded,x,y,event,frames;gc.collect();torch.cuda.empty_cache()
    report={'passed':True,'rows':results,'reproduced_controls':36,'audit_source_sha256':digest(__file__)}
    authenticated_json(ROOT/'regeneration_audit.json',report);print(json.dumps(report))


if __name__=='__main__':
    if '--regenerate' in sys.argv:regenerate_unsaved()
    else:main()
