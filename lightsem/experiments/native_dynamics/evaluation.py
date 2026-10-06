"""Frozen native-time evaluation; motion labels never enter generation inputs."""
from pathlib import Path
from collections import defaultdict
import json, math
import numpy as np
import torch
from .data import ROOT,read,write,sha,save_npz
from .dataset import Records,to_device
from .model import NativeDynamicsModel,Config
from .kinematics import observed_dynamics,project_native,masked_quantiles,GROUPS
from ..motion_study.music_model import MusicConfig
from ..dynamic_dance import diagnostics as physical

ARMS=('continued','dynamics','joint')

def setup():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)


def cpu_noise(flow,frames,seed,batch=1):
    """Declared CPU stationary draw, independent of the model's storage device.

    Same draw order and convolution as MusicCondFlow.noise on a CPU model;
    only the fixed kernel is copied, never the model or its weights.
    """
    generator=torch.Generator(device='cpu').manual_seed(seed)
    raw=torch.randn(batch*139,1,frames+24,device='cpu',generator=generator)
    kernel=flow.noise_kernel.detach().to(device='cpu',dtype=raw.dtype)
    return torch.nn.functional.conv1d(raw,kernel).reshape(batch,139,frames).transpose(1,2)


def load_final(root,arm,seed,device='cuda'):
    path=Path(root)/'learning/runs'/arm/f'seed_{seed}'/'final.pt'
    report=read(path.parent/'REPORT.json')
    if sha(path)!=report['artifacts'][str(path)] or report['steps']!=read(Path(root)/'TRAINING_PROTOCOL.json')['training']['steps']:raise ValueError('Incomplete or changed final checkpoint')
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=NativeDynamicsModel(MusicConfig(**state['base_config']),state['normalizers'],Config(**state['config']))
    model.load_state_dict(state['model'],strict=True);model=model.to(device).eval().requires_grad_(False)
    norms=[tuple(torch.tensor(state[name][k],device=device,dtype=torch.float32) for k in ('mean','std')) for name in ('audio_normalizers','extra_normalizers')]
    return model,norms,{'checkpoint':str(path),'sha256':sha(path),'identity_sha256':sha(path.parent/'IDENTITY.json'),'initial_checkpoint':state['identity']['initial'],'arm':arm,'seed':seed,'steps':state['step']}


def _average(value,mask,axes):
    value=np.asarray(value);mask=np.broadcast_to(mask,value.shape)
    count=mask.sum(axis=axes);total=np.where(mask,value,0).sum(axis=axes)
    return np.divide(total,count,out=np.full(np.shape(total),np.nan),where=count>0)


def available_mean(values):
    x=np.asarray(values);x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def metrics(generated,observed,mask):
    """Equal region/scale means; uncovered cells are explicitly unmeasurable."""
    with torch.no_grad():
        g=observed_dynamics(torch.as_tensor(generated)[None],torch.as_tensor(mask)[None]);o=observed_dynamics(torch.as_tensor(observed)[None],torch.as_tensor(mask)[None])
        valid=g['profile_valid']&o['profile_valid']
        error=(torch.log1p(g['profiles'])-torch.log1p(o['profiles'])).abs()
        cells=_average(error.numpy()[0],valid.numpy()[0],0)
        common=g['speed_valid']&o['speed_valid'];gq,good=masked_quantiles(g['speed'],common);oq,_=masked_quantiles(o['speed'],common)
        distribution=(torch.log1p(gq)-torch.log1p(oq)).abs().mean(-1)[0].numpy();distribution=np.where(good[0].numpy(),distribution,np.nan)
        spread=(gq[0,:,9]-gq[0,:,1]).numpy();reference_spread=(oq[0,:,9]-oq[0,:,1]).numpy()
        ratios=np.divide(spread,reference_spread,out=np.full(6,np.nan),where=reference_spread>1e-6)
        # Arms moving + lower body quiet is NOT an all-body rest.
        lm=(g['low_valid']&o['low_valid'])[0].numpy();ol=o['low'][0].numpy();gl=g['low'][0].numpy()
        eligible=lm[:,[0,3,4,5]].all(-1)&ol[:,0]&(~ol[:,3]|~ol[:,4])
        false_count=int((gl[:,5]&eligible).sum());den=int(eligible.sum())
        finite=bool(np.isfinite(generated).all())
        report={'finite':finite,'profile_error':available_mean(cells),'profile_error_by_region_scale':[[float(x) if np.isfinite(x) else None for x in row] for row in cells], 'registered_complete_coverage':bool(np.isfinite(cells).all() and good.all()), 'profile_error_scope':'available observed region/scale cells; full registered primary requires complete coverage','distribution_error':available_mean(distribution),'distribution_by_region':[float(x) if np.isfinite(x) else None for x in distribution],'spread_ratio_by_region':[float(x) if np.isfinite(x) else None for x in ratios],'false_hold_count':false_count,'false_hold_denominator':den,'false_hold_fraction':false_count/den if den else None,'observed_profile_coverage':float(valid.float().mean()),'constant_pose':bool(np.all(generated==generated[:1]))}
        traces={name:x[0].numpy() for name,x in g.items() if name in ('profiles','profile_valid','speed','low','low_valid')}
        traces.update(observed_profiles=o['profiles'][0].numpy(),observed_speed=o['speed'][0].numpy(),observed_low=o['low'][0].numpy())
        return report,traces


def bootstrap_paired(before,after,seed=7001,samples=5000):
    a=np.asarray(before,float);b=np.asarray(after,float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Paired finite song vectors required')
    rng=np.random.default_rng(seed);delta=b-a;means=delta[rng.integers(0,len(a),(samples,len(a)))].mean(1)
    return {'before':float(a.mean()),'after':float(b.mean()),'relative_reduction':float((a.mean()-b.mean())/max(a.mean(),1e-12)),'delta':float(delta.mean()),'ci95_delta':np.quantile(means,[.025,.975]).tolist(),'groups':len(a),'bootstrap_samples':samples,'bootstrap_seed':seed,'magnitude_pass':bool(b.mean()<=.95*a.mean() and np.quantile(means,.975)<0)}


def metric_controls(observed,mask):
    from scipy.ndimage import gaussian_filter1d
    shifted_mask=mask&np.roll(mask,len(mask)//2,axis=0)
    weights=gaussian_filter1d(mask.astype(np.float32),30.,axis=0)
    smooth=gaussian_filter1d(np.where(mask[...,None],observed,0.),30.,axis=0)/np.maximum(weights[...,None],1e-8)
    controls={'constant_pose':(np.zeros_like(observed),mask),
              'shifted_observation':(np.roll(observed,len(observed)//2,axis=0),shifted_mask),
              'oversmoothed_observation':(smooth,mask&(weights>=.8))}
    result={key:metrics(value,observed,common_mask)[0] for key,(value,common_mask) in controls.items()}
    d=observed_dynamics(torch.from_numpy(observed)[None],torch.from_numpy(mask)[None])
    p=d['profiles'][0].numpy();v=d['profile_valid'][0].numpy();constant=_average(p,v,0)[None]
    error=_average(np.abs(np.log1p(p)-np.log1p(constant)),v,0)
    result['constant_nonzero_speed_trace']={'profile_error':available_mean(error),'scope':'Numeric speed-trace control, not fabricated dance'}
    result['observed_identity']=metrics(observed,observed,mask)[0]
    return result


def _audio_inputs(batch,count=None):
    return (batch['audio'][:,:count],batch['coarse'],batch['coarse_valid'],batch['indices'][:,:count])

@torch.inference_mode()
def evaluate(root,arm,seed):
    setup();root=Path(root);study=read(root/'STUDY_PLAN.json');model,norms,identity=load_final(root,arm,seed)
    records=Records(root,'val',frames=960);lookup={(d,r['key']):r for d,rows in records.records.items() for r in rows}
    folder=root/'evaluation_v5'/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False);rows=[];controls=[]
    for w in study['evaluation']['windows']:
        record=lookup[w['domain'],w['key']];raw=records.pack(w['domain'],[(record,w['context_start'])]);batch=to_device(raw,*norms)
        offset=w['score_offset'];n=w['score_frames'];actual=w['context_frames'];s=slice(offset,offset+n);domain=w['domain']
        if domain=='jd':target=record['pose'][w['score_start']:w['score_start']+n];mask=record['mask'][w['score_start']:w['score_start']+n]
        else:
            target_tensor,truth_body=project_native(batch['native'],batch['valid']);target=target_tensor[0,s].cpu().numpy();mask=np.ones((n,13),bool)
        poses=[]
        for draw in study['evaluation']['draw_seeds']:
            noise=cpu_noise(model.flow,960,draw)[:,:actual].contiguous().to('cuda')
            native,plan=model.generate(*_audio_inputs(batch,actual),noise)
            projected,body=project_native(native,batch['valid'][:,:actual]);q=projected[0,s].cpu().numpy();poses.append(q)
            measured,traces=metrics(q,target,mask)
            row={**w,'arm':arm,'seed':seed,'draw_seed':draw,**measured,'noise_sha256':__import__('hashlib').sha256(noise.cpu().numpy().tobytes()).hexdigest()}
            if domain=='fd':
                world=body['positions'][0,s].cpu().numpy();truth=truth_body['positions'][0,s].cpu().numpy()
                row['physical']=physical.body_dynamics(world,np.arange(n)/30,contacts=native[0,s,-4:].cpu().numpy(),reference_body=truth,floor_y=float(np.quantile(truth[:,physical.FEET,1],.02)))
            if domain=='jd' and w['window']==1 and draw==7001:
                # Compact auditable traces: all24middlewindows, eacharm/seed.
                save_npz(folder/f"trace_{w['key']}.npz",time=np.arange(n)/30+w['score_start']/30,pose=q,observed=target,mask=mask,**traces)
                if arm=='continued' and seed==0:controls.append({'key':w['key'],'controls':metric_controls(target,mask)})
                # Fixed half-song feature shift. Full waveform interventions are
                # separately measured on both showcases with recomputed caches.
                shifted=dict(record);shifted['base']=np.roll(record['base'],record['length']//2,axis=0);shifted['extra']=np.roll(record['extra'],record['length']//2,axis=0);shifted['coarse']=np.roll(record['coarse'],record['bins']//2,axis=0)
                shifted_batch=to_device(records.pack(domain,[(shifted,w['context_start'])]),*norms)
                altered,_=model.generate(*_audio_inputs(shifted_batch,actual),noise);aq,_=project_native(altered,batch['valid'][:,:actual])
                row['half_song_feature_shift_error']=metrics(aq[0,s].cpu().numpy(),target,mask)[0]['profile_error']
                if arm=='joint':
                    override=plan['conditioning'].clone();bins=record['bins'];override[:,:bins]=torch.roll(override[:,:bins],bins//2,1)
                    altered,_=model.generate(*_audio_inputs(batch,actual),noise,plan_override=override);aq,_=project_native(altered,batch['valid'][:,:actual])
                    row['half_song_plan_shift_error']=metrics(aq[0,s].cpu().numpy(),target,mask)[0]['profile_error']
            rows.append(row)
        diversity=float(np.linalg.norm(poses[0]-poses[1],axis=-1)[mask].mean())
        for row in rows[-2:]:row['conditional_diversity']=diversity
        print(json.dumps({'arm':arm,'seed':seed,'domain':domain,'key':w['key'],'window':w['window'],'completed':len(rows)}),flush=True)
    report={'arm':arm,'seed':seed,'identity':identity,'study_sha256':sha(root/'STUDY_PLAN.json'),'rows':rows,'controls':controls,'input_sha256':records.inputs,'generation_uses':'audio/coarse_audio/valid_masks/frame_indices/noise only; no motion labels, retiming, or learned-clock correction','precision':'FP32 inference, deterministic mathattention','code_sha256':sha(__file__)}
    write(folder/'REPORT.json',report);return report


def aggregate(root):
    root=Path(root);reports={(a,s):read(root/'evaluation_v5'/a/f'seed_{s}'/'REPORT.json') for a in ARMS for s in (0,1)};comparisons={}
    def groups(report):
        values=defaultdict(list)
        for r in report['rows']:
            if r['domain']=='jd':values[r['group']].append(r)
        return {g:{k:float(np.mean([r[k] for r in rows])) for k in ('profile_error','distribution_error','conditional_diversity')} for g,rows in sorted(values.items())}
    means={k:groups(v) for k,v in reports.items()}
    for arm,base in [('dynamics','continued'),('joint','dynamics')]:
        seeds=[]
        for seed in (0,1):
            a=means[base,seed];b=means[arm,seed]
            if set(a)!=set(b) or len(a)!=24:raise ValueError('Missing paired24-song evaluation')
            fields={k:bootstrap_paired([a[g][k] for g in a],[b[g][k] for g in a]) for k in ('profile_error','distribution_error','conditional_diversity')}
            rows=[r for r in reports[arm,seed]['rows'] if r['domain']=='jd'];baserows=[r for r in reports[base,seed]['rows'] if r['domain']=='jd']
            ratios=[v for r in rows for v in r['spread_ratio_by_region'] if v is not None]
            def false(rs):
                den=sum(r['false_hold_denominator'] for r in rs);return sum(r['false_hold_count'] for r in rs)/den if den else None
            f,fb=false(rows),false(baserows)
            shifted=[r for r in rows if 'half_song_feature_shift_error' in r];correct=np.mean([r['profile_error'] for r in shifted]);shift=np.mean([r['half_song_feature_shift_error'] for r in shifted])
            fdrows=[r for r in reports[arm,seed]['rows'] if r['domain']=='fd'];fdbase={(r['key'],r['draw_seed']):r for r in reports[base,seed]['rows'] if r['domain']=='fd'}
            physical_rows=[]
            for r in fdrows:
                c=r['physical'];b0=fdbase[r['key'],r['draw_seed']]['physical'];g=physical.comparison_gates(c,b0)['gates'];names=('finite_body','bone_nonregression','supported_slide_nonregression','jerk_nonregression')
                support=c['contacts']['geometric_support_fraction']>=b0['contacts']['geometric_support_fraction']-.05
                floor=c['geometry']['floor_penetration_m']['p95']<=b0['geometry']['floor_penetration_m']['p95']+.02
                physical_rows.append({'key':r['key'],'draw_seed':r['draw_seed'],'gates':{k:g[k] for k in names},'support_coverage_pass':support,'floor_penetration_pass':floor,'pass':all(g[k]['status']=='pass' for k in names) and support and floor})
            gates={'magnitude':fields['profile_error']['magnitude_pass'] and all(r['registered_complete_coverage'] for r in rows+baserows),'complete_registered_coverage':all(r['registered_complete_coverage'] for r in rows+baserows),'distribution':fields['distribution_error']['after']<=fields['distribution_error']['before'],'spread':.5<=float(np.median(ratios))<=2,'false_holds':f<=fb+.02 if f is not None and fb is not None else None,'diversity':fields['conditional_diversity']['after']>=.95*fields['conditional_diversity']['before'],'audio_timing':bool(correct<shift),'finite_active':all(r['finite'] and not r['constant_pose'] for r in rows),'FD_physics':all(r['pass'] for r in physical_rows)}
            plan_gain=None
            if arm=='joint':
                plan_gain=float(np.mean([r['half_song_plan_shift_error']-r['profile_error'] for r in shifted]));gates['plan_timing']=plan_gain>0
            seeds.append({'seed':seed,'metrics':fields,'spread_ratio_median':float(np.median(ratios)),'false_hold_fraction':f,'reference_false_hold_fraction':fb,'audio_shift_error':float(shift),'audio_correct_error':float(correct),'plan_shift_gain':plan_gain,'gates':gates,'FD_physics':physical_rows})
        comparisons[arm]={'comparator':base,'seeds':seeds,'validation_gate_pass':all(all(v is True for v in s['gates'].values()) for s in seeds)}
    output={'analysis_sha256':sha(__file__),'comparisons':comparisons,'group_means':{f'{a}_seed{s}':v for (a,s),v in means.items()},'report_hashes':{str(root/'evaluation_v5'/a/f'seed_{s}'/'REPORT.json'):sha(root/'evaluation_v5'/a/f'seed_{s}'/'REPORT.json') for a,s in reports},'not_yet_included':'Full-song physical/DMX sensitivity, media verification and artistic review; validation gates do not promote a baseline.'}
    write(root/'reports/VALIDATION.json',output);return output
