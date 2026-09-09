"""Fixed complete evaluation of all12cells; native claims and rig changes stay separate."""
from pathlib import Path
from dataclasses import replace
import argparse,gc,json,time
import numpy as np
import torch
from .night_common import ROOT,BASE,STAGE,ARMS,SEEDS,base_context
from .night_training import study_context
from .night_model import NightModel
from .night_auxiliary import auxiliary_for_entry
from .night_render import ExpressionConfig,compose_scene,isolate_role,render_video,composition_observation,head_dimmers
from .night_calibration import screen_statistics
from .night_metrics import song_metrics,aggregate,off_interval_checks
from .beat_conditioned_model import noise_for,BeatConditionedModel
from .beat_conditioned_render import controls_to_dmx as old_map
from .quality_observer import beam_observation,video_observation,beam_series
from .stage11_common import (authenticated_json,read_authenticated,digest,artifact_manifest,verify_artifacts,gpu_lock,seed_all,json_digest)
from lightsem.conductor.rig import canonical_rig

DRAW_SEEDS=(101,102,103,104,105,106)
MODES=('full','noise2','zero','beat_only6','donor_full','donor_clock_fixed','shift_four_bars','aux_only_donor','core_only_donor')

@torch.no_grad()
def generate_groups(model,x,seeds=DRAW_SEEDS,steps=8,chunk=640):
    """Reuse the audio encoder; every fixture style is a whole temporal model draw."""
    if next(model.parameters()).device.type!='cuda':raise RuntimeError('GPU model required')
    model.eval();x=x.cuda();h=model.encode_audio_chunked(x,chunk,output_device='cuda')
    z=torch.cat([noise_for(x.shape[1],s) for s in seeds],0).cuda();hh=h.expand(len(seeds),-1,-1)
    radius=model.cfg.velocity_radius
    for step in range(steps):
        next_z=torch.empty_like(z)
        for start in range(0,x.shape[1],chunk):
            end=min(start+chunk,x.shape[1]);lo=max(0,start-radius);hi=min(x.shape[1],end+radius)
            v=model.velocity_from_h(hh[:,lo:hi],z[:,lo:hi],step/steps)
            next_z[:,start:end]=z[:,start:end]+v[:,start-lo:end-lo]/steps
        z=next_z
    a=model.a_head(h).expand(len(seeds),-1,-1);b=torch.sigmoid(model.b_head(h))[0]
    out=torch.cat((a,z),2)
    if not torch.isfinite(out).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite generated controls')
    return out.cpu().numpy(),b.cpu().numpy()

def conditioning_modes(raw,aux,donor,donor_aux,means,aux_mean):
    period=float(np.median(raw[:,91]));shift=int(round(16*period*40))
    # Avoid a full-period circular no-op on a short clip.
    shift=min(max(40,shift),len(raw)-40)
    modes={'full':(raw.copy(),aux.copy()),'noise2':(raw.copy(),aux.copy()),
           'donor_full':(donor.copy(),donor_aux.copy())}
    modes['zero']=(np.broadcast_to(means,raw.shape).copy(),np.broadcast_to(aux_mean,aux.shape).copy())
    beat=raw.copy();beat[:,:86]=means[:86]
    modes['beat_only6']=(beat,np.broadcast_to(aux_mean,aux.shape).copy())
    swapped=raw.copy();swapped[:,:86]=donor[:,:86]
    modes['donor_clock_fixed']=(swapped,donor_aux.copy())
    shifted=raw.copy();shifted[:,:86]=np.roll(raw[:,:86],shift,axis=0)
    modes['shift_four_bars']=(shifted,np.roll(aux,shift,axis=0))
    modes['aux_only_donor']=(raw.copy(),donor_aux.copy())
    modes['core_only_donor']=(swapped.copy(),aux.copy())
    for key in ('beat_only6','donor_clock_fixed','shift_four_bars','aux_only_donor','core_only_donor'):
        if not np.array_equal(modes[key][0][:,86:],raw[:,86:]):raise ValueError('Explicit rhythm channels changed')
    return modes,shift

def normalized_input(raw,aux,n,auxnorm,aux_missing=False):
    x=(raw.astype(np.float32)-np.asarray(n['x_mean'],np.float32))/np.asarray(n['x_std'],np.float32)
    if auxnorm is not None:
        a=aux.astype(np.float16).astype(np.float32)
        a=(a-np.asarray(auxnorm['mean'],np.float32))/np.asarray(auxnorm['std'],np.float32)
        if aux_missing:a[:]=0.
        x=np.concatenate((x,a),1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite normalized features')
    return torch.from_numpy(x.astype(np.float32,copy=False))[None]

def load_cell(arm,seed,root=ROOT):
    root=Path(root);p,m,n,identity=study_context(root);run=root/'runs'/arm/f'seed_{seed}'
    report=read_authenticated(run/'report.json');verify_artifacts(report['artifacts'])
    blob=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    if blob['step']!=p['training']['steps'] or blob['identity']['arm']!=arm or blob['identity']['seed']!=seed:
        raise ValueError('Checkpoint identity mismatch')
    if blob['identity']['protocol_sha256']!=identity['protocol_sha256']:raise ValueError('Checkpoint protocol mismatch')
    family,architecture=arm.split('_');dim=blob['identity']['model_config']['aux_dim'];seed_all(seed)
    model=NightModel(family,architecture,dim);model.load_state_dict(blob['model']);model=model.cuda().eval()
    if json_digest(model.config_dict())!=json_digest(blob['identity']['model_config']):raise ValueError('Architecture config differs')
    return model,digest(run/'final.pt'),p,n

def inputs(root=ROOT):
    rows=read_authenticated(BASE/'evaluation/input_manifest.json')['rows']
    for row in rows:
        if row['entry']['partition']!='val' or digest(row['path'])!=row['sha256']:raise ValueError('Evaluation cohort drift')
    return rows

def prepare_inputs(root=ROOT):
    root=Path(root);m,n=base_context();rows=inputs(root);result={}
    for family in ('base','fourier','muq'):
        auxnorm=None if family=='base' else read_authenticated(root/'features'/family/'training_array.json')
        loaded=[]
        for row in rows:
            with np.load(row['path']) as z:raw=z['x']
            loaded.append((raw,auxiliary_for_entry(family,row['entry'],root,row['start_frame'],row['frames'])))
        for i,row in enumerate(rows):
            raw,aux=loaded[i];tempo=1/np.median(raw[:,91])
            di=min([j for j in range(len(rows)) if j!=i],key=lambda j:(abs(np.log((1/np.median(loaded[j][0][:,91]))/tempo)),rows[j]['entry']['canonical_id']))
            modes,shift=conditioning_modes(raw,aux,*loaded[di],np.asarray(n['x_mean'],np.float32),
                         np.zeros(0,np.float32) if auxnorm is None else np.asarray(auxnorm['mean'],np.float32))
            files={};folder=root/'evaluation/inputs'/family/row['entry']['key'];folder.mkdir(parents=True,exist_ok=True)
            for mode,(xx,aa) in modes.items():
                path=folder/(('full' if mode=='noise2' else mode)+'.npz')
                if not path.exists():np.savez_compressed(path,raw_audio=xx,auxiliary=aa.astype(np.float16))
                else:
                    with np.load(path) as z:
                        if not np.array_equal(z['raw_audio'],xx) or not np.array_equal(z['auxiliary'],aa.astype(np.float16)):raise ValueError('Prepared input differs')
                files[mode]={'path':str(path),'sha256':digest(path),'aux_missing':mode in ('zero','beat_only6')}
            result[family+'/'+row['entry']['key']]={'files':files,'shift_frames':shift,'donor_key':rows[di]['entry']['key']}
    r={'rows':result,'source_sha256':digest(__file__),'selection':'fixed40VALcohort; closest-tempo donor among different canonical songs; no outcomes used'}
    authenticated_json(root/'evaluation/input_manifest.json',r);return r

def infer_cell(arm,seed,root=ROOT):
    root=Path(root);folder=root/'evaluation'/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    model,checkpoint,p,n=load_cell(arm,seed,root);rig=canonical_rig();cfg=ExpressionConfig(**read_authenticated(root/'expression_config.json'))
    family=arm.split('_')[0];auxnorm=None if family=='base' else read_authenticated(root/'features'/family/'training_array.json')
    rows=inputs(root);prepared=read_authenticated(root/'evaluation/input_manifest.json')
    reports=[];begin=time.monotonic()
    for i,row in enumerate(rows):
        key=row['entry']['key'];dest=folder/key;dest.mkdir(exist_ok=True);rp=dest/'report.json'
        if rp.exists():
            r=read_authenticated(rp)
            if r['checkpoint_sha256']!=checkpoint:raise RuntimeError('Old evaluation belongs to another checkpoint')
            verify_artifacts(r['artifacts']);reports.append(r);continue
        if any(dest.iterdir()):raise RuntimeError('Unverified partial inference: '+str(dest))
        with np.load(row['path']) as z:
            raw=z['x'];truth=(z['y'].astype(np.float32)-np.asarray(n['y_center'],np.float32))/np.asarray(n['y_scale'],np.float32);b=z['b']
        source=prepared['rows'][family+'/'+key]
        preds={};events={};dmx={};paths=[];full_trace=None
        for mode in MODES:
            file=source['files'][mode]
            if digest(file['path'])!=file['sha256']:raise RuntimeError('Prepared conditioning drift')
            with np.load(file['path']) as z:xx=z['raw_audio'];aa=z['auxiliary']
            seeds=tuple(s+1000 for s in DRAW_SEEDS) if mode=='noise2' else DRAW_SEEDS
            y,event=generate_groups(model,normalized_input(xx,aa,n,auxnorm,file['aux_missing']),seeds=seeds,steps=8)
            frames,trace,mapping=compose_scene(y,n['target_names'],len(raw)/40.,event_scores=event,rig=rig,config=cfg)
            preds[mode]=y;events[mode]=event;dmx[mode]=frames
            if mode=='full':full_trace=trace
            path=dest/(mode+'.npz');np.savez_compressed(path,standardized=y,event_scores=event,dmx=frames,
                noise_seeds=np.asarray(seeds),**(trace if mode=='full' else {}))
            paths.append(path)
        metrics=song_metrics(truth,preds,events,b,dmx,full_trace,raw,n['target_names'],rig)
        r={'key':key,'canonical_id':row['entry']['canonical_id'],'arm':arm,'seed':seed,'checkpoint_sha256':checkpoint,
           'input_sha256':row['sha256'],'donor_key':source['donor_key'],'shift_frames':source['shift_frames'],'metrics':metrics,
           'conditioning_inputs':source['files'],
           'artifacts':artifact_manifest(paths),'protocol_sha256':digest(root/'protocol.json')}
        authenticated_json(rp,r);reports.append(r)
        print(json.dumps({'stage':'evaluate','arm':arm,'seed':seed,'song':i+1,'energy_score':metrics['energy_score']['full'],
                          'controlled':metrics['controlled'],'alignment':metrics['alignment_advantage']}),flush=True)
    report={'arm':arm,'seed':seed,'rows':reports,'checkpoint_sha256':checkpoint,'gpu_task_seconds':time.monotonic()-begin,
            'source_sha256':digest(__file__),'conditional_audio_claim':'hybrid clock-held tests are diagnostic, not rhythm-free causal interventions'}
    authenticated_json(folder/'inference_report.json',report);del model;gc.collect();torch.cuda.empty_cache();return report

def render_cell(arm,seed,root=ROOT):
    from .stage11_data import load_manifest
    root=Path(root);folder=root/'evaluation'/arm/f'seed_{seed}';p=read_authenticated(root/'protocol.json');m,n=base_context()
    cfg=ExpressionConfig(**read_authenticated(root/'expression_config.json'));rig=canonical_rig();reports=[]
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']}
    for row in inputs(root):
        key=row['entry']['key']
        if key not in p['evaluation']['render_entries']:continue
        dest=folder/key;rp=dest/'video_report.json'
        if rp.exists():
            r=read_authenticated(rp);verify_artifacts(r['artifacts']);reports.append(r);continue
        with np.load(dest/'full.npz') as z:y=z['standardized'][:,:800];event=z['event_scores'][:800]
        d,t,_=compose_scene(y,n['target_names'],20.,event_scores=event,config=cfg)
        video=dest/'heads_only.mp4';r=records[row['entry']['recording_id']]
        render=render_video(isolate_role(d,'moving_head'),r['wav_path'],video,20.,audio_offset_s=row['entry']['offset_s']+row['start_frame']/40.)
        observed=video_observation(video)
        independent,itrace,_=compose_scene(y,n['target_names'],20.,event_scores=event,config=replace(cfg,composition='independent'))
        mirror,mtrace,_=compose_scene(y,n['target_names'],20.,event_scores=event,config=replace(cfg,composition='mirror'))
        mirror_error=max(float(abs(mtrace['targets'][:,i,0]+mtrace['targets'][:,5-i,0]).max()) for i in range(3))
        rays,_=beam_series(mirror,rig);angle_errors=[]
        for i in range(3):
            reflected=rays[:,5-i].copy();reflected[:,0]*=-1
            angle_errors.append(np.degrees(np.arccos(np.clip((rays[:,i]*reflected).sum(1),-1,1))))
        r={'key':key,'beam':beam_observation(d,rig),'screen':observed['summary'],'screen_brightness':screen_statistics(video),
           'composition':composition_observation(d,t,rig),'off_intervals':off_interval_checks(head_dimmers(d,rig)),
           'independent':composition_observation(independent,itrace,rig),'mirror_target_error_m':mirror_error,
           'mirror_emitted_ray_error_deg':float(np.max(angle_errors)),
           'render':render,'artifacts':artifact_manifest([video])}
        authenticated_json(rp,r);reports.append(r)
    authenticated_json(folder/'video_report.json',{'rows':reports,'source_sha256':digest(__file__)})
    return reports

def incumbent(root=ROOT):
    from .stage11_data import load_manifest
    root=Path(root);p=read_authenticated(root/'protocol.json');m,n=base_context();rig=canonical_rig()
    folder=root/'evaluation/incumbent_adapter';folder.mkdir(exist_ok=True);cfg=ExpressionConfig(**read_authenticated(root/'expression_config.json'))
    blob=torch.load(BASE/'run/final.pt',map_location='cpu',weights_only=False);seed_all(0)
    model=BeatConditionedModel();model.load_state_dict(blob['model']);model=model.cuda().eval()
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']};rows=[]
    for row in inputs(root):
        key=row['entry']['key']
        if key not in p['evaluation']['render_entries']:continue
        dest=folder/key;dest.mkdir(exist_ok=True);rp=dest/'report.json'
        if rp.exists():r=read_authenticated(rp);verify_artifacts(r['artifacts']);rows.append(r);continue
        with np.load(row['path']) as z:raw=z['x']
        y,b=generate_groups(model,normalized_input(raw,np.zeros((len(raw),0),np.float32),n,None))
        prior=read_authenticated(BASE/'evaluation/candidate'/key/'report.json');verify_artifacts(prior['artifacts'])
        with np.load(BASE/'evaluation/candidate'/key/'full.npz') as z:before=z['dmx'][:800];old_controls=z['standardized']
        reproduction=float(abs(y[0]-old_controls).max())
        if reproduction>5e-5:raise ValueError('Incumbent output reproduction failed')
        after,trace,_=compose_scene(y[:,:800],n['target_names'],20.,event_scores=b[:800],config=cfg)
        independent,it,_=compose_scene(y[:,:800],n['target_names'],20.,event_scores=b[:800],config=replace(cfg,composition='independent'))
        rr=records[row['entry']['recording_id']];r={'key':key,'reproduction_max_abs':reproduction,'checkpoint_sha256':digest(BASE/'run/final.pt'),'modes':{}};art=[]
        for mode,frames,tr in [('before',before,{}),('after',after,trace)]:
            v=dest/(mode+'_heads.mp4');info=render_video(isolate_role(frames,'moving_head'),rr['wav_path'],v,20.,audio_offset_s=row['entry']['offset_s']+row['start_frame']/40.)
            r['modes'][mode]={'beam':beam_observation(frames,rig),'screen':video_observation(v)['summary'],
                'composition':composition_observation(frames,tr,rig),'off_intervals':off_interval_checks(head_dimmers(frames,rig)),'render':info}
            art.append(v)
        array=dest/'controls.npz';np.savez_compressed(array,standardized=y,event_scores=b,before=before,after=after,**trace);art.append(array)
        r['independent']=composition_observation(independent,it,rig);r['artifacts']=artifact_manifest(art)
        authenticated_json(rp,r);rows.append(r)
    authenticated_json(folder/'report.json',{'rows':rows,'comparison':'same incumbent checkpoint; old single trajectory adapter versus new multi-draw group adapter',
        'source_sha256':digest(__file__)})

def summarize(root=ROOT):
    root=Path(root);cells=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=root/'evaluation'/arm/f'seed_{seed}';inf=read_authenticated(folder/'inference_report.json');vid=read_authenticated(folder/'video_report.json')
            if len(inf['rows'])!=40 or len(vid['rows'])!=8:raise ValueError('Incomplete evaluation matrix')
            if inf['arm']!=arm or inf['seed']!=seed or any(r['arm']!=arm or r['seed']!=seed or r['checkpoint_sha256']!=inf['checkpoint_sha256'] for r in inf['rows']):raise ValueError('Evaluation identity mismatch')
            if {r['key'] for r in inf['rows']}!={r['entry']['key'] for r in inputs(root)}:raise ValueError('Evaluation membership mismatch')
            result=aggregate(inf['rows'],vid['rows']);result.update(arm=arm,seed=seed,checkpoint_sha256=inf['checkpoint_sha256'])
            result['gates']['group_independence']=bool(np.mean([r['independent']['unrelated_velocity_abs_correlation'] is not None and r['independent']['unrelated_velocity_abs_correlation']<.85 for r in vid['rows']])>=.75)
            result['gates']['mirror_geometry']=all(r['mirror_target_error_m']<1e-8 and r['mirror_emitted_ray_error_deg']<.05 for r in vid['rows'])
            result['gates']['integrity']=True;result['all_gates_pass']=all(result['gates'].values())
            authenticated_json(folder/'summary.json',result);cells.append(result)
    eligible=[arm for arm in ARMS if all(c['all_gates_pass'] for c in cells if c['arm']==arm)]
    chosen=min(eligible,key=lambda a:np.mean([c['mean_energy_score'] for c in cells if c['arm']==a])) if eligible else None
    report={'cells':cells,'eligible_arms':eligible,'chosen_arm':chosen,'chosen_seed':0 if chosen else None,
            'selection':'lowest native energy score among arms passing all mandatory gates in BOTH seeds; seed0 fixed for demo',
            'fallback':'previous fixed checkpoint with new adapter if no new arm qualifies','source_sha256':digest(__file__)}
    authenticated_json(root/'evaluation/report.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','render','incumbent','summarize']);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.stage=='prepare':prepare_inputs();return
    if a.stage=='summarize':summarize();return
    with gpu_lock(ROOT):
        if a.stage=='incumbent':incumbent();return
        for seed in ([a.seed] if a.seed is not None else SEEDS):
            for arm in ([a.arm] if a.arm else ARMS):
                if a.stage=='infer':infer_cell(arm,seed)
                else:render_cell(arm,seed)

if __name__=='__main__':main()
