"""Raw output diagnostics and unchanged original-time development evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from ..dynamic_dance.diagnostics import body_dynamics

def raw_metrics(world,reference=None):
    w=np.asarray(world,float);q=w-w[:,:1];v=np.diff(q,axis=0)*30;j=np.diff(q,n=3,axis=0)*30**3
    smooth=gaussian_filter1d(q,1.5,axis=0,mode='nearest');slow=gaussian_filter1d(q,3.,axis=0,mode='nearest')
    slow_speed=np.linalg.norm(np.diff(slow,axis=0)*30,axis=-1).mean(-1)
    root=w[:,0,1];root_res=root-gaussian_filter1d(root,1.5,mode='nearest')
    f,p=welch(v,fs=30,nperseg=min(900,len(v)),axis=0);power=p.sum((1,2))
    ref=w if reference is None else np.asarray(reference,float)
    floor=float(np.quantile(ref[:,[7,8,10,11],1],.02));physics=body_dynamics(w,np.arange(len(w))/30,reference_body=ref,floor_y=floor)
    return {'raw_relative_jerk_p95_m_s3':float(np.quantile(np.sqrt(np.mean(np.sum(j*j,axis=-1),axis=-1)),.95)),
            'raw_relative_speed_mean_m_s':float(np.linalg.norm(v,axis=-1).mean()),
            'high_frequency_residual_RMS_m':float(np.sqrt(np.mean((q-smooth)**2))),
            'root_vertical_fast_residual_RMS_m':float(np.sqrt(np.mean(root_res**2))),
            'root_vertical_jerk_p95_m_s3':float(np.quantile(abs(np.diff(root,n=3)*30**3),.95)),
            'low_frequency_speed_spread_m_s':float(np.quantile(slow_speed,.9)-np.quantile(slow_speed,.1)),
            'velocity_power_above6Hz_fraction':float(power[f>=6].sum()/max(power[1:].sum(),1e-15)),
            'geometric_slide_p95_m_s':physics['contacts']['geometric_supported_slide_m_s']['p95'],
            'geometric_support_fraction':physics['contacts']['geometric_support_fraction'],
            'bone_drift_p95_m':physics['geometry']['bone_drift_m']['p95'],
            'floor_penetration_p95_m':physics['geometry']['floor_penetration_m']['p95'],
            'finite':bool(np.isfinite(w).all()),'constant_body':bool(np.all(w==w[:1]))}

def pair_metrics(world,reference):
    a=raw_metrics(world,reference);b=raw_metrics(reference,reference)
    ratios={k:a[k]/max(b[k],1e-12) for k in ('raw_relative_jerk_p95_m_s3','raw_relative_speed_mean_m_s','high_frequency_residual_RMS_m','root_vertical_fast_residual_RMS_m','low_frequency_speed_spread_m_s')}
    return {'candidate':a,'continued_A':b,'ratios':ratios}

def evaluate(root,seed,device="cuda",baseline_root=None):
    import torch
    from ..native_dynamics.evaluation import setup,load_final,cpu_noise,metrics
    from ..native_dynamics.dataset import Records,to_device
    from ..native_dynamics.kinematics import project_native
    from ..native_dynamics.data import write,read,sha,save_npz
    setup();root=Path(root);old=Path(baseline_root) if baseline_root is not None else root.parent;study=read(root/'STUDY_PLAN.json');model,norms,identity=load_final(root,'joint',seed,device=device);baseline,bnorms,bidentity=load_final(old,'continued',seed,device=device)
    assert all(torch.equal(a,b) for group1,group2 in zip(norms,bnorms) for a,b in zip(group1,group2))
    if device=='cpu':torch.set_num_threads(4)
    records=Records(root,'val',frames=960);lookup={(d,r['key']):r for d,rows in records.records.items() for r in rows};folder=root/'evaluation'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=False);rows=[]
    for w in study['evaluation']['windows']:
        record=lookup[w['domain'],w['key']];batch=to_device(records.pack(w['domain'],[(record,w['context_start'])]),*norms,device=device)
        n=w['score_frames'];actual=w['context_frames'];s=slice(w['score_offset'],w['score_offset']+n);domain=w['domain']
        if domain=='jd':target=record['pose'][w['score_start']:w['score_start']+n];mask=record['mask'][w['score_start']:w['score_start']+n]
        else:
            tp,truth=project_native(batch['native'],batch['valid']);target=tp[0,s].cpu().numpy();mask=np.ones((n,13),bool)
        poses=[]
        with torch.inference_mode():
            for draw in study['evaluation']['draw_seeds']:
                noise=cpu_noise(model.flow,960,draw)[:,:actual].contiguous().to(device);inputs=(batch['audio'][:,:actual],batch['coarse'],batch['coarse_valid'],batch['indices'][:,:actual])
                native,plan=model.generate(*inputs,noise);projected,body=project_native(native,batch['valid'][:,:actual]);q=projected[0,s].cpu().numpy();world=body['positions'][0,s].cpu().numpy();poses.append(q)
                a,_=baseline.generate(*inputs,noise);ap,abody=project_native(a,batch['valid'][:,:actual]);reference=abody['positions'][0,s].cpu().numpy()
                measured,traces=metrics(q,target,mask);bm,_=metrics(ap[0,s].cpu().numpy(),target,mask)
                row={**w,'seed':seed,'draw_seed':draw,**measured,'baseline_profile_error':bm['profile_error'],'raw3d':pair_metrics(world,reference),'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()}
                if domain=='jd' and w['window']==1 and draw==7001:
                    save_npz(folder/f"trace_{w['key']}.npz",time=np.arange(n)/30+w['score_start']/30,pose=q,observed=target,mask=mask,world=world,reference_world=reference,**traces)
                    shifted=dict(record);shifted['base']=np.roll(record['base'],record['length']//2,axis=0);shifted['extra']=np.roll(record['extra'],record['length']//2,axis=0);shifted['coarse']=np.roll(record['coarse'],record['bins']//2,axis=0)
                    sb=to_device(records.pack(domain,[(shifted,w['context_start'])]),*norms,device=device)
                    altered,_=model.generate(sb['audio'][:,:actual],sb['coarse'],sb['coarse_valid'],sb['indices'][:,:actual],noise);aq,aw=project_native(altered,batch['valid'][:,:actual]);row['shifted_profile_error']=metrics(aq[0,s].cpu().numpy(),target,mask)[0]['profile_error'];other=aw['positions'][0,s].cpu().numpy();row['audio_shift_relative_body_RMS_m']=float(np.sqrt(np.mean(((other-other[:,:1])-(world-world[:,:1]))**2)))
                rows.append(row)
        diversity=float(np.linalg.norm(poses[0]-poses[1],axis=-1)[mask].mean())
        for row in rows[-2:]:row['conditional_diversity']=diversity
        print(json.dumps({'seed':seed,'completed':len(rows),'key':w['key'],'domain':domain}),flush=True)
    write(folder/'REPORT.json',{'seed':seed,'candidate':identity,'baseline':bidentity,'rows':rows,'data_inputs':records.inputs,'study_sha256':sha(root/'STUDY_PLAN.json'),'raw3D_on_both_domains':True,'inference_device':device,'precision':'FP32 candidate and pairedA on same device','projection_coverage':'Same historical valid masks; observational metrics remain descriptive on available cells, no invented coverage.'})
