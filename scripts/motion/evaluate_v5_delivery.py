"""Paired checkpoint audit and emitted-DMX review; no training or source edits."""
from pathlib import Path
import argparse, hashlib, json, subprocess, shutil
import numpy as np

CAMPAIGN=Path('/mnt/lightsem-data/experiments/lightsem/native_dynamics_full_all_20261007_seed0_v5')
SOURCE=CAMPAIGN/'demo_eval_v8'
OUT=CAMPAIGN/'evaluation_delivery_v1'
SONGS=('shpongle','cure_for_me')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def arrays(path,**kw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**kw)
def world(path):
    with np.load(path) as z:return z['world'].copy()
def bodypath(role,song,condition='correct'):
    if role=='candidate' and condition in ('correct','gain'):
        return SOURCE/'bodies'/('C4' if condition=='correct' else 'C4__gain')/song/'body.npz'
    return OUT/'bodies'/role/song/(condition+'.npz')

def freeze():
    from lightsem.experiments.native_repair.reporting import LIMITS
    from lightsem.experiments.dynamic_dance.diagnostics import GATE_PARAMETERS
    cfg=read('/root/lightsem/configs/dance/fixed_c.json');plan=read(SOURCE/'PLAN.json')
    pins={}
    for p in [Path(__file__),Path('/root/lightsem/configs/dance/fixed_c.json'),Path(cfg['checkpoint']['path']),CAMPAIGN/'learning/runs/joint/seed_0_retry_cublas2/final.pt',SOURCE/'PLAN.json']:
        pins[str(p)]=sha(p)
    assert sha(cfg['checkpoint']['path'])==cfg['checkpoint']['sha256']
    for t in plan['tracks']:
        for cond in ('correct','gain'):
            paths=t if cond=='correct' else t['waveform_controls'][cond]
            for k in ('base92','music_timeline'):
                p=Path(paths[k]);pins[str(p)]=sha(p);pins[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
            p=bodypath('candidate',t['id'],cond);pins[str(p)]=sha(p);pins[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
    save(OUT/'PROTOCOL.json',dict(schema_version=1,config=cfg,tracks=plan['tracks'],pins=pins,draw_seed=7001,noise_frames=8320,
        gates={'raw_repair_ratios':LIMITS,'minimum_activity_ratio':.8,'minimum_variation_ratio':.8,'maximum_support_slide_ratio':1.2,'engineering':GATE_PARAMETERS},
        presentation={'start_s':90.,'duration_s':60.,'fps':30,'width':1280,'height':900},
        scope='Paired same-v2-features checkpoint comparison on two exposed showcases; all-data fit has no independent held-out evaluation. No artistic or generalization qualification.',
        supersedes='demo_eval_v8/evaluation used a historical reference, mismatched sample shapes, wrong channel lists and no dynamics limits; those reports are retained but are not the final verdict.'))

def verify():
    p=read(OUT/'PROTOCOL.json')
    for file,digest in p['pins'].items():
        if sha(file)!=digest:raise ValueError('Changed input '+file)
    if shutil.disk_usage(OUT).free<20*2**30:raise RuntimeError('Disk floor')
    return p

def generate():
    import torch
    from lightsem.dance.pipeline import accelerator_lock
    from lightsem.dance.baseline import load_model
    from lightsem.experiments.native_dynamics.showcase import input_arrays
    from lightsem.experiments.native_dynamics.evaluation import cpu_noise
    from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS
    from lightsem.experiments.motion_study.geometry import joints_from_features
    p=verify();rows=[]
    with accelerator_lock(),torch.inference_mode():
        for role in ('reference','candidate'):
            cfg=json.loads(json.dumps(p['config']))
            if role=='candidate':
                cp=CAMPAIGN/'learning/runs/joint/seed_0_retry_cublas2/final.pt';cfg['checkpoint']={'path':str(cp),'sha256':sha(cp)}
            model,norms,identity=load_model(cfg,device='cuda')
            for track in p['tracks']:
                song=track['id'];n=track['body_frames'];noise=cpu_noise(model.flow,8320,7001)[:,:n].contiguous().cuda()
                noise_sha=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                existing=read(bodypath('candidate',song).with_suffix('.json'))['metadata']
                if noise_sha!=existing['noise_sha256']:raise ValueError('Unpaired noise')
                for cond in (('correct','gain','repeat') if role=='reference' else ('repeat',)):
                    b,e,c,binding=input_arrays(track,cond)
                    fine=torch.cat(((torch.as_tensor(b,device='cuda')-norms[0][0])/norms[0][1],(torch.as_tensor(e,device='cuda')-norms[1][0])/norms[1][1]),-1)[None]
                    coarse=torch.zeros(1,CONTEXT_BINS,1061,device='cuda');valid=torch.zeros(1,CONTEXT_BINS,device='cuda',dtype=torch.bool)
                    coarse[0,:len(c)]=(torch.as_tensor(c,device='cuda')-norms[1][0])/norms[1][1];valid[0,:len(c)]=True
                    native,_=model.generate(fine,coarse,valid,torch.arange(n,device='cuda')[None],noise)
                    w=joints_from_features(native)[0].cpu().numpy();x=native[0].cpu().numpy()
                    if not np.isfinite(w).all():raise ValueError('Nonfinite body')
                    dest=bodypath(role,song,cond);arrays(dest,time=np.arange(n)/30,world=w,native=x,contacts=x[:,-4:])
                    meta=dict(role=role,song=song,condition=cond,checkpoint=identity,audio=binding,noise_sha256=noise_sha,body_sha256=sha(dest),fps=30,vertical_axis='Y')
                    save(dest.with_suffix('.json'),meta);rows.append(meta);print('GENERATED',role,song,cond,flush=True)
            del model;torch.cuda.empty_cache()
    save(OUT/'GENERATION.json',{'rows':rows,'protocol_sha256':sha(OUT/'PROTOCOL.json')})

def map_and_audit():
    from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
    from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
    from lightsem.dance.lighting import _load_p2_colors,_resample_body_full_interval
    from lightsem.conductor.body_transfer_v2 import make_outputs,map_body
    from lightsem.conductor.body_transfer import decode
    from lightsem.experiments.native_repair.assessment import pair_metrics
    from lightsem.experiments.native_repair.reporting import assess_pair
    from lightsem.experiments.dynamic_dance.diagnostics import body_dynamics,comparison_gates
    from lightsem.eval.training_health import collapse_metrics,input_sensitivity,stream_metrics
    p=verify();summary=[];common=min(t['body_frames'] for t in p['tracks'])
    for t in p['tracks']:
        song=t['id'];reference=world(bodypath('reference',song));candidate=world(bodypath('candidate',song));time30=np.arange(len(candidate))/30
        pairs=[]
        for label,mask in [('whole_song',np.ones(len(candidate),bool)),('90_150',(time30>=90)&(time30<150))]:
            pair=pair_metrics(candidate[mask],reference[mask]);pairs.append(dict(interval=label,metrics=pair,gates=assess_pair(pair)))
        physical={role:body_dynamics(w,time30,reference_body=reference,floor_y=0.) for role,w in [('reference',reference),('candidate',candidate)]}
        comparisons=comparison_gates(physical['candidate'],physical['reference'])
        extra={}
        for role,w in [('reference',reference),('candidate',candidate)]:
            repeat=world(bodypath(role,song,'repeat'));gain=world(bodypath(role,song,'gain'))
            extra[role]={'repeat_max_abs_m':float(abs(w-repeat).max()),'gain':input_sensitivity(w,{'gain':gain}),
                         'cross_song_matched_prefix':collapse_metrics([world(bodypath(role,s))[:common] for s in SONGS]),
                         'root_y':{'min':float(w[:,0,1].min()),'max':float(w[:,0,1].max()),'range':float(np.ptp(w[:,0,1]))},
                         'raw_peak_joint_speed':float(np.linalg.norm(np.diff(w,axis=0)*30,axis=-1).max())}
            time40=np.arange(t['lighting_frames'])/40
            w40=_resample_body_full_interval(w,time30,time40)
            asset=p['config']['cached_tracks'][song]
            l,r,palette,active,colorid=_load_p2_colors(asset['colors']['path'],t['audio'],len(time40),expected_sha256=asset['colors']['sha256'])
            loud,loudid=load_verified_loudness(t['loudness_receipt'],t['audio'])
            outputs=make_outputs(**p['config']['lighting']['counts']);transfer=dict(p['config']['lighting']['transfer']);assert transfer['apply_motion_limits'] is False
            dmx,patch,intent,bright,hard=compile_world(w40,time40,outputs,l,r,loud,transfer,mapper_id='01');decoded=decode(dmx,patch)
            folder=OUT/'emitted'/role/song;arrays(folder/'lights.npz',time=time40,universes=dmx,**decoded)
            save(folder/'patch.json',patch)
            # Identical channel layout for comparison; only the 13 moving fixtures
            # have eight channels. Bulb dimmer/RGB are audited separately.
            movers=[f for f in patch if f['kind']!='bulb'];bulbs=[f for f in patch if f['kind']=='bulb']
            def packed(fs):return np.stack([dmx[:,f['universe'],f['address']-1:f['address']-1+f['n_channels']] for f in fs],1)
            arrays(folder/'movers.npz',streams=packed(movers))
            lamp=stream_metrics(packed(movers),rgb_channels=[5,6,7],brightness_channels=[4]);bulb=stream_metrics(packed(bulbs),rgb_channels=[1,2,3],brightness_channels=[0])
            extra[role]['lighting']={'moving_fixtures':lamp,'bulbs':bulb,'hard_checks':hard,'limits_off':True,'palette_ids':np.unique(palette).tolist(),'palette_changes':int(np.count_nonzero(np.diff(palette))),'color':colorid,'loudness':loudid}
            # Mapper interventions hold audio/color/brightness fixed; inspect
            # requested targets and quantized moving angles, not dimmer changes.
            clip=w40[:240];frozen=np.repeat(clip[:1],len(clip),axis=0);altered=clip.copy();altered[:,20,0]+=.10
            raw0=map_body(frozen,outputs,'01',40.);raw1=map_body(clip,outputs,'01',40.);raw2=map_body(altered,outputs,'01',40.)
            d0,p0,_,_,_=compile_world(frozen,time40[:240],outputs,l[:240],r[:240],loud,transfer,mapper_id='01');dec0=decode(d0,p0)
            a0=dec0['angles_deg'][:,:len(movers)]
            shifts=np.max(np.linalg.norm(raw2['targets']-raw1['targets'],axis=-1),axis=0)
            extra[role]['mapper_controls']={'frozen_target_delta':float(abs(np.diff(raw0['targets'],axis=0)).max()),'frozen_emitted_angle_delta':float(abs(np.diff(a0,axis=0)).max()),'left_wrist_changed_fixture_indices':np.flatnonzero(shifts>1e-8).tolist(),'left_wrist_target_max_m':float(shifts.max())}
            for counts in ({'beams':4,'lasers':2,'bulbs':10},{'beams':10,'lasers':6,'bulbs':64}):
                test=map_body(frozen,make_outputs(**counts),'01',40.)
                if not np.allclose(test['targets'],test['targets'][:1]):raise ValueError('Autonomous mapper motion')
            # Flat authenticated provenance for the repo evaluator.
            cp=p['config']['checkpoint']['path'] if role=='reference' else str(CAMPAIGN/'learning/runs/joint/seed_0_retry_cublas2/final.pt')
            prov={}
            for key,path in {'checkpoint':cp,'audio':t['audio'],'body':bodypath(role,song),'stream':folder/'lights.npz','protocol':OUT/'PROTOCOL.json','base92':t['base92'],'muq':t['music_timeline'],'mapper':'/root/lightsem/lightsem/experiments/dynamic_dance_combo/campaign.py'}.items():prov[key+'_path']=str(path);prov[key+'_sha256']=sha(path)
            save(folder/'provenance.json',prov)
            samples={}
            for s in SONGS:
                path=OUT/'audit_samples'/role/(s+'.npz')
                if not path.exists():arrays(path,world=world(bodypath(role,s))[:common])
                samples[s]=str(path)
            # Recorded reference-derived limits are descriptive here; domain
            # repair gates above supply unchanged historical decision rules.
            manifest=dict(schema='lightsem.training_health.manifest.v1',body=str(bodypath(role,song)),stream=str(folder/'movers.npz'),samples=samples,interventions={'same_song_gain':str(bodypath(role,song,'gain'))},provenance=str(folder/'provenance.json'),required_provenance=list(prov),fps=30,rgb_channels=[5,6,7],brightness_channels=[4])
            save(folder/'audit_manifest.json',manifest)
        summary.append(dict(song=song,raw_repair=pairs,physics=physical,engineering=comparisons,diagnostics=extra))
    save(OUT/'DOMAIN_AUDIT.json',{'songs':summary,'baseline':p['config']['checkpoint'],'scope':p['scope'],'height_axis':'Y','baseline_changed':False,'missing_evidence':['Independent musical/choreographic ground truth','held-out generalization (all-data fit)','donor/shuffle waveform controls'], 'prior_reports_superseded':p['supersedes']})
    print('DOMAIN_AUDIT',[(x['song'],[v['gates'] for v in x['raw_repair']]) for x in summary],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','generate','audit']);a=ap.parse_args()
    {'freeze':freeze,'generate':generate,'audit':map_and_audit}[a.stage]()
