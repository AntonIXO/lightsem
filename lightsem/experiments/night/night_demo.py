"""Full user-track delivery and an isolated map-only comparison."""
from pathlib import Path
import argparse,gc,json,time
import numpy as np
import torch
from .night_common import ROOT,BASE,base_context
from .night_evaluation import generate_groups,normalized_input,load_cell,DRAW_SEEDS
from .night_render import ExpressionConfig,compose_scene,render_video,isolate_role,composition_observation
from .night_calibration import screen_statistics
from .night_auxiliary import interpolate_native
from .night_features import fourier_features
from ..beat_conditioned.beat_conditioned_model import BeatConditionedModel
from ..quality_observer import beam_observation,video_observation
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest,artifact_manifest,verify_artifacts,gpu_lock,seed_all
from lightsem.conductor.rig import canonical_rig

def demo_inputs():
    meta=read_authenticated(BASE/'inference/shpongle_input.json');audio=read_authenticated(BASE/'protocol.json')['demo']['audio_path']
    if digest(audio)!=meta['source_audio_sha256'] or digest(meta['features_path'])!=meta['features_sha256']:raise RuntimeError('Shpongle input drift')
    with np.load(meta['features_path']) as z:raw=z['x']
    return meta,audio,raw

def prepare_muq(root=ROOT):
    import soundfile as sf,soxr
    from muq import MuQ
    from .night_embeddings import extract_waveform,MUQ_REVISION
    root=Path(root);meta,audio,raw=demo_inputs();info=read_authenticated(root/'features/muq_model.json');verify_artifacts(info['artifacts'])
    layer=read_authenticated(root/'features/muq/layer_selection.json')['selected_layer'];x,sr=sf.read(audio,dtype='float32',always_2d=True);x=x.mean(1)
    if sr!=24000:x=soxr.resample(x,sr,24000)
    seed_all(0);model=MuQ.from_pretrained(info['path']).float().cuda().eval();begin=time.monotonic()
    out=extract_waveform(model,x)[layer];path=root/'features/shpongle_muq.npy';np.save(path,out)
    authenticated_json(root/'features/shpongle_muq.json',{'path':str(path),'sha256':digest(path),'layer':layer,'revision':MUQ_REVISION,
        'audio_sha256':meta['source_audio_sha256'],'native_hz':25,'gpu_seconds':time.monotonic()-begin})

def infer(root=ROOT):
    root=Path(root);folder=root/'delivery/shpongle';folder.mkdir(parents=True,exist_ok=True);m,n=base_context()
    meta,audio,raw=demo_inputs();cfg=ExpressionConfig(**read_authenticated(root/'expression_config.json'));decision=read_authenticated(root/'evaluation/report.json')
    old_report=read_authenticated(BASE/'run/report.json');verify_artifacts(old_report['artifacts'])
    blob=torch.load(BASE/'run/final.pt',map_location='cpu',weights_only=False);seed_all(0);model=BeatConditionedModel();model.load_state_dict(blob['model']);model=model.cuda().eval()
    begin=time.monotonic();old_y,old_b=generate_groups(model,normalized_input(raw,np.zeros((len(raw),0),np.float32),n,None))
    previous_report=read_authenticated(BASE/'previews/shpongle/full_inference.json');verify_artifacts(previous_report['artifacts'])
    with np.load(BASE/'previews/shpongle/full.npz') as z:previous=z['dmx'];previous_y=z['standardized']
    reproduce=float(abs(old_y[0]-previous_y).max())
    if reproduce>5e-5:raise RuntimeError('Incumbent reconstruction differs: '+str(reproduce))
    map_only,mt,mr=compose_scene(old_y,n['target_names'],meta['source_duration_s'],event_scores=old_b,config=cfg)
    np.savez_compressed(folder/'incumbent_new_adapter.npz',standardized=old_y,event_scores=old_b,dmx=map_only,**mt)
    del model,blob;gc.collect();torch.cuda.empty_cache()
    chosen=decision['chosen_arm'];checkpoint=digest(BASE/'run/final.pt')
    if chosen is None:y,b,dmx,trace,mapping=old_y,old_b,map_only,mt,mr
    else:
        model,checkpoint,p,n=load_cell(chosen,0,root);family=chosen.split('_')[0]
        norm=None if family=='base' else read_authenticated(root/'features'/family/'training_array.json')
        if family=='base':aux=np.zeros((len(raw),0),np.float32)
        elif family=='fourier':aux=fourier_features(raw)
        else:
            info=read_authenticated(root/'features/shpongle_muq.json')
            if digest(info['path'])!=info['sha256']:raise RuntimeError('Demo embedding drift')
            aux=interpolate_native(np.load(info['path']),np.arange(len(raw))/40.)
        y,b=generate_groups(model,normalized_input(raw,aux,n,norm));dmx,trace,mapping=compose_scene(y,n['target_names'],meta['source_duration_s'],event_scores=b,config=cfg)
        del model;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(folder/'full.npz',standardized=y,event_scores=b,dmx=dmx,noise_seeds=np.asarray(DRAW_SEEDS),**trace)
    np.save(folder/'previous_dmx.npy',previous)
    r={'chosen_arm':chosen,'uses_incumbent':chosen is None,'checkpoint_sha256':checkpoint,'incumbent_reproduction_max_abs':reproduce,
       'audio_path':audio,'input':meta,'mapping':mapping,'gpu_task_seconds':time.monotonic()-begin,
       'beam':beam_observation(dmx,canonical_rig()),'composition':composition_observation(dmx,trace),
       'artifacts':artifact_manifest([folder/'full.npz',folder/'incumbent_new_adapter.npz',folder/'previous_dmx.npy'])}
    authenticated_json(folder/'inference_report.json',r)

def render(root=ROOT):
    root=Path(root);folder=root/'delivery/shpongle';r=read_authenticated(folder/'inference_report.json');verify_artifacts(r['artifacts'])
    duration=r['input']['source_duration_s'];audio=r['audio_path'];rig=canonical_rig()
    with np.load(folder/'full.npz') as z:dmx=z['dmx']
    with np.load(folder/'incumbent_new_adapter.npz') as z:map_only=z['dmx']
    old=np.load(folder/'previous_dmx.npy');rows={}
    modes={'expression_v4':dmx,'heads_only':isolate_role(dmx,'moving_head'),'previous_same_renderer':old}
    if not r['uses_incumbent']:modes['incumbent_new_adapter']=map_only
    for name,frames in modes.items():
        rp=folder/(name+'_video.json')
        if rp.exists():record=read_authenticated(rp);verify_artifacts(record['artifacts']);rows[name]=record;continue
        video=folder/(name+'.mp4');info=render_video(frames,audio,video,duration,width=1920,height=1080)
        record={'video':str(video),'render':info,'artifacts':artifact_manifest([video])};authenticated_json(rp,record);rows[name]=record
        print(json.dumps({'stage':'full_demo_render','name':name,'seconds':info['gpu_task_seconds']}),flush=True)
    # Same ending interval and audio for role-isolated diagnosis. No generated ending fade.
    roles={};count=2400;start=len(old)-count;offset=start/40.
    for kind in ('moving_head','wash','blinder','strobe','projector'):
        rp=folder/('ending_'+kind+'.json')
        if rp.exists():record=read_authenticated(rp);verify_artifacts(record['artifacts']);roles[kind]=record;continue
        path=folder/('ending_'+kind+'.mp4');frames=isolate_role(old[start:],kind)
        info=render_video(frames,audio,path,60.,audio_offset_s=offset)
        record={'render':info,'screen':screen_statistics(path),'artifacts':artifact_manifest([path]),
                'interpretation':'isolation changes the renderer global light normalization; contributions are not additive'}
        authenticated_json(rp,record);roles[kind]=record
    authenticated_json(folder/'render_report.json',{'videos':rows,'ending_roles':roles,'source_sha256':digest(__file__)})

def observe(root=ROOT):
    root=Path(root);folder=root/'delivery/shpongle';r=read_authenticated(folder/'inference_report.json');videos=read_authenticated(folder/'render_report.json')
    ending={}
    for name in ('expression_v4','previous_same_renderer'):
        path=videos['videos'][name]['video'];ending[name]=screen_statistics(path,r['input']['source_duration_s']-60,60.)
    before=ending['previous_same_renderer']['border_luma_mean'];after=ending['expression_v4']['border_luma_mean']
    reduction=(before-after)/before if before>1e-8 else None
    heads=video_observation(videos['videos']['heads_only']['video'])
    result={'ending':ending,'border_luma_reduction':reduction,'ending_glare_gate':reduction is not None and reduction>=.3,
            'heads_only':heads,'beam':r['beam'],'composition':r['composition'],'uses_incumbent':r['uses_incumbent'],
            'no_ending_time_feature':True,'source_sha256':digest(__file__)}
    authenticated_json(folder/'observation_report.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare_muq','infer','render','observe']);a=p.parse_args()
    if a.stage=='observe':observe()
    else:
        with gpu_lock(ROOT):{'prepare_muq':prepare_muq,'infer':infer,'render':render}[a.stage]()
