"""Compare already GPU-rendered movies and observe the entire track on CPU."""
from pathlib import Path
import argparse,csv,json,subprocess,sys,time
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import authenticated_json,read_authenticated,digest,verify_artifacts,gpu_lock
FOLDER=ROOT/'delivery/shpongle'

def composite():
    if not read_authenticated(ROOT/'pipeline_state.json').get('gpu_pipeline_complete'):raise RuntimeError('Main GPU pipeline must finish first')
    r=read_authenticated(FOLDER/'render_report.json');a=r['videos']['previous_same_renderer'];b=r['videos']['expression_v4']
    verify_artifacts(a['artifacts']);verify_artifacts(b['artifacts']);dest=FOLDER/'before_after.mp4';rp=FOLDER/'comparison.json'
    if rp.exists():
        old=read_authenticated(rp)
        if digest(dest)!=old['sha256']:raise RuntimeError('Comparison drift')
        return
    font='/usr/share/fonts/TTF/DejaVuSans.ttf'
    filter_=(f"[0:v]scale=960:540,drawtext=fontfile={font}:text='Previous adapter':x=22:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[v0];"
             f"[1:v]scale=960:540,drawtext=fontfile={font}:text='Expression v4':x=22:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.65[v1];[v0][v1]hstack=inputs=2[v]")
    cmd=['ffmpeg','-v','error','-nostdin','-n','-threads','2','-i',a['video'],'-threads','2','-i',b['video'],
         '-filter_complex_threads','2','-filter_complex',filter_,'-map','[v]','-map','1:a:0','-c:v','av1_nvenc','-preset','p5',
         '-cq','26','-b:v','0','-pix_fmt','yuv420p','-c:a','copy',str(dest)]
    begin=time.monotonic();subprocess.run(cmd,check=True)
    authenticated_json(rp,{'path':str(dest),'sha256':digest(dest),'argv':cmd,'gpu_task_seconds':time.monotonic()-begin,
        'sources':{a['video']:digest(a['video']),b['video']:digest(b['video'])},
        'interpretation':'Only scaling, labels and composition of existing GPU stage frames. Original full movies remain1920x1080; comparison is1920x540 with both entire tracks.'})

def timeline(path):
    cmd=['ffmpeg','-v','error','-threads','2','-i',str(path),'-vf','fps=2,scale=214:120','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    raw=subprocess.check_output(cmd);rgb=np.frombuffer(raw,np.uint8).reshape(-1,120,214,3).astype(np.float32)/255
    if len(rgb)<500:raise RuntimeError('Incomplete full movie')
    luma=rgb@np.array([.2126,.7152,.0722]);border=np.concatenate((luma[:,:,:53],luma[:,:,161:]),axis=2)
    return {'time_s':np.arange(len(rgb))/2,'mean_luma':luma.mean((1,2)),'border_luma':border.mean((1,2)),
        'bright_area':(luma>.4).mean((1,2)),'white_clipping':(rgb.min(-1)>.95).mean((1,2))}

def observe():
    r=read_authenticated(FOLDER/'render_report.json');output={};rows=[]
    for mode in ('previous_same_renderer','expression_v4'):
        path=r['videos'][mode]['video'];data=timeline(path);output[mode]=data
        for i,t in enumerate(data['time_s']):rows.append({'mode':mode,**{k:float(v[i]) for k,v in data.items()}})
    if len(output['previous_same_renderer']['time_s'])!=len(output['expression_v4']['time_s']):raise ValueError('Unmatched comparison timeline')
    intervals=[];duration=read_authenticated(FOLDER/'inference_report.json')['input']['source_duration_s']
    for start in np.arange(0,duration,30):
        end=min(start+30,duration);values={}
        for mode,data in output.items():
            keep=(data['time_s']>=start)&(data['time_s']<end)
            values[mode]={k:float(v[keep].mean()) for k,v in data.items() if k!='time_s'}
        before=values['previous_same_renderer']['border_luma'];after=values['expression_v4']['border_luma']
        intervals.append({'start_s':float(start),'end_s':float(end),'values':values,'border_luma_reduction':float((before-after)/before) if before>1e-8 else None})
    path=FOLDER/'whole_track_luma.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    authenticated_json(FOLDER/'whole_track_observation.json',{'intervals':intervals,'frames_per_movie':len(output['expression_v4']['time_s']),
        'fps':2,'csv':str(path),'csv_sha256':digest(path),'source_sha256':digest(__file__),'interpretation':'Whole-track30s descriptive windows; original final60s glare gate remains unchanged.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d
    fig,axes=plt.subplots(2,1,figsize=(10,4),sharex=True)
    for mode,label,color in [('previous_same_renderer','Previous adapter','#8c8c8c'),('expression_v4','Expression v4','#285875')]:
        data=output[mode]
        axes[0].plot(data['time_s'],uniform_filter1d(data['border_luma'],10),label=label,color=color,lw=1)
        axes[1].plot(data['time_s'],uniform_filter1d(data['bright_area'],10),label=label,color=color,lw=1)
    for ax in axes:ax.axvspan(duration-60,duration,color='#ededed',zorder=-1);ax.spines[['top','right']].set_visible(False)
    axes[0].set(ylabel='Border luminance',title='Full Shpongle comparison; five-second display smoothing')
    axes[1].set(xlabel='Track time (seconds)',ylabel='Bright area fraction');axes[0].legend(frameon=False)
    fig.tight_layout();fig.savefig(FOLDER/'whole_track_luma.png',dpi=170);fig.savefig(FOLDER/'whole_track_luma.pdf');plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['composite','observe']);a=ap.parse_args()
    if a.stage=='observe':observe()
    else:
        with gpu_lock(ROOT):composite()
