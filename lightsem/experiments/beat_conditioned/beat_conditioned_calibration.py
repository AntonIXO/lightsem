"""Before training: demonstrate that separate gates reject known failures."""
from pathlib import Path
import gc
import json
import wave
import numpy as np
from .beat_conditioned_data import ROOT
from ..stage11.stage11_common import read_authenticated,authenticated_json,gpu_lock,digest
from .beat_conditioned_render import controls_to_dmx,isolate_heads,render_gpu
from ..quality_observer import beam_observation,video_observation,phase_explanation
from lightsem.conductor.rig import canonical_rig

def run(root=ROOT):
    root=Path(root);out=root/'calibration';out.mkdir(exist_ok=True)
    norms=read_authenticated(root/'normalizers.json');m=read_authenticated(root/'data_manifest.json')
    groups={}
    for e in sorted(m['entries'],key=lambda e:e['key']):
        if e['partition']=='train':groups.setdefault(e['canonical_id'],e)
    entries=[groups[k] for k in sorted(groups)[:3]]
    length=800;duration=20.;rig=canonical_rig();names=norms['target_names']
    silent=out/'silence.wav'
    if not silent.exists():
        with wave.open(str(silent),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(44100);f.writeframes(bytes(44100*20*4))
    cases={};identities={}
    for index,e in enumerate(entries):
        with np.load(e['path']) as z:
            start=(len(z['y'])-length)//4;y=(z['y'][start:start+length]-norms['y_center'])/norms['y_scale']
        label=f'train_oracle_{index}'
        cases[label]=isolate_heads(controls_to_dmx(y,names,duration)[0]);identities[label]={'key':e['key'],'sha256':e['sha256'],'start_frame':start}
    zeros=np.zeros((length,21));base=isolate_heads(controls_to_dmx(zeros,names,duration)[0])
    cases['constant']=base.copy()
    t=np.arange(length)/40.;phase=2*np.pi*2*t
    pulse=base.copy()
    for f in rig.fixtures:
        if f.kind=='moving_head':pulse[:,f.address+3]=np.rint(base[:,f.address+3]*(.55+.45*np.sin(phase))).astype('uint8')
    cases['brightness_pulse_only']=pulse
    periodic=zeros.copy();periodic[:,names.index('lateral_pos')]=np.sin(phase);periodic[:,names.index('vert_bounce')]=np.cos(phase)
    cases['bpm_only_sweep']=isolate_heads(controls_to_dmx(periodic,names,duration)[0])
    jitter=base.copy()
    for f in rig.fixtures:
        if f.kind!='moving_head':continue
        a=f.address-1;p=base[:,a].astype(int)*256+base[:,a+1].astype(int)
        ticks=np.rint(.12*np.sin(2*np.pi*10*t)*65535/f.pan_range_deg).astype(int);p+=ticks
        jitter[:,a]=(p>>8).astype('uint8');jitter[:,a+1]=(p&255).astype('uint8')
    cases['small_fast_jitter']=jitter
    records={}
    for label,frames in cases.items():
        path=out/(label+'.mp4');np.save(out/(label+'.npy'),frames)
        if not path.exists():
            render_gpu(frames,silent,path,duration,rig=rig)
            gc.collect()
        screen=video_observation(path);beam=beam_observation(frames,rig)
        from ..quality_observer import beam_series
        directions,_=beam_series(frames,rig)
        phase_fit=phase_explanation(directions.reshape(length,-1),np.sin(phase),np.cos(phase),(t*2/4)%1)
        row={'beam':beam,'screen':screen['summary'],'phase':phase_fit,'video':str(path)}
        records[label]=row;authenticated_json(out/(label+'.json'),row)
        print(json.dumps({'case':label,**row}),flush=True)
    report={'purpose':'TRAIN-only and known-null observer calibration before candidate training',
            'selection':'first3 sorted canonical TRAIN songs, first sorted coach, quarter-track 20s window',
            'identities':identities,'records':records,'observer_sha256':digest(Path(__file__).with_name('quality_observer.py'))}
    authenticated_json(out/'report.json',report)
    return report

if __name__=='__main__':
    with gpu_lock(ROOT):run()
