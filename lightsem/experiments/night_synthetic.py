"""Known-input rhythm checks and a separate raw-complex-STFT feature audit."""
from pathlib import Path
import json,time
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from .night_common import ROOT,STAGE,base_context
from .night_features import fourier_features,FOURIER_CHANNELS
from .stage11_common import authenticated_json,digest

def complex_novelty(audio,sr):
    import librosa
    s=librosa.stft(audio,n_fft=2048,hop_length=512);mag=np.abs(s);phase=np.angle(s)
    expected=mag[:,1:-1]*np.exp(1j*(2*phase[:,1:-1]-phase[:,:-2]))
    difference=np.abs(s[:,2:]-expected);freq=librosa.fft_frequencies(sr=sr,n_fft=2048)
    bands=[(0,sr/2),(40,150),(150,2000),(2000,sr/2)]
    columns=[]
    for lo,hi in bands:
        values=difference[(freq>=lo)&(freq<hi)].sum(0)
        columns.append(np.pad(values,(2,0)))
    return np.stack(columns,1),np.arange(s.shape[1])*512/sr

def run(root=ROOT):
    import librosa,mir_eval,soundfile as sf
    from lightsem.audio.online import extract_bus
    from .stage11_data import load_manifest
    root=Path(root);folder=root/'diagnostics/rhythm_stress';folder.mkdir(exist_ok=True);sr=44100;duration=40.
    cases={};regular=np.arange(.25,duration,60/120)
    cases['steady_120']=(regular,regular.copy())
    n=np.arange(100);a=.5*(40/60)/duration;b=100/60
    drift=(-b+np.sqrt(b*b+4*a*n))/(2*a);drift=drift[drift<duration]
    cases['tempo_drift_100_140']=(drift,drift.copy())
    step=np.r_[np.arange(.25,20,60/90),np.arange(20.25,duration,60/135)]
    cases['tempo_change_90_135']=(step,step.copy())
    cases['syncopated']=(regular,np.sort(np.r_[regular[::2],regular[1::2]+.25]))
    cases['swing']=(regular,np.sort(np.r_[regular,regular+1/3]))
    cases['silence_break']=(regular[(regular<12)|(regular>28)],regular[(regular<12)|(regular>28)])
    cases['meter_3_then_4']=(regular,regular.copy())
    rows=[]
    for name,(truth,onsets) in cases.items():
        y=np.zeros(int(sr*duration),np.float32)
        for j,t in enumerate(onsets):
            i=int(round(t*sr));count=min(int(.08*sr),len(y)-i)
            if count<=0:continue
            tt=np.arange(count)/sr;amplitude=.4
            if name=='meter_3_then_4':amplitude=.8 if j%(3 if t<20 else 4)==0 else .3
            y[i:i+count]+=amplitude*np.sin(2*np.pi*90*tt)*np.exp(-tt/.015)
        bus=extract_bus(y=y,sr=sr,fps_out=40.,causal=False)
        pulse=librosa.beat.plp(onset_envelope=bus.channels['onset'],sr=40,hop_length=1,win_length=321,tempo_min=30,tempo_max=240)
        estimate=find_peaks(pulse,distance=5,prominence=.1)[0]/40.
        scores={}
        for mode,pred in [('constant_grid',bus.beats),('local_pulse',estimate)]:
            ref=mir_eval.beat.trim_beats(truth);pred=mir_eval.beat.trim_beats(pred)
            scores[mode]={'f1':float(mir_eval.beat.f_measure(ref,pred)),'cmlt':float(mir_eval.beat.continuity(ref,pred)[1]),'events':len(pred)}
        novelty,times=complex_novelty(y,sr)
        rows.append({'name':name,'scores':scores,'clock_period_std':float(bus.channels['beat_period_s'].std()),
                     'complex_feature_finite':bool(np.isfinite(novelty).all()),'complex_feature_std':novelty.std(0).tolist()})
        np.savez_compressed(folder/(name+'.npz'),truth=truth,onsets=onsets,constant_grid=bus.beats,local_pulse=estimate,
                            complex_novelty=novelty,complex_times=times)
    # Real TRAIN excerpts: phase-sensitive novelty versus magnitude-derived onset,
    # with no claim of a trained feature benefit from this descriptive correlation.
    m,n=base_context();records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']};selected=[];seen=set()
    for e in sorted(m['entries'],key=lambda e:(e['canonical_id'],e['key'])):
        if e['partition']=='train' and e['canonical_id'] not in seen:
            selected.append(e);seen.add(e['canonical_id'])
        if len(selected)==8:break
    real=[]
    for e in selected:
        record=records[e['recording_id']];start=(e['n_frames']-2400)//4
        with sf.SoundFile(record['wav_path']) as f:
            sr=f.samplerate;f.seek(round((e['offset_s']+start/40)*sr));audio=f.read(round(60*sr),dtype='float32')
        if audio.ndim==2:audio=audio.mean(1)
        novelty,t=complex_novelty(audio,sr)
        with np.load(e['path']) as z:
            raw=z['x'][start:start+2400];truth_b=z['b'][start:start+2400]
        ns=np.stack([np.interp(np.arange(len(raw))/40,t,novelty[:,i]) for i in range(4)],1)
        index=n['feature_names'].index('onset');corr=np.corrcoef(ns[:,0],raw[:,index])[0,1]
        dest=folder/(e['key']+'_complex.npz');np.savez_compressed(dest,complex_novelty=ns,base=raw,event_truth=truth_b)
        real.append({'key':e['key'],'correlation_with_magnitude_onset':float(corr) if np.isfinite(corr) else None,
                     'path':str(dest),'sha256':digest(dest),'source_tensor_sha256':e['sha256']})
    result={'synthetic':rows,'real_train':real,'fourier_channel_names':[n['feature_names'][i] for i in FOURIER_CHANNELS],
            'interpretation':'Synthetic tests validate mechanics only. Real complex-phase novelty is an exploratory measurement, not a demonstrated generation improvement.',
            'source_sha256':digest(__file__)}
    authenticated_json(folder/'report.json',result);print(json.dumps({'synthetic_cases':len(rows),'real_train_cases':len(real)}))

if __name__=='__main__':run()
