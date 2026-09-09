"""Preregistered native, conditioning, brightness and composition diagnostics."""
import numpy as np
from .beat_conditioned_metrics import bootstrap_mean,physical_checks,direction_distance
from .quality_observer import beam_observation,phase_explanation,beam_series
from .night_render import composition_observation

def energy_score(draws,truth,window=80):
    """Unbiased ensemble energy score on nonoverlapping2s trajectory windows."""
    x=np.asarray(draws,float);y=np.asarray(truth,float)
    if x.ndim!=3 or x.shape[1:]!=y.shape or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid score inputs')
    scores=[]
    for start in range(0,len(y)-window+1,window):
        a=x[:,start:start+window].reshape(len(x),-1);b=y[start:start+window].ravel();scale=np.sqrt(len(b))
        fit=np.linalg.norm(a-b[None],axis=1).mean()/scale
        pair=sum(np.linalg.norm(a[i]-a[j])/scale for i in range(len(a)) for j in range(i+1,len(a)))
        scores.append(fit-pair/(len(a)*(len(a)-1)))
    if not scores:raise ValueError('No complete energy-score window')
    return float(np.mean(scores))

def alignment_error(pred,truth,names):
    ii=[names.index(k) for k in ('com_speed','vert_bounce_amp')]
    def smooth(a):return np.lib.stride_tricks.sliding_window_view(a[:,ii],40,axis=0).mean(-1)
    return float(np.square(smooth(pred)-smooth(truth)).mean())

def relative_advantage(correct,control):
    return float((control-correct)/control) if control>1e-12 else None

def effective_rank(x):
    x=np.asarray(x,float);s=np.linalg.svd(x-x.mean(0),compute_uv=False);p=s*s
    if p.sum()<=1e-12:return 0.
    p=p/p.sum();p=p[p>0];return float(np.exp(-(p*np.log(p)).sum()))

def native_dispersion(draws,truth):
    x=np.asarray(draws,float);y=np.asarray(truth,float);ref=y.std(0);supported=ref>1e-4
    ratio=x.std(1).mean(0)[supported]/ref[supported]
    return {'supported_channels':int(supported.sum()),'time_std_ratio':ratio.tolist(),
            'fraction_ratio_in_point25_to2':float(np.mean((ratio>=.25)&(ratio<=2.))) if len(ratio) else None,
            'mean_within_audio_draw_std':float(x.std(0).mean()),'effective_rank':effective_rank(x.reshape(-1,x.shape[-1])),
            'reference_effective_rank':effective_rank(y),'exact_constant_channels':np.all(np.ptp(x,axis=1)==0,axis=0).tolist()}

def off_interval_checks(dimmer):
    a=np.asarray(dimmer);long=[]
    for j in range(a.shape[1]):
        off=a[:,j]==0;d=np.diff(np.r_[False,off,False].astype(int));starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1)
        long.append(int(np.max(ends-starts,initial=0)))
    return {'longest_off_frames':long,'heads_with_off_at_least_150ms':sum(n>=6 for n in long)}

def song_metrics(truth,predictions,events,event_truth,dmx,trace,raw,names,rig):
    from sklearn.metrics import average_precision_score
    beam=beam_observation(dmx['full'],rig);composition=composition_observation(dmx['full'],trace,rig)
    noise=direction_distance(dmx['full'],dmx['noise2'],rig)
    donor=direction_distance(dmx['full'],dmx['donor_clock_fixed'],rig)
    shift=direction_distance(dmx['full'],dmx['shift_four_bars'],rig)
    errors={k:alignment_error(v.mean(0),truth,names) for k,v in predictions.items()}
    scores={k:energy_score(v[:,:,4:],truth[:,4:]) for k,v in predictions.items()}
    aps={}
    for k,v in events.items():
        aps[k]=[float(average_precision_score(event_truth[:,j],v[:,j])) if 0<event_truth[:,j].sum()<len(event_truth) else None for j in range(3)]
    paired=[aps['full'][j]-aps['beat_only6'][j] for j in range(3) if aps['full'][j] is not None]
    phase=phase_explanation(beam_series(dmx['full'],rig)[0].reshape(len(raw),-1),raw[:,87],raw[:,88],raw[:,90])
    response_ratio=donor/noise if noise>1e-8 else None
    controlled=bool(donor>=.15*beam['span_2s_median_deg'] and (donor>=.5*noise) and donor>1e-6)
    return {'beam':beam,'composition':composition,'phase':phase,'energy_score':scores,'alignment_errors':errors,
            'alignment_advantage':relative_advantage(errors['full'],errors['donor_clock_fixed']),
            'local_alignment_advantage':relative_advantage(errors['full'],errors['shift_four_bars']),
            'event_ap':aps,'event_ap_advantage':float(np.mean(paired)) if paired else None,
            'audio_response_deg':donor,'noise_response_deg':noise,'audio_to_noise_ratio':response_ratio,
            'shift_response_deg':shift,'controlled':controlled,
            'native_dispersion':native_dispersion(predictions['full'][:,:,4:],truth[:,4:]),
            'qualified_claim':'holding explicit clock channels does not hold all rhythm encoded in MuQ or spectral envelopes'}

def aggregate(rows,video_rows):
    values=[r['metrics'] for r in rows]
    def boot(key):
        v=[r[key] for r in values if r[key] is not None]
        return bootstrap_mean(v) if v else {'mean':None,'ci95':[None,None],'songs':0}
    alignment=boot('alignment_advantage');local=boot('local_alignment_advantage');event=boot('event_ap_advantage')
    fraction=float(np.mean([v['controlled'] for v in values]));physical=[];brightness=[]
    for row in video_rows:
        physical.append(all(physical_checks(row['beam'],row['screen']).values()))
        c=row['composition'];brightness.append(c['heads_with_span_ge_point4']>=4 and row['off_intervals']['heads_with_off_at_least_150ms']>=2)
    gates={'conditioning':fraction>=.75,
           'native_alignment':alignment['mean'] is not None and alignment['mean']>=.05 and alignment['ci95'][0]>0,
           'local_alignment':local['mean'] is not None and local['mean']>=.01 and local['ci95'][0]>0,
           'event_selectivity':event['mean'] is not None and event['mean']>=.01 and event['ci95'][0]>0,
           'visible_motion':bool(physical) and np.mean(physical)>=.75,
           'brightness':bool(brightness) and np.mean(brightness)>=.75,
           'no_fast_jitter':np.mean([v['beam']['direction_hf_above4hz_fraction'] for v in values])<=.15,
           'native_temporal_variation':np.mean([v['native_dispersion']['fraction_ratio_in_point25_to2'] or 0 for v in values])>=.8,
           'native_draw_variation':np.mean([v['native_dispersion']['mean_within_audio_draw_std'] for v in values])>=.05}
    gates={k:bool(v) for k,v in gates.items()}
    return {'gates':gates,'all_gates_pass':all(gates.values()),'controlled_song_fraction':fraction,
            'alignment':alignment,'local_alignment':local,'event_ap_advantage':event,
            'physical_clip_fraction':float(np.mean(physical)) if physical else None,
            'brightness_clip_fraction':float(np.mean(brightness)) if brightness else None,
            'mean_energy_score':float(np.mean([v['energy_score']['full'] for v in values])),
            'songs':len(rows),'clips':len(video_rows)}
