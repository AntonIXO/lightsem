"""Fixed, independent visible-motion and music-conditioning diagnostics."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import average_precision_score
from .quality_observer import beam_series,beam_observation,phase_explanation

THRESHOLDS={
    'lit_motion_min':.5,'span_min_deg':8.,'upward_max':.01,'hf_max':.15,
    'visible_motion_min':.5,'screen_brightness_min':.002,'screen_support_min':.01,
    'valid_clip_fraction_min':.75,'valid_clip_improvement_min':.25,
    'nonbeat_response_min':.15,'audio_to_noise_response_min':.5,
    'phase_r2_max':.90,'alignment_advantage_min':.05,'event_ap_advantage_min':.01,
    'conditioned_song_fraction_min':.75}

def physical_checks(beam,screen,thresholds=THRESHOLDS):
    return {'lit_motion':beam['lit_motion_fraction']>=thresholds['lit_motion_min'],
            'spatial_span':beam['span_2s_median_deg']>=thresholds['span_min_deg'],
            'on_stage':beam['upward_light_fraction'] is not None and beam['upward_light_fraction']<=thresholds['upward_max'],
            'no_fast_jitter':beam['direction_hf_above4hz_fraction']<=thresholds['hf_max'],
            'screen_motion':screen['visible_motion_frame_fraction']>=thresholds['visible_motion_min'],
            'screen_light':screen['mean_brightness']>=thresholds['screen_brightness_min'] and screen['support_fraction']>=thresholds['screen_support_min']}

def direction_distance(a,b,rig):
    va,wa=beam_series(a,rig);vb,wb=beam_series(b,rig)
    angles=np.rad2deg(np.arccos(np.clip(np.sum(va*vb,2),-1,1)))
    weight=np.minimum(wa,wb)
    return float((angles*weight).sum()/max(float(weight.sum()),1e-12))

def paired_diagnostics(reference,predictions,events,event_truth,dmx,raw_x,rig,names):
    base=beam_observation(dmx['full'],rig)
    noise=direction_distance(dmx['full'],dmx['noise2'],rig)
    audio=direction_distance(dmx['full'],dmx['swap_nonbeat'],rig)
    beat_only=direction_distance(dmx['full'],dmx['beat_only'],rig)
    phase=phase_explanation(beam_series(dmx['full'],rig)[0].reshape(len(raw_x),-1),
                            raw_x[:,87],raw_x[:,88],raw_x[:,90])
    indices=[names.index(k) for k in ('com_speed','vert_bounce_amp')]
    # Fixed one-second smoothing, valid support; metric never uses a test-fitted probe.
    def smooth(y):return np.lib.stride_tricks.sliding_window_view(y[:,indices],40,axis=0).mean(-1)
    ref=smooth(reference)
    errors={k:float(np.mean(np.square(ref-smooth(predictions[k])))) for k in ('full','swap_nonbeat','beat_only','shift_nonbeat')}
    advantages={k:(errors[k]-errors['full'])/max(errors[k],1e-12) for k in errors if k!='full'}
    aps={}
    for mode in ('full','swap_nonbeat','beat_only','shift_nonbeat'):
        aps[mode]=[float(average_precision_score(event_truth[:,j],events[mode][:,j])) if 0<event_truth[:,j].sum()<len(event_truth) else None for j in range(3)]
    ap_adv=[aps['full'][j]-aps['beat_only'][j] for j in range(3) if aps['full'][j] is not None]
    return {'beam':base,'phase':phase,'nonbeat_direction_response_deg':audio,
            'beat_only_direction_response_deg':beat_only,'noise_direction_response_deg':noise,
            'nonbeat_response_over_span':audio/max(base['span_2s_median_deg'],1e-9),
            'audio_to_noise_response_ratio':audio/max(noise,1e-9),
            'alignment_errors':errors,'alignment_advantages':advantages,'event_ap':aps,
            'event_ap_advantage':float(np.mean(ap_adv)) if ap_adv else None}

def bootstrap_mean(values,seed=991,n=10000):
    x=np.asarray(values,float)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Finite canonical-song values required')
    rng=np.random.default_rng(seed);means=x[rng.integers(len(x),size=(n,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'songs':len(x),'resamples':n}

def verdict(rows,video_rows,thresholds=THRESHOLDS):
    conditioned=[r['nonbeat_response_over_span']>=thresholds['nonbeat_response_min'] and
                 r['audio_to_noise_response_ratio']>=thresholds['audio_to_noise_response_min'] for r in rows]
    alignment=bootstrap_mean([r['alignment_advantages']['swap_nonbeat'] for r in rows])
    event=bootstrap_mean([r['event_ap_advantage'] for r in rows if r['event_ap_advantage'] is not None])
    valid={mode:float(np.mean([all(physical_checks(r[mode]['beam'],r[mode]['screen'],thresholds).values()) for r in video_rows]))
           for mode in ('full','incumbent')}
    gates={'visible_motion':valid['full']>=thresholds['valid_clip_fraction_min'] and valid['full']-valid['incumbent']>=thresholds['valid_clip_improvement_min'],
           'nonbeat_control':float(np.mean(conditioned))>=thresholds['conditioned_song_fraction_min'],
           'motion_audio_alignment':alignment['mean']>=thresholds['alignment_advantage_min'] and alignment['ci95'][0]>0,
           'event_selectivity':event['mean']>=thresholds['event_ap_advantage_min'] and event['ci95'][0]>0,
           'no_fast_jitter':float(np.mean([r['beam']['direction_hf_above4hz_fraction'] for r in rows]))<=thresholds['hf_max']}
    return {'gates':gates,'all_gates_pass':all(gates.values()),'valid_visible_clip_fraction':valid,
            'conditioned_song_fraction':float(np.mean(conditioned)),
            'alignment_advantage':alignment,'event_ap_advantage':event,
            'interpretation':'development diagnostic; human aesthetic preference unmeasured'}
