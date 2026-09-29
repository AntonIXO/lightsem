"""Offline audio rhythm clock. Not a learned beat/downbeat detector.

Broadband flux sets tactus (existing octave-aware corpus grid); kick-band flux
anchors phase. A conservative local peak tracker updates period, allowing drift
instead of imposing constant playback speed. Confidence gates free-running light
motion. Whole recording required; crop the resulting clock, never restart it.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import stft, resample_poly
from .beatgrid import fit_constant_grid

@dataclass(frozen=True)
class RhythmClock:
    time: np.ndarray
    beat: np.ndarray
    confidence: np.ndarray
    onset: np.ndarray
    beat_times: np.ndarray

    def __post_init__(self):
        t,b,c,o,bt = [np.asarray(v,float) for v in
                      (self.time,self.beat,self.confidence,self.onset,self.beat_times)]
        if t.ndim!=1 or len(t)<2 or any(v.shape!=t.shape for v in (b,c,o)):
            raise ValueError('Clock requires matching nonempty 1D frame arrays')
        if not all(np.isfinite(v).all() for v in (t,b,c,o,bt)):
            raise ValueError('Nonfinite rhythm clock')
        if np.any(np.diff(t)<=0) or np.any(np.diff(b)<0) or bt.ndim!=1 or np.any(np.diff(bt)<=0):
            raise ValueError('Clock times and beat index must be monotone')
        if np.any((c<0)|(c>1)) or np.any((o<0)|(o>1)):
            raise ValueError('Confidence/onset outside [0,1]')
        for name,v in zip(('time','beat','confidence','onset','beat_times'),(t,b,c,o,bt)):
            object.__setattr__(self,name,v.copy())

    def sample(self,times,phase_offset=0.):
        t=np.asarray(times,float)
        if not np.isfinite(t).all() or not np.isfinite(phase_offset):
            raise ValueError('Finite sampling time/phase required')
        if t.min(initial=self.time[0])<self.time[0]-1e-8 or t.max(initial=self.time[-1])>self.time[-1]+1e-8:
            raise ValueError('Clock does not cover requested recording')
        return (np.interp(t,self.time,self.beat)+phase_offset,
                np.interp(t,self.time,self.confidence),np.interp(t,self.time,self.onset))

    def save(self,path):
        np.savez_compressed(path,**{k:getattr(self,k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:return cls(**{k:z[k] for k in cls.__dataclass_fields__})


def _scale(x):
    x=np.maximum(np.asarray(x,float),0)
    return x/max(float(np.quantile(x,.95)),float(x.max(initial=0))*.02,1e-10)


def clock_from_audio(audio,sr,*,output_fps=40.):
    """Mono/stereo waveform -> full-track local clock; no source pose consulted.

    This conservative tracker permits local tempo drift around global tactus,
    not arbitrary abrupt tempo jumps. Ambiguous half/double tempo and syncopation
    remain explicit detector limitations. No automatic 4/4 downbeat claim.
    """
    import math
    y=np.asarray(audio,float)
    if y.ndim==2:y=y.mean(axis=1)
    if y.ndim!=1 or not np.isfinite(y).all() or len(y)<sr or sr<=0 or output_fps<=0:
        raise ValueError('Expected finite waveform >=1s, positive sample/output rates')
    duration=len(y)/sr
    g=math.gcd(int(sr),22050)
    y=resample_poly(y,22050//g,int(sr)//g)
    sr=22050;hop=128;fps=sr/hop
    freq,t,z=stft(y,fs=sr,nperseg=2048,noverlap=2048-hop,boundary='zeros',padded=True)
    mag=np.abs(z)
    # Log compression balances bass/transients without a trackwise noise floor.
    delta=np.maximum(np.diff(np.log1p(100*mag),axis=1,prepend=np.log1p(100*mag[:,:1])),0)
    broad=_scale(delta[(freq>=40)&(freq<8000)].sum(0))
    kick=_scale(delta[(freq>=40)&(freq<=120)].sum(0))
    evidence=.8*kick+.2*broad
    # Centered STFT rise occurs before an impulse peak. Correct the fixed window
    # response using its envelope center, not a fitted per-song AV offset.
    evidence=gaussian_filter1d(evidence,1.)
    times=np.arange(int(np.ceil(duration*output_fps))+1)/output_fps
    if np.max(np.abs(y),initial=0)<1e-7 or evidence.max(initial=0)<1e-8 or duration<4:
        zero=np.zeros_like(times)
        return RhythmClock(times,zero,zero,zero,np.array([])),{'status':'no_reliable_pulse'}
    grid=fit_constant_grid(broad,fps,phase_onset=kick)
    period=grid['period_s']; initial=period
    # Search near each predicted beat. Timing penalty suppresses off-beat jumps;
    # period correction makes local timing adapt rather than repeatedly restarting.
    first=float(grid['phase_s'])
    beats=[]; confidences=[]; current=first
    while current<duration:
        lo=max(0,int((current-.22*period)*fps));hi=min(len(t),int((current+.22*period)*fps)+1)
        if hi<=lo:break
        local=evidence[lo:hi];tt=t[lo:hi]
        score=local*np.exp(-.5*((tt-current)/(.16*period))**2)
        j=int(np.argmax(score)); strength=float(local[j])
        neighborhood=evidence[max(0,lo-int(period*fps)):min(len(t),hi+int(period*fps))]
        contrast=(strength-float(np.median(neighborhood)))/max(float(np.quantile(neighborhood,.9)),1e-8)
        conf=float(np.clip(contrast,0,1)*np.clip(strength/.5,0,1))
        observed=float(tt[j]) if conf>.15 else current
        # Fixed spectral onset latency correction (+17ms), declared and tested
        # on synthetic impulses, never selected against a real track.
        beat_time=max(0.,observed+.017)
        if beats and beat_time<=beats[-1]+.45*period:
            observed=current;beat_time=observed+.017
        if beat_time<duration:
            beats.append(beat_time);confidences.append(conf)
        error=observed-current
        period=float(np.clip(period+.22*conf*error,initial*.72,initial*1.35))
        current=observed+period
    beats=np.asarray(beats);confidences=np.asarray(confidences)
    if len(beats)<3:
        zero=np.zeros_like(times)
        return RhythmClock(times,zero,zero,zero,np.array([])),{'status':'insufficient_beats'}
    # Linear phase across measured beat intervals, with endpoint continuation.
    knots=np.r_[beats[0]-(beats[1]-beats[0]),beats,beats[-1]+(beats[-1]-beats[-2])]
    phase=np.interp(times,knots,np.arange(-1,len(beats)+1))
    confidence=np.interp(times,beats,confidences)
    confidence=gaussian_filter1d(confidence,.25*output_fps)
    onset=np.clip(np.interp(times,t,evidence)/max(float(np.quantile(evidence,.95)),1e-8),0,1)
    clock=RhythmClock(times,phase,np.clip(confidence,0,1),onset,beats)
    return clock,{'status':'tracked','global_bpm':grid['bpm'],'beat_count':len(beats),
                  'local_bpm_quantiles':np.quantile(60/np.diff(beats),[.05,.5,.95]).tolist(),
                  'mean_confidence':float(confidence.mean()),'fixed_onset_correction_s':.017,
                  'scope':'Offline local pulse tracker; beat annotations and downbeats unavailable'}
