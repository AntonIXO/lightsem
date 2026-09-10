"""Explanatory Fourier/autocorrelation/cyclic summaries on eight fixed TRAIN clips.

No learned component, candidate selection, or claim of improved motion generation.
"""
from pathlib import Path
import json,sys
import numpy as np
from scipy.ndimage import uniform_filter1d
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.night_features import fourier_features,FOURIER_FREQUENCIES,FOURIER_NAMES
from lightsem.experiments.night_common import base_context
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest

def main():
    import librosa
    manifest,n=base_context();entries={e['key']:e for e in manifest['entries']}
    stress=read_authenticated(ROOT/'diagnostics/rhythm_stress/report.json');out=ROOT/'diagnostics/periodicity';out.mkdir(exist_ok=True);rows=[]
    frequency=FOURIER_FREQUENCIES
    # Four octaves: .5–1,1–2,2–4,4–8Hz. Four phase bins match the sparse16-frequency basis.
    cyclic_bin=np.floor(np.mod(np.log2(frequency/.5),1.)*4+1e-8).astype(int)%4
    for fixed in stress['real_train']:
        e=entries[fixed['key']]
        if e['partition']!='train' or digest(e['path'])!=e['sha256']:raise RuntimeError('TRAIN periodicity input drift')
        with np.load(e['path']) as z:full=z['x'];start=(len(full)-2400)//4;x=full[start:start+2400]
        feature=fourier_features(x);bands=[]
        for band in range(3):
            offset=band*98+49
            magnitude=np.expm1(feature[:,offset:offset+48:3])
            normalized=feature[:,offset+1:offset+48:3]+1j*feature[:,offset+2:offset+48:3]
            coef=magnitude*normalized/np.maximum(abs(normalized),1e-12)
            bands.append(coef)
        bands=np.stack(bands,1);power=np.abs(bands).mean(1)
        cyclic=np.stack([power[:,cyclic_bin==k].mean(1) for k in range(4)],1)
        onset=x[:,n['feature_names'].index('onset')]
        autocorr=librosa.feature.tempogram(onset_envelope=onset,sr=40,hop_length=1,win_length=321,center=True,norm=np.inf)
        lags=np.arange(1,321);rates=40/lags;valid=(rates>=.5)&(rates<=8)
        ac=autocorr[lags[valid]].T;rate=rates[valid]
        # Weighted Fourier complex agreement over time, separately for every frequency.
        pairs={}
        for a,b in [(0,1),(0,2),(1,2)]:
            aa=bands[:,a];bb=bands[:,b]
            denominator=np.sqrt(np.sum(abs(aa)**2,axis=0)*np.sum(abs(bb)**2,axis=0))
            agreement=np.divide(np.sum(aa*np.conj(bb),axis=0),np.maximum(denominator,1e-12))
            pairs[FOURIER_NAMES[a]+'__'+FOURIER_NAMES[b]]={'coherence':abs(agreement).tolist(),'phase_rad':np.angle(agreement).tolist()}
        path=out/(e['key']+'.npz')
        np.savez_compressed(path,time_s=np.arange(2400)/40,frequency_hz=frequency,fourier_magnitude=power,cyclic_magnitude=cyclic,
            autocorrelation=ac,autocorrelation_frequency_hz=rate,band_complex=bands.astype(np.complex64))
        row={'key':e['key'],'canonical_id':e['canonical_id'],'start_frame':start,'path':str(path),'sha256':digest(path),'source_tensor_sha256':e['sha256'],
            'dominant_fourier_hz_median':float(np.median(frequency[np.argmax(power,axis=1)])),
            'dominant_autocorrelation_hz_median':float(np.median(rate[np.argmax(ac,axis=1)])),
            'input_constant_beat_hz':float(1/np.median(x[:,n['feature_names'].index('beat_period_s')])),
            'cross_band':pairs}
        rows.append(row)
    authenticated_json(out/'report.json',{'rows':rows,'source_sha256':digest(__file__),'windows_s':8,'fourier_frequencies_hz':frequency.tolist(),
        'cyclic_bins':cyclic_bin.tolist(),'method':'Main named-flux Fourier coordinates, normalized local onset autocorrelation, and four-bin octave-folded magnitude summaries. Cross-band complex agreement is amplitude-weighted over the full60s excerpt.',
        'interpretation':'TRAIN-only explanatory measurements. Autocorrelation has subharmonic ambiguities; sparse cyclic summaries discard absolute octave and do not increase information. Cross-band envelope phase is not original waveform STFT phase. No beat-truth or generation-gain claim.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,7))
    for row,ax in zip(rows[:3],axes):
        with np.load(row['path']) as z:
            fft=z['fourier_magnitude'];ac=z['autocorrelation'];f=z['frequency_hz'];ar=z['autocorrelation_frequency_hz']
        ax[0].pcolormesh(np.arange(2400)/40,f,fft.T,shading='nearest',cmap='Greys');ax[0].set(yscale='log',ylim=(.5,8),title=row['key']+' — Fourier',ylabel='Frequency (Hz)')
        order=np.argsort(ar);ax[1].pcolormesh(np.arange(2400)/40,ar[order],ac[:,order].T,shading='nearest',cmap='Greys',vmin=0,vmax=1)
        ax[1].set(yscale='log',ylim=(.5,8),title='Normalized autocorrelation',ylabel='Reciprocal lag (Hz)')
    for ax in axes[-1]:ax.set_xlabel('Excerpt time (seconds)')
    fig.tight_layout();fig.savefig(out/'periodicity.png',dpi=170);fig.savefig(out/'periodicity.pdf');plt.close(fig)
    print(json.dumps({'clips':len(rows),'report':str(out/'report.json')}))

if __name__=='__main__':main()
