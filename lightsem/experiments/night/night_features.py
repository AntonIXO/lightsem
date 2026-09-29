"""Audio feature and 2-D target preparation; no fitted values from VAL/TEST."""
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve
from .night_common import ROOT, STAGE, initial_setup, feature_path, verify_entry
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest, artifact_manifest
from ..dance_no_clock.dance_no_clock_data import hierarchical_weights
from ..beat_conditioned.beat_conditioned_data import X_NAMES

FOURIER_FREQUENCIES = np.geomspace(.5, 8., 16)
FOURIER_WINDOWS = (2., 8.)
FOURIER_NAMES = ('flux_lf', 'flux_mid', 'flux_hf')
FOURIER_CHANNELS = tuple(X_NAMES.index(name) for name in FOURIER_NAMES)
FOURIER_DIM = 294

def fourier_features(x, fps=40.):
    """Local Fourier coordinates, not a claim to add information to x92.

    Phase is represented by complex coefficients around each local time origin;
    angles are never interpolated. Reflection padding avoids an ending cue.
    """
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != 92 or not np.isfinite(x).all():
        raise ValueError('Expected finite x92')
    columns = []
    for channel in FOURIER_CHANNELS:
        raw = np.maximum(x[:,channel], 0.)
        for seconds in FOURIER_WINDOWS:
            size = int(round(seconds*fps)) | 1
            radius = size//2
            average = uniform_filter1d(raw, size, mode='reflect')
            novelty = raw-average
            win = np.hanning(size); win /= win.sum()
            padded = np.pad(novelty, (radius,radius), mode='reflect')
            energy = fftconvolve(np.pad(np.abs(novelty),(radius,radius),mode='reflect'),win,mode='valid')
            time_axis = np.arange(-radius,radius+1)/fps
            for frequency in FOURIER_FREQUENCIES:
                kernel = win*np.exp(2j*np.pi*frequency*time_axis)
                coef = fftconvolve(padded,kernel,mode='valid')
                supported = energy > 1e-8
                normalized = np.divide(coef,energy+1e-8,out=np.zeros_like(coef),where=supported)
                columns.extend((np.log1p(np.abs(coef)),normalized.real,normalized.imag))
            columns.append(np.log1p(np.maximum(average,0.)))
    out = np.stack(columns,1).astype(np.float32)
    if out.shape != (len(x),FOURIER_DIM) or not np.isfinite(out).all(): raise RuntimeError('Fourier feature failure')
    return out

def prepare_fourier(root=ROOT):
    m,n,_ = initial_setup(root); folder=Path(root)/'features/fourier';folder.mkdir(exist_ok=True)
    report_path=folder/'report.json'
    if report_path.exists(): return read_authenticated(report_path)
    train=[e for e in m['entries'] if e['partition']=='train'];weights=hierarchical_weights(train)
    mean=np.zeros(FOURIER_DIM);square=mean.copy();rows=[];started=time.monotonic()
    for i,e in enumerate(m['entries']):
        verify_entry(e);p=feature_path('fourier',e['key'],root)
        if not p.exists():
            with np.load(e['path']) as z: f=fourier_features(z['x'])
            np.save(p,f)
        else: f=np.load(p,mmap_mode='r')
        if f.shape != (e['n_frames'],FOURIER_DIM) or not np.isfinite(f).all():raise ValueError(e['key'])
        if e['partition']=='train':
            w=weights[e['key']];mean+=w*f.mean(0,dtype=np.float64)
            square+=w*np.square(f.astype(np.float64)).mean(0)
        rows.append({'key':e['key'],'path':str(p),'sha256':digest(p),'source_sha256':e['sha256']})
        if i%100==0: print(json.dumps({'stage':'fourier','entries':i+1,'elapsed_s':time.monotonic()-started}),flush=True)
    result={'dimension':FOURIER_DIM,'mean':mean.tolist(),'std':np.sqrt(np.maximum(square-mean**2,1e-10)).tolist(),
            'fit_partition':'train','rows':rows,'frequencies_hz':FOURIER_FREQUENCIES.tolist(),
            'windows_s':list(FOURIER_WINDOWS),'input_channels':list(FOURIER_CHANNELS),
            'new_waveform_information':False,'elapsed_s':time.monotonic()-started,'source_sha256':digest(__file__)}
    authenticated_json(report_path,result);return result

def prepare_poses(root=ROOT):
    from ..stage11.stage11_data import active_span, load_manifest
    from lightsem.motion.pose2d import to_skeleton
    m,n,_=initial_setup(root);folder=Path(root)/'features/pose';folder.mkdir(exist_ok=True)
    rp=folder/'report.json'
    if rp.exists():return read_authenticated(rp)
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']}
    rows=[];begin=time.monotonic()
    for i,e in enumerate(m['entries']):
        verify_entry(e);r=records[e['recording_id']];dest=folder/(e['key']+'.npz')
        if not dest.exists():
            if digest(r['pose_path'])!=r['pose_sha256']:raise RuntimeError('Pose source changed')
            with np.load(r['pose_path']) as z:
                k,c,t=z['K'],z['C'],z['t'];meta=json.loads(str(z['meta']))
            if k.ndim==3:k=k[:,None];c=c[:,None]
            f0,f1,offset,_,_=active_span(k,t,float(meta['fps_eff']))
            if abs(offset-e['offset_s'])>1e-6:raise ValueError('Pose/audio offset mismatch')
            coach=int(e['key'].rsplit('_c',1)[1]);ks=k[f0:f1,coach];cs=c[f0:f1,coach]
            stats={};sk=to_skeleton(ks.astype(float),cs.astype(float),float(meta['fps_eff']),stats=stats)
            pos=sk.pos[:,:,:2].copy();pos[:,:,0]-=sk.root[:,0].mean()
            source_t=np.arange(len(pos))/sk.fps;target_t=np.arange(e['n_frames'])/40.
            y=np.stack([np.interp(target_t,source_t,a) for a in pos.reshape(len(pos),34).T],1).astype(np.float32)
            good=np.isfinite(ks).all(-1)&(cs>=.3)
            ix=np.minimum(np.rint(target_t*sk.fps).astype(int),len(good)-1)
            mask=np.repeat(good[ix],2,axis=1)
            np.savez_compressed(dest,pose=y,mask=mask,source_time=target_t+offset)
        with np.load(dest) as z: coverage=float(z['mask'].mean())
        rows.append({'key':e['key'],'path':str(dest),'sha256':digest(dest),'pose_source_sha256':r['pose_sha256'],
                     'coverage':coverage,'offset_s':e['offset_s'],'canonical_id':e['canonical_id'],'partition':e['partition']})
        if i%100==0: print(json.dumps({'stage':'pose_targets','entries':i+1,'elapsed_s':time.monotonic()-begin}),flush=True)
    result={'rows':rows,'coordinate_system':'prepared 2D lateral/up in torso lengths, mean root x removed',
            'ground_truth':'detected 2D pose; not 3D','confidence_min':.3,'elapsed_s':time.monotonic()-begin,
            'source_sha256':digest(__file__)}
    authenticated_json(rp,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fourier','pose']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();{'fourier':prepare_fourier,'pose':prepare_poses}[a.stage](a.root)
