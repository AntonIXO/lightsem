"""Actual decoded stage-image footprint, fixed crop/band, every video frame.

Unlike analytic cone area, these are luminance-supported camera pixels. They
include residual non-head light and codec noise. Controls keep non-angle DMX
and renderer fixed. No adaptive crop or exposure normalization per variant.
"""
import json,subprocess,argparse
from pathlib import Path
import numpy as np
from lightsem.experiments.stage11.stage11_common import authenticated_json,digest


def run(path,out):
    cmd=['ffmpeg','-v','error','-threads','2','-c:v','libdav1d','-i',str(path),'-an','-vf','scale=320:180:flags=area','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # Same camera for all candidates. Floor and low-wall band: y96..137,
    # x80..240; excludes truss bulbs and most upper shafts.
    rows=[];i=0
    while True:
        raw=p.stdout.read(320*180*3)
        if not raw:break
        if len(raw)!=320*180*3:raise RuntimeError('partial frame')
        rgb=np.frombuffer(raw,np.uint8).reshape(180,320,3)
        luma=rgb@np.array([.2126,.7152,.0722])/255
        band=luma[96:138,80:240]
        # Subtract fixed dark floor, NOT per-frame arbitrary contrast scaling.
        energy=np.maximum(band-.003,0);mass=energy.sum();xmass=energy.sum(0)
        cdf=np.cumsum(xmass)/max(mass,1e-12)
        q=[int(np.searchsorted(cdf,v)) for v in (.1,.9)]
        support=band>.01
        rows.append({'time':i/30,'band_mass':float(mass),'band_luma':float(band.mean()),
            'horizontal_energy_span80_px':float(q[1]-q[0]) if mass>0 else 0.,
            'central_energy_fraction':float(energy[:,60:100].sum()/max(mass,1e-12)),
            'support_px':int(support.sum()),'support_x_columns':int(support.any(0).sum())})
        i+=1
    error=p.stderr.read().decode();code=p.wait();p.stdout.close();p.stderr.close()
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(path)]))['streams'][0]
    expected=int(info['nb_frames'])
    if code or len(rows)!=expected or abs(len(rows)/30-277.306667)>1/30:
        raise RuntimeError('Decode incomplete '+str(len(rows))+' '+error)
    segs={}
    for name,(lo,hi) in {'full':(0,277.31),'quiet':(0,20),'entry':(60,76),'active':(70,90),'ending':(250,277)}.items():
        chunk=[r for r in rows if lo<=r['time']<hi]
        segs[name]={k:float(np.median([r[k] for r in chunk])) for k in rows[0] if k!='time'}
    authenticated_json(out,{'video':str(path),'sha256':digest(path),'script_sha256':digest(__file__),'decoded_frames':i,'segments':segs,'rows':rows,
        'scope':'Actual video pixels, fixed lower-stage band at320x180; not beam-isolated, no causal attribution beyond matched geometry.'})
    print(json.dumps(segs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.video,a.out)
