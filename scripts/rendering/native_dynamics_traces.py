"""Measured, synchronized preview traces; never modifies body or lighting."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
from scipy.ndimage import uniform_filter1d
from PIL import Image,ImageDraw,ImageFont
from scripts.rendering import render_native_dynamics as base
from lightsem.experiments.dynamic_dance.contracts import DanceResult,MusicTimeline
from lightsem.experiments.dynamic_dance.diagnostics import dynamics_series

METHODS=('D7_reference','C3','C4')
ROWS=(('left_arm','Left arm (m/s)'),('right_arm','Right arm (m/s)'),
      ('lower','Lower body (m/s)'),('torso','Torso / head (m/s)'),
      ('root','Root speed (m/s)'),('requested','Requested motors (deg/s)'),
      ('emitted','Emitted motors (deg/s)'),('brightness','Beam dimmer (0–1)'))
SOURCE_COLORS=('#f9b35a','#a7df6c','#71a9ff','#de9edf')
REFERENCE='#91a0b1';CANDIDATE='#4bd6e6'
WIDTH,HEIGHT=1280,304

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def interval_speed(values,time):
    v=np.diff(values,axis=0)/np.diff(time).reshape((-1,)+(1,)*(values.ndim-1))
    speed=np.linalg.norm(v,axis=-1)
    if speed.ndim>1:speed=speed.mean(1)
    return np.r_[speed,speed[-1]]

def body_speeds(world,time):
    s=dynamics_series(world,time);v=np.linalg.norm(s['relative_velocity_m_s'],axis=-1)
    return {'left_arm':v[:,[16,18,20]].mean(1),'right_arm':v[:,[17,19,21]].mean(1),
            'lower':v[:,[1,2,4,5,7,8,10,11]].mean(1),'torso':v[:,[3,6,9,12,15]].mean(1),
            'root':s['root_speed_m_s']}

@lru_cache(maxsize=2)
def song_traces(root,song):
    root=Path(root);plan=base.read(root/'PLAN.json');track=next(t for t in plan['tracks'] if t['id']==song)
    music=MusicTimeline.load(track['music_timeline']);pins={str(root/'PLAN.json'):sha(root/'PLAN.json'),track['music_timeline']:sha(track['music_timeline'])}
    all_values={};times=None
    for method in METHODS:
        pair=base.load_pair(root,plan,method,track);body_path=root/'bodies'/method/song/'body.npz';body=DanceResult.load(body_path)
        times=pair.times;raw={k:np.interp(times,body.time,x) for k,x in body_speeds(body.world,body.time).items()}
        movers=np.array([p['kind']!='bulb' for p in pair.patch]);beams=np.array([p['kind']=='beam' for p in pair.patch])
        p=root/'timelines'/method/song/'timeline.npz'
        with np.load(p) as z:targets=z['targets'][:,movers].copy()
        vec=targets-np.array([p['position'] for p in pair.patch])[movers];length=np.linalg.norm(vec,axis=-1)
        if np.any(length<=1e-12):raise ValueError('Requested target coincides with fixture')
        requested=np.stack((np.rad2deg(np.arctan2(vec[...,0],-vec[...,1])),np.rad2deg(np.arcsin(np.clip(vec[...,2]/length,-1,1)))),-1)
        raw['requested']=interval_speed(requested,times);raw['emitted']=interval_speed(pair.controls['angles_deg'][:,movers],times)
        raw['brightness']=pair.controls['dimmer'][:,beams].mean(1)
        display={k:(x.copy() if k=='brightness' else uniform_filter1d(x,size=10,mode='nearest')) for k,x in raw.items()}
        all_values[method]={'raw':raw,'display':display}
        for item in [body_path,body_path.with_suffix('.json'),p,p.parent/'report.json',p.parent/'patch.json']:pins[str(item)]=sha(item)
    if any(not np.isfinite(x).all() for v in all_values.values() for group in v.values() for x in group.values()):raise ValueError('Nonfinite measured trace')
    selected=(times>=90)&(times<=150);scales={k:max(float(all_values[m]['display'][k][selected].max()) for m in METHODS) for k,_ in ROWS}
    scales['brightness']=1.;scales['requested']=scales['emitted']=max(scales['requested'],scales['emitted']);scales={k:max(v,1e-8) for k,v in scales.items()}
    p95=np.maximum(np.quantile(music.sources[:,:,0],.95,axis=0),1e-9)
    sources=np.stack([np.interp(times,music.time,music.sources[:,j,0]/p95[j]) for j in range(4)],-1)
    return {'time':times,'selected':selected,'values':all_values,'scales':scales,'sources':sources,'source_scale':max(1.,float(sources[selected].max())),'source_p95':p95,'sections':music.sections,'inputs':pins}

class Strip:
    def __init__(self,root,method,song):
        self.data=song_traces(str(root),song);self.method=method;self.left=254;self.right=1268
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
        self.image=Image.new('RGB',(WIDTH,HEIGHT),(12,18,27));d=ImageDraw.Draw(self.image)
        d.text((10,3),'Measured behaviour | A reference: gray; this model: cyan | speed curves: 250 ms display average',font=self.font,fill='#cfdaea')
        d.text((10,22),'Sources / full-song p95',font=self.font,fill='#bccadc')
        for j,(label,color) in enumerate(zip(('drums','bass','other','vocals'),SOURCE_COLORS)):
            d.text((600+j*100,20),label,font=self.font,fill=color)
            self.line(d,self.data['sources'][:,j],39,self.data['source_scale'],color)
        d.text((10,39),'Estimated source RMS',font=self.font,fill='#91a0b1')
        ref=self.data['values']['D7_reference']['display'];current=self.data['values'][method]['display']
        for i,(key,label) in enumerate(ROWS):
            y=65+i*26;scale=self.data['scales'][key]
            d.text((10,y+3),label,font=self.font,fill='#bccadc');d.text((216,y+3),f'{scale:.2g}',font=self.font,fill='#91a0b1')
            d.line((self.left,y+21,self.right,y+21),fill='#354152')
            self.line(d,ref[key],y,scale,REFERENCE);self.line(d,current[key],y,scale,CANDIDATE)
        for sec in self.data['sections']:
            if 90<=sec['start']<=150:d.line((self.x(sec['start']),39,self.x(sec['start']),269),fill='#546174')
        for stamp in range(90,151,10):d.text((min(self.x(stamp)-12,self.right-30),272),str(stamp),font=self.font,fill='#b7c5d7')
        d.text((10,288),'Song seconds | section lines estimated | scales shared across A/B/C; motor rows share scale | native clock 1×, no retiming',font=self.font,fill='#b7c5d7')
    def x(self,t):return round(self.left+(float(t)-90)/60*(self.right-self.left))
    def line(self,d,values,y,scale,color):
        xs=np.arange(self.left,self.right+1);ts=90+(xs-self.left)/(self.right-self.left)*60
        values=np.interp(ts,self.data['time'],values)
        yy=y+20-np.clip(values/scale,0,1)*19
        d.line(list(zip(xs.tolist(),yy.tolist())),fill=color,width=1)
    def frame(self,stamp):
        image=self.image.copy();d=ImageDraw.Draw(image);x=self.x(stamp);d.line((x,39,x,270),fill='#fff2bb',width=2)
        d.text((1090,3),f'Song {stamp:.2f} s',font=self.font,fill='#fff2bb');return image
    def evidence(self):
        return {'inputs':self.data['inputs'],'scales':self.data['scales'],'source_scale':self.data['source_scale'],'source_p95':self.data['source_p95'].tolist(),'body_speed':'Native30Hz forward differences; region speeds hip-relative, root shown separately; interpolate measured speeds to40Hz for chart.','motor_speed':'Mean over13moving fixtures of pan/tilt velocity norm; requestedtargetangles versus decodedDMX.','smoothing':'Ten40Hzsamples(250ms) moving average for displayed speed traces only. No body/DMX processing. Raw series retained.','meaning':'Behaviour traces, not error against music; no groundtruth showcase choreography. No fabricated fragment-rate or motif-cut markers.'}
    def export(self,path):
        s=self.data['selected'];values={'time':self.data['time'][s],'source_rms_over_p95':self.data['sources'][s].astype('f4')}
        for label,method in [('reference','D7_reference'),('current',self.method)]:
            for kind in ('raw','display'):
                for key,x in self.data['values'][method][kind].items():values[f'{label}_{kind}_{key}']=x[s].astype('f4')
        with Path(path).open('xb') as f:np.savez_compressed(f,**values)
