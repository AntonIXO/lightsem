"""Read-only audit of frozen rev02 signals. No inference, train, render or source edits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from scipy.signal import welch
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery')
REV=ROOT/'surface_revision_02'
sys.path.insert(0,str(REV/'source'))
from lightsem.experiments.motion_study.geometry import joints_from_features,rotation_6d_to_matrix,OFFSETS,PARENTS
from lightsem.experiments.motion_study.lighting_map import ema,surface_targets,LightingConfig
from lightsem.experiments.quality_observer import beam_series
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.beat_conditioned.beat_conditioned_render import _angles
OUT=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/dance_mapper_audit_20260911');OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
def load(path):
 with np.load(path) as z:return {k:z[k] for k in z.files}
trace=load(REV/'articulated_surface_trace.npz');dmx=load(REV/'articulated_surface.npz')['dmx']
features=load(ROOT/'final_diagnostic/inference/shpongle.npz')['flow0']
t=trace['time'];fps=40
j0=joints_from_features(torch.from_numpy(features)).numpy()
def interp(a):
 return np.stack([np.interp(t,np.arange(len(a))/30,x) for x in a.reshape(len(a),-1).T],axis=1).reshape(len(t),*a.shape[1:])
j=interp(j0);rot=interp(rotation_6d_to_matrix(torch.from_numpy(features[:,3:9])).numpy())
rig=canonical_rig();heads=sorted(rig.by_kind('moving_head'),key=lambda f:f.position[0]);pos=np.array([h.position for h in heads]);directions,_=beam_series(dmx,rig)
hit=pos[None]+directions*(-pos[None,:,1]/directions[:,:,1])[:,:,None]
raw=trace['raw_targets'][:,:6];smooth=trace['targets'][:,:6];reg=trace['regional']
activity=trace['activity'];cfg=LightingConfig(**json.loads((REV/'design_preregistered.json').read_text())['mapper_config'])
neutral=np.zeros((22,3))
for i in range(1,22):neutral[i]=neutral[PARENTS[i]]+OFFSETS[i]
root=j[:,0];sr=ema(root,12,40);travel=root-sr
shared_x=.5*np.tanh(sr[:,0]/5)+.9*np.tanh(travel[:,0]/1.2);shared_z=.7*np.tanh(travel[:,2]/1.5)
unclipped=[];rest_values=[];reconstructed=[]
for rank,f in enumerate(heads):
 pair=min(rank,5-rank);side=-1 if rank<3 else 1
 ids,weights=(([21] if side<0 else [20]),[1.]) if pair==0 else ((([11,5] if side<0 else [10,4]),[.65,.35]) if pair==1 else (([17,15,12] if side<0 else [16,15,12]),[.5,.3,.2]))
 rest=np.sum(neutral[ids]*np.array(weights)[:,None],axis=0);rest_values.append(rest)
 amp=(.25+.75*activity)*1.8*(2 if pair==2 else 1)
 delta=reg[:,rank]
 unclipped.append(np.column_stack((.95*f.position[0]+shared_x+amp*2.8*delta[:,0],np.zeros(len(t)),(-2.,-4.1,-2.6)[pair]+shared_z+amp*(2.6*delta[:,2]+2.2*delta[:,1]+.7*np.tanh((root[:,1]-.9)/.3)))))
 reconstructed.append(surface_targets(j,f.position[0],rank,activity,cfg)[0])
unclipped=np.stack(unclipped,1);reconstructed=np.stack(reconstructed,1)
emitted_angles=[]
for f in heads:
 a=f.address-1;ticks=np.stack((dmx[:,a].astype(float)*256+dmx[:,a+1],dmx[:,a+2].astype(float)*256+dmx[:,a+3]),-1)
 emitted_angles.append((ticks/65535-.5)*[f.pan_range_deg,f.tilt_range_deg])
emitted_angles=np.stack(emitted_angles,1)
def target_angles(a):
 pan,tilt=_angles(a-pos[None]);return np.stack((pan,tilt),-1)
angles={'raw':target_angles(raw),'smooth':target_angles(smooth),'emitted':emitted_angles}
bands=[(.1,.5),(.5,1),(1,2),(2,4)]
def power(a):
 f,P=welch(a.reshape(len(a),-1),fs=40,nperseg=min(len(a),400),axis=0,detrend='linear')
 P=P.sum(1)
 return {f'{lo}-{hi}':float(P[(f>=lo)&(f<hi)].sum()) for lo,hi in bands}
def lagfit(x,y):
 # Output y is compared to past input x; positive lag means emitted behind source.
 x=x.reshape(len(x),-1);y=y.reshape(len(y),-1)
 scores=[]
 for lag in range(0,81):
  a=x[:-lag] if lag else x;b=y[lag:] if lag else y
  a=a-a.mean(0);b=b-b.mean(0)
  scores.append(float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())))
 q=int(np.argmax(scores));return {'lag_s':q/40,'corr_at_lag':scores[q],'zero_lag_corr':scores[0]}
def rms_speed(a):return float(np.sqrt(np.mean(np.sum((np.diff(a,axis=0)*40)**2,axis=-1))))
result={'scope':'Single saved Shpongle flow0, frozen rev02. Descriptive, unpreregistered; no human or audio fit test.', 'files':{},'reconstruct_raw_max_abs_m':float(np.max(np.abs(raw-reconstructed))), 'regional_rest':np.array(rest_values).tolist(),'segments':{}}
for path in [REV/'articulated_surface_trace.npz',REV/'articulated_surface.npz',ROOT/'final_diagnostic/inference/shpongle.npz',REV/'source/lightsem/experiments/motion_study/lighting_map.py']:
 result['files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
for label,(lo,hi) in {'full':(0,277.3),'quiet':(0,20),'active':(70,90),'contraction':(194,201),'precontraction':(180,190),'ending':(250,277)}.items():
 sl=(t>=lo)&(t<hi);r={}
 r['activity_quantiles']=np.quantile(activity[sl],[.1,.5,.9]).tolist()
 limit=3+27*activity[sl]**2;r['local_speed_limit_deg_s_quantiles']=np.quantile(limit,[.1,.5,.9]).tolist()
 for name,a in {'regional':reg,'raw_floor':raw,'smoothed_floor':smooth,'emitted_floor':hit}.items():
  r[name]={'rms_speed':rms_speed(a[sl]),'median_fixture_10_90_extent_m':float(np.median(np.linalg.norm(np.quantile(a[sl],.9,axis=0)-np.quantile(a[sl],.1,axis=0),axis=-1)))}
 r['angle_lag_raw_to_emitted']=lagfit(angles['raw'][sl],angles['emitted'][sl]);r['angle_lag_smooth_to_emitted']=lagfit(angles['smooth'][sl],angles['emitted'][sl])
 pp={k:power(v[sl]) for k,v in angles.items()};r['angle_band_power']=pp
 r['emitted_raw_band_amplitude_ratio']={b:float(np.sqrt(pp['emitted'][b]/pp['raw'][b])) for b in pp['raw']}
 r['smooth_raw_band_amplitude_ratio']={b:float(np.sqrt(pp['smooth'][b]/pp['raw'][b])) for b in pp['raw']}
 r['raw_emitted_floor_error_median_m']=float(np.median(np.linalg.norm(raw[sl]-hit[sl],axis=-1)))
 r['raw_clipped_fraction_by_fixture']=np.any(np.abs(unclipped[sl]-raw[sl])>1e-6,axis=-1).mean(0).tolist()
 r['raw_x_span_quantiles']=np.quantile(np.ptp(raw[sl,:,0],axis=1),[.1,.5,.9]).tolist()
 r['emitted_x_span_quantiles']=np.quantile(np.ptp(hit[sl,:,0],axis=1),[.1,.5,.9]).tolist()
 r['median_regional_x']=np.median(reg[sl,:,0],axis=0).tolist()
 r['median_root_forward_xyz']=np.median(rot[sl,:,2],axis=0).tolist()
 # exact null vector of 3D -> 2D regional map; independent of scalar gains
 v=np.diff(reg[sl],axis=0)*40;n=np.array([0.,2.6,-2.2]);n/=np.linalg.norm(n)
 r['regional_velocity_energy_in_exact_map_nullspace_fraction']=float(np.sum((v@n)**2)/np.sum(v*v))
 yv=v[:,:,1];zv=v[:,:,2]
 r['depth_projection_cancellation_ratio']=float(np.sum((2.2*yv+2.6*zv)**2)/(np.sum((2.2*yv)**2)+np.sum((2.6*zv)**2)))
 vel=np.diff(angles['emitted'][sl],axis=0)*40;speed=np.linalg.norm(vel,axis=-1)
 r['emitted_speed_median_p95']=np.quantile(speed,[.5,.95]).tolist()
 r['speed_above_90percent_local_limit_fraction']=float(np.mean(speed>.9*limit[1:,None]))
 r['per_fixture_emitted_rms_speed_m_s']=[rms_speed(hit[sl,i:i+1]) for i in range(6)]
 result['segments'][label]=r
# General structural controls, not production content: null-space regional delta and stationary T-pose yaw.
result['structural']={'regional_projection_matrix':[[2.8,0,0],[0,2.2,2.6]],'exact_null_vector':[0,2.6,-2.2], 'floor_vertical_target_range_m':[float(raw[:,:,1].min()),float(raw[:,:,1].max())]}
neutral_body=neutral.copy();neutral_body[:,1]+=.95
turns={}
for deg in [0,45,90,135,180]:
 rad=np.deg2rad(deg);c,s=np.cos(rad),np.sin(rad);R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
 body=(neutral@R.T);body[:,1]+=.95;body=np.repeat(body[None],80,axis=0)
 target=np.stack([surface_targets(body,f.position[0],rank,np.ones(80),cfg)[0][-1] for rank,f in enumerate(heads)])
 turns[str(deg)]={'x_span':float(np.ptp(target[:,0])),'targets':target.tolist()}
result['structural']['stationary_neutral_turn_activity1']=turns
(OUT/'metrics.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(OUT/'audit_arrays.npz',time=t,regional=reg,raw=raw,smoothed=smooth,emitted=hit,unclipped=unclipped,emitted_angles=emitted_angles)
print(json.dumps(result,indent=2))
