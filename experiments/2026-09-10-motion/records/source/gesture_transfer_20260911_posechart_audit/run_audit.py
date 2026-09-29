import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import ConvexHull
from lightsem.experiments.motion_study.geometry import forward_kinematics, OFFSETS, PARENTS
from lightsem.experiments.motion_study.articulated_stage import stage_targets,infer_root_rotation
from lightsem.experiments.motion_study.lighting_map import LightingConfig
from pose_chart import pose_chart

OUT=Path(__file__).resolve().parent
N=241
p=np.linspace(0,1,N)
FX=np.linspace(-5,5,6)
CFG=LightingConfig(target_space='articulated_stage',target_smoothing_s=0,stage_base_unfolded_m=(7.7,3.2,8.),stage_lean_gain_m=0.)

def rot(axis,degrees):
    th=np.deg2rad(np.broadcast_to(degrees,(N,)))
    c,s=np.cos(th),np.sin(th)
    r=np.broadcast_to(np.eye(3),(N,3,3)).copy()
    i,j={'x':(1,2),'y':(2,0),'z':(0,1)}[axis]
    r[:,i,i]=c;r[:,j,j]=c;r[:,i,j]=-s;r[:,j,i]=s
    return r

def fk(spec,yaw=0,roll=0):
    r=np.broadcast_to(np.eye(3),(N,22,3,3)).copy()
    r[:,0]=rot('y',yaw)@rot('z',roll)
    for j,axis,angle in spec:r[:,j]=rot(axis,angle)
    root=np.tile([0,.95,0.],(N,1))
    return forward_kinematics(torch.from_numpy(r),torch.from_numpy(root)).numpy(),r[:,0]

def old(body):
    values=[stage_targets(body,x,k,np.full(N,.7),CFG) for k,x in enumerate(FX)]
    t=np.stack([v[0] for v in values],1)
    d=np.stack([v[1] for v in values],1)
    return t,np.stack((t[:,:,0],1-t[:,:,2]+t[:,:,1]),-1),d

def spread(t,c):
    area=[]
    for pts in c:
        area.append(float(ConvexHull(pts).volume) if np.linalg.matrix_rank(pts-pts.mean(0))==2 else 0.)
    distances=np.linalg.norm(t[:,:,None]-t[:,None,:],axis=-1)[:,np.triu_indices(6,1)[0],np.triu_indices(6,1)[1]]
    return {'x_span_minmax_m':[float(np.ptp(c[:,:,0],axis=1).min()),float(np.ptp(c[:,:,0],axis=1).max())],
      'chart_hull_area_minmax_m2':[min(area),max(area)],
      'surface_pair_distance_min_m':float(distances.min()),
      'x_order_inversion_frames':int(np.any(np.diff(c[:,:,0],axis=1)<-1e-8,axis=1).sum()),
      'hand_order_inversion_frames':int((c[:,0,0]>c[:,5,0]+1e-8).sum()),
      'leg_order_inversion_frames':int((c[:,1,0]>c[:,4,0]+1e-8).sum()),
      'max_step_surface_m':float(np.linalg.norm(np.diff(t,axis=0),axis=-1).max())}

probes={
 'neutral':([],0,0),
 'hands_fold_cross':([(16,'y',90*p),(17,'y',-90*p),(18,'y',90*p),(19,'y',-90*p)],0,0),
 'hands_front_reach':([(16,'y',90+0*p),(17,'y',-90+0*p),(18,'y',90*(1-p)),(19,'y',-90*(1-p))],0,0),
 'hands_raise':([(16,'z',90*p),(17,'z',-90*p)],0,0),
 'hands_raise_yaw90':([(16,'z',90*p),(17,'z',-90*p)],90,0),
 'legs_cross':([(1,'z',-25*p),(2,'z',25*p)],0,0),
 'torso_signed_bend':([(3,'z',70*p-35)],0,0),
 'rigid_yaw':([],360*p-180,0),
 'rigid_signed_roll':([],0,40*p-20),
 'leaf_wrist_rotation':([(20,'x',180*p)],0,0),
}
results={}; arrays={}; maxbone=0.; maxq=0.
for name,(spec,yaw,roll) in probes.items():
    body,q=fk(spec,yaw,roll)
    bone=np.linalg.norm(body[:,1:]-body[:,np.array(PARENTS[1:])],axis=-1)
    maxbone=max(maxbone,float(np.max(np.abs(bone-np.linalg.norm(OFFSETS[1:],axis=-1)))))
    maxq=max(maxq,float(np.max(np.abs(infer_root_rotation(body)-q))))
    ot,oc,od=old(body); nt,nc,nd=pose_chart(body)
    results[name]={'existing':spread(ot,oc),'candidate':spread(nt,nc),
      'existing_endpoint_x_delta_m':(oc[-1,:,0]-oc[0,:,0]).tolist(),
      'candidate_endpoint_x_delta_m':(nc[-1,:,0]-nc[0,:,0]).tolist(),
      'existing_endpoint_chart_displacement_m':np.linalg.norm(oc[-1]-oc[0],axis=-1).tolist(),
      'candidate_endpoint_chart_displacement_m':np.linalg.norm(nc[-1]-nc[0],axis=-1).tolist(),
      'local_right_left_wrist_x_endpoint':(body[-1,[21,20],0]-body[-1,0,0]).tolist(),
      'existing_min_lateral_derivative_m_per_m':float(np.min(2.8*.775/np.cosh(.775*2.8*od[:,:,0]/.65)**2)),
      'existing_articulation_delta_max_m':float(np.max(np.abs(od)))}
    for key,v in [('body',body),('existing_target',ot),('candidate_target',nt),('existing_chart',oc),('candidate_chart',nc)]:arrays[name+'_'+key]=v
# Stateless tests: repeated poses, permuted frames, isolated regional ownership.
body,_=fk(probes['hands_fold_cross'][0]); nt,nc,_=pose_chart(body)
perm=np.random.default_rng(9).permutation(N)
pt,pc,_=pose_chart(body[perm])
neutral,_=fk([]); base=pose_chart(neutral)[1]
left,_=fk([(16,'y',90*p),(18,'y',90*p)])
leftc=pose_chart(left)[1]
unchanged=[0,1,2,3,4]
# Signed lean and rigid rest recovery already use proper root FK.
# Seam continuity through the public chart (candidate coordinate limits leave it unchanged).
from lightsem.experiments.motion_study.articulated_stage import unfolded_targets
seam=unfolded_targets(np.zeros(3),np.array([7-1e-8,7,7+1e-8]))
checks={'max_bone_length_error_m':maxbone,'max_inferred_Q_error':maxq,
 'permutation_target_error_m':float(np.max(np.abs(pt-nt[perm]))),
 'left_arm_only_other_region_max_change_m':float(np.max(np.abs(leftc[:,unchanged]-base[:,unchanged]))),
 'seam_adjacent_max_distance_m':float(np.linalg.norm(np.diff(seam,axis=0),axis=-1).max()),
 'all_saved_arrays_finite':all(np.isfinite(v).all() for v in arrays.values()),
 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'torch_cuda_initialized':torch.cuda.is_initialized()}
# Exact FK counterexample to the shared 3D->2D regional projection.
# Reflect a shoulder-to-wrist vector along its chart-null direction (0,3,4).
ra=rot('z',60)[0]
v=OFFSETS[18].astype(float)+OFFSETS[20].astype(float)
va=ra@v
normal=np.array([0.,3.,4.])/5
vb=va-2*np.dot(va,normal)*normal
av,bv=va/np.linalg.norm(va),vb/np.linalg.norm(vb)
w=np.cross(av,bv); dot=np.dot(av,bv)
k=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
align=np.eye(3)+k+k@k/(1+dot)
rb=align@ra
nullbodies=[]
for shoulder in [ra,rb]:
    rr=np.broadcast_to(np.eye(3),(N,22,3,3)).copy();rr[:,16]=shoulder
    nullbodies.append(forward_kinematics(torch.from_numpy(rr),torch.from_numpy(np.tile([0.,.95,0.],(N,1)))).numpy())
na,nb=nullbodies
nulltest={'left_wrist_position_difference_m':float(np.linalg.norm(nb[0,20]-na[0,20])),
 'left_wrist_difference_xyz_m':(nb[0,20]-na[0,20]).tolist(),
 'existing_max_target_difference_m':float(np.max(np.abs(old(na)[0]-old(nb)[0]))),
 'candidate_max_target_difference_m':float(np.max(np.abs(pose_chart(na)[0]-pose_chart(nb)[0]))),
 'left_elbow_difference_m':float(np.linalg.norm(nb[0,18]-na[0,18]))}
arrays['nullspace_pose_a']=na[:1];arrays['nullspace_pose_b']=nb[:1]
# Finite/continuous proper-FK stress probe (not anatomical plausibility).
rr=np.broadcast_to(np.eye(3),(N,22,3,3)).copy()
rng=np.random.default_rng(47)
for j in range(22):rr[:,j]=rot('x',rng.uniform(-180,180,N))@rot('y',rng.uniform(-180,180,N))@rot('z',rng.uniform(-180,180,N))
stress=forward_kinematics(torch.from_numpy(rr),torch.from_numpy(np.tile([0.,.95,0.],(N,1)))).numpy()
st,sc,_=pose_chart(stress)
checks.update({'stress_targets_finite':bool(np.isfinite(st).all()),'stress_x_minmax_m':[float(sc[:,:,0].min()),float(sc[:,:,0].max())],
 'stress_u_minmax_m':[float(sc[:,:,1].min()),float(sc[:,:,1].max())], 'projection_nullspace_counterexample':nulltest})

# Nonlinear saturation table is derivative, not a data-based fit.
saturation=[{'world_delta_x_m':d,'lane_shift_m':float(.65*np.tanh(.775*2.8*d/.65)),
 'derivative_m_per_m':float(.775*2.8/np.cosh(.775*2.8*d/.65)**2)} for d in [0,.1,.25,.5,.75,1,1.25]]
sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['lightsem/experiments/motion_study/articulated_stage.py','lightsem/experiments/motion_study/geometry.py',str(OUT/'pose_chart.py')]}
report={'instrument':'synthetic FK raw target geometry; no controller, rendered pixels, real clips or music', 'N_per_probe':N,'activity':.7,'config':CFG.__dict__,'checks':checks,'saturation':saturation,'probes':results,'sources_sha256':sources}
(OUT/'measurements.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(OUT/'synthetic_probes.npz',**arrays)
print(json.dumps(report,indent=2))
