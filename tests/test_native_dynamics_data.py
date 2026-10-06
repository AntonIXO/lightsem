import json
from pathlib import Path
import numpy as np
from lightsem.experiments.native_dynamics.data import pose_observations,Quality

def fixture(path,n=600):
    k=np.zeros((n,1,17,2));k[...,0]=400;k[...,1]=400
    k[:,:,5:7,1]=300;k[:,:,11:13,1]=500
    k[:,:,5,0]=450;k[:,:,6,0]=350;k[:,:,11,0]=430;k[:,:,12,0]=370
    c=np.ones((n,1,17));box=np.tile([200,100,600,800],(n,1,1));t=np.arange(n)/30
    def save():np.savez(path,K=k,C=c,box=box,t=t,meta=json.dumps({'height':1000,'width':1000}))
    save();return k,c,box,t,save

def test_confident_real_held_poses_are_not_erased(tmp_path):
    p=tmp_path/'pose.npz';fixture(p)
    a,meta=pose_observations(p,20.)
    assert a['mask'].all();assert meta['constant_pose_frames_for_review']>0
    assert np.max(np.abs(np.diff(a['pose'],axis=0)))==0

def test_missing_observation_is_not_a_zero_speed_label(tmp_path):
    p=tmp_path/'pose.npz';k,c,box,t,save=fixture(p);k[100:120]=np.nan;c[100:120]=0;save()
    a,_=pose_observations(p,20.)
    assert not a['mask'][100:120].any();assert np.isfinite(a['pose']).all()
    assert a['mask'][150].all()

def test_fixed_record_scale_does_not_normalize_away_motion(tmp_path):
    p=tmp_path/'pose.npz';k,c,box,t,save=fixture(p);k[300:,:,5:7,1]=200;save()
    a,meta=pose_observations(p,20.)
    assert abs(a['pose'][400,1,1])>abs(a['pose'][100,1,1])
    assert meta['torso_scale_px']==250

def test_invalid_joint_does_not_mask_unrelated_observed_limbs(tmp_path):
    p=tmp_path/'pose.npz';k,c,box,t,save=fixture(p);c[:,0,9]=0;save()
    a,_=pose_observations(p,20.)
    assert not a['mask'][:,5].any();assert a['mask'][:,6].all()
