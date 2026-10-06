import numpy as np
from scripts.rendering.native_dynamics_traces import body_speeds,interval_speed

def test_translation_is_not_articulation_and_no_fake_terminal_stop():
    t=np.arange(80)/40.;p=np.zeros((80,22,3));p[:,:,0]=t[:,None]
    v=body_speeds(p,t)
    for name in ('left_arm','right_arm','lower','torso'):np.testing.assert_allclose(v[name],0,atol=1e-12)
    np.testing.assert_allclose(v['root'],1)

def test_joint_motion_affects_only_relevant_region():
    t=np.arange(80)/40.;p=np.zeros((80,22,3));p[:,20,0]=t
    v=body_speeds(p,t);np.testing.assert_allclose(v['left_arm'],1/3)
    for name in ('right_arm','lower','torso','root'):np.testing.assert_allclose(v[name],0)

def test_motor_speed_vector_norm_and_stationary_input():
    t=np.arange(80)/40.;a=np.zeros((80,2,2));a[:,:,0]=3*t[:,None];a[:,:,1]=4*t[:,None]
    np.testing.assert_allclose(interval_speed(a,t),5)
    np.testing.assert_array_equal(interval_speed(np.zeros_like(a),t),0)
