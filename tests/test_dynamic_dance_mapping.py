"""Geometry preservation at the actual renderer/DMX sample rate."""
import numpy as np

from lightsem.conductor.body_transfer_v2.skeleton import PARENTS
from lightsem.experiments.dynamic_dance.mapping import lighting_body


def test_lighting_resampling_preserves_each_bone_during_fast_turn():
    n=61
    body=np.zeros((n,22,3),dtype=np.float64)
    body[:,0,1]=1
    # This synthetic spatial turn is a geometry diagnostic, never demo motion.
    angles=np.linspace(0,2.7,n)
    for j in range(1,22):
        direction=np.stack([np.sin(angles),np.cos(angles),np.zeros(n)],axis=1)
        body[:,j]=body[:,PARENTS[j]]+(0.05+j*.002)*direction
    time,out=lighting_body(body)
    assert len(out)==round(n/30*40)
    assert np.allclose(np.diff(time),1/40)
    for j in range(1,22):
        assert np.allclose(np.linalg.norm(out[:,j]-out[:,PARENTS[j]],axis=1),0.05+j*.002,atol=1e-12)
    assert np.allclose(out[-1],body[-1],atol=1e-12)


def test_lighting_resampling_keeps_frozen_body_exact():
    rng=np.random.default_rng(8)
    pose=rng.normal(size=(22,3))
    body=np.repeat(pose[None],60,axis=0)
    _,out=lighting_body(body)
    assert np.array_equal(out,np.repeat(pose[None],80,axis=0))
