import unittest
import numpy as np
from lightsem.experiments.motion_study.articulated_stage import NEUTRAL,stage_targets
from lightsem.experiments.motion_study.lighting_map import LightingConfig
class LeanTests(unittest.TestCase):
 def test_opposite_rigid_leans_are_distinct_but_static_stays_static(self):
  for axis in ('roll','pitch'):
   values=[]
   for deg in (-30,30):
    c,s=np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))
    q=np.array([[c,-s,0],[s,c,0],[0,0,1]]) if axis=='roll' else np.array([[1,0,0],[0,c,-s],[0,s,c]])
    b=np.tile((NEUTRAL@q.T+np.array([0,.95,0]))[None],(80,1,1))
    points=np.stack([stage_targets(b,x,i,np.ones(80),LightingConfig(stage_lean_gain_m=2.,stage_base_unfolded_m=(7.7,3.2,8.)))[0] for i,x in enumerate((-5,-3,-1,1,3,5))],1)
    np.testing.assert_allclose(points,np.broadcast_to(points[:1],points.shape),atol=1e-12)
    self.assertGreaterEqual(np.min(np.ptp(points[:,:,0],axis=1)),8.2-1e-12)
    values.append(points[0])
   component=(lambda p:p[:,0]) if axis=='roll' else (lambda p:1-p[:,2]+p[:,1])
   self.assertGreater(float(np.max(np.abs(component(values[0])-component(values[1])))),1.9)
if __name__=='__main__':unittest.main()
