import unittest
import numpy as np
from lightsem.experiments.motion_study.offline_angles import project_angles
from lightsem.experiments.motion_study.lighting_map import LightingConfig
class OfflineTests(unittest.TestCase):
 def test_static(self):
  x=np.tile([20.,-30.],(200,1));ticks=project_angles(x,[540,270],np.ones(200),LightingConfig())
  np.testing.assert_array_equal(ticks,np.broadcast_to(ticks[:1],ticks.shape))
 def test_feasible_nonperiodic_and_infeasible_paths(self):
  for scale in (1.,10.):
   t=np.arange(400)/40;x=scale*np.column_stack([np.sin(2*np.pi*.7*t),.5*np.sin(2*np.pi*.83*t)])
   y=(project_angles(x,[540,270],np.ones(400),LightingConfig())/65535-.5)*[540,270]
   self.assertLessEqual(np.linalg.norm(np.diff(y,axis=0)*40,axis=-1).max(),30+1e-8)
   self.assertLessEqual(np.linalg.norm(np.diff(y,n=2,axis=0)*1600,axis=-1).max(),80+1e-8)
   if scale==1.:self.assertLess(np.sqrt(np.mean((y-x)**2)),.02)
if __name__=='__main__':unittest.main()
