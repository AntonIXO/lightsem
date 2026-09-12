"""Counterexamples for opt-in offline tracking, not aesthetic acceptance."""
import unittest
import numpy as np
from lightsem.experiments.motion_study.lighting_map import LightingConfig,limited_angles,smooth_targets
class TimingTests(unittest.TestCase):
 def test_zero_phase_does_not_shift_low_frequency_gesture(self):
  fps=40;t=np.arange(1600)/fps
  x=np.column_stack([np.sin(2*np.pi*.8*t),np.zeros(len(t))])
  y=smooth_targets(x,.1,fps,'zero_phase')
  sl=slice(200,-200)
  lag=np.argmax(np.correlate(y[sl,0],x[sl,0],mode='full'))-(len(x[sl])-1)
  self.assertEqual(lag,0)
  self.assertGreater(np.std(y[sl,0])/np.std(x[sl,0]),.7)
 def test_static_targets_and_short_sequences(self):
  for n in (1,2,100):
   x=np.tile([12.,-20.],(n,1));y=smooth_targets(x,.1,40,'zero_phase')
   np.testing.assert_allclose(y,x,atol=1e-12)
   ticks=limited_angles(y,[540,270],np.ones(n),LightingConfig(angular_feedforward=True))
   np.testing.assert_array_equal(ticks,np.broadcast_to(ticks[:1],ticks.shape))
 def test_feedforward_reduces_trackable_gesture_error(self):
  t=np.arange(1600)/40;x=np.column_stack([3*np.sin(2*np.pi*.7*t),np.zeros(len(t))])
  errors=[]
  for ff in (False,True):
   ticks=limited_angles(x,[540,270],np.ones(len(t)),LightingConfig(angular_feedforward=ff))
   angles=(ticks/65535-.5)*[540,270]
   errors.append(np.sqrt(np.mean((angles[200:]-x[200:])**2)))
  self.assertLess(errors[1],errors[0]*.2)
 def test_feedforward_cannot_bypass_mechanics(self):
  x=np.zeros((600,2));x[100:250]=[70,-50];x[250:450]=[-70,50]
  cfg=LightingConfig(angular_feedforward=True)
  ticks=limited_angles(x,[540,270],np.ones(len(x)),cfg)
  angles=(ticks/65535-.5)*[540,270];v=np.diff(angles,axis=0)*40
  self.assertLessEqual(np.linalg.norm(v,axis=-1).max(),30+1e-9)
  self.assertLessEqual(np.linalg.norm(np.diff(v,axis=0)*40,axis=-1).max(),80+1e-9)
if __name__=='__main__':unittest.main()
