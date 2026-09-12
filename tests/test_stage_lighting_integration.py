import unittest
import numpy as np
from lightsem.experiments.motion_study.lighting_map import compose,LightingConfig
from lightsem.conductor.rig import canonical_rig
class StageIntegration(unittest.TestCase):
 def test_static_source_and_photometric_parity(self):
  x=np.zeros((90,139),np.float32);x[:,2]=.95;x[:,3:135]=[1,0,0,0,1,0]*22
  loud={'time':[0.,3.],'momentary':[-20.,-20.],'short_term':[-20.,-20.]}
  common=dict(target_smoothing_s=1e-6,angular_controller='offline_projection')
  old,_,_=compose(x,loud,3.,config=LightingConfig(target_space='articulated_surface',**common),dynamics=False)
  new,_,_=compose(x,loud,3.,config=LightingConfig(target_space='articulated_stage',**common),dynamics=False)
  mask=np.ones(512,bool)
  for f in canonical_rig().by_kind('moving_head'):mask[f.address-1:f.address+3]=False
  np.testing.assert_array_equal(new[:,mask],old[:,mask])
  np.testing.assert_array_equal(new,np.broadcast_to(new[:1],new.shape))
 def test_refuses_cartesian_smoothing(self):
  x=np.zeros((90,139),np.float32);x[:,2]=.95;x[:,3:135]=[1,0,0,0,1,0]*22
  loud={'time':[0.,3.],'momentary':[-20.,-20.],'short_term':[-20.,-20.]}
  with self.assertRaises(ValueError):compose(x,loud,3.,config=LightingConfig(target_space='articulated_stage'))
if __name__=='__main__':unittest.main()
