import unittest
import numpy as np
from lightsem.sim.renderer import RenderConfig

def smooth(a,b,x):
 t=np.clip((x-a)/(b-a),0,1);return t*t*(3-2*t)
def integrals(full,soft=.6,reference=14.):
 edge=np.cos(np.deg2rad(reference/2));low=edge-.018;high=edge+.0072;w=high-low;q=np.clip((1-low)/w,0,1)
 ref=w*(q**3-.5*q**4)+max(1-high,0)
 outer=np.cos(np.deg2rad(full/2+soft));inner=np.cos(np.deg2rad(max(full/2-soft,0)))
 return ref,1-.5*(inner+outer),outer,inner
class FluxTests(unittest.TestCase):
 def test_integrated_profile_not_peak_normalization(self):
  c=np.linspace(-1,1,500001)
  for full in (6.,14.,32.):
   ref,area,outer,inner=integrals(full)
   measured=np.trapezoid(smooth(outer,inner,c)*(ref/area),c)
   self.assertAlmostEqual(measured,ref,places=7)
  self.assertGreater(integrals(6)[0]/integrals(6)[1],8.)
 def test_default_is_unchanged_and_invalid_combinations_fail(self):
  self.assertEqual(RenderConfig().beam_flux_reference_deg,0.)
  for bad in (-1.,91.,float('nan')):
   with self.assertRaises(ValueError):RenderConfig(beam_edge_profile='angular',beam_flux_reference_deg=bad)
  with self.assertRaises(ValueError):RenderConfig(beam_flux_reference_deg=14.)
if __name__=='__main__':unittest.main()
