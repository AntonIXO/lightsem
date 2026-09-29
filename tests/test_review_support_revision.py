import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'scripts/motion/review_music_physical_support_revision.py';s=importlib.util.spec_from_file_location('support_review',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_no_extrapolation_or_invented_poses(self):
  times=np.array([0.,.025,.05]);v=np.arange(3*17*2).reshape(3,17,2).astype(float);old={'times':times,'pose8':v,'pose32':v+1}
  poses,valid=m.supported_legacy_poses(old,4)
  np.testing.assert_array_equal(valid,[True,True,False,False]);self.assertTrue(np.all(poses['pose8'][~valid]==0))
  expected=np.stack([np.interp(np.arange(4)/30,times,v.reshape(3,34)[:,j]) for j in range(34)],1).reshape(4,17,2)
  np.testing.assert_array_equal(poses['pose8'][valid],expected[valid])
 def test_timestamp_errors_not_hidden(self):
  for t in ([0,0],[1,0],[0,float('nan')]):
   with self.assertRaises(ValueError):m.legacy_pose_support(t,4)
 def test_unsupported_body_is_explicit(self):
  from PIL import ImageFont
  poses={'valid':np.array([False])}
  image=m.body_image('old_incumbent',None,poses,0,ImageFont.load_default())
  self.assertEqual(image.size,(640,360))
if __name__=='__main__':unittest.main()
