import unittest
import numpy as np
from lightsem.experiments.beat_conditioned.beat_conditioned_render import controls_to_dmx,isolate_heads
from lightsem.experiments.beat_conditioned.beat_conditioned_data import NAMES,event_targets,X_NAMES
from lightsem.experiments.quality_observer import beam_observation
from lightsem.conductor.rig import canonical_rig

class BeatConditionedTests(unittest.TestCase):
    def test_constant_controls_make_constant_light(self):
        frames,_,_=controls_to_dmx(np.zeros((200,21)),NAMES,5.,event_scores=np.full((200,3),.3))
        np.testing.assert_array_equal(frames,np.repeat(frames[:1],len(frames),axis=0))

    def test_extreme_gestures_remain_below_truss_and_within_speed(self):
        y=np.random.default_rng(8).normal(0,8,(800,21));rig=canonical_rig()
        frames,_,report=controls_to_dmx(y,NAMES,20.)
        observed=beam_observation(frames,rig)
        self.assertEqual(observed['upward_light_fraction'],0.)
        for head in report['heads']:
            self.assertLessEqual(head['pan_speed_deg_s'],180.)
            self.assertLessEqual(head['tilt_speed_deg_s'],120.)
        only=isolate_heads(frames)
        for f in rig.fixtures:
            if f.kind!='moving_head':self.assertEqual(only[:,f.address-1:f.address-1+f.n_channels].max(),0)

    def test_tolerance_does_not_wrap_across_sequence_boundary(self):
        b=np.zeros((20,3),np.float32);b[0,1]=1;b[10,0]=1
        t=event_targets(b)
        self.assertEqual(t[:,1].sum(),3);self.assertEqual(t[-1,1],0)
        self.assertEqual(t[:,0].sum(),5)
        self.assertEqual(X_NAMES[-5:],['beat_phase_sin','beat_phase_cos','beat_accent','bar_phase','beat_period_s'])

if __name__=='__main__':unittest.main()
