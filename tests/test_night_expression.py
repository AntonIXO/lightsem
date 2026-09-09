import unittest
from dataclasses import replace
import numpy as np
from lightsem.experiments.night_render import ExpressionConfig,compose_scene,group_targets
from lightsem.experiments.night_features import fourier_features,FOURIER_DIM
from lightsem.experiments.beat_conditioned_data import NAMES
from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.fixtures import Rig

class ExpressionTests(unittest.TestCase):
    def test_common_initial_weights_match_incumbent(self):
        import torch
        from lightsem.experiments.night_model import NightModel
        from lightsem.experiments.beat_conditioned_model import BeatConditionedModel
        torch.manual_seed(17);old=BeatConditionedModel().state_dict()
        for family,architecture,dim in [('base','once',0),('fourier','film',294),('muq','film',1024)]:
            torch.manual_seed(17);new=NightModel(family,architecture,dim).state_dict()
            for k,v in old.items():self.assertTrue(torch.equal(v,new[k]),k)

    def test_energy_score_separates_wrong_prediction(self):
        from lightsem.experiments.night_metrics import energy_score,off_interval_checks
        target=np.ones((160,17));matched=np.repeat(target[None],6,axis=0)
        self.assertEqual(energy_score(matched,target),0.)
        self.assertGreater(energy_score(matched+1,target),.9)
        dim=np.ones((100,6));dim[10:11]=0
        self.assertEqual(off_interval_checks(dim)['heads_with_off_at_least_150ms'],0)
        dim[20:30,:2]=0
        self.assertEqual(off_interval_checks(dim)['heads_with_off_at_least_150ms'],2)

    def test_constant_inputs_settle_and_dark_activity_can_switch_heads_off(self):
        z=np.zeros((6,160,21));z[:,:,NAMES.index('qom')]=-3;z[:,:,NAMES.index('weight')]=-3
        d,t,_=compose_scene(z,NAMES,4.,event_scores=np.zeros((160,3)))
        self.assertTrue(np.array_equal(d[80:],np.repeat(d[80:81],80,axis=0)))
        self.assertTrue(np.all(t['dimmer'][:,:6]==0))

    def test_mirror_targets_and_dimmers(self):
        z=np.random.default_rng(2).normal(0,.5,(6,160,21));rig=canonical_rig();cfg=ExpressionConfig(composition='mirror')
        targets,_,_=group_targets(z,NAMES,rig,cfg)
        d,t,_=compose_scene(z,NAMES,4.,config=cfg)
        for a,b in ((0,5),(1,4),(2,3)):
            self.assertTrue(np.allclose(targets[:,a,0],-targets[:,b,0]))
            self.assertTrue(np.allclose(targets[:,a,1:],targets[:,b,1:]))
            self.assertTrue(np.array_equal(t['dimmer'][:,a],t['dimmer'][:,b]))

    def test_rig_permutation_does_not_change_addressed_output(self):
        z=np.random.default_rng(3).normal(0,.5,(6,160,21));rig=canonical_rig()
        d,_,_=compose_scene(z,NAMES,4.,rig=rig)
        p,_,_=compose_scene(z,NAMES,4.,rig=Rig(list(reversed(rig.fixtures))))
        self.assertTrue(np.array_equal(d,p))

    def test_constant_envelopes_have_no_fourier_modulation(self):
        x=np.ones((640,92));f=fourier_features(x)
        self.assertEqual(f.shape,(640,FOURIER_DIM))
        # Each block is 16 * (magnitude, real, imaginary) plus mean activity.
        for start in range(0,FOURIER_DIM,49):self.assertLess(float(abs(f[:,start:start+48]).max()),1e-6)

    def test_fourier_tracks_a_local_periodic_envelope(self):
        x=np.zeros((1280,92));x[:,3]=1+np.sin(2*np.pi*2*np.arange(1280)/40.)
        f=fourier_features(x)
        energy=f[400:-400,49:97:3].mean(0)
        from lightsem.experiments.night_features import FOURIER_FREQUENCIES
        self.assertLess(abs(FOURIER_FREQUENCIES[energy.argmax()]-2),.3)

if __name__=='__main__':unittest.main()
