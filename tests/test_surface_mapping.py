"""Spatial controls and coordinate conventions for actual illumination, not targets."""
import unittest
from pathlib import Path
import numpy as np
import torch
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.motion_study.spatial_observer import (controls,spatial_series,summary,gates,first_hits,encode_targets)
from lightsem.experiments.motion_study.lighting_map import compose,LightingConfig,surface_targets
from lightsem.experiments.motion_study.geometry import joints_from_features,matrix_to_rotation_6d
from lightsem.experiments.quality_observer import beam_series
from lightsem.experiments.beat_conditioned_render import _angles
from lightsem.sim.renderer import _decode_fixture,_beam_direction


class SurfaceContracts(unittest.TestCase):
    def test_failure_controls_are_rejected_and_separated_motion_passes(self):
        rig=canonical_rig()
        for name,dmx in controls(rig).items():
            result=gates(summary(spatial_series(dmx,rig),0,len(dmx)/40))
            self.assertEqual(all(result.values()),name=='separated_motion',(name,result))
            if name=='translation':self.assertFalse(result['not_just_translation'])
            if name in ('static_spread','jitter','flashing_spread'):self.assertFalse(result['trajectory'])

    def test_angle_roundtrip_counterexamples_and_renderer_parity(self):
        vectors=np.array([[0,-6,0],[5,-6,-4],[-5,-6,4],[0,3,-5],[4,3,2],[0,-.01,-5]],float)
        rig=canonical_rig();f=rig.by_kind('moving_head')[0]
        pan,tilt=_angles(vectors)
        for i,(p,t) in enumerate(zip(pan,tilt)):
            actual=np.array(_beam_direction(f,.5+p/f.pan_range_deg,.5+t/f.tilt_range_deg))
            np.testing.assert_allclose(actual,vectors[i]/np.linalg.norm(vectors[i]),atol=2e-9)
        targets=np.broadcast_to(vectors[:,None],(6,6,3))+np.array([h.position for h in rig.by_kind('moving_head')])
        dmx=encode_targets(targets,rig);v,_=beam_series(dmx,rig)
        np.testing.assert_allclose(v[:,0],vectors/np.linalg.norm(vectors,axis=1,keepdims=True),atol=.00011)
        for row,d in zip(v[:,0],dmx):
            p=_decode_fixture(f,d)
            np.testing.assert_allclose(row,_beam_direction(f,p['pan'],p['tilt']),atol=2e-9)

    def test_first_hit_uses_wall_before_extrapolated_floor_and_finite_scene(self):
        p=np.array([[0,6,-1],[0,6,-1],[0,6,-1]])
        v=np.array([[0,-1,-2],[0,-6,-1],[100,1,1]],float)
        v/=np.linalg.norm(v,axis=1,keepdims=True)
        hit,surf,floor=first_hits(v,p)
        np.testing.assert_array_equal(surf,[1,0,-1])
        np.testing.assert_allclose(hit[0],[0,3.5,-6])
        self.assertLess(floor[0,2],-6)
        self.assertTrue(np.isnan(hit[2]).all())

    def test_neutral_anatomy_and_turns_are_not_mirrored_or_swapped(self):
        f=torch.zeros(120,139);f[:,2]=.95;f[:,3:135]=torch.tensor([1.,0,0,0,1,0]*22)
        body=joints_from_features(f).numpy()
        self.assertGreater(body[0,20,0],0);self.assertLess(body[0,21,0],0)
        cfg=LightingConfig(target_space='articulated_surface')
        neutral,delta=surface_targets(body,-5,0,np.ones(120),cfg)
        np.testing.assert_allclose(delta,0,atol=1e-6)
        np.testing.assert_allclose(neutral[:,0],-4.75,atol=1e-6)
        # Turning the body changes world right-wrist displacement; fixed fixture
        # identity means it is NOT reassigned to whichever wrist is now leftmost.
        rotation=torch.diag(torch.tensor([-1.,1.,-1.]))
        f[:,3:9]=matrix_to_rotation_6d(rotation)
        turned=joints_from_features(f).numpy()
        target,delta=surface_targets(turned,-5,0,np.ones(120),cfg)
        self.assertGreater(delta[0,0],1.)
        self.assertGreater(target[0,0],neutral[0,0])

    def test_static_motion_has_no_invented_chase_and_photometry_is_identical(self):
        f=np.zeros((120,139),np.float32);f[:,2]=.95;f[:,3:135]=np.tile([1,0,0,0,1,0],22)
        loud={'time':np.arange(120)/30,'momentary':np.full(120,-15.),'short_term':np.full(120,-15.)}
        old,_,_=compose(f,loud,4.)
        new,tr,_=compose(f,loud,4.,config=LightingConfig(target_space='articulated_surface'))
        mask=np.ones(512,bool)
        for h in canonical_rig().by_kind('moving_head'):mask[h.address-1:h.address+3]=False
        np.testing.assert_array_equal(old[:,mask],new[:,mask])
        np.testing.assert_array_equal(new,np.broadcast_to(new[:1],new.shape))
        self.assertGreater(np.ptp(spatial_series(new,canonical_rig())['hit'][0,:,0]),9.)

    def test_saved_rejected_hybrid_is_spatial_regression(self):
        path=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic/lighting/hybrid.npz')
        if not path.exists():self.skipTest('Historical artifact integration test; explicit delivery audit requires this input')
        with np.load(path) as z:d=z['dmx']
        s=summary(spatial_series(d,canonical_rig()),0,len(d)/40)
        self.assertAlmostEqual(s['floor_x_span']['median'],1.81737638,places=5)
        self.assertFalse(gates(s)['spatial_spread'])
        self.assertFalse(gates(s)['cone_separation'])
        self.assertGreater(s['focus_fraction'],.97)


if __name__=='__main__':unittest.main()
