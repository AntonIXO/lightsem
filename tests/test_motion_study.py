"""Physical counterexamples and indexing contracts for the new motion path."""
import unittest

import numpy as np
import torch

from lightsem.experiments.motion_study.geometry import (
    FPS, OFFSETS, PARENTS, rotation_6d_to_matrix, matrix_to_rotation_6d,
    forward_kinematics, root_from_features, joints_from_features, contacts_from_joints)
from lightsem.experiments.motion_study.legacy_review import visibility
from lightsem.experiments.motion_study.metrics import compare, fidelity_gates
from lightsem.experiments.motion_study.prepare import Groups, fingerprint_similarity
from lightsem.experiments.motion_study.phase_model import PhaseCodec, PhaseConfig


class MotionContracts(unittest.TestCase):
    def test_phase_codec_preserves_native_root_contacts_and_static_holds(self):
        rng = np.random.default_rng(91)
        norms = {'mean':[0.]*139,'std':[1.]*139}
        for arm in ('frame24','phase64','phase128','phase256'):
            cfg = PhaseConfig(arm)
            fit = {'dc_mean':np.zeros(132),'basis':[
                np.linalg.qr(rng.standard_normal((132,132)))[0] for _ in cfg.ranks]}
            model = PhaseCodec(cfg,norms,fit)
            value = torch.from_numpy(rng.standard_normal((1,481,139)).astype(np.float32))
            value[...,135:] = value[...,135:].sigmoid()
            decoded = model(value)
            torch.testing.assert_close(decoded[...,:3],value[...,:3],atol=0,rtol=0)
            torch.testing.assert_close(decoded[...,135:],value[...,135:],atol=0,rtol=0)
            torch.testing.assert_close(decoded,model.forward_chunked(value,32),atol=2e-6,rtol=2e-6)
            self.assertEqual(model.encode(value).shape[-1],sum(cfg.ranks)+7*cfg.stride)
            still = model(value[:,:1].expand(-1,480,-1))
            torch.testing.assert_close(still,still[:,:1].expand_as(still),atol=2e-6,rtol=2e-6)

    def test_music_global_noise_and_chunked_ode_agree(self):
        from lightsem.experiments.motion_study.music_model import MusicModel, MusicConfig
        torch.set_num_threads(2)
        torch.manual_seed(19)
        norms = {'mean':[0.]*139,'std':[1.]*139,'delta_scale':[1.]*139}
        model = MusicModel(MusicConfig(width=16,dilations=(1,2,4),steps=8),norms).eval()
        # Nonzero vector field: a zero output layer makes seam tests vacuous.
        torch.nn.init.normal_(model.output.weight,std=.03)
        audio = torch.randn(2,211,92)
        with torch.inference_mode():
            full = model.generate(audio,seed=7,chunked=False)
            original = model.vector_field_chunked
            model.vector_field_chunked = lambda a,s,t: original(a,s,t,core_frames=47)
            chunks = model.generate(audio,seed=7,chunked=True)
            other = model.generate(audio,seed=8,chunked=True)
        torch.testing.assert_close(full,chunks,atol=2e-5,rtol=2e-5)
        self.assertGreater(float((full-other).abs().max()),.1)

    def test_motion_energy_score_rejects_frozen_and_shifted_controls(self):
        from lightsem.experiments.motion_study.music_data import motion_summary,energy_score
        f = torch.zeros(240,139)
        f[:,3:135] = torch.tensor([1.,0,0,0,1,0]*22)
        f[:,2] = .9+.1*torch.sin(torch.arange(240)/30*2*torch.pi*.7)
        target = motion_summary(f)
        self.assertEqual(energy_score(target[None].expand(6,-1,-1),target),0.)
        frozen = f[:1].expand_as(f)
        shifted = torch.roll(f,45,0)
        self.assertGreater(energy_score(motion_summary(frozen)[None],target),.01)
        self.assertGreater(energy_score(motion_summary(shifted)[None],target),.01)

    def test_generated_contact_labels_cannot_hide_sliding_without_losing_coverage(self):
        from lightsem.experiments.motion_study.music_evaluation import physical
        f = torch.zeros(1,120,139)
        f[...,0] = .5
        f[...,2] = 1.
        f[...,3:135] = torch.tensor([1.,0,0,0,1,0]*22)
        f[...,135:] = 1.
        claimed,_ = physical(f)
        self.assertAlmostEqual(claimed['sliding'],.5,places=5)
        self.assertAlmostEqual(claimed['contact_coverage'],1.,places=5)
        f[...,135:] = 0.
        hidden,_ = physical(f)
        self.assertEqual(hidden['contact_coverage'],0.)
        self.assertAlmostEqual(hidden['kinetic'],claimed['kinetic'],places=6)

    def test_quantized_rig_limits_hold_through_target_reversals(self):
        from lightsem.experiments.motion_study.lighting_map import limited_angles,LightingConfig
        cfg = LightingConfig()
        targets = np.zeros((500,2))
        targets[50:200] = [65.,-35.]
        targets[200:350] = [-65.,35.]
        ranges = np.asarray([540.,270.])
        ticks = limited_angles(targets,ranges,np.ones(500),cfg)
        emitted = (ticks/65535-.5)*ranges
        velocity = np.diff(emitted,axis=0)*cfg.fps
        acceleration = np.diff(velocity,axis=0)*cfg.fps
        self.assertLessEqual(float(np.linalg.norm(velocity,axis=1).max()),cfg.maximum_speed_deg_s+1e-6)
        self.assertLessEqual(float(np.linalg.norm(acceleration,axis=1).max()),cfg.maximum_acceleration_deg_s2+1e-5)
        self.assertGreater(float(np.ptp(emitted[:,0])),10.)
        static = limited_angles(np.zeros((100,2)),ranges,np.zeros(100),cfg)
        self.assertTrue(np.array_equal(static,np.broadcast_to(static[:1],static.shape)))

    def test_rotation_round_trip_and_degenerate_predictions(self):
        torch.manual_seed(1)
        q, _ = torch.linalg.qr(torch.randn(50, 3, 3))
        q[:, :, 2] *= torch.linalg.det(q)[:, None]
        decoded = rotation_6d_to_matrix(matrix_to_rotation_6d(q))
        torch.testing.assert_close(decoded, q, atol=1e-6, rtol=1e-6)
        bad = torch.tensor([[0., 0, 0, 0, 0, 0], [1., 0, 0, 2, 0, 0]])
        decoded = rotation_6d_to_matrix(bad)
        torch.testing.assert_close(decoded@decoded.transpose(-1, -2), torch.eye(3).expand(2, 3, 3))
        torch.testing.assert_close(torch.linalg.det(decoded), torch.ones(2))

    def test_articulation_keeps_bones_and_root_time_convention(self):
        torch.manual_seed(2)
        features = torch.zeros(120, 139)
        features[:, 0] = 1.
        features[:, 2] = 1.
        features[:, 3:135] = torch.randn(120, 132)
        root = root_from_features(features)
        torch.testing.assert_close(root[:, 0], torch.arange(120)/FPS)
        joints = joints_from_features(features)
        for j, p in enumerate(PARENTS):
            if p >= 0:
                target = torch.full((120,), float(np.linalg.norm(OFFSETS[j])))
                torch.testing.assert_close(torch.linalg.vector_norm(joints[:, j]-joints[:, p], dim=-1), target,
                                           atol=1e-5, rtol=1e-5)

    def test_ground_proximity_without_stationarity_is_not_contact(self):
        j = torch.zeros(90, 22, 3)
        self.assertTrue(bool((contacts_from_joints(j) == 1).all()))
        j[:, :, 0] = torch.arange(90)[:, None]/FPS
        self.assertTrue(bool((contacts_from_joints(j) == 0).all()))

    def test_source_mask_cannot_hide_finite_prediction(self):
        pose = np.zeros((10, 17, 2))
        mask = np.zeros((10, 17), bool)
        self.assertTrue((visibility(pose, mask, False) == 1).all())
        self.assertTrue((visibility(pose, mask, True) == 0).all())
        pose[3, 5] = np.nan
        self.assertEqual(visibility(pose, mask, False)[3, 5], -1)

    def test_duplicate_closure_and_shifted_fingerprint(self):
        groups = Groups(['a', 'b', 'c'])
        groups.join('a', 'b'); groups.join('b', 'c')
        self.assertEqual(groups.find('a'), groups.find('c'))
        rng = np.random.default_rng(3)
        a = rng.integers(0, 2**32, 400, dtype=np.uint32)
        b = np.concatenate((rng.integers(0, 2**32, 31, dtype=np.uint32), a))
        self.assertEqual(fingerprint_similarity(a, b)[0], 1.)
        self.assertLess(fingerprint_similarity(a, rng.integers(0, 2**32, 400, dtype=np.uint32))[0], .85)

    def test_identity_passes_but_frozen_motion_fails_fidelity(self):
        f = np.zeros((120, 139), np.float32)
        f[:, 0] = .5
        f[:, 2] = 1
        f[:, 3:135] = np.tile([1, 0, 0, 0, 1, 0], 22)
        perfect = compare(f, f, .5)
        self.assertTrue(all(fidelity_gates(perfect).values()))
        static = f.copy(); static[:, 0] = 0
        wrong = compare(f, static, .5)
        self.assertFalse(fidelity_gates(wrong)['travel_energy'])
        self.assertFalse(fidelity_gates(wrong)['velocity_fidelity'])
        still = static.copy()
        self.assertTrue(all(fidelity_gates(compare(still, still, .5)).values()))


if __name__ == '__main__':
    unittest.main()
