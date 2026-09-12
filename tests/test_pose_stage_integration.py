"""CPU-only engineering checks. No corpus, model, renderer, or study run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from lightsem.experiments.motion_study.geometry import (
    forward_kinematics, joints_from_features, matrix_to_rotation_6d)
from lightsem.experiments.motion_study.lighting_map import LightingConfig, compose, compose_world_joints
from lightsem.experiments.motion_study.articulated_stage import NEUTRAL, REGIONS
from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.decode import decode_stream

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_features():
    # Nonconstant deterministic CPU sequence: simultaneous travel, yaw, lean,
    # and independent shoulder/leg articulation. No RNG or external recording.
    t = np.arange(60, dtype=np.float64) / 30
    r = np.broadcast_to(np.eye(3), (len(t), 22, 3, 3)).copy()
    for joint, axis, gain in ((0, 1, .4), (3, 2, .5), (16, 2, 1.2),
                              (17, 2, -.9), (1, 0, .8), (2, 0, -.6)):
        angle = gain*np.sin(t*2.3 + joint*.13)
        c, s = np.cos(angle), np.sin(angle)
        i, j = ((1,2), (2,0), (0,1))[axis]
        r[:,joint,i,i] = c; r[:,joint,j,j] = c
        r[:,joint,i,j] = -s; r[:,joint,j,i] = s
    x = np.zeros((len(t), 139), np.float32)
    x[:,0] = .2*np.sin(t); x[:,1] = .1*np.cos(t)
    x[:,2] = .95 + .08*np.sin(t*2)
    x[:,3:135] = matrix_to_rotation_6d(torch.from_numpy(r)).numpy().reshape(len(t),132)
    return x


def loudness():
    return {"time": [0., .7, 2.], "momentary": [-26., -18., -23.],
            "short_term": [-24., -21., -22.]}


def fingerprints(result):
    frames, trace, report = result
    def digest(x):
        x = np.asarray(x)
        return {"shape": list(x.shape), "dtype": x.dtype.str,
                "sha256": hashlib.sha256(x.tobytes()).hexdigest()}
    return {"frames": digest(frames), "trace": {k: digest(v) for k,v in trace.items()},
            "report": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()}

from lightsem.experiments.motion_study.pose_stage import pose_chart, stage_targets, BASE, LEAN_M


class PoseStageIntegration(unittest.TestCase):
    def setUp(self):
        self.x = synthetic_features()
        with torch.inference_mode():
            self.body = joints_from_features(torch.from_numpy(self.x)).numpy()
        self.cfg = LightingConfig(target_space="pose_stage", target_smoothing_s=1e-6)

    def test_exact_reviewed_prototype_parity_on_synthetic_fk(self):
        spec = importlib.util.spec_from_file_location("pose_reference", FIXTURES / "pose_stage_reference.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        bodies = [self.body]
        # FK root rotation includes roll, pitch, yaw and yaw wrap; no fitted data.
        from scipy.spatial.transform import Rotation
        rotations = np.broadcast_to(np.eye(3), (15,22,3,3)).copy()
        rotations[:,0] = Rotation.from_euler("xyz", np.linspace([-1.,-.8,-np.pi], [1.,.8,np.pi],15)).as_matrix()
        rotations[:,16] = Rotation.from_euler("z", np.linspace(-2.,2.,15)[:,None]).as_matrix()
        bodies.append(forward_kinematics(torch.from_numpy(rotations),
                      torch.tensor([[0.,.95,0.]]*15,dtype=torch.float64)).numpy())
        for body in bodies:
            for activity in (.7, np.linspace(-.2,1.2,len(body))):
                for actual, expected in zip(pose_chart(body,activity),reference.pose_chart(body,activity)):
                    np.testing.assert_array_equal(actual,expected)

    def test_actual_default_dmx_and_every_trace_byte_unchanged(self):
        expected = json.loads((FIXTURES / "pose_stage_default_baseline.json").read_text())
        self.assertEqual(fingerprints(compose(self.x,loudness(),2.)),expected)
        self.assertEqual(fingerprints(compose(self.x,loudness(),2.,config=LightingConfig(target_space="legacy"))),expected)

    def test_same_source_photometry_and_decoded_constraints_both_controllers(self):
        before = self.x.copy()
        source_before = self.body.copy()
        original, _, _ = compose(self.x,loudness(),2.)
        rig = canonical_rig()
        nonangle = np.ones(512,bool)
        for f in rig.by_kind("moving_head"):
            nonangle[f.address-1:f.address+3] = False
        for controller in ("chase", "offline_projection"):
            cfg = LightingConfig(target_space="pose_stage", target_smoothing_s=1e-6,
                                 angular_controller=controller)
            frames, trace, report = compose(self.x,loudness(),2.,config=cfg)
            np.testing.assert_array_equal(frames[:,nonangle],original[:,nonangle])
            # Existing EMA at 1e-6 has alpha=1, but subtraction/addition can
            # round by a few ulps. Do not modify this shared legacy path.
            np.testing.assert_allclose(trace["targets"],trace["raw_targets"],rtol=0,atol=2e-15)
            decoded = decode_stream(rig,frames)
            for i,f in enumerate(rig.fixtures):
                if f.kind != "moving_head":
                    continue
                normalized = np.column_stack((decoded["pan"][:,i],decoded["tilt"][:,i]))
                self.assertTrue(np.all((normalized >= 0) & (normalized <= 1)))
                angles = (normalized-.5)*[f.pan_range_deg,f.tilt_range_deg]
                speed = np.diff(angles,axis=0)*cfg.fps
                accel = np.diff(speed,axis=0)*cfg.fps
                self.assertLessEqual(np.linalg.norm(speed,axis=1).max(),cfg.maximum_speed_deg_s+1e-9)
                self.assertLessEqual(np.linalg.norm(accel,axis=1).max(),cfg.maximum_acceleration_deg_s2+1e-9)
        np.testing.assert_array_equal(self.x,before)
        np.testing.assert_array_equal(joints_from_features(torch.from_numpy(self.x)).numpy(),source_before)

    def test_world_seam_exact_raw_chart_and_no_mutation(self):
        before = self.body.copy()
        frames, trace, _ = compose_world_joints(self.body,loudness(),2.,config=self.cfg,dynamics=False)
        times = trace["time"]
        flat = self.body.reshape(len(self.body),-1)
        body40 = np.stack([np.interp(times,np.arange(len(flat))/30,col) for col in flat.T],axis=1).reshape(-1,22,3)
        targets, _, delta = pose_chart(body40,.7)
        rig = canonical_rig()
        heads = sorted(rig.by_kind("moving_head"),key=lambda f:(f.position[0],f.name))
        for rank,f in enumerate(heads):
            index = next(i for i,g in enumerate(rig.fixtures) if f.name==g.name)
            np.testing.assert_array_equal(trace["raw_targets"][:,index],targets[:,rank])
        np.testing.assert_array_equal(trace["regional"],delta)
        np.testing.assert_array_equal(self.body,before)

    def test_fixed_constants_config_semantics_static_and_batch_independence(self):
        np.testing.assert_array_equal(BASE,[7.7,3.2,8.])
        self.assertEqual(LEAN_M,1.7)
        body = np.repeat((NEUTRAL+[0,.95,0])[None],8,axis=0)
        _, chart, _ = pose_chart(body)
        np.testing.assert_allclose(chart[:,:,1],np.broadcast_to(BASE[[0,1,2,2,1,0]],(8,6)),atol=1e-14)
        for full, single in zip(pose_chart(self.body),pose_chart(self.body[23:24])):
            np.testing.assert_array_equal(full[23:24],single)
        for value in pose_chart(body):
            np.testing.assert_array_equal(value,np.broadcast_to(value[:1],value.shape))
        other = LightingConfig(stage_base_unfolded_m=(1.,2.,3.),stage_lean_gain_m=0.,surface_gain=99.,torso_gain=99.)
        for rank in range(6):
            for actual, expected in zip(stage_targets(body,-5.,rank,.7,self.cfg),
                                        stage_targets(body,5.,rank,.7,other)):
                np.testing.assert_array_equal(actual,expected)

    def test_static_world_body_emits_static_dmx_for_both_controllers(self):
        body = np.repeat((NEUTRAL+[0,.95,0])[None],60,axis=0)
        for controller in ("chase","offline_projection"):
            cfg = LightingConfig(target_space="pose_stage",target_smoothing_s=1e-6,
                                 angular_controller=controller)
            frames, _, report = compose_world_joints(body,loudness(),2.,config=cfg,dynamics=False)
            np.testing.assert_array_equal(frames,np.broadcast_to(frames[:1],frames.shape))
            self.assertEqual(report["pose_stage_geometry"]["base_unfolded_m"],[7.7,3.2,8.])
            self.assertEqual(report["pose_stage_geometry"]["signed_lean_gain_m"],1.7)

    def test_crossed_fk_hands_keep_region_ownership_not_fixture_x_order(self):
        r = np.broadcast_to(np.eye(3),(1,22,3,3)).copy()
        r[:,16] = r[:,17] = np.diag([-1.,1.,-1.])
        body = forward_kinematics(torch.from_numpy(r),torch.tensor([[0.,.95,0.]],dtype=torch.float64)).numpy()
        _, chart, _ = pose_chart(body)
        self.assertGreater(body[0,21,0],body[0,20,0])
        self.assertGreater(chart[0,0,0],chart[0,5,0])
        self.assertEqual(REGIONS[0][1],(21,)); self.assertEqual(REGIONS[5][1],(20,))

    def test_signed_rigid_lean_and_chart_bounds(self):
        from scipy.spatial.transform import Rotation
        for axis in ("x","z"):
            q = Rotation.from_euler(axis,[[-.5],[.5]]).as_matrix()
            body = np.einsum("tij,kj->tki",q,NEUTRAL)+[0,.95,0]
            _, chart, _ = pose_chart(body,root_rotation=q)
            self.assertGreater(np.max(np.abs(chart[0]-chart[1])),.5)
        extreme = self.body.astype(float)*1000
        targets, chart, _ = pose_chart(extreme)
        self.assertTrue(np.isfinite(targets).all())
        self.assertTrue(np.all(np.abs(chart[:,:,0])<=7.5))
        self.assertTrue(np.all((chart[:,:,1]>=0)&(chart[:,:,1]<=12.5)))
        self.assertTrue(np.all((targets[:,:,1]==0)|(targets[:,:,2]==-6)))

    def test_projection_nullspace_and_close_endpoints_are_not_repaired(self):
        body = (NEUTRAL+[0,.95,0])[None].copy()
        moved = body.copy()
        moved[:,21] += [0.,.03,.04]  # 4*dy - 3*dz = 0 in this 3D->2D chart.
        np.testing.assert_allclose(pose_chart(body)[0],pose_chart(moved)[0],rtol=0,atol=2e-14)
        body[:,21,0] = .0001
        body[:,20,0] = -.0001
        _, chart, _ = pose_chart(body)
        self.assertGreater(chart[0,0,0],chart[0,5,0])
        self.assertLess(chart[0,0,0]-chart[0,5,0],.002)

    def test_invalid_geometry_and_rotation_inputs(self):
        for body in (np.zeros((0,22,3)),np.zeros((2,21,3)),np.full((2,22,3),np.nan),np.zeros((2,22,3))):
            with self.subTest(shape=body.shape), self.assertRaises(ValueError): pose_chart(body)
        for q in (np.zeros((3,3)),np.diag([-1.,1.,1.]),np.eye(3)*2,
                  np.full((3,3),np.nan),np.full((3,3),np.inf),np.zeros((2,2))):
            with self.subTest(q=q), self.assertRaises(ValueError): pose_chart(self.body,root_rotation=q)
        from lightsem.experiments.motion_study.articulated_stage import infer_root_rotation
        inferred = infer_root_rotation(self.body)
        for a,b in zip(pose_chart(self.body),pose_chart(self.body,root_rotation=inferred)):
            np.testing.assert_array_equal(a,b)
        for activity in (np.nan,np.inf,-np.inf,np.zeros(3)):
            with self.assertRaises(ValueError): pose_chart(self.body,activity)
        for rank in (-1,6,.5):
            with self.assertRaises(ValueError): stage_targets(self.body,0,rank,.7,self.cfg)
        for x in (np.nan,np.inf,7.):
            with self.assertRaises(ValueError): stage_targets(self.body,x,0,.7,self.cfg)

    def test_mapper_rejects_unknown_modes_smoothing_bad_world_and_inputs(self):
        for cfg in (LightingConfig(target_space="pose_stag"),LightingConfig(target_space="pose_stage"),
                    LightingConfig(target_space="pose_stage",target_smoothing_s=np.nan),
                    LightingConfig(target_space="pose_stage",target_smoothing_s=-1.),
                    LightingConfig(target_space="pose_stage",target_smoothing_s=1e-6,angular_controller="unknown"),
                    LightingConfig(target_space="pose_stage",target_smoothing_s=1e-6,target_smoothing_mode="unknown")):
            with self.assertRaises(ValueError): compose(self.x,loudness(),2.,config=cfg)
        for world in (self.body[:2],np.full_like(self.body,np.nan),np.zeros_like(self.body)):
            with self.assertRaises(ValueError): compose(self.x,loudness(),2.,config=self.cfg,world_joints_override=world)
        with self.assertRaises(ValueError):
            compose(self.x,loudness(),2.,config=self.cfg,world_joints_override=self.body,
                    independent_features=np.repeat(self.x[None],6,axis=0))
        for x in (np.zeros((60,138)),np.full_like(self.x,np.nan)):
            with self.assertRaises(ValueError): compose(x,loudness(),2.,config=self.cfg)


if __name__ == "__main__":
    unittest.main()
