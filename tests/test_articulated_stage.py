"""Synthetic geometry contracts; no model, music, or aesthetic evaluation."""
import unittest
from types import SimpleNamespace

import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.motion_study.articulated_stage import (
    NEUTRAL, REGIONS, infer_root_rotation, stage_targets, unfolded_targets,
)
from lightsem.experiments.motion_study.geometry import OFFSETS, PARENTS
from lightsem.experiments.motion_study.spatial_observer import encode_targets, first_hits
from lightsem.experiments.quality_observer import beam_series


def rotation(angle, axis="y"):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose(root_q=None, raised=False):
    local = np.broadcast_to(np.eye(3), (22, 3, 3)).copy()
    if root_q is not None:
        local[0] = root_q
    if raised:
        local[16] = rotation(np.pi/2, "z")
        local[17] = rotation(-np.pi/2, "z")
    world = np.empty_like(local)
    body = np.empty((22, 3))
    body[0], world[0] = [0, .95, 0], local[0]
    for joint in range(1, 22):
        parent = PARENTS[joint]
        body[joint] = body[parent] + world[parent] @ OFFSETS[joint]
        world[joint] = world[parent] @ local[joint]
    return body


class ArticulatedStageContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace()
        self.rig = canonical_rig()
        self.heads = sorted(self.rig.by_kind("moving_head"), key=lambda f: (f.position[0], f.name))

    def targets(self, body, activity=1., **kwargs):
        return np.stack([stage_targets(body, f.position[0], rank, activity, self.cfg, **kwargs)[0]
                         for rank, f in enumerate(self.heads)], axis=1)

    def test_static_pose_has_no_clock_or_sequence_dependent_motion(self):
        for raised in (False, True):
            body = np.repeat(pose(raised=raised)[None], 240, axis=0)
            target = self.targets(body)
            np.testing.assert_array_equal(target, np.broadcast_to(target[:1], target.shape))
            np.testing.assert_array_equal(target[117:118], self.targets(body[117:118]))

    def test_root_turn_removes_rest_contraction_but_retains_facing(self):
        angles = np.linspace(-np.pi, np.pi, 361)
        q = np.stack([rotation(v) for v in angles])
        body = np.stack([pose(v) for v in q])
        inferred = infer_root_rotation(body)
        np.testing.assert_allclose(inferred, q, atol=1e-12)
        target = self.targets(body)
        self.assertGreaterEqual(np.ptp(target[:, :, 0], axis=1).min(), 6.)
        np.testing.assert_allclose(np.ptp(target[:, :, 0], axis=1), 9.5, atol=1e-12)
        self.assertGreater(np.max(np.abs(target[0]-target[180])), .6)  # back vs front
        self.assertGreater(np.max(np.abs(target[90]-target[180])), .3)  # lateral facing
        for rank, f in enumerate(self.heads):
            _, delta = stage_targets(body, f.position[0], rank, 1., self.cfg)
            np.testing.assert_allclose(delta, 0., atol=1e-12)
        np.testing.assert_allclose(target, self.targets(body, root_rotation=q), atol=1e-12)
        np.testing.assert_allclose(target[0], target[-1], atol=1e-12)  # yaw wrap

    def test_full_root_orientation_recovery_not_world_up_approximation(self):
        q = rotation(.8) @ rotation(.4, "x") @ rotation(-.3, "z")
        body = pose(q, raised=True)[None]
        np.testing.assert_allclose(infer_root_rotation(body)[0], q, atol=1e-12)
        np.testing.assert_allclose(self.targets(body), self.targets(body, root_rotation=q), atol=1e-12)

    def test_raised_hands_reach_wall_while_planted_legs_stay_on_floor(self):
        body = np.stack([pose(), pose(raised=True)])
        target = self.targets(body)
        np.testing.assert_array_equal(target[0, :, 1], 0.)
        self.assertTrue((target[1, [0, 5], 1] > .5).all())
        np.testing.assert_array_equal(target[1, [0, 5], 2], -6.)
        np.testing.assert_array_equal(target[:, [1, 4], 1], 0.)
        np.testing.assert_array_equal(target[0, [1, 4]], target[1, [1, 4]])
        # Separate synthetic depth intervention demonstrates non-height response.
        backward = pose()[None]
        backward[:, 21, 2] -= .7
        self.assertGreater(self.targets(backward)[0, 0, 1], .5)

    def test_surface_chart_and_pose_path_are_continuous_at_seam(self):
        eps = 1e-7
        target = unfolded_targets(2., np.array([7.-eps, 7., 7.+eps]))
        np.testing.assert_allclose(target[1], [2., 0., -6.])
        np.testing.assert_allclose(np.linalg.norm(np.diff(target, axis=0), axis=1), eps, atol=1e-12)
        body = np.repeat(pose()[None], 3, axis=0)
        body[:, 21, 1] += (7.-5.7)/4. + np.array([-eps, 0., eps])
        path = self.targets(body)[:, 0]
        np.testing.assert_allclose(path[1, 1:], [0., -6.], atol=1e-12)
        self.assertLess(np.linalg.norm(np.diff(path, axis=0), axis=1).max(), 4.01*eps)

    def test_first_hit_and_quantized_optics_roundtrip(self):
        body = np.stack([pose(), pose(raised=True), pose(rotation(np.pi)), pose(rotation(.8), raised=True)])
        # Force both chart endpoints with finite synthetic positional controls.
        low, high = pose(), pose()
        low[21, 1] -= 10.
        high[21, 1] += 10.
        body = np.concatenate((body, low[None], high[None]))
        target = self.targets(body)
        positions = np.array([f.position for f in self.heads])
        direction = target-positions
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        hit, surface, _ = first_hits(direction, positions)
        self.assertTrue(np.isfinite(hit).all())
        self.assertTrue((surface >= 0).all())
        np.testing.assert_allclose(hit, target, atol=1e-12)
        # Encoder/beam_series use rig order; do not assume sorted order matches.
        order = [self.heads.index(f) for f in self.rig.by_kind("moving_head")]
        dmx = encode_targets(target[:, order], self.rig)
        emitted, _ = beam_series(dmx, self.rig)
        hit, surface, _ = first_hits(emitted, positions[order])
        self.assertTrue(np.isfinite(hit).all())
        self.assertTrue((surface >= 0).all())
        np.testing.assert_allclose(hit, target[:, order], atol=.002)

    def test_identities_persist_and_lateral_residual_cannot_cross_lanes(self):
        self.assertEqual(tuple(r[0] for r in REGIONS), (
            "right_hand", "right_leg", "right_torso", "left_torso", "left_leg", "left_hand"))
        q = rotation(np.pi)
        neutral = pose(q)[None]
        moved = neutral.copy()
        moved[:, 21, 1] += .6  # anatomical RIGHT wrist is now on positive world x
        base, changed = self.targets(neutral), self.targets(moved)
        self.assertGreater(np.linalg.norm(changed[:, 0]-base[:, 0]), .5)
        np.testing.assert_array_equal(changed[:, 1:], base[:, 1:])
        _, delta = stage_targets(moved, -5., 0, 1., self.cfg)
        np.testing.assert_allclose(delta, [[0., .6, 0.]], atol=1e-12)
        extreme = pose()[None]
        # Non-root anchors preserve recovered Q; extreme pose saturates both ways.
        for ids in ((21, 11, 5, 17, 15, 12), (20, 10, 4, 16)):
            extreme[:, ids, 0] += 100. if 21 in ids else -100.
        x = self.targets(extreme)[0, :, 0]
        self.assertGreaterEqual(np.diff(x).min(), .6-1e-12)
        self.assertGreaterEqual(np.ptp(x), 8.2-1e-12)

    def test_invalid_rotation_and_degenerate_anchors_fail_explicitly(self):
        body = pose()[None]
        with self.assertRaises(ValueError):
            self.targets(body, root_rotation=np.zeros((3, 3)))
        body[:, 1] = body[:, 2]
        with self.assertRaises(ValueError):
            self.targets(body)
        self.assertTrue(np.isfinite(self.targets(body, root_rotation=np.eye(3))).all())


if __name__ == "__main__":
    unittest.main()
