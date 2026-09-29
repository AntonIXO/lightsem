# Articulated stage candidate: implementation

Scope: synthetic geometry engineering only. No Shpongle tuning, model inference,
training, GPU work, vault writes, video review, or performance claim.
Read `CLAUDE.md`, `outputs/dance_investigation_mapper.md`, and `RESEARCH.md`
(the requested lowercase `research.md` does not exist).

## New files and interface

- `lightsem/experiments/motion_study/articulated_stage.py`
- `tests/test_articulated_stage.py`
- This report.

No changes to `lighting_map.py` or `offline_angles.py` by this worker. Parent
integration must keep this geometry opt-in.

```python
stage_targets(body, fixture_x, rank, activity, cfg, *, root_rotation=None)
# -> (target_xyz[T,3], regional_delta[T,3])
```

`cfg` is accepted for call compatibility. Legacy geometry/timing gains do not
alter the fixed candidate constants. Rank means increasing canonical fixture x.
The fixture identities stay right hand, right leg, right torso, left torso,
left leg, left hand. There is no nearest-side sorting during body turns.

## Geometry

Root-child hip and spine1 offsets determine an orthonormal body frame. Its
rotation relative to the exact asymmetric rest frame recovers root Q. This
uses canonical FK position semantics, not a world-up/yaw approximation.
Alternatively supply proper column-vector rotation matrices as `(3,3)` or
`(T,3,3)`. Degenerate inferred anchors and invalid explicit matrices fail.

The returned residual is `regional_world - Q @ neutral_region`. It removes the
old `(Q-I) @ rest` contraction term while preserving world-space directions of
articulated deviations. It does not cancel gestures by re-expressing every
frame in a facing-normalized coordinate system.

Fixed engineering choices:

- Lane center: `0.95 * fixture_x`.
- Lateral residual: `0.65 * tanh(activity_gain * 2.8 * delta_x / 0.65)` m.
- Activity gain: `0.25 + 0.75 * clip(activity, 0, 1)`.
- Explicit shared facing: `0.35 * Q[0,2]` m in x and
  `0.35 * (1 - Q[2,2])` m in unfolded coordinate u.
- Bounded shared root translation: `0.25*tanh(root_x/3)` m in x and
  `-0.4*tanh(root_z/3)` m in u. No EMA, moving origin, or history dependence.
- Gesture u displacement: activity gain times `4*delta_y - 3*delta_z`.
- Shared root height: `0.8*tanh((root_y-0.95)/0.4)` m in u.
- Hand/leg/torso base u: `5.7 / 3.2 / 5.7` m.
- u limits: `[0, 12.5]`; x limits: `[-7.5, 7.5]` m.

The continuous chart is `(x, max(u-7,0), 1-min(u,7))`. Thus floor z decreases
from +1 to -6, then wall y rises from 0 to 5.5. The seam has a geometric corner,
not a jump. Raised and backward gestures can reach the wall; planted leg
regions can remain on the floor. This remains a 3D-to-2D projection with a
null direction. It does not encode every pose, contact, or wrist orientation.

Canonical lane centers are 1.9 m apart. Bounded lateral articulation leaves
at least 0.6 m between adjacent raw target lanes, and at least 8.2 m outer span.
Shared cues shift all lanes together. These bounds concern the canonical rig,
shared body/activity, and raw targets, not independently generated bodies,
actuator trajectories, beam-cone separation, pixel coverage, or music quality.
The function accepts fixture x within +/-6.5 m, but lane-separation claims do
not extend to arbitrary custom rigs.

## Validation

Native command:

```text
bash run_torch.sh -m unittest discover -s tests -p test_articulated_stage.py -v
```

Result: **8 tests passed**, exit 0 (test runtime 0.030 s).

- Static neutral and raised holds are exactly stable and chunk-independent.
- 361 synthetic root-only yaw poses retain 9.5 m raw x span, versus the requested
  minimum 6 m. Corrected regional residuals are zero to 1e-12 m tolerance.
  Facing still changes endpoints, including front/back and lateral turns.
- Full pitch/roll/yaw root recovery matches supplied Q to 1e-12 tolerance.
- Raising both arms moves hand endpoints above 0.5 m on the back wall while
  planted leg targets remain unchanged on the floor. A separate backward wrist
  displacement also changes the surface family.
- Chart and wrist-height paths are continuous across the floor/wall seam.
- Finite synthetic target rays round-trip through `first_hits` to 1e-12 m.
  Encoding and decoding current DMX optics keeps tested first-hit positions
  within 0.002 m; tested cases include both chart limits and turned poses.
- Anatomical right-wrist intervention still changes only the right-hand fixture
  after a 180-degree turn. Extreme lateral controls do not cross lanes.
- Invalid rotations and degenerate inferred anchors fail explicitly.

The round-trip test uses the current finite floor/back-wall axis observer. It
is not a rendered visibility, truss-occlusion, cone-support, or aesthetic test.

## Integration caution

Do not interpret this geometry test as controller acceptance. Cartesian target
smoothing across the chart corner can create intermediate points above the
floor and in front of the wall. If smoothing is retained, smooth unfolded
coordinates and refold, or use the proposed zero-target-smoothing path and
verify emitted finite first hits separately. Quantized motor projection can
also create trajectories distinct from these raw targets. No clock, sine,
chase, randomness, tempo script, or time fade is implemented here; trigonometry
appears only in synthetic orientation conversion tests.
