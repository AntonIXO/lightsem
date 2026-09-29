# Opt-in pose-stage mapper integration — 2026-09-11

## Status and scope

Engineering integration only. `LightingConfig()` still selects `legacy`.
No real recordings, train/VAL/TEST data, models, GPU, renderer, vault, or study
run were used. No frozen experiment, other worker output, commit, or default
promotion was changed. A new real-record evaluation requires the parent's
preregistration and freeze first.

This supplies an alternative to fixed fixture-x lanes. It is **not a measured
dance fix**. Raw endpoint ordering does not establish every 3D or emitted-beam
intersection, tracking quality, visual separation, or aesthetic quality.

## Files

- `lightsem/experiments/motion_study/pose_stage.py` — new stateless chart and
  mapper adapter; finite body/activity/proper-rotation and rig/rank checks.
- `lightsem/experiments/motion_study/lighting_map.py` — add only opt-in
  `pose_stage` dispatch, seam-smoothing guard, settings comments and opt-in
  effective-geometry report. Existing dirty-tree edits were retained.
- `tests/test_pose_stage_integration.py` — CPU synthetic checks.
- `tests/fixtures/pose_stage_reference.py` — byte-for-byte copy of the reviewed
  outputs-only prototype, independent of the new implementation.
- `tests/fixtures/pose_stage_default_baseline.json` — actual pre-edit default
  DMX, every trace array, and serialized report fingerprints for the fixed
  60-native-frame synthetic sequence. This is not a fabricated expected DMX.
- `outputs/dance_pose_stage_integration.md` — this handoff.

Reference source: `outputs/gesture_transfer_20260911_posechart_audit/pose_chart.py`.
SHA256: `ae58ac0fa70405a21f305fc1a4e3e95ce9712d9f5929b730b12091fb28b62e2c`.
Pre-task mapper SHA256 (including prior dirty-tree work):
`c6215a234c1b085ae43765a6e2ec8f0d22cb47cd9ac2cd99a2c8765aef6dc38b`.

## Use and configuration semantics

```python
cfg = LightingConfig(target_space="pose_stage", target_smoothing_s=1e-6)
frames, trace, report = compose(features, loudness, duration_s, config=cfg)
# Existing complete-world-body seam is also supported:
frames, trace, report = compose_world_joints(world, loudness, duration_s, config=cfg)
```

- `pose_stage` fixes prototype anchors at **(7.7, 3.2, 8.0) m** and signed lean
  at **1.7 m**. They are not estimated from source recordings.
- `stage_base_unfolded_m` and `stage_lean_gain_m` still apply only to
  `articulated_stage`. `pose_stage` ignores them explicitly, as documented in
  `LightingConfig`, the chart module, and `report['pose_stage_geometry']`.
  It also ignores `surface_gain` and `torso_gain` for geometry.
- Equations, saturations, root-local regional composition, and region ownership
  match the prototype exactly. X gain is 8; x bound is 7.5; chart half-range is
  6.25; the floor/wall seam is u=7; chart u is in [0,12.5]. No fixture-x lane,
  rest blending, endpoint reordering, separation floor, or nullspace repair was
  added. Fixture x validates the same +/-6.5 m rig support as articulated_stage.
- `trace['regional']` contains **root-local** deltas for this mode. The
  articulated-stage mode retains its existing world-direction deltas.
- Geometry has no clock, phase, history, or sequence statistic. Existing audio
  activity, photometry, interpolation, palette, and controller paths remain.
- Target smoothing must be finite and in [0,1e-6]. Larger Cartesian smoothing
  is forbidden across the unfolded floor/wall seam, as in articulated_stage.
  Shared causal EMA at 1e-6 has alpha=1 but can add a few floating-point ulps;
  the legacy smoother was not changed. Exact prototype parity is for raw chart
  outputs. Tests allow 2e-15 absolute roundoff after the unchanged smoother.
- `chase` remains the default angular controller. `offline_projection` remains
  available; it uses the full supplied angular sequence, not a new timed motif.
  Both paths use the existing DMX encoder and are checked through the decoder.
- The optional `pose_chart(..., root_rotation=Q)` seam checks finite proper
  rotations. Without Q, hip/spine1 offsets infer root orientation. This does
  not edit source joints. Complete-world-body overrides still cannot mix
  independent native draws.

## Validation

Run from `/root/lightsem` with CUDA disabled:

```bash
CUDA_VISIBLE_DEVICES="" bash run_torch.sh -c "import sys,unittest; sys.path.insert(0,'tests'); unittest.main(module=None,argv=['unittest','test_pose_stage_integration','test_articulated_stage','test_stage_lighting_integration','test_stage_root_lean','test_offline_angles','-v'])"
```

**24 tests passed** (11 new + 13 existing), 0.280 s test time; exit code 0.
The whole dedicated-environment command took 1.67 s.

The new checks cover:

1. Exact array equality with the reviewed prototype on synthetic FK, root
   roll/pitch/yaw, yaw endpoints, local articulation, and varying activity.
2. Actual pre-edit default/explicit-legacy DMX bytes, every trace array byte,
   and report fingerprint equality on the same moving synthetic sequence.
3. Unchanged non-angle DMX bytes and source features/FK joints; decoded
   mechanical position, speed <=30 deg/s and acceleration <=80 deg/s2 for
   both existing controllers on that sequence.
4. Full-world-body seam raw-target/regional parity and input immutability.
5. Fixed anchors/config semantics, frame/batch independence, static raw
   geometry and static emitted DMX for both controllers.
6. Crossed synthetic FK hand endpoint ordering with fixed region ownership.
7. Signed opposite rigid leans, finite chart bounds, surface membership.
8. Explicit preservation of one 3D-to-2D nullspace and sub-2-mm raw endpoint
   separation in constructed world bodies. These checks do not claim that
   arbitrary constructed bodies are valid FK or that beam intersections follow.
9. Invalid rotations, degenerate/nonfinite bodies, nonfinite activity, bad
   rank/fixture bounds, unknown target/controller/smoothing names, prohibited
   smoothing, bad world override, and invalid feature inputs.

## Limitations

- Synthetic engineering only; no corpus-level generalization or visual claim.
- Millimetre pair overlaps and projection nullspaces remain by design. Saturation
  can suppress large movements. No universal separation guarantee is offered.
- Controller motion can differ from raw targets. No emitted-beam crossing or
  cone-overlap metric was introduced or claimed for this mode.
- The adapter currently computes the full six-region chart per fixture call.
  This favors literal prototype parity; no long-record throughput was measured.
- Fingerprint regression is tied to this synthetic sequence and numerical
  environment. It is not an exhaustive proof for every legacy input. Existing
  target modes keep their prior equations, config defaults, and execution paths.
- Initial test development found a SciPy Euler-angle input-shape mismatch and
  expected EMA roundoff; only tests were corrected. An attempted dotted
  `tests.*` runner failed because `tests` is not an importable local package;
  the command above uses its explicit path. No failure was hidden by relaxing
  the exact prototype/default-byte checks.
