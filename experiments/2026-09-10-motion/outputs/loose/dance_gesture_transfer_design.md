# Gesture transfer: lateral mechanism audit and one pose-chart prototype

**Unpreregistered synthetic engineering diagnosis, not a study or visual acceptance.** Study label `20260911`. Instrument: proper canonical FK joints and raw six-head target geometry. No real recordings, VAL/TEST, source model, music, GPU, renderer, controller, DMX, or vault access. Production files/defaults were not changed. The previous real-source root-translation audit is context only; no root-gain fix was attempted.

## Answer

**The existing stage02 mapper structurally blocks a real gesture class: changed left/right endpoint ordering.** Every crossed-hand, crossed-leg, reach, raise, and turn input must retain the fixture-lane x order. It also strongly compresses already-folded arm reach. A root-local candidate admits anatomical crossings without swapping ownership. **It is not a dance fix:** it can bring targets within millimetres, and a proper-FK counterexample moving the wrist **49.1 cm** is invisible in **both** mappings.

These are statements about raw surface endpoints. Ordered endpoint x does not prove non-crossing 3D beam segments or ordered rendered centroids. Moving shafts, illuminated surface spread, projected 2D spread, and dance quality are separate instruments. No new study should start before parent preregistration.

## Existing mapping and mathematical losses

Comparator: live `articulated_stage.stage_targets`, stage02 role anchors `(7.7,3.2,8.0)`, lean gain **0**, constant activity **0.7** (`a=.775`), canonical fixtures x=`[-5,-3,-1,1,3,5]`. No smoothing or motor limits. Source hashes are in `measurements.json`.

Let `r_k` be the regional weighted root-relative position, `n_k` its fixed canonical rest position, and `Q` the root rotation. Existing code correctly computes

```
d_world = r_world - Q n = Q d_local
x_k = .95 fixture_x[k] + shared_x + .65 tanh(a 2.8 d_world_x / .65)
shared_x = .35 forward_x + .25 tanh(root_x/3)
u_k = base_k + .35(1-forward_z) + a(4 d_world_y - 3 d_world_z)
      + .8 tanh((root_y-.95)/.4) - .4 tanh(root_z/3)
```

The surface chart uses `u=0..7` for floor `z=1-u`, and `u=7..12.5` for back wall `y=u-7,z=-6`. It clips x to ±7.5 and u to `[0,12.5]`.

1. **Order cannot change.** Neighbouring anchors are 1.9 m apart. Each residual is strictly inside ±0.65 m for finite inputs, so `x[k+1]-x[k] > 1.9-1.3 = .6 m`. Shared cues cancel. At lean gain 0 the total x range remains inside ±6 m, so final x clipping is inactive. This is a proof for **all** finite canonical FK poses on this rig/config, not just the samples. Nonadjacent gaps are even larger. A wrist crossing the body midline cannot cross its partner endpoint. Saturation also prevents hand endpoints from entering neighbouring role lanes.
2. **World projection changes gesture direction with facing.** For root yaw θ, `d_world_x=cosθ*d_local_x+sinθ*d_local_z`. At 90°, local lateral motion contributes zero to x. It can still enter u through world z. This is directional transfer/loss, not total gesture erasure. Root-local correction does not currently mean root-local mapping: subtracting `Q n` removes the old `(Q-I)n` rest error but leaves `Q d_local` in world axes.
3. **Strong lateral saturation.** The derivative is `a*2.8*sech²(a*2.8*d_x/.65)`. At activity .7 its values for `d_x=0,.25,.5,.75,1 m` are **2.170, 1.158, .287, .0573, .0109 m/m**. A 1 m articulated displacement is not literally erased, but further motion there has only 0.5% of the zero-displacement slope. Our folded-arm/reach probe reaches a minimum derivative **.0338 m/m**. No u endpoint clipping occurred in the named gesture probes: their compression cannot be blamed on u clipping.
4. **Signed rigid lean is removed at stage02 lean=0.** Root roll of ±20° produces different world joints but zero regional articulation; forward remains `(0,0,1)`. Existing targets are constant to floating-point precision. Upper-spine articulation, in contrast, is retained. This is not a reason to restore the incorrect rotated-rest term.
5. **Regional and dimensional bottlenecks remain.** Hands use wrists only; legs use `.65 foot + .35 knee`; torsos use `.5 shoulder + .3 head + .2 neck`. Wrist orientation is invisible. Weighted averages lose independent joint configurations and cancelling changes. Even a full regional point becomes only `(x, 4y-3z)`: direction `(0,3,4)` is an exact nullspace before clipping. Two torso heads also share their head/neck terms; they do not represent six independent body channels.

## Synthetic probes and measurements

`run_audit.py` uses production `forward_kinematics` with proper local rotation matrices and fixed canonical bone offsets. Each diagnostic has 241 static parameter samples, not learned motion. Linear angle ramps only enumerate input poses; the mapper never reads a clock. Root stays `(0,.95,0)` except root rotation. This is bone-valid FK, **not** collision, contact, balance, or human range-of-motion validation.

- Hand fold/cross: left/right shoulder yaw ±0→90°, elbow yaw ±0→90°.
- Front reach: shoulders held at ±90°, elbows unfold ±90→0°.
- Raise: shoulder roll ±0→90°, repeated at root yaw 0° and 90°.
- Leg cross: hip roll ∓0→25°.
- Signed torso bend: spine1 roll -35→35°.
- Turns/lean: neutral local joints, root yaw -180→180° or roll -20→20°.
- Null: isolated leaf wrist rotation; static neutral. A separate exact FK projection-null pair is described below.

| Probe / measurement | Existing stage02 | Candidate |
|---|---:|---:|
| Hand-cross ramp: inverted anatomical hand endpoint frames / 241 | 0 | 40 |
| Hand-cross end: `left_x-right_x`, m | +8.211 | **-1.580** |
| Front reach: right-hand target x change, m | -0.0238 | -2.090 |
| Raise facing 90°: right-hand x transfer, m | 0 (roundoff) | +3.411 |
| Leg-cross ramp: inverted leg endpoint frames / 241 | 0 | 149 |
| Rigid roll -20→20°: right-torso x change, m | 0 (roundoff) | -0.891 |

The folded-to-reached right wrist moves **38.1 cm in 3D**, including **26.3 cm laterally**. Existing x moves only **2.38 cm**, while u moves **64.0 cm**. Thus “no gesture transfer” would be false; the lateral reach transfer is strongly compressed. At yaw 90° the raised right wrist moves **74.9 cm in 3D**; existing x is unchanged but u changes **2.877 m**. Again, this is not an all-coordinate collapse.

**Surface/2D spread is separate:** in the hand-cross ramp, existing unfolded `(x,u)` convex-hull area is **34.81–38.87 m²**; candidate area is **8.51–27.66 m²**. Candidate x-span contracts from its rest span **9.69 m** to a minimum **1.85 m**, which accompanies the synthetic fold/cross, not necessarily failure. Existing raised-at-90° area ranges **35.91–46.73 m²** despite zero lateral hand transfer. These are intrinsic chart-area measures, **not** world XY, camera pixels, beam footprint areas, or visual diversity scores. Surface Euclidean pair distance is measured separately in the JSON.

**Important candidate negative:** minimum surface pair distance is **1.68 cm** during fold/cross, **4.06 mm** during front reach, and **6.69 mm** during leg crossing. No target-repulsion/spacing guarantee exists. This may yield unwanted overlap or collapse in the renderer. Crossing and useful distributed lighting are not synonymous.

## One stateless alternative: anatomical root-local pose chart

Prototype: `outputs/gesture_transfer_20260911_posechart_audit/pose_chart.py`. No production integration. Fixture rank always owns the same region `(R hand,R leg,R torso,L torso,L leg,L hand)` regardless of current side, facing, or crossing. No sorting, swaps, lane anchors, time, oscillator, phase, sequence statistics, learned model, or per-recording fit.

Compute `local_j = Q.T (world_j-root)`, `r_k = Σ weights*local_j`, `d_k=r_k-n_k`. The canonical hip/spine frame recovers Q, including rest asymmetry; it is not a world-up yaw approximation. Invalid/degenerate frames are rejected. The prototype uses

```
a = .25 + .75 clip(activity,0,1)
sx = .35 forward_x + .25 tanh(root_x/3) + a*1.7 up_x
su = .35(1-forward_z) + .8 tanh((root_y-.95)/.4)
     - .4 tanh(root_z/3) - a*1.7 up_z
x_k = 7.5 tanh((8*r_k,x + sx)/7.5)
du_k = a*(4*d_k,y - 3*d_k,z) + su
u_k = 6.25 + 6.25 tanh(atanh((base_k-6.25)/6.25) + du_k/6.25)
```

Map `(x,u)` through the same floor/back-wall chart. The chart is continuous at `u=7` but has a derivative corner. There is no interpolation shortcut through free space. Smooth bounds remove hard u clipping, not the geometric corner. For finite ordinary inputs they keep endpoints within finite surfaces. Algebraically bounded tanh can round to its bounds at extreme floats; no claim of universal strictly positive machine derivative is made.

**Fixed scale choices:** 8 m/m comes from a 6 m half-stage linear extent divided by the canonical ~0.75 m half arm span. The smooth lateral bound is the established ±7.5 m chart boundary. Lean gain 1.7 m is one canonical body height, preserving sign without reinstating `Q n-n`. Vertical/depth gains 4 and 3, root bounds, facing bounds, and stage02 role anchors are retained engineering constants. U half-range 6.25 m is half the floor-plus-wall chart. No parameters were fitted to source motion, diagnostics, or visual scores. This is one design, not a gain sweep; its larger lateral gain and changed coordinates are not a clean single-factor causal experiment.

Absolute `r_x`, rather than only an offset inside fixture lanes, lets poses dictate composition. Since the same monotone x function and shared cue apply to all regions, **regional x ordering is preserved**, including legitimate anatomical inversions. Lateral activity scaling is intentionally absent: intensity can diminish without algebraically pulling crossed hands back toward rest. Activity still modulates u and lean as shown.

**Recorded engineering revision:** the first implementation used `8*(n_x+a*d_x)`. At activity .7 it blocked the modest synthetic wrist crossing despite inverted source wrists. The saved `initial_activity_blend_measurements.json` reports 0/241 hand inversions. It was replaced with absolute `8*r_x`, not tuned by fitting a gain. This was an unpreregistered design iteration, not a preregistered win. All constants stayed fixed.

Root-local articulation intentionally stays consistent through yaw; facing is a small separate cue, not a rotating formation. A root-only turn does not sweep neutral limb offsets around the rig. For rigid rotation, `d_local=0`, avoiding the old `(Q-I)n` artifact. Signed up-vector lean is explicit. The x bound means a shared pre-bound facing cue is not an exactly rigid post-bound translation. This candidate does **not** fully represent world facing/direction.

## Exact FK counterexample retained by the candidate

Rotate a left shoulder by 60° about z, leaving its elbow at rest. Let `v` be its shoulder-to-wrist vector. Reflect it across the plane normal to `n=(0,3,4)/5`, then realize that vector with a proper shoulder rotation (Rodrigues alignment; other local joints unchanged). Both poses have the same wrist x and `4y-3z`.

Measured change between these two proper-FK bodies:

- Left wrist: **49.1197 cm**, difference `(0,-.294718,-.392958) m`.
- Left elbow: **24.1510 cm**.
- Maximum target difference across all six heads: existing **0**, candidate **4.44e-16 m**.

The weighted torso shoulder position does not move when its own rotation changes, so no other region rescues this signal. Thus the prototype still compresses a nontrivial arm gesture class. A point chart cannot encode complete 3D articulation; this counterexample is not grounds to claim all arm motion is lost. Leaf wrist rotation is also invisible to both, as expected from a joints-only positional input.

## Verification, artifacts, and limits

CPU command (CUDA disabled):

```
CUDA_VISIBLE_DEVICES='' bash run_torch.sh outputs/gesture_transfer_20260911_posechart_audit/run_audit.py
CUDA_VISIBLE_DEVICES='' bash run_torch.sh outputs/gesture_transfer_20260911_posechart_audit/plot_diagnostics.py
```

- Bone lengths preserved: max error **1.71e-8 m** (float32 canonical offsets).
- Inferred Q versus FK root Q: max element error **1.11e-15**.
- Permuting input frames permutes candidate targets exactly: max error **0**.
- Left-arm-only articulation changes no other owned region: max error **0**.
- Adjacent surface samples at `u=7±1e-8` differ by **1e-8 m**, no seam jump.
- All saved probe arrays finite. Additional 241 arbitrary proper-rotation FK stress poses remain finite and in bounds; these are not biomechanically plausible motion tests.
- CUDA initialized: **false**. No model checkpoints/data files opened.

Artifacts in `outputs/gesture_transfer_20260911_posechart_audit/`:
`pose_chart.py`, `run_audit.py`, `measurements.json`, `synthetic_probes.npz`, `chart_trajectories.png`, `fk_nullspace_counterexample.png`, `plot_diagnostics.py`, logs, the first-iteration JSON, and `source_snapshot/` copies of the audited production mapper/FK files. The audited production source hashes are unchanged at audit completion. `chart_trajectories.png` shows raw intrinsic surface coordinates only. `fk_nullspace_counterexample.png` shows different arm geometry with identical targets.

No optical, controller, safety, pixel, music-causality, perceptual, contact, or dance acceptance was tested. Faster/larger target paths may fail existing motor limits. Ownership is guaranteed by fixed region assignment, not semantic recognition. World-turn composition is deliberately simplified. Target overlap is a serious unresolved negative. This audit justifies questioning the fixed-lane lateral mapping; it does not justify promoting this prototype, reopening root gains, or claiming lights now dance.
