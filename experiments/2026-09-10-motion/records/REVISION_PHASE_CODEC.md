# Native-frame codec revision

This revision follows failure of all three originally scheduled codecs. It tests a
compound representation change; it cannot isolate interpolation or BF16 as the cause
of the previous failure. The existing numerical fidelity thresholds are retained.

Four native 30 Hz frames are packed into each 7.5 Hz token using an orthonormal Haar
transform. TRAIN-fitted linear projections compress the rotation channels within
each temporal band. The average band has a fitted mean; difference bands have fixed
zero offsets so a constant pose does not acquire periodic motion from a learned bias.
No temporal interpolation is needed to recover the native frame phases.

Horizontal root velocities, height and four contact channels are explicit latent
channels. Their full cost is counted below. A future music model must generate these
channels too; the decoder receives no source-motion side input during playback.

| Arm | Rotation ranks by temporal band | Total channels per token | Token Hz | Values/s |
|---|---|---:|---:|---:|
| Frame24 linear baseline | 24 | 31 | 30 | 930 |
| Phase64 | 32,16,12,4 | 92 | 7.5 | 690 |
| Phase128 | 64,32,20,12 | 156 | 7.5 | 1170 |
| Phase256 | 100,72,52,32 | 284 | 7.5 | 2130 |

All ranks use identical full TRAIN-record exposure, with equal canonical-song and
recording weights. Float64 covariance eigendecomposition provides an analytic fit;
there are no gradient updates, hidden schedule extensions or selectable training
seeds. Encode/decode runs in FP32. Random orthogonal projections are retained as a
descriptive initialization control. The original nonlinear Temporal128 is a frozen
historical reference, not a comparison matched for optimization cost or exposure.

The primary hypothesis is fixed to Phase128 versus the original Temporal128 on full
VAL velocity NRMSE. All four revision arms are evaluated before selecting the lowest
total bitrate that passes every gate; another rank cannot replace the primary arm
after outcomes are seen.

Evaluation retains all 16-second windows, all complete VAL recordings and the twelve
fixed visual clips. Direct versus chunked reconstruction must agree within 5e-5.
Two-second quiet and stationary windows are now explicitly included, each requiring
at least one eligible window. Maximum excess quiet speed must be <=.02 m/s and
stationary travel error <=.034 m. Physical foot sliding/penetration, source quality
and external visual review remain separate requirements. Missing visual reviews
prevent selection.

Outputs are isolated under
`/mnt/lightsem-data/experiments/lightsem/motion_representation_phasecodec_20260910`.
Scientific execution uses `run_phase_motion_study.py`, which imports an immutable
source snapshot and verifies frozen input hashes. Hypothesis:
`h-motion-native-phase-codec-20260910` in `/root/lightsem-vault`.
