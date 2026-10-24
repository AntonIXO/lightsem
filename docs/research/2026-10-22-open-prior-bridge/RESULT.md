# Open appearance prior bridge — 2026-10-22

## Scope

This arm connects the frozen SpeedNet prior trained on **RPMC-L2 + Roskilde** (699 + 224 = 923 samples, common 36-D MIR input) to the legacy MuQ+Base92 audio-to-light foundation. The prior emits only two appearance-derived controls; it is not pan/tilt/DMX supervision.

The legacy `foundation_v3_jerk_safe` checkpoint is frozen. The treatment trains only a zeroable 2-D prior-control → hidden residual adapter with the existing canonical RGB/brightness/event loss. Seed `0`; canonical 80/20 song-group split; no sweep.

## Result

Vibescience experiment: `exp-open-appearance-prior-bridge-20261022`

Verdict: **null / powered sub-threshold result** for the preregistered appearance-transfer claim (`h-open-appearance-prior-bridge-20261022`).

- Frozen control held-out composite appearance error: `0.4529533607`
- Treatment held-out composite appearance error: `0.4497962783`
- Absolute gain: `0.0031570824`
- Preregistered MDE: `0.005`
- Relative improvement: `0.697%`
- The gain is below the committed bar; it is not a confirmation.

Component movement:

- Brightness L1: `0.4049016132 -> 0.3950860014`
- RGB L1: `0.3351288613 -> 0.3364621041` (slightly worse)
- Event BCE: `0.6188296076 -> 0.6178407296`

## Integration and safety

The adapter is live, not a silent replica:

- Zero-prior ablation max absolute difference: `1.1920929e-07`
- Live held-out RGB difference: `0.0053167343`
- Own-song RGB swap: `0.0375576653 -> 0.0387706421`
- Fresh finite-difference raw audit passed on Shpongle and Cure For Me:
  - maximum treatment speed p99: `0.0109336023`
  - maximum treatment acceleration p99: `0.0167890514`
  - maximum treatment jerk p99: `0.1716723616`
  - limits: `0.18 / 0.50 / 1.50`

The noncollapse gate was not passed because Shpongle latent effective rank stayed below the preregistered `2.0` bar (`1.6054` baseline, `1.6178` treatment). This is an incumbent/harness limitation, not evidence that the bridge caused collapse; the baseline already fails that gate.

No renderer call was made: the blocking appearance gate failed, and the open sources still contain no authored physical-light targets. No physical-motion or authored-choreography claim follows.

## Artifacts

- Training receipt: `/mnt/lightsem-data/experiments/lightsem/foundation_open_prior_bridge_20261022_v1/TRAINING.json`
- Adapter checkpoint: `/mnt/lightsem-data/experiments/lightsem/foundation_open_prior_bridge_20261022_v1/prior_bridge_adapter.pt`
- Fresh treatment raw audit: `/mnt/lightsem-data/experiments/lightsem/foundation_open_prior_bridge_20261022_v1/raw_audit/RAW_AUDIT.json`
- Fresh baseline raw audit: `/mnt/lightsem-data/experiments/lightsem/foundation_open_prior_bridge_20261022_v1/raw_audit_baseline/RAW_AUDIT.json`
- Bridge runner: `/root/lightsem/scripts/experiments/run_open_prior_bridge.py`
- Runner SHA256: `9841da6a3052c37295a48a18e58021ee1722e7fb5203630a29ff0127d5fb4db7`
- W&B analysis run: https://wandb.ai/antonix-itmo/lightsem/runs/ihvftunh

The exact seed-0 recipe must not be retried unchanged. A future arm needs a changed learning interface or target weighting, and must rebase the noncollapse gate against a control that passes it.
