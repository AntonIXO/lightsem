# StereoAudioV1 pilot

This directory documents the versioned stereo/audio-conditioning branch.  It
is diagnostic infrastructure; it does not replace the selected native C or P2
baseline.

## Contract

`lightsem.audio.stereo_v1` retains the original L/R waveform and exports:

- 40 Hz L/R, mid/side, ILD, width, balance, coherence, phase, and spectral
  flux descriptors;
- 40 Hz event descriptors;
  - 10 Hz stereo-image descriptors;
  - 2 Hz relation descriptors.

Use `resample_stream` with explicit timestamps when moving the 40 Hz event
stream to the native 30 Hz body clock or the 2 Hz planner clock; do not
interpolate native velocity/rotation channels.

Stereo descriptors describe a mastered stereo image.  They are not physical
3D location labels.  MuQ/Base92 remain the existing pinned mono/mid input.

## Local smoke

```bash
bash run.sh scripts/motion/prepare_stereo_features.py \
  --manifest /path/to/train_dev_manifest.json \
  --out /mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1

TORCH_ENV=/mnt/lightsem-data/venv-torch-fixed bash run_torch.sh \
  scripts/motion/train_stereo_conditioner.py \
  --manifest /mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1/STEREO_FEATURE_MANIFEST.json \
  --out /mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1/learning/R2 \
  --arm factorized --steps 3000 --batch-size 16 --split train --device cuda

TORCH_ENV=/mnt/lightsem-data/venv-torch-fixed bash run_torch.sh \
  scripts/motion/audit_stereo_dance.py \
  --manifest /mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1/STEREO_FEATURE_MANIFEST.json \
  --checkpoint /mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1/learning/R2/final.pt \
  --split dev \
  --out /mnt/lightsem-data/experiments/lightsem/audio_stereo_spatial_20261007_pilot_v1/audit/R2.json
```

The audit intentionally leaves the scientific gate pending until paired
timing, spatial, relation, and silent-source comparison metrics are supplied.
One-seed pilot outputs cannot promote a baseline.

For a sequential guarded run, prepare the manifest first and then use
`scripts/motion/run_stereo_pilot.py --stage preflight` followed by
`--stage train`.  The launcher enforces the 20 GiB free-space floor, refuses
TEST rows, caps each representation arm at 3,000 updates, and takes the
existing GPU curator lock when `--device cuda` is selected. Add
`--wait-for-gpu --max-wait-hours 24` to queue behind an active audit instead of
failing immediately.

For a persistent launch whose completed state remains inspectable, use:

```bash
bash scripts/motion/launch_stereo_pilot_systemd.sh
systemctl --user status lightsem-stereo-pilot.service
```

The launcher uses `--remain-after-exit`; a curator can therefore distinguish a
completed run from a crashed or externally stopped run.
