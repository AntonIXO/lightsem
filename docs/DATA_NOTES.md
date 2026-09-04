# LightSem — data notes (measured, not assumed)

Everything here was verified against the on-disk corpus by
`audit_data_claims.py` and `smoke_stage_a.py`. Two claims inherited from the
handoff brief were **wrong or imprecise** and are corrected below.

## Corpus inventory (verified 2026-08-07)

| Item | Value |
|---|---|
| Keypoint files | **1408** `.pkl` in `data/kp3d/keypoints3d/` |
| Audio files | **60** `.wav` in `data/wav/wav/` |
| Distinct music ids | 60 |
| Distinct dancers | 30 |
| Distinct choreographies | 21 |
| Usable after policy | **1408 / 1408** (0 rejected) |

### CORRECTION 1 — the "2817 files" figure was macOS junk

The handoff recorded *"2817 files, 884 MB"*. The real count is **1408**.

`keypoints3d.zip` lists 2818 entries, of which 2816 match `*.pkl`:
**1408 genuine files + 1408 `__MACOSX/._<name>.pkl` AppleDouble resource
forks.** The extracted directory contains only the 1408 real files
(`extensions: {'.pkl': 1408}`, no other entries). Nothing is missing — the
original count double-counted archive metadata.

### CORRECTION 2 — NaN depends on *which key you read*

The handoff said *"Keypoints contain NaN frames"* without qualification. Each
pickle is a dict with **two** arrays, and they differ:

| Key | Files with NaN (of 202 sampled) | Mean NaN frame fraction |
|---|---|---|
| `keypoints3d` | **39** (19.3 %) | 0.0071 |
| `keypoints3d_optim` | **0** | 0.000000 |

`AistDataset` defaults to **`keypoints3d_optim`**, which is NaN-free across the
whole sample — so in the default path the interpolation policy never fires.
It is retained because it *is* required for `keypoints3d`, and because the
`>10 % bad frames → reject` gate is cheap insurance.

**Practical consequence:** do not cite "NaN handling" as if it were doing work
in the default pipeline. It is a guard, not an active transform.

## Shapes and constants

- Keypoints: `(T, 17, 3)` float64, `KP_FPS = 60.0`
- Audio: `SR = 22050`, `HOP = 512`
- Filename regex must accept `cAll` as well as `c01`:
  `r"(g\w+?)_(s\w+?)_(c[\w\d]+?)_(d\d+)_(m\w+?)_(ch\d+)"` — 1408/1408 parse.

## Duration mismatch — the silent data-loss trap

Pose clips are **short**; audio tracks are long.

| | min | median | max |
|---|---|---|---|
| Pose sequence duration (s) | 7.10 | **9.60** | 47.97 |

Audio tracks run 30–51 s. **Any audio analysis must be clipped to the
sequence's own `duration_s`.** Ignoring this once produced `n = 0` usable
pairs after a "≥20 beats" filter — a total, silent data loss with no error
raised. `SequenceMeta.duration_s` is the authority.

(Measured median here is 9.60 s; the handoff said 8.7 s. Same order, minor
drift — the 9.60 figure is what the current 1408-file corpus reports.)

## Artifacts

- `data/cache/manifest.parquet` — per-sequence identity + quality record
- `data/cache/manifest_summary.json` — aggregate counts and reject reasons
