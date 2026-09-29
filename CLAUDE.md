# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Current authorization (September 12, 2026): the product is LIGHT. Dance is a
visual diagnostic of the intermediate motion representation, not a separate
product or an optimization target. Implement direct audio rhythm controls for
lighting timing, using internal motion for spatial style rather than literal
limb tracking. Do not retime or retrain the dancer to satisfy a lighting request.
See `docs/research/2026-09-10-motion/RHYTHM_COMPANION_PLAN.md` and
`RHYTHM_COMPANION.md`. The new offline entry point is `run_rhythm_companion.py`;
existing mapper defaults remain unchanged. Deliver native 1080p AV1 with audio.

Earlier authorization (September 10, 2026): follow
`docs/research/2026-09-10-motion/PLAN.md`. Music-only playback may generate an internal
motion sequence, and direct audio dynamics may control brightness/activity. This
explicit user update supersedes the older inference restriction below.

Historical architecture: LightSem turns sound into stage lighting for concerts and festivals. Dance datasets are the
**training signal only**: paired (music, motion) teaches what light should do, and at inference
the system takes audio and emits DMX. There is no dancer, pose or dance generator in the
inference path, and proposing one is a repeat of a mistake already corrected once.

`PLAN.md` is the current plan (rev. 5) and section 1.2/1.3 hold the measured results that
constrain it. The **system of record is the vibescience vault**, not this repo.

## Running things

Two interpreters, and which one you need depends on torch:

```bash
bash run.sh   <script.py> [args]   # py3.12: librosa, moderngl/EGL, sklearn. No torch.
bash run_torch.sh <script.py>      # /mnt/lightsem-data/venv-torch: torch 2.14+cu130, ultralytics
```

Never invoke a venv python binary directly from a tool shell; it fails with "embedded null byte".
Use `uv` for installs, never system pip. `run_gpu.sh` is retired (it borrowed an unrelated
project's venv).

Rebuild the caches in this order; each is fast and everything downstream reads them:

```bash
bash run.sh build_caches.py audio       # 60 AIST tracks -> causal feature bus + beat grid
bash run.sh build_caches.py motion      # 1408 clips -> 30 descriptor channels + events
bash run.sh build_targets.py scalers    # corpus median/IQR the conductor calibrates against
bash run.sh -m lightsem.model.light_data  # the training tensor
```

Verification suite (14 checks, each aimed at a specific silent-failure mode):

```bash
bash run_torch.sh verify_all.py            # all
bash run_torch.sh verify_all.py beat grid  # only checks whose name matches a filter word
```

Stage drivers, each writing JSON under `outputs/`: `run_stage0a.py {metrics,render}`,
`run_stage0b.py {build,fit,shared}`, `run_burger_replication.py`, `run_oslo_audit.py`,
`run_stage0c.py {category,late}`, `train_light.py`, `run_stage1_eval.py`, `run_stage2.py`,
`train_operator_head.py`, `run_stage4.py`, `run_pose_validation.py`.

Vault writes go through `log_to_vault.Client` (stdio MCP with the vault path pinned); see the
`log_*_to_vault.py` scripts for the call sequence and `docs/VIBESCIENCE_API_NOTES.md` for the
argument names the server validates hard.

## Architecture

One direction of flow, and each arrow is a separate module with its own test:

```
audio ──► lightsem/audio/{online,beatgrid}      causal feature bus + constant-tempo beat grid
      ──► lightsem/model/{light_trunk,infer}    causal Conformer, three heads   [inference only]
      ──► motion descriptors (30 channels)      same object a real dancer produces
      ──► lightsem/conductor/*                  descriptors -> fixture parameters
      ──► lightsem/dmx/{encode,decode}          -> 512-channel universe at 40 fps
      ──► lightsem/sim/renderer                 -> MP4 (headless EGL)
```

The descriptor bundle is the seam. `lightsem/motion/` produces it from mocap
(`skeleton.py` + `descriptors.py`) or from video (`pose2d.py`), and `lightsem/model/infer.py`
produces the same object from audio alone. Anything downstream cannot tell which.

**Three target blocks, three objectives** (`lightsem/model/light_data.py`): A regressed
deterministically, B an event hazard, C sampled by a conditional rectified flow. The membership
of A and C is set by measurement, not by the literature: `BLOCK_A_V2` is the six channels Stage 1
found predictable. Regressing an unpredictable channel emits the corpus median, which makes light
that flashes uniformly and scores at chance; sample it instead.

**Three conductors**, and the distinction matters for every comparison:
`conductor/v0.py` is motion-only and must stay that way (it is the strict oracle for the category
test); `conductor/hybrid.py` takes envelopes from the dancer and transient phase from the audio,
and is what ships; `conductor/beat_trigger.py` is the 30-line baseline every claim is priced
against.

**Every metric carries its own null** (`lightsem/eval/`). Light-BAS is reported against a
metronome ceiling and a circular-shift floor, FID_light is computed on delta features against a
split-half floor and a block-shuffle null, and A/B pairs are brightness-matched with an
over-bright foil. A bare number here will be challenged.

## Things that have already gone wrong

- **Memory authorization (September 29, 2026).** The user explicitly states that
  memory changes do not require separate permission. Proceed within the task's
  scope and verify preservation. This supersedes earlier handoff approval blocks.
  The installed experiment writer can drop custom `##` sections in Notes. The
  physical-objective and optics-full records were restored using the isolated
  recovery MCP; preserve their full note text on any subsequent write. See
  `docs/research/2026-09-28-end-to-end/interpretation_debt_20260928_v1/repair_v1/REPAIR_AUDIT_V1.json`.
- **Vault isolation.** Write only to `/root/lightsem-vault`. `/root/vibescience-vault` belongs to
  an unrelated project and must remain unchanged. Capture its file hashes before
  and after writes; historical count assertions may be stale (695 at this study).
- **Disk.** Root is ~80% full. All datasets, caches and checkpoints live under
  `/mnt/lightsem-data/`.
- **Beat grid.** Use `audio/beatgrid.py`, not `librosa.beat.beat_track`: the latter returned
  octave-wrong tempi on this corpus and locked phase to the off-beat. Period comes from broadband
  onset, phase from 40-120 Hz kick flux at a 5.8 ms hop.
- **Never regress absolute light levels or raw kinetic energy.** Both are recorded rule-outs.
  Targets are gain-normalised envelopes; on absolute levels the Oslo data goes from R2 +0.30
  to -0.72.
- **Streaming context is 26 s, not 4 s.** Attention windows compose across layers, so the trunk's
  receptive field is `n_blocks * (attn_window - 1 + conv_kernel - 1) + 1`. Priming a chunk with
  one window makes the same audio produce different light depending on how it was sliced.
- **Corpus validity is not interchangeable.** AIST++ movement is explained by genre, not by
  audio, so it cannot train block A; free dancing to EDM (`data/oslo.py`) can; the RPMC-L2
  operator data carries the strongest audio-to-light coupling in the project. Check
  `PLAN.md` section 1.2 before assuming a corpus supports a claim.
