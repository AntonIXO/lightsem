# LightSem — prototype report

Status as of 2026-08-07. Every number below came from a real run on this box;
nothing here is estimated or projected. Where something is weak or stubbed it
says so.

## What exists and runs

One command, audio in → rendered MP4 out:

```bash
cd /root/lightsem
bash run.sh -m lightsem.cli render --wav data/cache/synth_edm.wav \
    --out outputs/lightsem_demo.mp4
bash run.sh -m lightsem.cli render --aist mBR0 --duration 25
bash run.sh -m lightsem.cli analyse --aist mBR0
bash run.sh -m lightsem.cli manifest
```

| Stage | Module | Smoke test | Status |
|---|---|---|---|
| A data | `lightsem/data/` | `smoke_stage_a.py` | PASS |
| B audio | `lightsem/audio/` | `smoke_stage_b.py` | PASS |
| C DMX | `lightsem/dmx/` | `smoke_stage_c.py` | PASS |
| D sim | `lightsem/sim/` | `smoke_stage_d.py` | PASS |
| E e2e | `lightsem/cli.py` | `verify_output.py` | PASS |

## The delivered artifact

`/root/lightsem/outputs/lightsem_demo.mp4` — **verified by decoding it**, not
by trusting the writer:

- 4.44 MB (4,436,345 bytes) · h264 **960×540 @ 30 fps** · **1440 frames**
- **48.00 s**, aac 22050 Hz mono, peak **−0.9 dB** (non-silent)
- Luminance mean 26.60, std 14.04, per-frame |Δ| mean 2.679 → genuinely animated
- **Intro 10.17 vs drop 38.31 luminance** → the light tracks the structure

Second artifact on real AIST audio: `outputs/aist_mBR0_show.mp4`, 2.33 MB,
25.00 s, 750 frames, drop 23.15 vs intro 12.77.

Render cost: **6.8 s wall for 48 s of video** (~7× faster than realtime) on the
RTX 5070 via standalone EGL — no X/Wayland needed.

## What genuinely works

**The anticipation layer** — this is the novel part and it survives honest
scoring. On synthetic EDM with drops at known times:

| | precision | nearest-cue lead |
|---|---|---|
| **LightSem (causal)** | **1.000** | **2.78 s ± 0.08 s** |
| Rate-matched random | 0.495 ± 0.208 | 4.36 s ± 2.48 s |
| Metronome, same rate | 0.500 ± 0.106 | 3.37 s ± 1.66 s |

Precision **z = +2.42** vs random. The lead-time *std* is the sharper result:
0.08 s vs 2.48 s — the cue lands at a consistent phase of the build, not
merely inside a window. Null-track control: on drop-free audio the cue rate
collapses **0.146 → 0.021 Hz**.

**Causal vs oracle is enforced, not asserted.** `anticipation.py` has two
modes: CAUSAL (one-sided EMA + backward-difference slope, reads only t′ ≤ t;
time-to-drop is extrapolated from the tension slope) and ORACLE (full-track
lookahead, exact by construction — the ceiling, not a prediction). The smoke
test asserts `max|causal − oracle| > 1e-6` so a refactor cannot silently leak
the future.

**Structure → scene, not waveform → DMX.** Honours the recorded rule-out.
Measured dimmer levels: intro 64.7 < breakdown 98.8 < build-up 137.3 < drop
153.0. Micro layer is beat-locked (dimmer peak in phase-bin 0 of 8), and
intensity lifts **1.91×** in the 3 s before a drop.

**The renderer.** Raymarched volumetric cones in haze, 6 moving heads that
actually articulate (pan spread 35.79 across heads during a fan), washes,
blinders, strobe, and a gobo-projecting projector. The renderer consumes the
**DMX universe**, decoding fixture state back out of the 512 channels — so a
wrong encoding shows up in the picture.

## Bugs found by controls (all fixed)

Six real defects, each caught by a control rather than by reading code:

1. **`librosa.util.sync` off-by-one** → `IndexError` crash. It emits one more
   column than beat frames.
2. **Quantile labelling** called a 0.613-energy segment "breakdown" when track
   max was 0.620. Drop localisation error 3.23 s/6.66 s → **0.44 s/0.57 s**
   after switching to track-range-relative thresholds.
3. **The metric punished better anticipation.** Credit window was
   `lookahead_s + tolerance_s` = 3.0 s, so a cue **6.29 s** before a drop
   scored a MISS while 2.86 s counted. Recall read 0.000 for a working
   system. Fixed by separating `max_lead_s` from `lookahead_s`.
4. **moderngl uniform arrays** are exposed as `u_lpos`, *not* `u_lpos[0]`. My
   `if name in prog` guard silently skipped every per-light write — all frames
   rendered identically. Now writes whole arrays with `.write()`.
5. **Zoom 0.0 pinned beams to 6°** — invisible pencils. Clamped to ≥ 0.18.
6. **Blown-out white render.** Scatter scales with lit-fixture count, so a
   drop pushed ~10× the intro's energy through the tonemap. Fixed with a
   lit-count normaliser + ACES tonemap (a plain 1/2.2 gamma lifted blacks to
   grey fog).

**Lesson worth keeping:** #3 and #4 both passed their numeric assertions while
being completely wrong. Only a per-cue *trace* and a *visual* inspection
caught them. A metric that disagrees with a visibly working system is a
suspect metric.

## What is baseline-quality

- Beat/tempo tracking, boundary detection — standard librosa. Works.
- **Functional labelling (intro/build-up/drop/breakdown/outro) — heuristic**,
  validated only on synthetic audio. Weakest link: it still tags **3 spurious
  "drops"** on drop-free steady-state audio. Upgrade path: All-In-One,
  EDMFormer, Beat This!.
- Downbeat inference is a bar-phase energy heuristic, not a trained tracker.

## What is stubbed / unverified

- **Art-Net output** (`ArtNetSender`) is wire-format correct but **never
  tested against a real node or BlenderDMX**. Reported as untested.
- **Live desktop mode** not implemented; headless MP4 only (as specified).
- **No learned model.** The macro layer is a rule table. The architecture
  reserves the slot for one — that is the point of the macro/micro split — but
  nothing is trained.
- The fixture personalities are plausible, not GDTF-accurate.

## The corpus finding (recorded as a formal rule-out)

**AIST audio cannot validate this system's core claim.** Its energy dynamic
range is ~5× too small:

| | RMS p90/p10 |
|---|---|
| Synthetic EDM control | **18.27** |
| AIST, median of 10 | **3.76** |

Anticipation recall on AIST was 0.000 on 3/5 tracks; on the one track scoring
0.800 the random baseline was 0.625. Segment energies live in a 0.475–0.518
band. These are loudness-normalised dance loops with no build-up/drop contour.

Both hypotheses were tested and **both were partly true**: the corpus lacks
structure *and* the labeller had a real defect. The synthetic control is what
separated them.

**Consequence:** AIST stays right for pose/choreography work and is fine as a
rendering input. Structure and anticipation claims need **EDM-98 / EDM-CUE**
(arXiv 2603.08759, HF `disco-eth/edm-cue`) — *not yet downloaded*. This is the
single highest-value next step.

## Honest limits on the anticipation claim

- Validated on **synthetic audio I generated**. The synthesiser and detector
  share my assumptions about what a build-up looks like — that is a real
  circularity, and it is why the EDM-98 validation matters.
- The `n=2 drops` per synthetic track is tiny. `z = +2.42` is encouraging, not
  established.
- The causal forecast extrapolates a tension slope to a threshold. It is
  genuinely causal, but it is linear extrapolation, not a learned predictor.

## Corrections to the handoff brief

- **"2817 keypoint files"** → the real count is **1408**. The zip lists 2816
  `.pkl` entries = 1408 real + 1408 macOS `__MACOSX/._` resource forks.
- **"Keypoints contain NaN frames"** → only in the `keypoints3d` array (19.3%
  of sampled files). `keypoints3d_optim`, which the loader defaults to, is
  **NaN-free across the whole sample**. The interpolation policy is a guard,
  not an active transform.
- Median pose clip duration measures **9.60 s** here, not 8.7 s.

Details in `docs/DATA_NOTES.md` and `docs/STAGE_B_FINDINGS.md`.

## Infrastructure fix (outside the repo)

`vibescience-mcp` hardcoded `FastMCP(name="vibescience-mcp")`, so running two
instances against different vaults collapsed them into one tool namespace and
**one vault silently shadowed the other** — LightSem records would have been
written into the optiHealth vault. Patched `server.py` to honour
`VIBESCIENCE_SERVER_NAME`; config now maps
`vibescience-lightsem → /root/lightsem-vault`. Verified both instances
announce distinct identities and resolve distinct vaults.

Two rule-outs written to `/root/lightsem-vault` (corpus, metric) via
`log_to_vault.py`.
