# Stage F — training the macro layer on RPMC-L2: NEGATIVE RESULT

All numbers below are real execution output. Artifacts:
`data/cache/{baselines,ceiling_probe,macro_results}.json`, `logs_train_macro.txt`.

## Claim under test

A causal temporal encoder over openl3 audio embeddings can predict the stage
lighting scene vector of real lighting-engineer demonstrations (RPMC-L2), i.e.
the trained macro layer replaces the hand-written rule table.

**Result: refuted on this target formulation.** Audio contributes nothing
measurable once the lighting's own persistence is accounted for.

## Setup

- 140 songs, 379,125 light-rate frames -> 126,332 frames after decimating to
  the 10 Hz openl3 grid (30 fps light / 10 Hz audio = 3).
- Split BY SONG (hash): 99 train / 23 val / 18 test. No song straddles splits.
- Target: 6-dim scene vector, **per-song centred**. Centring is essential —
  35–65% of raw target variance is *between* songs (venue palette identity),
  which is given by the show, not predicted from audio.
- Metric: R2_within on held-out songs.

## Baselines (computed before training, `baselines_macro.py`)

| baseline | mean R2_within |
|---|---|
| global mean | -0.0000 |
| **per-song mean (oracle)** | **0.0000** ← zero point |
| ridge, 1 frame | -0.4888 |
| ridge, pooled 6.4 s | -1.2456 |
| ridge, centred, 1 frame | -0.1091 |
| ridge, centred, pooled 6.4 s | -0.5592 |

No linear model beats holding the song's mean colour. The bar is R2_within > 0.

## Trained model (`train_macro.py`, 40 epochs, 12k steps, 76 s on RTX 5070)

2,359,769 params. Causality verified by perturbation test, not by inspection
(`smoke_model.py`: past changes output, future provably cannot leak).

```
ep  0  loss 0.1554  val_R2_now -0.0044   <- best
ep 15  loss 0.0531  val_R2_now -0.3761
ep 39  loss 0.0418  val_R2_now -0.4450
```

Train loss fell 3.7x while validation R2 collapsed. Classic memorisation of 99
songs. Best checkpoint was **epoch 0** — before learning anything.

Final test R2_within: **-0.0071** (now), -0.0395 at +4 s.
Worse at longer horizons, i.e. no anticipation either.

## Why: the ceiling probe (`probe_signal_ceiling.py`)

The decisive measurement. Ridge on per-song-centred data, held-out songs:

| predictor | mean R2_within |
|---|---|
| audio only | -0.1091 |
| **persistence only (y's own past, lags 0.1–4.0 s)** | **+0.7083** |
| AR + audio | +0.7035 |

**Audio's marginal contribution over persistence: -0.0049.** Negative — adding
512 audio dimensions on top of the autoregressive term does not help at all;
it costs a little through variance.

Lighting is ~71% predictable from its own recent past. That is the number any
future model must beat, and it requires no audio whatsoever.

### Time-shift control (rules out the earlier correlation as evidence)

Correlation of audio with light, y shifted within the same song:

```
lag -30 s  0.27325   <- HIGHEST
lag -10 s  0.17049
lag   0 s  0.14104   <- true alignment
lag +10 s  0.13384
lag +30 s  0.17044
```

Real moment-to-moment coupling must peak at lag 0. It **minimises** near zero
and rises at large |lag|. So the 20–36x "audio↔light correlation vs shuffled
control" reported during the data audit was **slow co-drift** — both signals
wander over minutes — not the frame-level coupling that would let a model
predict lighting from music. That audit conclusion is hereby retracted; the
shuffled control was too weak because it destroyed within-song structure too.

## What this rules out

1. **Direct regression of audio -> absolute scene vector is not learnable here.**
   Not a capacity problem: the linear model fails the same way, and the AR
   result shows the target is highly predictable by other means.
2. **The 6-dim reduction may itself be the problem.** Frame-level HSV
   histogram summaries of stage *video* include camera cuts, crowd, haze, and
   the engineer's arbitrary palette. That is not the same object as a DMX cue.
3. **The apparent correlation from the data audit is not evidence.** Any future
   claim of audio->light signal must clear a lag-shift control, not a shuffle.

## What is still open (not tested, do not assume refuted)

- Predicting **changes/events** (scene transitions, hits) rather than absolute
  levels. Persistence dominates levels by construction; a change-point target
  removes that advantage and is what a lighting desk actually fires.
- Predicting the **residual** after the AR term — explicitly asking audio to
  explain only what persistence cannot.
- 140/699 songs used. More data would not fix a lag-structure problem, but the
  event formulation has not been tried at any scale.
