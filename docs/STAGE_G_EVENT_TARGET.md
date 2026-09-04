# Stage G — the event reformulation: a real signal, and what it isn't

All numbers from real execution. Artifacts: `data/cache/{event_probe_cv,
cut_vs_light,family_attribution,horizon_sweep}.json`, logs `logs_event_cv.txt`,
`logs_families140.txt`, `logs_horizon.txt`.

Read `STAGE_F_MACRO_TRAINING.md` first — this stage only makes sense as the
answer to that one's failure.

## The move

Stage F refuted audio → lighting **levels**: persistence explained R²_within
+0.708 with no audio at all, and audio's marginal contribution was −0.005.

That comparison was rigged by construction. A slowly-varying target is
trivially predicted by its own past, and persistence is useless at exactly the
moments an operator acts. So the level result says less than it appears to.

Reformulate the target as the operator's **action**: when does the scene
change? Two consequences — persistence loses its structural edge, and the task
becomes the one a lighting desk actually performs (fire a cue, or don't).

`multimodal-ai` lecture 06 names the original error directly: *"Applying
discrete-alignment methods to continuous data. Sensor streams have no natural
elements. Pretending fixed windows are semantic units is a silent correctness
bug."* Segment first, then align.

## Method

- **Events**: per-song z-scored 6-dim light vector; novelty = ‖future 1.0 s mean
  − past 1.0 s mean‖; peaks above the 90th percentile, 2.0 s minimum separation.
  140 songs → **1376 events**, median 6.5/min, median inter-event 4.8 s.
- **Competitor is NOT persistence** (it cannot predict change) but the **rhythm
  of past changes**: time since last event, 30 s event rate, phase against the
  song's own median inter-event interval. A desk firing every N seconds is a
  real deployable baseline that needs no audio.
- **Metric**: per-song average precision / base rate, 5-fold CV by song so every
  song is held out once.
- **Null**: the event train **circularly shifted** within each song. Preserves
  event count and rhythm exactly; destroys only alignment to audio. AP's
  small-sample upward bias appears identically in the null and cancels.

That last point matters: a first pass on a single 18-song split gave a random
floor of 1.141x, not 1.0x. Raw lift is not calibrated and cannot be read as
skill. Everything below is z against a matched permutation null.

## Result 1 — the coupling is real

```
rhythm-only baseline (no audio)   1.3241x
audio only                        1.2974x
rhythm + audio                    1.4151x
audio's MARGINAL lift            +0.0909x

null (40 reps)  -0.1568 ± 0.0876  ->  z = +2.83, p = 0.000
```

The null mean is **negative**: under shift, adding audio features *hurts*
through added variance. So the positive marginal is not an artifact of extra
parameters — parameters cost you here, and audio still wins.

The same audio features that were worthless on levels are significant on
events. **The failure in Stage F was the target formulation, not the features
and not the model.**

## Result 2 — it is not camera editing

The target comes from stage *video* histograms, so a camera cut looks like a
lighting change, and concert video is edited on the music. If cuts dominated,
this would be audio→editing.

Measured at native 30 fps, 36 songs, 968 events (`probe_cut_vs_light.py`):

| signature | value | reading |
|---|---|---|
| median rise time | 18 frames = 0.60 s | a cut is ~1 frame |
| rise in ≤2 frames | 13.5% | minority is cut-like |
| revert to pre-state within 4.5 s | 8.4% | cuts often cut back |
| move to a new state and stay | 57.6% | lighting-like |

Gradual and persistent. This does not *prove* the absence of edit
contamination — that needs shot-boundary labels the dataset doesn't ship — but
it bounds it as a minority.

## Result 3 — it is NOT loudness (the important one)

Every family gets identical treatment and its **own** null, because
dimensionality changes overfitting bias. 140 songs, horizon 1.0 s.

| family | dim | lift | marginal | z | p |
|---|---|---|---|---|---|
| energy (loudness + flux) | 2 | 1.4704 | +0.1463 | **+0.66** | 0.240 |
| harmonic (chroma) | 2 | 1.3832 | +0.0590 | −1.41 | 0.960 |
| openl3 (learned) | 512 | 1.4151 | +0.0909 | +2.52 | 0.000 |
| spectral (centroid/bw/rolloff/contrast) | 10 | 1.7060 | +0.3818 | **+4.21** | 0.000 |
| timbre (MFCC) | 128 | 1.6203 | +0.2962 | **+4.98** | 0.000 |

Three findings:

1. **Energy is the only family that fails its null.** Loudness does not explain
   when an operator changes the scene. This contradicts the assumption behind
   every FFT-threshold rig. The industry's L2 approach isn't at the ceiling
   here — it is keying on the wrong feature.

2. **Timbre and spectral character carry the signal.** Operators respond to the
   sound's *character* changing — distortion entering, texture shifting — not
   to it getting louder. On Rock/Punk/Metal livehouse material that is musically
   sensible: these genres are loud throughout, so loudness is uninformative.
   This is the defensible claim that an L3 model has headroom, and it *names*
   the features it must encode.

3. **openl3 (512-d, learned) is beaten by 10 hand-picked spectral descriptors.**
   The general-purpose embedding is not the right representation for this task.
   A bespoke encoder over spectral/timbral descriptors is better motivated —
   and far cheaper — than fine-tuning openl3.

## Result 4 — it is short-range, NOT anticipatory

| horizon | base | marginal lift | z | p |
|---|---|---|---|---|
| 0.2 s | 0.023 | **+0.4824** | +3.73 | 0.000 |
| 0.5 s | 0.056 | +0.2085 | +3.41 | 0.000 |
| 1.0 s | 0.110 | +0.0909 | +3.15 | 0.000 |
| 2.0 s | 0.217 | +0.0561 | +3.37 | 0.000 |
| 4.0 s | 0.380 | **+0.0031** | +3.27 | 0.000 |
| 8.0 s | 0.565 | **−0.0056** | +2.20 | 0.000 |

`probe_horizon_sweep.py` printed "ANTICIPATORY" because z > 2 at 4 s and 8 s.
**That automated verdict is wrong and is rejected.** Read the effect size: the
marginal lift decays ~99% by 4 s and goes negative by 8 s. z stays high only
because the null is itself negative and its spread tightens as the base rate
grows, so "less bad than the null" registers as significant while the absolute
lift is zero.

**High z with a null effect size is not a usable anticipation window.** The
z-based rule in that script is a bug — it is recorded here rather than silently
patched, because it produced a claim that had to be overturned by hand.

Peak lift is at the **shortest** horizon tested. That is the signature of
detection, not prediction.

## What this means for the product

**Supported:** better *detection* — reacting to timbral change rather than
loudness, at ~1 s latency. A real improvement over threshold triggers, which
key on the feature this data says does not work.

**Not supported:** the original L3 pitch of light that *leads* a build-up by
4–8 s. This corpus does not carry that signal.

The 2.78 s lead the DSP baseline showed on synthetic EDM does not transfer.
RPMC-L2 is Rock/Punk/Metal livehouse material, which lacks the long telegraphed
build-ups that make EDM anticipation possible. **The lead-time claim may be
genre-specific and is untested on EDM with real lighting ground truth.** That is
the single highest-value next experiment, and it needs a different corpus.

## Process note

Two experiments closed with an auto-computed verdict of `supports` while their
stated hypotheses were *refuted*. `close_experiment` computes the verdict from
predicted-effect direction versus measurement, so a number that moved up reads
as support even when the claim failed. The mechanism that exists for this is
**gates** — a failing blocking gate downgrades `supports` → `directional_only` —
and no gate was preregistered on those two. Corrected in the problem record.
Preregister a gate for every claim, not just a diagnostic direction.
