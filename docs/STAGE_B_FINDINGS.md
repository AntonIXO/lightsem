# Stage B — audio structure & anticipation: measured findings

Every number here came from a real run (`probe_control_synthetic.py`,
`probe_anticipation_baselines.py`, `smoke_stage_b.py`). Where a result is
weak, it is reported as weak.

## Headline

The anticipation layer works, **on audio that has structure to anticipate**.
It fires cues with **precision 1.000** and a **2.78 s ± 0.08 s** lead before
synthetic drops, against a rate-matched random baseline of 0.495 ± 0.208
(**z = +2.42**) and a metronome baseline of 0.500.

## Three real bugs found and fixed by controls

These were found *because* a synthetic track with known drop times was run
through the identical pipeline. None was visible from the AIST results alone.

### BUG 1 — `librosa.util.sync` off-by-one → IndexError
`sync` emits one MORE column than there are beat frames (it closes the final
open segment), so a cluster-change index can equal `len(beat_times)`.
Crashed on the second track tested. Fixed by mapping an out-of-range index to
the clip end.

### BUG 2 — quantile labelling mislabels loud segments
The labeller used quantiles over the segment list, so on a track where most
segments are loud a **0.613-energy segment was labelled "breakdown" while the
track maximum was 0.620**. Drop localisation error was **3.23 s / 6.66 s**.

Fixed by labelling **relative to the track's own energy range** (`rel = (e -
emin)/(emax - emin)`) plus an explicit *drop-follows-build-up* rule.
Localisation error → **0.44 s / 0.57 s**.

### BUG 3 — the metric punished better anticipation
The scoring window was `lookahead_s + tolerance_s` = 3.0 s. A cue arriving
**6.29 s** before a drop was scored a MISS while one at 2.86 s counted — the
metric rewarded *later* warning, which is backwards for a system whose whole
purpose is leading the build-up. Recall read 0.000 while the system was in
fact cueing correctly.

Fixed by separating `max_lead_s` (how early still counts, default 12 s) from
`lookahead_s` (when the rig fires). Recall 0.000 → 0.875.

**Lesson:** a metric that disagrees with a working system is a suspect
metric. The trace (`probe_cue_trace.py`) is what settled it.

## Why AIST audio is the wrong corpus for this stage

| Corpus | RMS p90/p10 | std/mean |
|---|---|---|
| Synthetic EDM control | **18.27** | 0.726 |
| AIST (median of 10) | **3.76** | ~0.45 |

AIST tracks are short, loudness-normalised dance loops with **~5× less energy
dynamic range**. On them, anticipation recall was 0.000 on 3/5 tracks, and on
the one track scoring 0.800 the random baseline was 0.625 — i.e. no real
skill. Segment energies sat in a 0.475–0.518 band: there is essentially no
build-up/drop structure to find.

**This is a corpus limitation, not (only) a detector limitation.** The
synthetic control proves the detector localises real drops to <1 s. Both
hypotheses were tested; both were partly true, and the fix for the detector
half is already in.

**Consequence:** AIST is the right corpus for pose/choreography work and fine
as a *rendering* input, but EDM-98 / EDM-CUE is the right corpus for
validating structure detection. Not yet downloaded.

## Causal vs oracle — the line that must not blur

`anticipation.py` implements two explicitly separated modes:

- **CAUSAL** — one-sided EMA + backward-difference slope; at time *t* reads
  only samples ≤ *t*. Time-to-drop is a genuine forecast (extrapolate the
  tension slope to a threshold crossing). This is what a real rig could run.
- **ORACLE** — full-track lookahead; time-to-drop is exact *because it reads
  the answer*. Not a prediction; it is the ceiling.

The smoke test asserts `max|causal − oracle| > 1e-6` so a future refactor
cannot silently leak the future into the causal path. Measured correlations
between the two curves on AIST: 0.435, 0.005, 0.193, −0.069 — they are
genuinely different signals.

## Null-track control

On steady-state audio with no drops, the cue rate collapses from
**0.146 Hz → 0.021 Hz**. The system is not trigger-happy.

**Residual weakness, stated plainly:** the labeller still tags **3 spurious
"drops"** on that same drop-free audio. Boundary detection is solid; the
functional labeller is the weakest link and is a rule-based heuristic that
has never been scored against annotated ground truth. EDM-98 is the way to
fix that, and it has not been done.

## What is baseline-quality vs novel

- Beat/tempo tracking, boundary detection — **standard librosa baseline**,
  works.
- Functional labelling (intro/build-up/drop/breakdown/outro) — **heuristic**,
  validated only on synthetic audio.
- Causal tension curve + forecast time-to-drop — **the novel bit**, and the
  part with the strongest evidence (precision z = +2.42, tight 2.78 s lead).
