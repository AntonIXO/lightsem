# LightSem v2 — sound → light for concerts and festivals, trained on dance

Date: 2026-09-04, **rev. 8** (Anton's three directions after the Stage 8 result — richer audio
features, more authored-lighting corpora, transformer/multimodal methods — and his mixture-of-light
idea, run through one Perplexity deep-research pass and four reasoning passes with the full project
brief, the MIT MAS.S60 multimodal skill, and the Skip-BART and SeqLight papers read in full;
section 6.9 is the new stage, section 5 gains the authored-lighting rows, section 9 gains its
rule-outs). Previous header:
Date: 2026-09-04, **rev. 7** (category test rated, section 1.5; the Just Dance corpus
Anton downloaded becomes Stage 8, section 6.8; section 5 gains its row). Previous header:
Date: 2026-09-04, **rev. 6** (Anton's call: the product is a **pre-rendered show**, not a live
desk. Section 0.1 states what that changes, section 1.4 records the repair pass that preceded it,
and section 6.7 is the new stage. Previous header:
Date: 2026-09-04, **rev. 5** (implementation pass run; section 1.3 records what was built and what each stage gate returned). Previous header:
Date: 2026-09-04, **rev. 4** (rev. 3 sections kept; section 1.2 and the marked
paragraphs record what Stages 0a and 0b actually measured, and the data table in section 5 is
re-ordered accordingly). Previous header:
Date: 2026-09-04, **rev. 3** (rev. 2 corrected by Anton: dance is the training set, no dance
generation at inference; rev. 3 integrates four Perplexity reasoning passes run 2026-09-04 with
the full project context — audio→light design, music-induced-movement science, causal model and
latency, dataset audit — plus Anton's note that off-the-shelf pose models can extract the motion
data from video). Language: English by request.
System of record: vibescience vault `/root/lightsem-vault`, problem
`dancing-light-generative-20260903`, runbooks `rb-ls2-*`. This file is secondary provenance.

---

## 0. The task, stated once

**Dance datasets are the training set. The product is a causal sound → light model for concerts
and festivals.** Paired (music, dance motion) → motion descriptors → light targets on a canonical
rig (fixed mapping) → train **audio → light** directly. At inference: sound in, DMX out; no
dancer, no pose, no dance generator. Motion may come from mocap datasets **or be extracted from
video with existing pose models** (YOLO-pose / RTMPose / ViTPose for 2D, GVHMR / SMPLest-X for
3D, as InfiniteDance did) — the dancer is a training-time-only modality either way.

What rev. 3 changes, in one paragraph: the movement-science literature says audio determines a
**shared energy/periodicity skeleton** of dance (how much, at which metrical level, bounce, head
and hip response to bass, energy contrast at drops) plus a **distribution over style** (which
limb, which side, which phase, hands, floor path). A 24-d target trained with one loss would mix
the two and either blur (MSE) or sample a dance prior with a music watermark (flow). So the
target is split into three blocks with three objectives — **deterministic skeleton, event
point-process, sampled style residual** — and the rig mapping follows the body: big slow fixtures
follow the torso/head (bass), small fast fixtures follow the hands (highs), vertical channels
bounce at the beat, horizontal sweeps run at the bar. Latency science (ITU-R BT.1359: sound
leading light is detectable at +45 ms) forces a two-path deployment: onset→strobe transients on a
10–20 ms DSP path, the learned model painting the slower channels.

### 0.1 The look-ahead axis (rev. 6)

**The product is a pre-rendered show for a track that is known in advance.** A DJ set with a
prepared list, a band with a fixed setlist, a festival slot programmed the day before. The whole
track is available before the first frame of light is emitted.

That does not delete the causal work; it relabels it. Every number from here on carries its
position on one axis:

| L | what it means | how it is bought |
|---|---|---|
| `L = 0` | strictly causal, nothing after *t* | free; the live desk case |
| `L = 250 ms` | bounded look-ahead | a house delay line; IMAG already forces 50–150 ms |
| `L = whole track` | offline | pre-rendering; no delay in the room at all |

**A score is comparable only to a score at the same L.** An offline number quoted without its
causal counterpart is not a result — it is the same mistake as the leak in section 1.4, made on
purpose. Three consequences follow immediately:

1. `extract_bus(causal=False)` is the offline feature path and already exists. The flag was kept
   when the bus was made causal, so the centred STFT, the centred HPSS median and the
   interpolating resample are one argument away.
2. The five `ORACLE_CHANNELS` stop being a debt. A whole-track constant-tempo comb grid is simply
   correct offline; `flags["oracle_beat"]` now marks a property rather than a compromise.
3. Section 4's latency budget is no longer a product gate. It stays as the `L = 0`
   characterisation and for any live-augmented variant.

The honest reason to prefer pre-rendering is not that the numbers are higher. It is section 6.7's
H-C: the largest measured effect in this project's data is the drop/breakdown movement contrast
(1.68–2.53×, Cohen's *d* up to +2.55), and using it requires knowing a drop is coming, which is
exactly what a causal model cannot do and a human operator with a setlist always does.

---

## 1. What exists today (verified on disk 2026-09-03/04)

| piece | status |
|---|---|
| `lightsem/data/aist.py` — AIST++ loader/manifest (1408 seq, 60 tracks, 30 dancers, 21 choreos) | PASS |
| `lightsem/data/rpmc_remote.py`, `extract_rpmc.py` — RPMC-L2 range reader; **699/699 songs cached** (`cache/rpmc_all`: OpenL3 512-d at 10 Hz + the 6-d scene vector; no waveform) | PASS |
| `lightsem/audio/*` — librosa beats/downbeats, laplacian segmentation, EDM-role labeller (heuristic), causal/oracle tension | PASS |
| `lightsem/dmx/*` — club rig, structure→scene rules, beat micro-layer, 512-ch universe @40 fps, cue list, Art-Net sender (untested on hardware) | PASS |
| `lightsem/sim/*` — headless moderngl/EGL volumetric renderer → MP4 (~7× realtime) | PASS |
| `lightsem/model/*`, `train_macro.py` — causal patched transformer audio→6-d light scene vector (Stage F) | NEGATIVE; keep as regression baseline harness |
| Data: AIST++ keypoints3d + wav (`data/`), AIST++ SMPL `motions.zip` (`/mnt/lightsem-data/aistpp/`), RPMC-L2 140 songs (`data/cache`) | on disk |
| Pose tooling: `mediapipe 1.0.0` + `cv2` in the project venv; no ultralytics/RTMPose/GVHMR yet; AIST multi-view video reachable (research licence) | to install |

Root disk 89 % full → bulk data and the torch venv go to `/mnt/lightsem-data/` (README there).

### 1.1 Measured facts that bind the plan (vault, Aug 2026)

1. Audio → one dancer's kinetic energy by regression fails: held-out R² = 0.002 at 0.1 s, negative
   at 2–8 s; inter-dancer r = 0.024 / 0.13 / 0.50 at 0.1 / 2 / 8 s; beat-synchronous
   same-choreography agreement real but small (r = +0.131, R² ≈ 1.7 %).
2. Audio → absolute lighting level fails on RPMC-L2 (persistence R²_within +0.708, audio −0.005).
3. Audio → lighting **event** is real (AP lift +0.09, z +2.83), carried by timbre/spectral shape,
   not loudness; reactive (≤ 1–2 s).
4. Sound-pivot chain ≤ 0; AIST audio lacks EDM structure; timing metrics need a metronome null;
   z without effect size is unusable.

### 1.2 What Stages 0a and 0b measured (2026-09-04, this repo)

Run with `run_stage0a.py`, `run_stage0b.py`, `run_burger_replication.py`, `run_oslo_audit.py`;
recorded in the vault as experiments 001-003.

**Two corpus bugs, fixed before anything else.** librosa's beat tracker returned octave-wrong
tempi on this corpus (161.5 BPM for an 80 BPM track) and locked the grid phase to the off-beat
whenever snare and hat flux exceeded the kick's. Replaced by a constant-tempo comb with a
log-normal prior, period from broadband onset and phase from 40-120 Hz kick flux at 5.8 ms
resolution (`lightsem/audio/beatgrid.py`); it recovers AIST's true 80/90/100/110/120/130 BPM
design on 58 of 60 tracks. The first footfall detector fired 2.1 times per musical beat on
planted-foot marker jitter; contact hysteresis brought it to 0.69 per beat and moved footfall
beat-locking from z = 1.4 to z = 5.2.

**Dance carries the beat's period, not its phase (Stage 0a).** Footfalls are tightly periodic
per clip (Rayleigh R = 0.55, median z = 5.2) but the preferred phase belongs to the
choreography, not the music. Light mapped from motion alone therefore scores at its own
circular-shift null (+0.008, CI [-0.010, +0.026]) while a 30-line audio beat trigger scores
+0.094 (CI [+0.083, +0.106]). Splitting the sources - envelopes and amplitudes from the dancer,
transient phase from the audio - beats both: +0.124 (CI [+0.114, +0.135]), while staying close
to the dance-derived look (FID_light 43 to the motion-only oracle versus 794 to the beat
trigger). This is the same two-path split section 4 derives from latency, reached independently.
A per-track audio-motion sync offset was fitted, validated on held-out clips, and then
**refuted** because it did not transfer to a channel it was not fitted to
(`lightsem/data/sync.py` keeps the refutation).

**AIST++ cannot supply block A (Stage 0b).** Burger 2013's stimulus-level correlations do not
replicate here and several invert: pulse clarity to quantity of motion +0.62 published against
-0.078 measured, low-band flux to head speed +0.73 against -0.102, high-band flux to hand speed
+0.65 against -0.321. The reason is the corpus design, not the method: the genre label alone
explains 0.835 of between-track variance in quantity of motion, 0.916 in centre-of-mass speed
and 0.955 in footfall rate, and with genre and tempo partialled out every audio-movement
correlation collapses to within +/-0.21 of zero. AIST++ is choreographed dance to genre-matched
music, so genre drives both sides. Within a track, causal audio predicts a little of the shared
dynamics (laterality R2 0.19, hand speed 0.18, centre-of-mass speed 0.13, vertical bounce 0.09)
and that is the honest ceiling for anything trained on it.

**Free dancing to EDM does supply block A (Stage 0b, Oslo).** On the Oslo club-dancing corpus
(29 dancers, two DJ mixes, CC-BY-4.0): dancers hearing the same mix move together (split-half
reliability of the across-dancer mean 0.61-0.93, mean pairwise inter-dancer r 0.16-0.61);
movement at the drop is 1.68x to 2.53x the breakdown level (Cohen's d +1.30 to +2.55, all CIs
clear of zero); and causal acoustic features predict the shared movement curve on **held-out
tracks** at R2 +0.30 (CI [+0.225, +0.373]), two to three times the best AIST++ number. One
correction on the way there: absolute movement regressed on absolute loudness scores negative
R2 across tracks, because each track in a mix sits at its own level. Both sides need the causal
trailing gain normalisation a real desk has; that alone moved the same data from -0.72 to +0.30.

**Consequence for the plan.** Block A is alive, in the concert domain, on gain-normalised
envelopes. AIST++ keeps three jobs - genre style priors, motion realism as an FID reference,
and validating the video-to-pose path against mocap - and loses the fourth. The binding
constraint is no longer "is there a signal" but "how much free-dance-to-concert-music data can
we get", which promotes the Oslo sets and own-footage capture ahead of more choreography corpora.

### 1.3 Implementation pass, 2026-09-04 — what was built and what the gates said

Built and verified (14 of 14 checks in `verify_all.py`): descriptor extractor for mocap and
video, constant-tempo beat grid, three conductors on a canonical rig (motion-only oracle,
hybrid, beat-trigger baseline), evaluation suite with metronome / circular-shift / block-shuffle
nulls, a 9.3 M-parameter causal Conformer with regression, hazard and rectified-flow heads,
streaming inference, and a brightness-matched blind A/B harness.

| stage | gate | result |
|---|---|---|
| 0a | FID_light discriminates | **pass** — split-half floor 1.8, block-shuffle null 110 (60x) |
| 0a | dance-derived light beats its shift null | **fail** — +0.008, CI [-0.010, +0.026]; the hybrid split reaches +0.124, CI [+0.114, +0.135] |
| 0a G3 | video pose matches mocap r > 0.8 | **fail on the average, passes per channel** — vertical family 0.79-0.87, shape 0.78-0.81; twist 0.14, hip yaw 0.19, footfall rate 0.30 |
| 0b | descriptors beat dancer identity on AIST++ | **fail** — genre explains 0.65-0.96 of between-track movement variance |
| 0b (Oslo) | audio predicts shared movement in free dancing | **pass** — held-out-track R2 +0.30, CI [+0.225, +0.373]; drop/breakdown 1.7-2.5x, d up to 2.55 |
| 1 | causality leak zero | **pass** — exactly 0.0 |
| 1 | model light beats its null / FID / diversity | **fail** — +0.017 vs real-mapped +0.216; FID 70 vs half-null 67; diversity 0.42x real |
| 2B | operator-head event lift >= +0.09, z > 3 | **pass** — +0.3145x, CI [+0.149, +0.479], z 3.66, 699 shows, shift control peaks at 0 s |
| 4 | transient <= 20 ms, learned <= 65 ms, causal | **pass** — 10.8 ms, 62.5 ms p99, leak 0.0 |
| 0c (ii) | dance-derived light beats the beat trigger on >= 1 question, CI excluding 0.5 | **pass, directional** — 'lighting engineer' 8-1-1, CI [0.518, 0.997]; 'on the music' 5-4-1, chance; one rater, 10 pairs (section 1.5) |
| 0c (i), 3 | late lights; full human A/B | **materials generated, awaiting raters** |

Three corrections the work forced on this plan:

1. **The KV cache is not 3-4 s.** Attention windows compose across layers: six blocks of a
   160-frame window plus a 15-tap causal convolution look back 1039 frames, 26 s. Priming a
   chunk with one window instead gave the same audio different light depending on how it was
   sliced (block A differing by 0.79). Section 4's cache figure is wrong; 26 s is the number.
2. **The strongest signal is the operator, not the dancer.** Audio's marginal lift for
   predicting an operator's scene change is +0.31 across 699 professional shows; the best
   dance-derived number anywhere is a within-track R2 of about 0.2. The conductor should take
   look changes from the operator head, transient phase from the beat grid, and envelopes from
   the dance-trained head — three sources, each used for what it measurably carries.
3. **Regressing an unpredictable channel is worse than sampling it.** A regression head asked
   for a channel it cannot predict emits the corpus median, the light then flashes uniformly at
   6.5 events per second, and a uniform train sits at its own circular-shift null however well
   each flash is phased. Moving those channels to the sampled head and making it autoregressive
   moved model light from +0.0009 to +0.0172 against the null and FID from 114 to 70.

### 1.4 Repair pass, 2026-09-04 — a cloud review, and what the check written for it found

Ten review findings were fixed (index truncation on partial cache rebuilds, a `bpm / ratio * ratio`
no-op, a broadcast crash on a weak Oslo fit, an unequal-length concatenation, a silently
no-op command dispatch, a `[:-0]` slice, a dead `max_gap`, a double DMX decode, a p99 computed
from one measurement repeated 100 times). The regression check written to hold the worst of them
then failed on something larger.

**The audio bus was not causal.** Three librosa defaults, each offline-correct and each reading
the future:

| channel | look-ahead | cause |
|---|---|---|
| `percussive_ratio` | 375 ms, **15× its own SD** | HPSS's centred 31-frame median |
| every spectral channel | 25 ms | `center=True` timestamps a 2048-sample window at its midpoint |
| `onset`, `pulse_clarity`, mel | whole clip (thin) | `top_db` clamps against the clip's maximum |

Plus a 23 ms leak in the analysis→control resample and up to 70 ms in `trailing_pulse_clarity`'s
interpolation. All fixed; the check is now in `verify_all.py` (17 checks, 0 failures) and asserts
that the five beat-grid channels are still flagged oracle rather than silently exempt.

**The cost, which is the point.** Held-out block A R² fell **+0.0438 → +0.0192** after retraining
on honest features. Half of the previously recorded number was bought with look-ahead a live
system cannot have. The model's own causality probe was passing throughout and never had a chance
of catching this: it perturbs the model's input, not the input's construction.

**Two measurements that redirect the work.**

1. *Event density.* The model emits 85.8 visible light events per clip against 30.1 for
   light mapped from the real dance. A train denser than the beat grid has a high circular-shift
   null by construction, so correct phasing cannot show above it. Block-B thresholds are now
   fitted to the corpus event rate (footfall 0.898 ev/s against 0.896, hit 1.352 against 1.353)
   and stored in the checkpoint — but a clean control on identical features shows this **lowered**
   Light-BAS (+0.0212 → +0.0179) while improving FID (69.54 → 67.29) and diversity (2.57 → 2.73).
   The predicted mechanism was wrong and the light event count did not move.
2. *Where the density actually comes from.* The model's **sampled** channels carry 11× to 452× the
   corpus's above-6 Hz energy (`section_energy` 116×, `space_eff` 202×, `weight` 343×,
   `hand_spread` 452×), while the **regressed** block A channels sit at 0.3–5×. The targets are
   mocap through a 12 Hz zero-phase Butterworth and are physically smooth; a flow head sampled per
   frame is not, and `ar_flow` was already enabled, so conditioning on its own previous output is
   not sufficient. That jitter crosses the light's visibility threshold on the wash fixtures.

Two diagnostics were registered so neither failure can hide again: `light-event-rate-ratio` and
`descriptor-hf-energy-ratio`, both target-1.0.

### 1.5 Category test, rated 2026-09-04 — dance is not a category error for light

Stage 0c (ii), `outputs/stage0c/category/`, scored by `score_category.py` against the sealed key;
vault `exp-010`. Ten AIST++ pairs, 8 s, brightness-matched (mean ratio 1.03–1.17, p95 ratio
0.84–0.96), dance-derived **hybrid** conductor against the 30-line **beat trigger**, one rater
(Anton), FOIL trials skipped.

| question | hybrid | beat trigger | equal | win rate (decided) | exact 95 % CI | sign test |
|---|---|---|---|---|---|---|
| which looks like a **lighting engineer** did it | 8 | 1 | 1 | 0.889 | [0.518, 0.997] | p = 0.039 |
| which is **on the music** | 5 | 4 | 1 | 0.556 | [0.212, 0.863] | p = 1.0 |

The tie on timing is by construction: after Stage 0a the hybrid takes its transient phase from
the same audio beat grid the beat trigger uses, so the two arms are phased identically and the
first question cannot separate them. The second question is a dynamics question — does the
light vary in amount, width and rate the way a designed show does — and that is where the
dancer's envelopes show. **Block C stays in the product.** The verdict is directional, not
final: one rater of five, ten tracks of twenty, no club/EDM half, and the file letter was a
constant label across trials. The preregistered bar remains open for Stage 3.

The rater's notes are the useful part. T05, beat trigger: *"too fast changes for light, but on
the music actually."* T07, both bad: *"A does not moving, B is moving boring"* — the hybrid was
static on that clip and lost both questions. The hybrid loses when its envelopes are flat; the
beat trigger loses when it flashes on every beat. That is the model's own largest measured
defect (section 1.4, 85.8 light events per clip against 30.1) named independently by a human,
and it is why H-B in section 6.7 is aimed at what a viewer sees rather than at a metric.

### 1.6 Stage 7 results, 2026-09-04 evening — H-B first, because it is the largest lever found

**H-B, spectral matching (vault exp-011).** Four arms on identical descriptors from the causal
checkpoint, 40 val_track clips, 3 seeds:

| arm | Light-BAS − null | FID | diversity (clips / seeds) | events per clip | worst HF ratio |
|---|---|---|---|---|---|
| raw (Stage 1 as recorded) | +0.018 | 67.3 | 2.73 / 1.74 | 85.8 | 117× (`hand_spread`) |
| causal one-pole, fitted per channel | **+0.293** | **19.8** | 2.90 / 1.26 | 35.5 | 5.1× (`lateral_pos`) |
| zero-phase 12 Hz (the filter the hypothesis named) | +0.061 | 46.5 | 2.90 / 1.79 | 63.9 | 72× |
| zero-phase at the corpus-matched cutoffs | +0.304 | 23.4 | 2.86 / 1.29 | 35.8 | 5.0× |

Real-mapped light on the same clips: +0.211; beat trigger +0.061; FID half-null 67.5, split-half
floor 6.3. Two readings. The mechanism is confirmed and it is large: the sampled channels' jitter
was what sank every Stage 1 light gate, and matching their spectrum to the corpus moves model light
from chance to above the real-mapped value on Light-BAS and to a third of the null on FID, with
diversity across clips not falling. The filter the hypothesis *named* does almost nothing, because
the mismatch is not at 12 Hz: the corpus channels are smooth well below that and the fitted cutoffs
sit near 0.2 Hz (eight channels hit the 1 s cap). The causal one-pole already captures the gain,
so **this is not an offline-only win**; the offline arm removes the one-pole's lag and scores the
same. Caveats that travel with the number: eight channels are nearly flattened; the model's light
is now more beat-locked than real-dance light because most of its events come from the
audio-phased transient path; diversity across style seeds fell from 1.74 to 1.26, which is the
sampled variation that was jitter. Whether the result reads as better light is Stage 3's human
question. The checkpoint that carries the fitted constants is `light_trunk_v3causal_smooth.pt`,
and `outputs/stage0c/category_model/` must be re-rendered from it.

**H-A, the look-ahead ceiling (vault exp-012).** Same 9.3 M Conformer, same 749,173 frames, splits,
12k steps; the only changes are the symmetric attention window, symmetric convolution padding and
`extract_bus(causal=False)` (leak 0.29 by design against 0.0). Both gates **fail**, and that is the
informative outcome:

| | causal (L = 0) | offline (whole track) |
|---|---|---|
| held-out block A R², val_track / test_genre / test_dancer | +0.019 / −0.270 / +0.054 | **−0.009 / −0.218 / +0.002** |
| event AP val_track (footfall, jump, hit) | 0.363 / 0.047 / 0.426 | 0.364 / 0.044 / 0.428 |
| Light-BAS − null, raw | +0.018 | +0.019 |
| Light-BAS − null, with H-B smoothing | +0.293 | +0.276 |
| FID raw / smoothed | 67.3 / 19.8 | 66.4 / 18.2 |
| diversity across clips raw / smoothed | 2.73 / 2.90 | 3.06 / 3.06 |

With the entire track visible, nothing moves. The gap between model light and real-mapped light is
therefore not a causality tax and not a context-length problem; it is the corpus, as Stage 0b said
(AIST++ movement is explained by genre, not audio). Consequences: H-C is tested on Just Dance,
where songs have structure, never on AIST++; the pre-rendered path's honest advantages are H-B's
lag removal and H-C/H-D anticipation, not accuracy; and Stage 8 is the accuracy lever.

**H-D, operator head with anticipation (vault exp-013).** Audio-only arms on the 699-show split,
margin = model lift over the causal rhythm-of-past-changes baseline:

| arm | lift | rhythm | margin | 95 % CI |
|---|---|---|---|---|
| causal, horizon 0 (exp-008 audio-only) | 1.35× | 1.25× | +0.096 | [−0.05, +0.26] |
| causal, horizon +4 s | 1.39× | 1.31× | +0.082 | [−0.08, +0.25] |
| offline, horizon +4 s | 1.29× | 1.31× | **−0.021** | [−0.16, +0.14] |
| offline, horizon 0 | 1.41× | 1.25× | +0.158 | [−0.06, +0.41] |

**G-antic-4s fails**: with the whole show audible, audio alone does not predict an operator's scene
change four seconds ahead better than the rhythm of past changes. The offline head at horizon 0 is
numerically above the causal one and its CI spans zero. A first version of these arms scored a
**10× lift** at +4 s and was discarded as a leak before it was quoted: the hazard features are
computed from the target's own past events, legitimate for a causal head and a window into the
future the moment attention went symmetric; the target-shift control caught it (a genuine
anticipator degrades smoothly away from its horizon, a leak collapses to 1× one second either
side). Rule kept: an offline model may only see inputs that do not derive from its own target.

**Stage 8.2, the Just Dance predictability audit (vault exp-014).** 294 held-out songs, 453 coach
channels, the Oslo recipe per coach (both sides causally gain-normalised, 13 causal bus scalars
at three time constants, ridge by song fold):

| | audio → coach, held-out songs, 2 s | inter-coach r (93 multi-coach songs) |
|---|---|---|
| quantity of motion | R² −0.016, r +0.21 | 0.27 |
| vertical bounce amplitude | R² +0.004, r +0.21 | 0.42 |
| head speed | R² −0.027, r +0.17 | 0.22 |
| hand speed | R² +0.005, r +0.24 | — |

**G8.2-r2 fails** (gate +0.20; Oslo's crowd curve gave +0.30 from published audio scalars): a
linear map from causal audio scalars to one coach's movement correlates weakly but consistently
across 294 songs and explains none of the variance, at any ridge penalty. The ceiling it is
under is the inter-coach agreement — coaches in the same routine hearing the same music agree at
r 0.22–0.42 — which is what a single-performer target allows. What the corpus does carry, at the
section level: coach movement in a song's loudest third of sections is **1.49–1.57× the quietest
third** (CI [1.44, 1.63], 453 channels), the Oslo drop/breakdown range, and 70 % of channels
bounce at the beat or half-time. G8.2-genre could not be measured (no genre or Effort labels were
ever attached to the corpus) and is recorded as not passed. Reading: the frame-level coupling is
weak on this corpus as on AIST++; the section-level coupling is real, which is exactly the
information H-C's structure channels carry, and the Conformer arms decide whether a nonlinear
model with the full mel bank does better than the linear audit.

**Stage 8.5 and H-C, the pipeline trained on Just Dance (vault exp-015).** Three arms of the same
9.3 M causal Conformer on the Just Dance tensor (3.2 M frames, 453 coach channels, splits by song;
`hip_yaw_speed` and `footfall_rate` out of the loss), each with its own event thresholds and H-B
smoothing fitted on the validation songs, scored on 40 clips of held-out songs; the AIST++-trained
model scored on the same clips:

| model | held-out block A R² | event AP (footfall / jump / hit) | Light-BAS − null | FID | diversity | events per clip |
|---|---|---|---|---|---|---|
| AIST++-only (v3causal, smoothed) | −0.266 | 0.28 / 0.12 / 0.30 | +0.242 | 125.7 | 7.10 | 545 |
| Just Dance, no structure | −0.225 | 0.31 / 0.14 / 0.34 | **+0.135** [0.121, 0.150] | **36.5** | 7.44 | 796 |
| Just Dance + structure channels | −0.178 | 0.36 / 0.14 / 0.34 | +0.107 | **27.6** | 6.65 | 869 |
| Just Dance + shuffled structure (control) | −0.189 | 0.35 / 0.14 / 0.35 | +0.120 | 33.1 | 7.35 | 806 |
| light mapped from the real coach | — | — | +0.087 | 8.1 (split-half floor) | 6.26 | 881 |
| beat trigger | — | — | +0.092 | — | — | — |

Base event rates are 0.020 / 0.007 / 0.024 per frame, so the event head lifts them 14–20×.

Readings, in order of weight. **The pipeline trains on Just Dance and works at the light level:**
the no-structure model beats its shift null by more than light mapped from the real coach does,
passes all three Stage 1 light gates on held-out songs (BAS ≥ 0.9 × real, FID below half the
null, diversity ≥ 0.7 × real), emits the corpus's event rate within 10 %, and transfers to the
Oslo crowd (8.5 above). **Block A envelopes do not generalise across songs** for any model,
including the AIST++ one; what generalises is events and section-level energy, which is what
light uses. **H-C is refuted on its gate**: the structure channels lower Light-BAS by 0.028 (paired
over the 40 clips, CI [−0.042, −0.017]) and diversity, while improving FID from 36.5 to 27.6; the
shuffled control reaches 33.1, so part of the FID gain is real structure and none of it is
beat-locking. **A caution on the metric:** the AIST++ model is the most beat-locked of all
(+0.242) and the furthest from the corpus's light (FID 126, 545 events against 881), because a
model that predicts nothing about the envelopes leaves the audio-phased transient path to do all
the flashing; Light-BAS alone rewards that, so the number is read with FID, the event rate and the
human test, never on its own.

**Stage 8 gates.** 8.1 fails on the letter (196 single-coach songs, 250 asked) with 294 songs and
453 channels in hand. 8.2 fails (frame-level R² ≈ 0; section contrast 1.5× real). 8.3: 70 % of
channels bounce at the beat or half-time, no per-file offset fitted. 8.5 passes on both halves:
the Just Dance model above the AIST++ one on held-out songs, and Oslo transfer r +0.335 against
+0.07. H-C fails. **H-F passes (vault exp-016):** the Just Dance background, with coaches and UI
masked and reduced to RPMC-L2's 6-d scene vector, is predictable from causal audio the way an
operator's scene changes are, and more so — on 38 held-out songs the same 4-block head with
hazard features lifts scene-change AP 3.17× against the rhythm-of-past-changes baseline's 1.48×,
margin +1.69, CI [1.19, 2.27], leak 0, shift curve peaking at the trained horizon (RPMC-L2's 699
operators gave +0.555). Game backgrounds cut on bars, so they are more regular than a human
operator; the palette (hue KL) half of the gate was not implemented and is recorded as not
measured. HEAD B now has a second training corpus.

**Where this leaves the plan (2026-09-04, 21:10).** Every hypothesis queued in the vault has been
run and recorded: exp-010 (category test), exp-011 (H-B), exp-012 (H-A), exp-013 (H-D), exp-014
(H-E, the audit), exp-015 (H-C), exp-016 (H-F). The pipeline trains on the Just Dance corpus and
its light passes the Stage 1 gates on held-out songs and transfers to a real crowd. The shipped
causal model is `light_trunk_jd_none_smooth.pt` for pop and EDM songs, with
`light_trunk_v3causal_smooth.pt` as the AIST++-trained reference; the demo renders are under
`outputs/stage8/demo/`. Open items are the ones that need something this session did not have:
raters for 0c(i) and Stage 3 and for `outputs/stage0c/category_model/`, a DMX node for Stage 4's
hardware half, a learned conductor (Stage 5) and EDM lighting ground truth (Stage 6); and the
measured next levers are more songs with several performers each, 3D lifting for the channels 2D
pose cannot supply, early stopping on validation songs, and the H-F visuals as a second target.

---

## 2. What audio determines in a dancing body — and what it does not

Free-movement mocap studies (many people, no choreography) measured the shared map directly.
Burger, Thompson, Luck, Saarikallio & Toiviainen 2013 (N = 60, 30 pop clips,
doi:10.3389/fpsyg.2013.00183), stimulus-level correlations:

| audio feature | movement feature | r | shared across people? |
|---|---|---|---|
| pulse clarity | centre-of-mass speed / shoulder wiggle / total quantity of motion / hip wiggle / foot speed | 0.67 / 0.67 / 0.62 / 0.58 / 0.55 | yes — whole body |
| low-frequency spectral flux (50–100 Hz, kick/bass) | **head speed** | **0.73** | yes |
| high-frequency flux (6.4–12.8 kHz, hats) | hand speed / head speed / hand distance | 0.65 / 0.64 / 0.57 | mixed (residual is large, cultural) |
| percussiveness | head speed / shoulder wiggle | 0.67 / 0.61 | mixed |
| tempo (73–154 BPM) | any amplitude | n.s. | tempo sets *which metrical level* people lock to, not how much they move |

Period/phase locking (Burger et al. 2014, doi:10.3389/fnhum.2014.00903; Toiviainen, Luck &
Thompson 2010, doi:10.1525/mp.2010.28.1.59): vertical bounce locks to the **tactus**; horizontal
sway and torso rotation lock to **2-beat and bar** levels; faster tempo shifts energy up the
metric hierarchy (bar-level ML r = 0.86). **Period is a group fact; phase is a person fact** —
period-locked dancers scatter around the phase circle. Bass loudness raises quantity of motion,
hips and torso (Van Dyck et al. 2013, doi:10.1525/mp.2013.30.4.349; Cameron et al. 2022,
doi:10.1016/j.cub.2022.06.035, inaudible VLF bass +12 % dancing). At drops in club EDM, quantity
of motion and vertical head movement surge; at breakdowns they collapse (Solberg & Jensenius
2017, doi:10.1177/1029864917713785). Groove is an inverted-U gain on all of the above (Witek et
al. 2014, doi:10.1371/journal.pone.0094446). In dyads, music drives head bobs and the vertical
bounce; the partner drives hands and lateral motion (Bigand et al. 2024,
doi:10.1016/j.cub.2024.05.055).

**Reading of our own numbers.** r = 0.50 at 8 s is the section-energy channel above; r = 0.024 at
0.1 s says the *amplitude* of instantaneous kinetic energy is music-independent — it is the phase
and style scatter, not a missing model. A generative objective escapes the conditional-mean
collapse for diversity/FID, but it cannot invent a coupling that is not in p(y | audio): a flow
model of 10 Hz kinetic energy is an unconditional dance prior with a music watermark. What audio
*does* determine is exactly the set of things a lighting desk already uses: intensity contour,
bounce rate, which metrical level, transients, energy contrast. That is the supervision to keep.

### 2.1 Modality profile and invariances (MIT MAS.S60 lectures 03/06/08)

| | audio (input) | dance motion (train-time only) | light (output) |
|---|---|---|---|
| elements | frames → beats → bars → sections | poses → movements → phrases | DMX frames → cues → looks |
| information | tempo, phase, energy, timbre, structure | kinematics, effort, space use, contacts | intensity, colour, direction, beam, rhythm |
| task relevance | shared part → **predicted** | unique part → **sampled** or dropped | target |

Invariances: root-relative, torso-normalised features; mirror equivariance; per-song loudness
normalisation; causal time-shift equivariance. Tempo is not an invariance (it selects the metric
level). Cross-modal transfer moves only redundant information (lecture 08) — hence predict the
skeleton, sample the rest. Segment before aligning (lecture 06): beat/bar-synchronous features.

---

## 3. Targets: three blocks, three objectives

Computed from joints (SMPL FK from mocap, or 2D/3D pose from video), torso-normalised, mirror-aware,
resampled to 40 fps and beat-synchronised. **Do not average 30 dancers into one vector**; average
only the shared scalars per bar and keep per-clip residuals as samples of p(style | music, genre).

**Block A — shared skeleton (deterministic; MSE/L1 on z-scored features + a periodicity head)**
- quantity of motion; centre-of-mass, head and foot speed (not hands)
- vertical bounce amplitude and **period** (spectral peak of the vertical channel at tactus)
- embodied metrical level (½-beat / beat / 2-beat / bar; one-hot or log-period)
- Laban Weight proxy (bass, pulse clarity, RMS, break/drop contrast); Laban Time proxy (percussiveness, attack slope)
- footfall rate near the beat; section energy contrast

**Block B — events (point process; Bernoulli/hazard head, AP/F1 at 0.2 s and 1 s)**
- footfalls, jumps, hits (from dance); scene changes and palette shifts (from RPMC-L2 operators)

**Block C — style residual (sampled; conditional flow matching on the residual after subtracting A)**
- hand speed / spread; laterality *amplitude* weakly from audio, *sign and phase* sampled;
  travel direction; symmetry; spatial spread residual; Laban Space/Flow; jerk series
- conditioned on a genre/style token; sampled once per "dancer" for organic variation, or dropped

**Noise (not in any loss):** exact trajectories, gesture identity, microtiming, partner-driven
laterality, frame-level contacts beyond "a footfall happened near a beat".

### 3.1 Rig mapping (conductor v0 rule table, follows the body)

| body / block | light channel | rationale |
|---|---|---|
| quantity of motion, Laban Weight (A) | master intensity, haze density, saturation | Burger 2013 pulse-clarity/bass → whole body |
| vertical bounce at tactus (A) | dimmer / tilt bounce on the beat | vertical locks to the tactus |
| mediolateral sway at 2-beat/bar (A + C sign) | pan sweep period at half-bar/bar; sweep direction sampled | horizontal locks to the bar; phase is personal |
| head/torso speed ← LF flux (A) | big slow fixtures (washes, large heads) | bass drives head/hips |
| hands ← HF flux (C, mean from A) | small fast fixtures (LED bars, strobes, pixel effects) | hats drive hands |
| footfalls, jumps, hits (B) | strobe bursts, tilt snaps, shutter | events, not envelopes |
| scene change, palette (B, RPMC-L2) | look change, hue distribution | operator data, timbre-driven |
| spread residual (C) | zoom / beam width | phrase-scale only |
| Laban Time (A) | crossfade time (sudden vs sustained) | percussiveness |

Per-fixture slew limits; canonical rig (6 heads L/C/R × 2 trusses, washes, blinders, strobe,
LED bars) so targets are rig-independent; a rig adaptor maps to any venue's patch.

---

## 4. Model and deployment

**Backbone.** Causal Conformer, 6–8 blocks, d_model 256–384, ~15–40 M params, 40 fps control
rate, KV cache 3–4 s (Alexanderson et al. 2023 replaced dilated convs with Conformers for this
task class, arXiv 2211.09707). Three heads over one trunk: A (regression + periodicity), B
(hazard), C (conditional rectified flow / CFM over the residual, 1–4 Euler steps; MotionStreamer
reports MSE-head FID 59.2 vs diffusion-head 10.7 on the same AR trunk, arXiv 2503.15451). No VQ
tokenizer on a 24-d bus (quantisation banding; codebook collapse at 120 h). Diffusion forcing
(DiscoForcing, arXiv 2605.28491) only if long-horizon memory turns out to matter. Aux losses:
beat-gated L1 on hit channels, velocity penalty on wash channels. Baselines run every time:
MSE on all 24 dims (must collapse in diversity), beat-trigger rules, Stage-F harness.

**Conditioning (causal, ~80–200 d).** 40–64 log-mel or MFCC + RMS + spectral flux (LF and HF
bands separately — they drive different body parts) + half-wave-rectified onset; beat phase from
an online tracker (PLP zero-latency TISMIR 2024, 11.6 ms; BeatNet ~20 ms) as sin/cos φ, inter-beat
interval, stability; optional frozen *causal* music SSL. Never a bidirectional Jukebox/MERT/MuQ
block at inference.

**Two-path deployment (the latency finding).** ITU-R BT.1359-1: sound ahead of picture is
detectable at **+45 ms** (acceptable +90); flash–tone JNDs ~20–70 ms; a light that follows a kick
by 80 ms reads as a miss. Budget: 12 ms audio block + 8 ms model + 5 ms net + 20–40 ms fixture
≈ 45–65 ms. Therefore: (1) a **transient path** — onset/beat-phase → strobe/hit channels on a
10–20 ms DSP path (block B + HEAD C rules); (2) the **learned path** paints blocks A and C at
40 fps; (3) a 100–250 ms PA delay line is viable in the house feed (IMAG already forces 50–150 ms)
but never in monitors; FOH air delay is ~3 ms/m. Lookahead sweep in Stage 4 decides how much of
(3) is worth buying. Causality is verified mechanically (future-perturbation test: splice a click
at t+Δ, output at t must be bit-identical; gradient probe ∂y_t/∂audio_{t+k} = 0), latency with a
click-track impulse and a photodiode, reported as p50/p99.

**Heads in the product (unchanged naming):** HEAD A = dance-trained light head (blocks A, C);
HEAD B = operator head on RPMC-L2 (block B events + hue/value distributions); HEAD C =
deterministic beat micro-layer (transient path). Conductor v0 = §3.1 table with a multiplicative
strobe gate (accent × hit/drop); v1 learned (Stage 5).

---

## 5. Data (research licences; no YouTube scraping)

| source | what | licence | use |
|---|---|---|---|
| **Oslo club-dancing EDM mocap (have: Zenodo 21945721 + 21945706, CC-BY-4.0)** | 2015: 29 dancers, 3 groups, 2 DJ mixes, 200 Hz, one marker each, drop/build/breakdown times, redistributable acoustic features at 86 fps. 2014: 16 dancers, group quantity of motion only (their per-dancer tracking failed) | CC-BY-4.0 | **block A**: the only corpus here where audio predicts movement on held-out tracks (R2 +0.30). Audio itself is not redistributable — features only |
| Oslo standstill/micromotion (Zenodo 21964686, downloading) | 42 adults, 6 EDM excerpts, 22-marker mocap | CC-BY-4.0 | block A ceiling on "how much" with no dance |
| AIST++ (have) + AIST multi-view video (reachable) | 5.2 h, 10 genres, SMPL + 9-view video | annotations CC-BY-4.0; video research DB | **not block A** (see 1.2): genre style priors, motion realism reference, and video → 2D-pose validation against mocap |
| FineDance | 7.7 h public (14.6 h total), 22 genres, SMPL-H + wav; splits @Genre / @Dancer | research (README gives no commercial terms) | genre breadth; held-out-dancer split |
| InfiniteDance (HF `huuuuuuuuu/InfiniteDance`) | 100.7 h, 30 genres, SMPL-X from video (FRDM-repaired), MuQ features | non-commercial academic | scale; kinematic noise expected |
| OpenDanceSet (arXiv 2506.07565) | 100.3 h, 14 genres, 147 subjects, 2D + 3D; poses + source URLs, not pixels | research; ToS on the user | scale, if URLs resolve |
| PopDanceSet, Let's Dance (arXiv 1801.07388), BRACE (2207.10120) | 3.6 h / ~1000 clips / breaking, pose from video | research | 2D-descriptor path |
| RPMC-L2 (HF `RS2002/RPMC-L2`) | 699 songs from 35 livehouse shows, 10 Hz OpenL3 embeddings + HSV histograms / scene vector, 42.5 GB; **all 699 cached**; no waveform, so no stems and no beat grid of our own | CC-BY-4.0 | HEAD B events + palette; the only public stage-domain lighting corpus (Skip-BART / SeqLight train on it) |
| **Just Dance gameplay captures (have: `/mnt/lightsem-data/justdance/`, 338 files, 19 GB)** | JD1–JD2020, 22.0 h, full commercial songs (median 236 s), one filmed and stylised coach per routine (multi-coach routines common: 252 of the 453 coach channels), 720p 234 / 1080p 80 / 480p 21, 29.97 fps dominant, stereo audio on all but 6 muted files | YouTube-sourced; Ubisoft choreography and art; **research only** (section 5 legal line) | **Stage 8**: block A/C targets from the coach via 2D pose; the background is a designed visual show (HEAD B-style events and palette); full songs give H-C its structure labels. Not a product training set |
| **xLights free `.xsq` sequences** (xlightsseq.com: ≈ 556 free — 336 Christmas, 98 Halloween, 89 music, 33 misc; ~25–40 h) | per-model effect layers (bars, shockwave, strobe, twinkle, wash, faces) with palettes and parameters on a 25 ms grid — the authored object, remappable onto our groups; **audio not included** (media path only) | sequence = sequencer's copyright, song = separate copyright; vendor files single-home | **Stage 9** authored-light target, only where the song is legally on disk; research only; `.fseq` renders not wanted |
| **Beat Saber lightshow maps** (BeatSaver API, `?tags=lightshow`; zip = `Info.dat` + `<diff>.beatmap.dat` + `.egg` audio) | v3 `lightColorEventBoxGroups` / `lightRotationEventBoxGroups`: group-level events with brightness, colour, strobe and a distribution / phase-spread filter per box (one map: 348 colour boxes, 9 082 events, 21 groups, 3.6 min); exact BPM grid; maps uploaded daily | per-map, varies; DMCA process; mapper-uploaded audio | **Stage 9** authored-light target after the 9.2 gate; research only |
| console show files (grandMA2/3 XML + MVR patch, MagicQ, Onyx) | cues, sequences, effects, presets, fixtures, timecode — the professional object; **no public archive exists** | by contract only | the only commercially clean product corpus; business-development item, section 6.9.1 |
| Raveform (TISMIR 2026) | 4 902 DJ-mix links, 1 423 tracks with expert metrical / functional structure (build, drop, breakdown) | research | EDM section conditioning for the 6.9.4 planner, never a target; no lights |
| own recordings (events, consent) | video → pose | own | the only fully commercial-clean path |

**Video → descriptors pipeline (Anton's YOLO idea).** 2D pose (YOLO11-pose / RTMPose / ViTPose)
normalised by torso length supplies speed, planar laterality, spread, hands vs head split, and
periodicity — i.e. most of block A and the useful part of C. It cannot supply clean jumps, foot
contacts or Laban Weight (depth, camera tilt); those come from 3D lifting (GVHMR, SMPLest-X) or
are dropped. Reject multi-person/occluded frames. Validate the pipeline on AIST video against
AIST++ mocap before trusting it anywhere else. Legal line: datasets with video licensed for
research, or own footage; YouTube-scraped sets are usable for a paper, not a product.

Infra: dedicated torch venv on `/mnt` (uv, cu130); `hf` CLI; ultralytics/RTMPose install;
Perplexity Pro Search quota is back; the one Deep Research call of the month was spent on the rev. 8 pass (2026-09-04 evening).

---

## 6. Stages (each with a kill/keep gate; sized for one RTX 5070 12 GB)

Numbers are preregistered here and re-committed in the vault at run time (`propose_hypothesis`).

### Stage 0a — descriptors, light targets, metrics, oracle renders (1–2 days) — `rb-ls2-00`
FK + descriptor extractor (mocap path) and the 2D pose path on AIST video, cross-validated;
conductor v0 (§3.1) → canonical-rig targets; `lightsem/eval/`: Light-BAS with metronome and
circular-shift nulls, FID_light on **delta features** (onsets, colour jumps, velocities — not
levels), Div_light among samples of the same track, split-half floor. Render 5 real-dance→light
clips. Gates: Light-BAS(real) ≥ 0.9 × metronome; FID split-half × 3 < shift null; Anton's blind
look ≥ 4/5 read as dancing (if not, fix the mapping, not the idea).

### Stage 0b — predictability audit (a weekend of numpy, decides the block assignment) — `rb-ls2-00b`
For every descriptor at 0.1 / 0.5 / 2 / 8 s on AIST++ **and** FineDance: inter-dancer r,
audio R² on held-out tracks, and R² of a **dancer-identity baseline**. A channel enters block A
only if audio beats dancer identity; block C if inter-dancer r is small but the audio-conditioned
*spread* is stable; otherwise noise. Also measure period-locking of the vertical channel at the
tactus and the horizontal channels at 2-beat/bar. Gate: the Burger-predicted channels (quantity
of motion, vertical bounce, head speed, metrical level) beat the identity baseline with audio
R² > 0.3 at 2–8 s; if none does, the dance-derived skeleton is dead and only events survive.

### Stage 0c — late-lights A/B and category-error test (2 days) — `rb-ls2-00c`
(i) Delay a working beat-trigger by 0 / 40 / 80 / 120 ms; blind A/B on screen + headphones;
decides how much of the transient path must be DSP. (ii) 20 tracks: oracle dance-derived DMX vs
a 30-line beat-trigger, 5 raters, **brightness-matched**. If the beat-trigger wins, dance keeps
only blocks A/B (energy + events) and block C is dropped from the product.

### Stage 1 — train the three-block model on AIST++ + FineDance (1 week, GPU) — `rb-ls2-01`
Trunk + heads A/B/C; baselines MSE-all-24 (must collapse: Div < 0.5 × real), beat-trigger,
Stage-F. Splits by track **and** genre **and** dancer (AIST++ same-song-different-dancer is a
choreography leak, not a generalisation test). Gates: block A held-out R² ≥ audit ceiling × 0.8;
block B AP at 0.2 s beats the rhythm baseline with z > 3; Light-BAS(model) ≥ 0.9 × real-mapped
reported next to the metronome; FID_light(delta) < 0.5 × shift null; Div ≥ 0.7 × real.

### Stage 2 — scale data and add the operator head (1–2 weeks) — `rb-ls2-02`
(A) video→descriptor pipeline over InfiniteDance/OpenDanceSet/PopDanceSet/Let's Dance + Oslo EDM
mocap for block A on concert music; gates: EDM collapse rate < 0.10, FID_light on EDM within 1.5×
the AIST++ value. (B) full RPMC-L2 (699 songs) → HEAD B; gates: event AP lift ≥ +0.09, z > 3;
hue KL < per-song-mean; no anticipation claim unless lift at 4 s > +0.05 with CI excluding 0.

### Stage 3 — conductor v0 end-to-end + blind human A/B (3–5 days) — `rb-ls2-03`
20 tracks (10 dance, 10 club/EDM) × {beat-trigger, hue/value-only Skip-BART/SeqLight reproduction,
LightSem v2, real-dance-derived upper reference}; ≥ 20 raters; **brightness-matched pairs (mean
and 95th-percentile LUT), 4–8 s clips, a too-bright foil**; two forced choices: "which is on the
music?" and "which is the lighting engineer?". Gates: v2 > beat-trigger with win rate > 0.6 and CI
excluding 0.5; v2 > hue/value-only with CI excluding 0.5; not significantly below 0.4 vs the
real-dance reference.

### Stage 4 — live (1–2 weeks) — `rb-ls2-04`
Two-path deployment; lookahead sweep L ∈ {0, 40, 80, 250} ms; click-track + photodiode latency
(p50/p99); future-perturbation causality test; Art-Net/sACN verified on a node or BlenderDMX;
festival multi-universe rig adaptor. Gates: transient path ≤ 20 ms; learned path ≤ 65 ms
audio-in→DMX-out at L = 0; Light-BAS at L = 0 ≥ 0.9 × offline; causality test bit-identical.

### Stage 5 — learned conductor, preference fine-tuning (research) — `rb-ls2-05`
SeqLight-style decomposition across fixtures; GRPO/DPO on A/B pairs; additive-projection audit.

### Stage 6 — EDM anticipation, only with real EDM lighting ground truth — `rb-ls2-06`
EDM-98 ships as `pip install edm98==0.1.0` (labels intro/build-up/drop/breakdown as [time, label]
pairs + train/eval splits, MIT; code CC-BY-4.0; github.com/25ohms/EDM-98) with **Deezer track IDs,
no audio** — the tracks must be sourced separately before the structure labeller can be scored.

### 6.7 Stage 7 — the offline pre-rendered path (rev. 6, next work) — `rb-ls2-07-offline-prerendered`

Four hypotheses, queued in the vault, **in this order**. H-A decides whether the other three are
worth running at all; do not reorder for convenience.

**H-A — the look-ahead ceiling. Run this first.**
Same 9.3 M Conformer, same 749k frames, same splits, same 12k steps, same seed. The only changes
are that the attention mask and the causal convolution padding come off and `extract_bus` is
called with `causal=False`. One training run, ~15 minutes.
*Gates:* offline held-out block A R² ≥ +0.04 (double the causal +0.0192); offline Light-BAS
≥ +0.105 (half of real-mapped +0.2106); both numbers quoted next to their `L = 0` counterparts.
*Why first:* it separates "the model is constrained" from "the corpus is exhausted". If an
offline model with the entire track still cannot clear +0.105, the binding constraint is data —
free dance to concert music — and three more architecture experiments would spend a week
confirming that slowly. Sweep `L ∈ {0, 250 ms, whole track}` only if the endpoints differ.

**H-B — zero-phase spectral matching.** No retraining. Filter each predicted channel with the
same 12 Hz zero-phase Butterworth the targets were built with; compare three arms on identical
descriptors — unfiltered, causal one-pole (`calibrate_output_smoothing`, already implemented, the
online fallback), and zero-phase.
*Gates:* `light-event-rate-ratio` within 1.3 of real-mapped; no channel above 2× corpus above-6 Hz
energy; FID below the unfiltered arm; **diversity not below the unfiltered arm** — a rate win
bought with dead light fails, and smoothing trivially reduces the rate, so this gate is the one
that matters.

**H-C — whole-track structure conditioning.** Section label (intro / build / drop / breakdown /
core) and signed seconds to the next drop as conditioning channels. Labels from whole-track
Laplacian segmentation plus the existing EDM role labeller, validated against the Oslo record's
hand-marked breakdown/buildup/drop times — the only place in this corpus where the truth is
written down. Retrain with and without, everything else identical.
*Gates:* Light-BAS above the unconditioned model with a paired-bootstrap CI excluding zero;
rendered drop/breakdown intensity ratio within 1.3× of the corpus 1.68–2.53×; and a **shuffle
control** — section labels permuted across tracks must destroy the gain, or the model is using the
channel as a bias term and the structure claim is void.

**H-D — operator head with anticipation.** The cheapest of the four: features extracted, causal
head trained, the change is the attention mask. Section 9's ban on anticipation claims was written
for a live product where anticipation had to be inferred; pre-rendering removes the reason for the
rule but not its standard.
*Gates:* event AP lift at a +4 s horizon > +0.05 with CI excluding zero (the number preregistered
in section 6, Stage 2B, unchanged); offline marginal lift above the causal head's +0.3145 on the
same 86-show test split; time-shift control still peaks within 2 s of zero.

### 6.8 Stage 8 — the Just Dance corpus (rev. 7, `rb-ls2-08-justdance`)

What is on disk (manifest `/mnt/lightsem-data/cache/justdance_manifest.json`): 338 captures,
22.0 h, all with an audio stream, 6 of them silent; h264 176 / AV1 147 / VP9 15 (AV1 decodes
through PyAV since `c7c0935`; `av` is installed in the torch venv by hand and is not yet in a
requirements file); constant frame rate on all 338; 29.97 fps for 200 files, 59.94/60 for 85,
25 for 33. One coach per routine except five duet/trio/crew files. 15 songs appear in more than
one version (30 files), so **splits are by song, never by file.** Some audio is Ubisoft's
in-house cover (Studio Allstars, Sweat Invaders, …), not the released master — harmless for
pairing, which needs synchrony, not the original.

What the game is, from sources (Perplexity pass 2026-09-04; Switch-JustDance arXiv:2511.17925
is the one peer-reviewed corpus built from this material): the coach is a **filmed dancer** on
green screen, stylised in post, white body paint standard from JD4; a minority of JD2018+ maps
are CGI. Coach and background are one **pre-rendered video** to which pictograms are timed, so
audio and motion are locked inside the game; only the capture chain can add an offset. Wiki and
emulator threads say the coach layer is 30 fps and 60 fps captures are UI composites; **measured,
that is false for these files**: over 400 frames of the coach region, even/odd frame-difference
ratios are 1.01–1.07 on the 60 fps files against 1.01–1.03 on 29.97 fps controls, with no
duplicated frames, so the 60 fps captures carry real per-frame motion and are used at their
native rate (descriptors resample to 40 fps as always). The coach faces the camera and the
player mirrors it — for us that is a performer seen from the audience, the same geometry as the
frontal AIST video the pose path was validated on, so no sign flip is expected; but whether the
coach footage itself was flipped in post is asserted from documentation, not measured, and is
recorded that way (`meta["view"] = "frontal"`, provenance *asserted*). It only matters for
channels whose sign survives into a loss; see the laterality note in 8.1. Ubisoft's
video artists **hand-animate the background lighting to the music**, and Gold Moves in
JD2015–2018 "often occur at the start of the chorus or drop" with a frame-filling flash — a
designed highlight marker readable from the pixels. The game's own timing files (moves, beats)
exist only as internal tools or ripped UbiArt assets and are **not used**; the Just Dance wiki's
per-song Difficulty and Effort ratings are legal metadata and supply the confound baseline.

Why this corpus is different from AIST++, and the one way it might not be. AIST++ is 60
structure-less loops with genre-typical improvisation; its movement is explained by genre
(section 1.2). Just Dance is full songs with verses, choruses, bridges and drops, and the
choreography is made to *that* song section by section — a designed response to structure,
which is the lighting designer's job description and exactly what H-C needs. The way it can
fail is the same way AIST++ failed: if between-song movement is explained by the song's genre
or difficulty rating rather than by its audio, the corpus trains a genre prior. That is a
measurement, not an assumption, and it is the first thing Stage 8 makes.

Three signals from one corpus, in the order they are tested:

1. **Coach motion → block A/C targets** (2D pose path, section 5). The channels that transfer
   from video are the vertical and energy family (section 1.3 G3: vertical 0.79–0.87, shape
   0.78–0.81); twist, hip yaw and footfall rate do not, so block A on this corpus is
   `vert_bounce_amp`, `vert_period_s`, `com_speed`, `qom`, `head_speed`, `spread`, with
   `hip_yaw_speed` and `footfall_rate` masked out of the loss for Just Dance clips. The coach is
   mirrored relative to the player; laterality signs are already sampled, and `meta["mirrored"]`
   records it. Pose on the stylised coach works: under a uniform protocol (`start_s` 40, 750
   frames, imgsz 960) 6 of 10 probed clips reach coverage ≥ 0.98 with keypoint confidence
   0.91–0.95; the earlier probe's failures were the AV1 decoder and title screens, not the
   coach. Two clips are genuinely weak (Funkytown 0.31 with 24 tracks; a 480p JD1 clip 0.57),
   and the pictogram strip (bottom right, small human figures), the score band (top), lyrics
   (bottom left) and the star meter (left) were the suspects. Measured on Funkytown: masking the
   bottom 18 % and top 12 % moves coverage 0.31 → 0.48 and tracks 23 → 21, so the mask is kept
   but does not explain that clip. The Gold Move flash fills the frame around the coach and cannot be
   masked; frames with a frame-wide brightness spike are marked low-confidence and the spike
   itself is kept as the highlight marker. **Multi-coach routines are channels, not noise**
   (Anton, 2026-09-04): a duet, trio or quartet is several dancers hearing the same music, the
   structure AIST++ has, so every coach becomes its own descriptor entry (`dancer_id` = coach
   index, `music_id` = the capture). Coaches are followed as parallel tracks; the coach count
   comes from the game's own pictogram strip, one small figure per coach, because box size
   cannot tell a coach from a background extra (Havana's two extras persist the whole song).
   Filenames name only 5 multi-coach files; the frames show more (Wild Wild West and On Stage
   are quartets, Don't Wanna Know and HandClap trios). Gate each coach on coverage ≥ 0.8. Lead-in menus and the results screen are cut by
   `start_s` and by the end of the coach track.
   **Trial, 2026-09-04 (`extract_justdance.py trial`, six clips × 900 frames).** YOLO11m,
   YOLO26m at 960 and 640, YOLO26x at 960, and RTMPose-m on CPU as an independent reference.
   On clean solos every arm agrees with YOLO26x within 0.03–0.2 torso lengths and RTMPose
   within 0.07, so the keypoints are trustworthy; YOLO26's trajectories carry 3–10× less
   above-6 Hz energy than YOLO11's on the same frames. The two hard clips are detection, not
   pose: the Funkytown alien costume is found by RTMPose's separate detector in 99.6 % of frames
   and by YOLO26m at conf 0.25 in 52 %; a 0.10 floor plus fragment merging lifts it to 72 %.
   Selection is tracker-free: every coach-sized figure is followed as a track anchored on the
   mid-shoulder keypoint (a full-body and an upper-body box of the same coach sit 0.4–0.8
   heights apart and would otherwise split one figure into alternating fragments), duplicates
   are suppressed by overlap, containment and anchor distance, and sequential fragments are
   merged at the junction across camera cuts. Coaches are the big, sharp, persistent tracks
   (coverage ≥ 0.6, height ≥ 0.8 and confidence ≥ 0.92 of the best); YOLO cannot see the
   pictogram figures, so the strip's hue count is recorded as a diagnostic only. Result on the
   six: solos and multi-coach routines at ≥ 0.998 coverage per coach with the right count
   (HandClap 3, Wild Wild West 4, Havana 1 with two extras dropped). Bounce period on the
   beat where the coach bounces periodically (Wild Wild West 1.04×, Funkytown 2.04×).
   **Production**: YOLO26m-pose, conf 0.10, three GPU workers, 60 fps files at stride 2. The first
   20 files ran at imgsz 960 (~98 frames/s aggregate once the GPU was free, 7.5 h for the rest);
   the remaining 312 run at 640 (~182 frames/s, ~4 h), which the trial showed agrees with the
   largest model as well as 960 does. `meta["imgsz"]` records which. Launched 15:00, restarted
   at 640 at 16:05.
2. **Background visuals → a designed show.** With coach and UI masked, the background is an
   art-directed light show: beams and flashes on the drop, sparkles on gold moves, palette
   changes at the chorus. Extract per-frame brightness and HSV histograms exactly as RPMC-L2
   stores them, detect scene changes and palette shifts, and run the operator-head test
   unchanged: audio's marginal lift over the rhythm-of-past-changes baseline. If it clears the
   Stage 2B bar, HEAD B gains 22 h of pop/EDM/latin designed visuals next to 699 concert shows.
3. **Full songs → structure labels for H-C.** Whole-track Laplacian segmentation and the EDM
   role labeller run on real songs here for the first time; the coach's energy contrast between
   chorus and verse is the corpus's analogue of Oslo's drop/breakdown 1.68–2.53×, and the
   background's brightness contrast is a second, independent reading of the same sections.

Order of work and gates:

- **8.1 Usability manifest** (done for audio and codecs; pose coverage pending): drop the 6
  silent files from the paired corpus, not only from the audio side; song-key dedupe; per-file
  `fps`, `start_s`, `coverage`, `n_coaches`, `view`. **Result, 2026-09-04 19:40:** extraction complete
  (332 captures, 3.3 h at 640 after the first 20 at 960); 453 coach channels across 294 songs
  pass the 0.8 gate over the coach's active span, 27.9 h of motion, median coverage 0.93, no
  capture without a usable coach. The gate as written fails on the letter: 196 single-coach
  songs, not 250, because duets, trios and quartets are far more common than the filenames
  said (252 of the 453 channels are from multi-coach routines). Laterality: a uniform flip of the coach
  footage would give every Just Dance clip a laterality sign opposite to AIST and Oslo with
  nothing in the data to reveal it, so until one human glance at a one-handed move settles the
  flip, sign-carrying channels (`lateral_pos`, `laterality`) from this corpus stay out of any
  regression loss — which costs nothing, because their sign is sampled in block C by design.
  Gate: ≥ 250 usable single-coach songs with coverage ≥ 0.8.
- **8.2 Predictability audit on held-out songs** (the decision; `rb-ls2-00b` recipe unchanged):
  audio → coach movement curve at 0.5 / 2 / 8 s, R² on held-out songs against a song-identity
  baseline and with genre partialled out; chorus/verse movement ratio with CI. *Gate:* held-out
  song R² ≥ +0.20 on the vertical/energy family at 2 s (Oslo gave +0.30, AIST++ ≤ 0.21 and
  collapsing under genre), the genre-partialled correlation not collapsing to within ±0.1 of
  zero, chorus/verse ratio CI clear of 1. If this fails the coach motion is another AIST++
  and only signals 2 and 3 survive from the corpus.
- **8.3 Sync check** before any block B use: the choreography is on the beat by design, so the
  vertical-bounce phase against the beat grid should sit at one phase across files; a per-file
  scatter flags capture offsets. Any fitted offset must transfer to a channel it was not fitted
  on (section 1.2's refutation still stands).
- **8.4 Visual-show test** (signal 2): event AP lift ≥ +0.09, z > 3 on held-out songs; hue KL
  below the per-song mean; time-shift control peaking at 0.
- **8.5 Train** the offline trunk (section 6.7 H-A variant, structure channels from H-C) on
  AIST++ + Just Dance with a corpus token, Oslo held out entirely as the concert-domain test.
  *Gate:* held-out-song block A R² on Just Dance above the AIST++-only model, **and** Oslo
  shared-movement R² not below the AIST++-only model — the second is the one that matters,
  because Oslo is real people hearing real EDM and Just Dance is one performer per song.
  The test exists (`run_oslo_transfer.py`: both mixes rendered from the parallel session's
  placements, the model's predicted energy against the crowd's mean speed inside each track
  slot). **AIST++-only baseline, 2026-09-04:** raw r(model qom, crowd) +0.07 over eight slots
  (+0.30 to +0.37 on the well-placed Ladykiller and Unlock Down slots, −0.48 on Joyride in
  mix 2, whose placement margin is −0.16), held-out-track ridge from the model's channels
  R² −0.12 [−0.21, −0.05] against +0.30 from published audio scalars. Two slots are badly
  placed and drag the mean; the well-placed five are the ones to read.
  **Just Dance-trained model (no structure channels, H-B smoothing), 2026-09-04 20:30:** raw
  r(model qom, crowd) **+0.335** over the same eight slots — Ladykiller +0.73 / +0.63,
  Unlock Down +0.45 / +0.54, Icarus +0.34 / −0.05, Joyride +0.41 / −0.38 — with no fitting at
  all; the two negatives are the badly placed slots. Held-out-track ridge from its channels
  R² −0.08 [−0.36, +0.16], r +0.24, not below the AIST++-only model's −0.12. **The 8.5 concert
  gate passes**: a model trained on filmed game coaches predicts the energy of a real crowd
  hearing real EDM, and the AIST++-trained one does not.

Split of labour (2026-09-04): the parallel session owns video → descriptors (pose2d, masks,
coach selection, the motion cache); this session owns the audio bus on the captures, the
structure labels, the audit, the trunk and the vault records.

**Dependencies and things that are not blocked by this.**
Oslo audio reconstruction (a parallel effort) caps at roughly ±1 s alignment because the record
publishes 12 scalar channels and no audio, and the DJ desk moves exactly the channels published.
±1 s is usable for block A envelopes at 2–8 s EMA and useless for block B or beat phase, so the
corpus must carry per-track alignment error as an explicit field and downstream code must gate
channels on it. The Oslo R² +0.30 result does **not** depend on that reconstruction — it is
computed from the published series, already on the mix clock. The human A/B materials
(`outputs/stage0c/`) are unaffected by everything in this section except `category_model/`, which
was re-rendered on 2026-09-04 (evening) from `light_trunk_v3causal_smooth.pt`, the shipped causal
model with H-B's smoothing, and is ready to rate: model against the beat trigger, ten
brightness-matched pairs plus ten foils, same two questions as section 1.5.

### 6.9 Stage 9 — from decent to designed: authored light as the target, the conductor as the model (rev. 8, `rb-ls2-09-designed-light`)

**Where rev. 7 left the product, in the numbers that matter.** On held-out Just Dance songs the
shipped model's light lands on the beat better than the coach's own motion does (Light-BAS
+0.135 [0.121, 0.150] against +0.087 for the real coach and +0.092 for the 30-line beat
trigger), its feature distribution is still 4.5× off a real dancer's (FID_light 36.5 against a
split-half floor of 8.1), and the descriptors it regresses are not predictable from audio at all
(held-out block A R² −0.22; two coaches on the same song agree at r 0.22–0.42, which is the
ceiling of any dance-trained model). Look-ahead did nothing (H-A), section channels into the
per-frame model hurt (H-C), anticipation on the operator data failed (H-D), and the largest
lever found so far was post-hoc smoothing (H-B). Anton's reading, 2026-09-04 evening: the result
is decent and clearly short of a human designer.

**Diagnosis, shared by five outside passes and the two 2025–26 stage-lighting papers.** The
bottleneck is not the trunk, not causality, and not the amount of dance. It is three things:
(i) the *target* — dance is a weak proxy for lighting, and coaches disagree with each other;
(ii) the *hand-written conductor*, which owns every artistic decision (palette, fixture roles,
cue grammar) while the model only steers 30 envelope channels; (iii) *no human preference*
anywhere in the loop. Stage 9 attacks all three, in that order, and it does it with authored
light as the target rather than with more dance.

**Provenance of this section.** Anton's three directions (richer audio features, more concert
corpora, transformer/multimodal methods) and his mixture-of-light idea were run through one
Perplexity deep-research pass and four reasoning passes (audio decomposition, datasets in two
halves, multimodal methods, mixture of light) with the full project brief attached
(answers kept verbatim with the brief and sources under `docs/research/stage9/`; the rhythm-game half
of the dataset pass returned empty twice and is covered by the deep pass plus a direct BeatSaver probe), against the MIT MAS.S60 `multimodal-ai` skill, and against
two papers read in full: **Skip-BART** (arXiv 2506.01482, ICLR 2026: the RPMC-L2 authors treat
lighting as generation, BART over OpenL3 at 10 Hz with a music→light skip connection; 38 raters
put it at 4.35 vs ground truth 4.51 vs rule-based 2.67 on a 7-point scale) and **SeqLight**
(arXiv 2605.03660, May 2026: Skip-BART predicts the *full* hue/value distribution of all lights,
then a goal-conditioned imitation-learned policy decomposes it onto eight simulated point lights
per venue with no per-fixture demonstrations; 30 raters, 4.54 vs Skip-BART 3.90 vs rule 2.70;
fixed light order, saturation fixed at 100 %, hue and value only, no pan/tilt/gobo). SeqLight is
the closest published system to what Stage 9 builds, and its two-stage split (venue-independent
intent, venue-specific realisation) is the design this section adopts.

#### 9.0 What the passes said, and what is kept

| direction | verdict | kept | dropped |
|---|---|---|---|
| 1. audio decomposition | only features that *disambiguate sources the mix collapses* can add anything; the 92-channel bus already holds Burger's mix-level flux bands, RMS, onset and beat phase; a new family that is mostly "section energy" fails the way H-C did | HTDemucs 4-stem envelopes/onsets/2-band flux; per-stem Burger flux for the head/hands split; vocal *activity* (not lyrics); bar phase (not section labels); one frozen music-encoder layer group PCA'd to 32–64 d, only against a same-width random projection and a song-ID leak probe | chords/key/tension, LUFS/LRA, mix timbre, MSAF/SongFormer/EDMFormer *labels* as bus channels, Jukebox, CLAP, RoFormer separators (until stem bleed is shown to cap onset AP) |
| 2. datasets | there is **no public, commercially clean corpus of full-song professional DMX programming**; RPMC-L2 is the only public stage-domain set (SeqLight adds none); authored light exists at scale in two hobby/game ecosystems | RPMC-L2 all 699 songs (now cached); xLights free `.xsq` sequences; Beat Saber lightshow maps; Raveform DJ-mix structure as *conditioning*; console show files by partnership | laser (ILDA/Pangolin), pyro (Finale 3D), fountain, VJ packs, SoundSwitch autoscripts (a beat-trigger, not a human), MVD-MIX except as a weak proxy; scraping anything for a product |
| 3. transformers / multimodal | a contrastive audio↔light critic captures only *shared* information, which on this data is beat + loudness — the mode the model already occupies; an AR "light LM" does not replace the three-head design (Skip-BART regresses absolute hue/value, needed 2×4090+A100, and needed a skip connection to lock rhythm); Lodge-style two-level planning belongs on the *conductor*, not in the per-frame model | factorised critic with beat-trigger negatives as a best-of-N reranker; section planner writing conductor knobs; discrete look tokens as a planner; Bradley–Terry on ≤ 12 conductor knobs from 200–400 pairwise ratings; frozen encoder only as an offline dump | AR light-LM as decoder; RL on the flow head before the critic is validated; contrastive pretraining as the objective; diffusion forcing as a new head |
| 4. mixture of light | of the three architectures — (a) one head per fixture *type*, (b) fixture-token set transformer, (c) group-level effect parameters + deterministic engine — **(c) wins**: it is how shows are programmed (groups, palettes, effects engine, cues), it is permutation- and mirror-safe by construction, and it is the only one trainable from data that has *no per-fixture DMX*; type is an output *mask*, not an expert (MAS.S60: hard-partitioned experts under-share) | group-effect head with type masks; loss on a rig-invariant observable through a cheap differentiable histogram renderer (SeqLight's move); random-rig dropout; rig-transfer test | type-MoE in the trunk; fixture-token decoder until console logs exist; learning pan/tilt/gobo from HSV histograms (invisible through haze and cameras); set-matching losses |

The one sentence the deep-research pass ends on, and it agrees with everything measured here:
*another music-to-motion transformer will not close the gap; the project needs authored lighting
intent, a learned hierarchical planner, and a preference model that can tell deliberate
composition from technically synchronised motion.*

#### 9.1 Targets: authored light, three corpora, one format

Every Stage 9 target is a **group-level light event stream plus per-group intensity/colour
envelopes at 40 fps**, the same object for all three corpora so one audit script scores them all.

- **RPMC-L2, 699/699** (`cache/rpmc_all`, CC-BY-4.0 on Hugging Face; OpenL3 512-d at 10 Hz +
  the 6-d scene vector; **no waveform**, so no stems and no beat grid of our own — the operator
  head stays on its published features). One mixed "fixture group", 10 Hz, hue mode + value
  with S fixed at 100 %. Baseline to beat: with-hazard event lift 1.81× (rhythm-only 1.25×,
  Δ +0.56 [0.38, 0.74]), audio-only Δ +0.10 [−0.05, +0.26].
- **Just Dance visual** (`cache/justdance_visual`, 332 captures, wav on disk): background HSV
  histograms with coach and UI masked; exp-016 lift 3.17× vs shift 1.48×, Δ +1.69 [1.19, 2.27],
  shift peak 0. The one corpus here with waveform *and* an authored visual target, so it is where
  H-G's stems are priced.
- **xLights free `.xsq` sequences** (xlightsseq.com lists ≈ 556 free: 336 Christmas, 98
  Halloween, 89 music, 33 misc; ~25–40 h): XML with per-model effect layers (bars, shockwave,
  strobe, twinkle, wash, faces) with palettes and parameters on a 25 ms timing grid — *the
  authored object itself*, remappable onto our groups. **Audio is not included**: the file
  stores a media path and the song must be owned; sequences are the sequencer's copyright and
  vendor files are single-home licences. Research only; keep only sequences whose song is
  legally on disk; drop singing-face visemes. Conversion: parse `ElementEffects` →
  `{t_start, t_end, model, effect, params, palette}`, map models to fixture groups (roofline,
  matrix, tree, megatree → wash, pixel bar, spot, blinder), sample at 25 ms, hold at 40 fps.
  Falcon `.fseq` renders are *not* wanted (per-channel 8-bit frames, effects destroyed).
- **Beat Saber lightshow maps** (BeatSaver, `api.beatsaver.com`, `?tags=lightshow`, zip with
  `Info.dat`, `<Difficulty>.beatmap.dat`, `.egg`/`.ogg` audio). Verified on one map
  (`414ae`, v3 schema, 103.5 BPM, 3.6 min): `lightColorEventBoxGroups` 348 boxes over 21
  environment group ids carrying 9 082 events with beat offset, colour index, brightness and
  strobe fields and a *distribution/phase-spread filter per box* — i.e. a group effects engine,
  exactly the abstraction 9.3 emits; `lightRotationEventBoxGroups` 126 boxes (the moving-light
  analogue); the v2 `basicBeatmapEvents` lane (5 fixture groups: back lasers, rings, left/right
  lasers, centre) is 16 events in this map, so the v3 groups are where the show is. Lightshow
  maps are uploaded daily (ids 5417b–54184 on 2026-09-04). Per-map licences vary, BeatSaver runs
  a DMCA process, audio is the mapper's upload: **research only**, and one corpus gate (9.2)
  decides whether it is a lighting corpus or a beat-trigger corpus. Conversion: environment
  group id → fixture role (per environment table), box filter → group cardinality and phase
  spread, events → per-group intensity/colour at 40 fps on the map's BPM grid (BPM is exact,
  so alignment error is zero — the cleanest beat grid in the project).
- **Console show files** (grandMA2/3 XML export + MVR patch geometry, MagicQ, Onyx): no public
  archive exists; the deep pass's verdict is that *fifty licensed shows with cue hierarchy beat
  tens of thousands of dubious videos*. This is a business-development item, not a scraping
  item: offer programmers an export/backup tool, pay for a training licence, prefer
  production-owned or commissioned music, keep cue names, executors, palettes and effects, and
  render everything onto the canonical rig. Recorded here as the only route to a
  commercially clean product corpus; nothing in Stage 9 depends on it.

Not targets: Rock Band / Guitar Hero `VENUE` tracks (proprietary assets and redistributed
recordings; a legally clean corpus would have to be commissioned from former RBN authors),
lasers, pyro, fountains, SoundSwitch autoscripts, VJ loop packs.

#### 9.2 The predictability gate every corpus must pass first (H-H)

The cross-modal audit (`rb-ls2-00b` recipe, on the new target format) runs before any training
on a corpus, split by song and by show, with the two nulls the outside passes agree on:

1. **Time-shift null**: same audio, lights shifted by a random offset in {±2, ±4, ±8} s; event
   AP must collapse.
2. **Beat-trigger baseline**: our own `conductor/beat_trigger.py` (beats and bar onsets fire a
   flash / colour jump) — this is SoundSwitch's autoscript and Beat Saber's auto-lighting; it
   will look decent on drops.

*Gate G9.2*: AP_audio − AP_shift ≥ +0.15 **and** AP_audio ≥ AP_beat-trigger + 0.05, on both the
event stream and the per-group envelope R², causal and offline reported together. A corpus that
fails is fitting tempo, not lighting authorship, and stays out of every loss. xLights and Beat
Saber are audited separately; Beat Saber additionally gets a *group-count* control (does the
number of active groups track anything beyond loudness).

#### 9.3 The model: a group-effects decoder replaces the rule table (H-I, the mixture of light)

Anton's mixture-of-light idea, in the form the fourth pass and SeqLight both land on:

```
audio bus ──► frozen trunk + operator head ──► GROUP-EFFECT HEAD (0.5–2 M params, shared,
              type-conditioned FiLM / output mask)
              per group role g ∈ {wash, spot, beam, strobe, blinder, pixel bar, laser*, haze}:
                intensity envelope (gain-normalised), palette / hue, effect id,
                rate in beats, phase spread, participation density, look-change hazard
         ──► DETERMINISTIC EFFECTS ENGINE (today's conductor with the if-then table removed):
              binds groups to whatever is patched, sorted by (type, stage x), fans the waveform
              over the group with the phase spread, keeps the H-B one-pole per channel
         ──► DMX ──► EGL renderer
```

- **Why not per-type experts or a fixture-token transformer.** Six moving-head slots concatenated
  is a canonical-rig overfit that cannot chase or mirror; a set transformer is the right geometry
  but the wrong *target* — the corpora hold mixed HSV and group events, never pan/tilt, so a
  decoder that emits pan/tilt/gobo is inventing degrees of freedom the loss cannot see. Type
  enters as a parameter mask (a wash has no gobo, a strobe has no CMY), not as a router.
- **Loss on a rig-invariant observable.** Ground truth is not a set of fixtures, so no
  Hungarian/Chamfer matching. A cheap differentiable renderer (each active group an additive
  coloured blob; spots as 2-D Gaussians on the stage plane) produces the per-frame HSV histogram
  and the same delta-light features `eval/fid.py` already uses; train against RPMC-L2 histograms,
  Just Dance visual histograms and, once 9.2 passes, xLights/Beat Saber group envelopes.
  Per-type terms: wash/LED → hue+value histogram, never absolute level; strobe/blinder → onset
  times and duty cycle, not brightness; spot/beam → histogram contribution plus an effect-rate
  term, pan/tilt engine-procedural until console logs exist; pixel bars later (needs a 1-D
  renderer). Auxiliary look-change cross-entropy from the operator head. Random rig: dropout
  groups, jitter cardinality, shuffle positions within a type, sometimes remove the strobe —
  because the loss lives on the render, two washes swapping places costs zero, which is the
  invariance wanted. **No distillation of the current conductor's DMX**: that would clone the
  bottleneck and turn random rigs into noise.
- **The dance descriptors are demoted.** They stay as a training signal through the frozen
  trunk (the product claim is unchanged) but the decoder reads the trunk and the operator head,
  not the 30 channels — a decoder sitting on channels with R² −0.22 clones a bad clock.
- **Gate G9.3**, held-out songs, mean Value matched to the reference and onset density capped to
  the human band: FID_light vs the hand-written conductor must beat a shuffled-donor control by
  > 2× (H-C fake-won 36.5 → 27.6 with shuffle at 33.1; that trap is pre-registered away);
  Light-BAS not below the conductor *without raising onset density* (BAS bought by flashing every
  beat is a fail); brightness-matched human A/B on "fits the music" vs the conductor with a
  win-rate CI excluding 50 % at n ≥ 20 pairs, "looks like an engineer did it" and a new
  "too metronomic" question kept separate; **rig transfer**: freeze the decoder, patch an unseen
  rig (different N, no strobe, extra LED bars) and beat channel-copy on HSV-histogram distance to
  the canonical render. Miss any two and the conductor was not the bottleneck: stop decorating
  it and go get stems or console logs.
- **Cost**: MVP in a week on the 5070 (frozen trunk, tiny head, existing expander, histogram
  renderer in torch; no EGL in the loop; no pan/tilt learning).

#### 9.4 The planner: section-level intent written onto the conductor knobs (H-J)

H-C refuted *section channels into the per-frame model* (paired BAS −0.028, FID gain explained
by a shuffled donor). It did not test the thing a lighting designer actually does, which is
decide the *look* of a section before it starts and hold it. Lodge-style two-level: an
audio-only, offline planner (bars and sections; Raveform-style role labels as conditioning; may
look one bar ahead — the show is pre-rendered) emits scene family, palette, intensity class,
strobe density and transition type per section as *knobs of the 9.3 engine* (or of the current
conductor while 9.3 is built); the frame model is untouched. *Gate G9.4*: paired Light-BAS not
below the no-planner model; FID_light and the human "feels intentionally structured" question
must beat a **shuffled-section-plan control** (the real plan applied to the wrong sections) and
a causal planner; time-to-drop is never an input (the H-D leak pattern).

#### 9.5 The critic: factorised, beat-trigger-negative, best-of-N first (H-K)

A dual encoder C(audio, light) on **bar-synchronous** tokens (audio = the 92-channel bus; light =
HSV histogram + look-change flag + strobe density, not raw dimmer), positives from RPMC-L2, Just
Dance visual and the 9.2 corpora, and three negative families — circular time-shift ≥ 1–2 bars,
other-song light on this audio, and **beat-trigger light on this audio** (without the third,
InfoNCE's cheapest solution is a beat detector and the reranker rewards the mode the model
already occupies). Two heads, C_R (redundant: beat, RMS) and C_U (unique: look-change timing,
palette jumps, section contrast); rank by C_U − λ·C_R, and require the ranking to *change* when
the critic is projected to its nearest additive model (MAS.S60 lecture 07: otherwise it learned
RMS and onsets). Use: best-of-N over the rectified-flow samples of block C (N ∈ {8, 16}, scored
through the conductor, not the renderer). *Gate G9.5*: critic accuracy real-vs-beat-trigger
clearly above chance (real-vs-shift alone is not enough); additive projection changes the
best-of-8 order; best-of-8 beats random-of-8 on held-out human "fits the music". Only after that
gate does the critic become a *reward*, and then on the conductor knobs (10–20 scalars) first,
never on the flow head while BAS is already above the coach's +0.087.

#### 9.6 Human preference on the knobs (H-L)

Bradley–Terry P(i ≻ j) = σ(wᵀ(φ_i − φ_j)) over the ≤ 12 knobs of the rule table (or the 9.3
engine), regularised toward today's defaults, fitted separately for the two questions that
dissociated in section 1.5 ("fits the music" tied, "looks like an engineer did it" 8–1–1).
Detectability: an 8–1 gap shows at ~30 decided pairs; a 60 % true win rate needs ~200 decided
pairs to separate from 50 %; 55 % needs ~1000. So the study is 200–400 pairs, brightness-matched,
generated by `render_song_demo.py`-style pairs on held-out songs with knob perturbations, rated
by Anton and whoever he recruits. *Gate G9.6*: held-out preference accuracy > 60 % at n = 200;
the fitted w must not raise BAS toward the metronome; a fresh batch prefers fitted knobs over the
defaults. These ratings are *not* spent on DPO of the flow head.

#### 9.7 Richer audio, gated by audit before any training (H-G)

Order from the first pass, each with the one experiment that decides it, all on song-split
beat-synchronous ridge / gradient boosting against the 92-channel bus, with a shuffled-donor and
an L = 0 counterpart; nothing enters the Conformer until the audit beats the bus:

1. **HTDemucs 4-stem** envelopes + onsets + 2-band flux (skip RoFormer until bleed is shown to
   cap onset AP; Demucs chunks are non-causal ≈ 8 s, so the causal counterpart is the chunked
   stem). Decider: Δ event-AP on Just Dance visual over the bus, and Δ 2 s R² on coach hand
   speed vs footfall. Threshold +0.02 AP with a source-specific split (drums → feet/head,
   hats/vocals → hands), else dead. Cost ≈ 4–8 h for 28 h of wav on the 5070, ~6–8 GB.
   `demucs` is not in the torch venv yet (`uv pip install demucs`).
2. **Per-stem Burger flux** as the explicit head/hands prior — free after 1; kill if the stem
   split adds nothing over mix flux.
3. **Vocal activity / phrase onsets** from the vocal stem (no lyric tokens: text overfits song
   identity, Whisper hallucinates, JD is multilingual); keep only if vocals win a subset drums
   lose; Oslo as the negative control (EDM, near-zero expected).
4. **Bar phase** (All-In-One / Beat This! downbeats) as sin/cos next to the kick-flux beat phase;
   70 % of coaches bounce at 1× or 2× the beat, so *bar* is the open question for metric level
   and wash/pan-at-the-bar. Kill if ΔR² < 0.02. Section labels are not re-imported.
5. **One frozen embedding**, PCA'd to 32–64 d — MuQ layers 9–12 or MERT-330M layers 10–13
   (semantic/hue) plus MERT ~15–16 (rhythm), never the last layer, never Jukebox or CLAP —
   against a same-width random projection of the bus and a song-ID kNN leak probe (pretraining
   corpora contain these pop hits). Kill if residual ΔAP < 0.02 after subtracting the song-mean
   embedding: it is a hash of the catalogue. (H-M in the vault; lowest priority.)

RPMC-L2 has no waveform, so the stem arms are priced on Just Dance visual, the coach split, the
Oslo mixes (±1 s alignment, 2 s envelopes only) and, after 9.2, Beat Saber (audio in the zip).

#### 9.8 Order of work, cost, and what this stage does not do

| week | item | GPU | decides |
|---|---|---|---|
| 1 | H-G audit (Demucs on the JD wavs, features, ridge) ‖ H-H pull + audit (BeatSaver `lightshow` top-rated ~300 maps; xLights free `.xsq` only where the song is on disk) | an evening | which features and which corpora exist for the project |
| 2 | H-I MVP: histogram renderer, group-effect head on the frozen trunk, RPMC-L2 + JD visual targets, random-rig dropout | 1–2 days | whether the conductor was the bottleneck |
| 3 | H-J planner on the knobs; H-K critic + best-of-8 | 1 day each | structure and reranking |
| open | H-L preference study (200–400 pairs; needs raters) ; H-M frozen encoder ; console-file partnership | — | human-level or not |

Every number is quoted with its L = 0 counterpart, priced against the beat trigger and the
hand-written conductor, and split by song and show. Human A/B stays brightness-matched with the
over-bright foil.

**Not in Stage 9:** an autoregressive light LM as the decoder (Skip-BART regresses absolute
hue/value, which is a recorded rule-out here); RL or DPO on the flow head before G9.5 passes;
mixture-of-experts by fixture type inside the trunk; section labels, chords, loudness range or
timbre descriptors as per-frame bus channels (H-C's trap); Jukebox or CLAP; a fixture-token
decoder before per-fixture console data exists; more dance corpora as a first lever (the ceiling
is the coaches' own agreement, r 0.22–0.42); any product training on BeatSaver, xLights, RPMC
video or YouTube material (research only, section 5); scraping. Oslo remains the concert-domain
test for every arm.

---

## 7. Evaluation protocol (what convinces a sceptic)

| metric | null / baseline | trap it guards against |
|---|---|---|
| Light-BAS reported as Δ vs metronome | metronome (upper), circular shift (lower) | a metronome maximises it |
| event AP / F1 at 0.2 s and 1 s | rhythm-of-past-changes | persistence, level-following |
| FID_light on delta features | shifted motion, split-half real | FID on levels rewards holding a look |
| Div_light among samples of one track | real-dance value; MSE baseline | mode collapse hidden by FID |
| EDM collapse rate | real 5th percentile | in-the-wild generalisation |
| audio→DMX latency p50/p99, causality test | — | offline-only results, leaked future |
| human pairwise win rate, brightness-matched, two questions | beat-trigger, Skip-BART/SeqLight, real-dance upper ref, too-bright foil | raters prefer brighter |

Splits by track, genre and dancer. Effect sizes with CIs, never z alone.

---

## 8. Risks and the cheapest test for each

1. **Category error (dance ≠ light).** Stage 0c (ii). If the beat-trigger wins, keep only blocks A/B.
2. **Sampling a prior.** Stage 0b audit; any channel whose inter-dancer r ≈ 0 at the emitted
   timescale and whose audio R² does not beat dancer identity is removed from the loss.
3. **Late lights.** Stage 0c (i); transient path on DSP regardless of the model.
4. **Dataset music is studio dance, product music is club/rock/EDM.** Oslo EDM mocap + the
   video→descriptor pipeline on concert footage; EDM collapse rate as the watch metric.
5. **Metric gaming.** §7 nulls; brightness-matched human A/B as the only headline.
6. **Licences.** InfiniteDance/OpenDanceSet research-only; commercial product needs own footage
   (consent) or licensed motion; Oslo sets are CC-BY-4.0 but music must be sourced separately.

## 9. Not in this plan

Regressing raw kinetic energy at 10 Hz; regressing absolute light levels; scraping YouTube; a
dancer, pose model or dance generator in the inference path
(music-to-dance generators remain, at most, an offline pseudo-labelling teacher gated by the
same audits).

**Changed in rev. 6:** anticipation on RPMC-L2 moves from *excluded* to *gated* (section 6.7 H-D),
because the product now knows the track in advance. The preregistered bar is unchanged: lift at
4 s > +0.05 with a CI excluding zero, and the shift control peaking at 0. Also newly forbidden:
**quoting any offline number without its `L = 0` counterpart on the same splits.**

**Changed in rev. 8:** the dance descriptors are demoted from the decoder's input to a training
signal through the frozen trunk (the product claim is unchanged); authored light becomes the
target. Newly forbidden, from the five outside passes and section 6.9.8: an autoregressive light LM
as the decoder (absolute hue/value regression); RL or DPO on the flow head before the critic gate
G9.5 passes; mixture-of-experts by fixture type inside the trunk; section labels, chords, loudness
range or timbre descriptors as per-frame bus channels; Jukebox or CLAP encoders; a fixture-token
decoder before per-fixture console data exists; product training on BeatSaver, xLights, RPMC video
or YouTube material.

## 10. File and vault map

Code to add: `lightsem/motion/` (FK, 2D/3D pose extraction, descriptor blocks A/B/C),
`lightsem/conductor/` (v0 table, rig adaptor), `lightsem/eval/` (Light-BAS, FID_light-delta,
nulls, A/B tooling, latency/causality tests), `lightsem/audio/online.py` (causal features, PLP/
BeatNet phase), `lightsem/model/light_trunk.py` + heads. Vault: problem
`dancing-light-generative-20260903`; runbooks `rb-ls2-infra`, `00`, `00b`, `00c`, `01…06`, `07-offline-prerendered`, `08-justdance`, `09-designed-light`;
diagnostics `light-bas`, `fid-light-kinetic`, `light-diversity-kinetic`, `edm-collapse-rate`,
`human-preference-winrate`, `audio-to-dmx-latency-ms`, `descriptor-audio-r2-over-identity`,
`critic-pair-accuracy`, `rig-transfer-hist-distance`, `preference-heldout-accuracy` (rev. 8);
papers linked (incl. Burger 2013/2014, Toiviainen 2010, Van Dyck 2013, Solberg & Jensenius 2017,
Witek 2014, Bigand 2024, Alexanderson 2023, MotionStreamer, ITU-R BT.1359-1, Skip-BART
arXiv:2506.01482, SeqLight arXiv:2605.03660, FactorCL arXiv:2306.05268, Lodge arXiv:2403.10518).
