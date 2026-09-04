# LightSem v2 — sound → light for concerts and festivals, trained on dance

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
| `lightsem/data/rpmc_remote.py`, `extract_rpmc.py` — RPMC-L2 range reader; 140/699 songs cached | PASS |
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
| 0c, 3 | human A/B | **materials generated, awaiting raters** |

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
| RPMC-L2 (HF `RS2002/RPMC-L2`) | 699 songs, ~30 fps HSV histograms, 42.5 GB | CC-BY-4.0 | HEAD B events + palette |
| own recordings (events, consent) | video → pose | own | the only fully commercial-clean path |

**Video → descriptors pipeline (Anton's YOLO idea).** 2D pose (YOLO11-pose / RTMPose / ViTPose)
normalised by torso length supplies speed, planar laterality, spread, hands vs head split, and
periodicity — i.e. most of block A and the useful part of C. It cannot supply clean jumps, foot
contacts or Laban Weight (depth, camera tilt); those come from 3D lifting (GVHMR, SMPLest-X) or
are dropped. Reject multi-person/occluded frames. Validate the pipeline on AIST video against
AIST++ mocap before trusting it anywhere else. Legal line: datasets with video licensed for
research, or own footage; YouTube-scraped sets are usable for a paper, not a product.

Infra: dedicated torch venv on `/mnt` (uv, cu130); `hf` CLI; ultralytics/RTMPose install;
Perplexity Pro Search quota is back (Deep Research still 0 on 2026-09-04).

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

**Dependencies and things that are not blocked by this.**
Oslo audio reconstruction (a parallel effort) caps at roughly ±1 s alignment because the record
publishes 12 scalar channels and no audio, and the DJ desk moves exactly the channels published.
±1 s is usable for block A envelopes at 2–8 s EMA and useless for block B or beat phase, so the
corpus must carry per-track alignment error as an explicit field and downstream code must gate
channels on it. The Oslo R² +0.30 result does **not** depend on that reconstruction — it is
computed from the published series, already on the mix clock. The human A/B materials
(`outputs/stage0c/`) are unaffected by everything in this section except `category_model/`, which
was rendered from the pre-repair checkpoint and needs re-rendering.

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

## 10. File and vault map

Code to add: `lightsem/motion/` (FK, 2D/3D pose extraction, descriptor blocks A/B/C),
`lightsem/conductor/` (v0 table, rig adaptor), `lightsem/eval/` (Light-BAS, FID_light-delta,
nulls, A/B tooling, latency/causality tests), `lightsem/audio/online.py` (causal features, PLP/
BeatNet phase), `lightsem/model/light_trunk.py` + heads. Vault: problem
`dancing-light-generative-20260903`; runbooks `rb-ls2-infra`, `00`, `00b`, `00c`, `01…06`;
diagnostics `light-bas`, `fid-light-kinetic`, `light-diversity-kinetic`, `edm-collapse-rate`,
`human-preference-winrate`, `audio-to-dmx-latency-ms`, `descriptor-audio-r2-over-identity`;
papers linked (incl. Burger 2013/2014, Toiviainen 2010, Van Dyck 2013, Solberg & Jensenius 2017,
Witek 2014, Bigand 2024, Alexanderson 2023, MotionStreamer, ITU-R BT.1359-1).
