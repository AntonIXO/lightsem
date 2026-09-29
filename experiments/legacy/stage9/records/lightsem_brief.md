# LightSem — project brief for outside reasoning (2026-09-04, rev. 7 state)

## What the product is
Sound in, stage light out (DMX, 512 channels at 40 fps) for concerts, festivals and DJ sets.
The product is a PRE-RENDERED show: the whole track is known in advance, so the model may look
ahead (offline, "L = whole track"); a strictly causal variant (L = 0) is kept as a comparison and
for live use. No dancer, pose model or dance generator exists in the inference path. Dance data
is used ONLY as training signal: paired (music, dancer motion) teaches what light should do.

## Pipeline today
audio -> causal feature bus (92 channels at 40 fps: log-mel, RMS, spectral flux in a low band
50-100 Hz and a high band 6-13 kHz, half-wave onset, kick-flux beat phase as sin/cos, tempo)
-> 6-block Conformer trunk (d_model 256, 4 heads, depthwise conv kernel 15, attention window
4 s per layer composing to ~26 s receptive field, ~9 M params, offline flag makes attention and
conv symmetric) -> three heads emitting 30 "motion descriptor" channels: block A regressed
(6 channels that proved predictable: footfall rate, metric level, vertical bounce period and
amplitude, ...), block B event hazard (footfall, jump, hit), block C sampled by a conditional
rectified flow (hand speed/spread, laterality, travel, symmetry, ...). Then fitted per-channel
causal one-pole smoothing and calibrated event thresholds -> a HAND-WRITTEN "hybrid conductor"
(rule table: torso/head energy -> big slow washes, hands -> small fast fixtures and strobes,
vertical bounce -> dimmer/tilt bounce on the beat, sway -> pan sweeps at the bar, events ->
strobe bursts) -> canonical rig (6 moving heads on two trusses, washes, blinders, strobe, LED
bars) -> DMX -> headless EGL renderer to MP4.
A separate "operator head" (1-D causal conv net) is trained on RPMC-L2 (699 songs, per-frame HSV
histograms of real music-video/stage lighting) and predicts lighting EVENTS (look changes) and
hue distributions from audio; it is the strongest audio-to-light coupling measured in the project.

## Training corpora on disk
- AIST++ (5.2 h, 10 street-dance genres, mocap SMPL, 60 tracks). Measured: movement is explained
  by genre, not by audio; cannot train block A.
- Oslo club-dancing EDM mocap (29 dancers, 2 DJ mixes, one marker each, CC-BY): the only corpus
  where audio predicts held-out movement (R^2 +0.30 on energy at 2 s). Used as the CONCERT test.
- RPMC-L2 (HSV histograms of real lighting, 699 songs): operator head, events AP lift +0.31.
- Just Dance gameplay captures (338 videos, 22 h, 294 unique songs, YouTube-sourced, research
  use only): 2D pose (YOLO26-pose) of every coach -> 453 coach channels, 27.9 h of motion.
  Trained the current best model. Its BACKGROUND is a designed visual show; an HSV-histogram
  "visual" target from it gives event AP lift 3.17x vs an audio-shift control 1.48x.
- FineDance, InfiniteDance, OpenDanceSet, PopDanceSet: research-licence dance sets, not yet used.
- Hardware: one RTX 5070 12 GB, root disk nearly full, bulk data on a second disk.

## Measured results (every number carries its null)
- Held-out Just Dance songs, Light-BAS (beat alignment of light onsets minus circular-shift
  null): model +0.135 [0.121, 0.150]; the real coach's motion mapped to light +0.087; a 30-line
  beat-trigger baseline +0.092. So the model is MORE metronomic than a human dancer.
- FID_light on delta features: model vs real 36.5; split-half floor 8.1; block-shuffle null ~70.
- Diversity across seeds/clips passes (7.4 vs real 6.3).
- Held-out block A R^2 is negative (-0.22): audio barely predicts the coach's motion frame by
  frame. Per-coach audit: R^2 ~ 0, r +0.17 to +0.24 at 2-8 s; inter-coach agreement between
  two coaches on the same song only r 0.22-0.42 (this is the ceiling of any dance-trained
  model); section contrast (loud thirds vs quiet thirds) only 1.5x; 70% of coaches bounce at
  1x or 2x the beat.
- Offline look-ahead (H-A): no gain over causal on descriptors (R^2 -0.009 vs +0.019) or on
  light BAS. The corpus binds, not the causality.
- Feeding song-structure channels (section role, energy, time-to-rise/drop) into the per-frame
  model (H-C): paired BAS -0.028 [-0.042, -0.017], FID better 36.5 -> 27.6 but a shuffled-donor
  control gets 33.1. Refuted as an input channel; an offline section PLANNER acting on the
  conductor (scene, palette, intensity per section) is untested.
- Anticipation on RPMC-L2 (H-D, predicting a look change 4 s early with an offline window):
  margin -0.021, CI includes zero; fails. Target-derived inputs under a symmetric window leak
  (10x lift), so offline arms must be audio-only.
- Post-hoc smoothing (H-B): the largest single lever found; fitted causal one-pole per channel
  moved BAS +0.018 -> +0.29 and FID 67 -> 20 on AIST, similar on Just Dance.
- Oslo transfer: the Just Dance-trained model's predicted energy correlates r +0.335 with the
  real crowd's mean speed across DJ-mix track slots (AIST-trained: +0.07).
- Human category test (10 rated pairs): light from real dancer motion beats beat-trigger 8-1-1
  on "looks like a lighting engineer did it"; tie on "fits the music".
- Older: audio -> absolute light level fails (persistence wins); regressing raw kinetic energy at
  10 Hz fails; both are permanent rule-outs. Timing metrics need a metronome null.

## The user's assessment and the three directions to plan
Result is decent but well short of what a human lighting designer does. Three directions:
1. DECOMPOSE THE TRACKS to get more training features: stems (drums/bass/vocals/other),
   per-stem onsets and energy, chords/key/harmonic tension, vocal activity and lyrics, song
   structure (verse/chorus/drop/breakdown), pretrained music embeddings (MERT, MuQ, CLAP,
   Jukebox), timbre descriptors, per-stem beat/downbeat, dynamics.
2. MORE CONCERT / LIGHTING DATASETS: anything that pairs music with what a lighting designer or
   VJ actually did: DMX or console logs, show files, rhythm-game lighting tracks (Beat Saber
   lighting events in beatmaps, Rock Band / Guitar Hero VENUE lighting authoring, Audio Trip,
   Synth Riders), music-video lighting datasets (RPMC-L2 is one), laser-show files, concert
   video corpora, festival streams, LED-wall content, academic corpora (SeqLight / Skip-BART /
   "music-to-lighting" papers), light-show design corpora from Light-o-rama or xLights
   sequences, Christmas light shows, fountain shows. Licence and legal reuse matters: the
   product cannot be trained on scraped YouTube; research-only sets are fine for a paper.
3. TRANSFORMER / MULTIMODAL ideas: we compare relevance between movement, sound and what the
   director did. Candidates: a cross-modal relevance/critic model (audio <-> light, audio <->
   motion) used as a reranker over flow samples or as a reward for preference fine-tuning;
   contrastive pretraining; music-to-dance transformers (FACT, Bailando, EDGE, Lodge, POPDG,
   MotionStreamer) and what to borrow; music-to-lighting models (Skip-BART, SeqLight); diffusion
   forcing; frozen music foundation models as encoders; masked-token pretraining on light logs.
4. "MIXTURE OF LIGHT": the rig as a SET of N heterogeneous light sources with different types
   (wash, spot/moving head, strobe, blinder, LED bar/pixel, laser) each with different parameter
   spaces; multi-head or mixture-of-experts style outputs, one expert per fixture type or a
   set-equivariant decoder where each fixture is a token with a type embedding and the model
   adapts to the rig during training (random rig sampling), so a show can be rendered for any
   venue's patch. The hand-written conductor is the current bottleneck for "human quality".

## Constraints that any plan must respect
- Every metric must be priced against a trivial baseline (beat trigger, persistence, shuffle).
- Never regress absolute light levels or raw kinetic energy.
- Any offline number must be quoted with its causal (L = 0) counterpart.
- Cross-modal transfer moves only redundant information: if lighting decisions depend on
  information not in the audio, no model can recover them from audio alone; a corpus whose
  targets are a function of the audio at the needed resolution is required.
- Segment before aligning on continuous modalities; beat/bar/section-synchronous features.
- Splits by song, never within a song.
- One consumer GPU; training runs of hours, not days.
