# LightSem — Prior Art Review

Deep research run 2026-08-07 via Perplexity (`perplexity_research`, sources: web + scholar),
seeded with everything already verified in-session so the model could not just restate it.

**Headline: the field is tiny.** Two generative papers, one dataset. Almost every capability
listed below as "does not exist" is absent not because it's hard, but because nobody has
assembled existing pieces.

---

## 1. EDM / club lighting — DOES NOT EXIST

- No EDM lighting dataset, no EDM lighting model.
- RPMC-L2 is explicitly skewed to rock / punk / metal / core — the authors list this as a
  limitation themselves.
- EDM structure data exists but carries **no light channel**:
  - `EDM-98` — 98 professionally annotated EDM tracks, 5 tempo categories (120–180 BPM),
    released with EDMFormer (arXiv 2603.08759). Targets drops and buildups specifically.
  - `EDM-CUE` — 4710-song source corpus, HuggingFace `disco-eth/edm-cue`.
  - GiantSteps+ EDM Key (Zenodo 1095691), Kaggle beatsdataset — key/genre only.
- Drop detection has its own small literature: ISMIR 2014 content-based drop detection;
  a 2023 Twente thesis with hand-labelled drops + CNN.

## 2. Anticipatory / predictive lighting — DOES NOT EXIST

Important nuance found: **Skip-BART is technically non-causal** — bidirectional BART sees the
whole track. But that is an offline convenience, not lookahead control; the authors explicitly
name real-time / online control as an open problem.

Closest working primitives:
- Zero-latency beat tracker with explicit lookahead (TISMIR 2024, PLP-based).
- BeatNet (ISMIR 2021) CRNN + particle filtering, online.
- Video-prediction diffusion (MCVD, NeurIPS 2022) — adjacent, unapplied here.
- Music-psychology grounding for *why* anticipation matters: UC Press *Music Perception* 42(3)
  on EDM build-up tension → drop pleasure.

## 3. Real-time ML DMX inference — DOES NOT EXIST

- Not one published latency figure for audio → DMX with a real trained model.
- Transport infrastructure is mature and open: Open Lighting Architecture (OLA),
  BlenderDMX Art-Net/sACN, OSC2DMX, OpenLyght.
- Nobody has connected a learned model to it.

## 4. Datasets — RPMC-L2 IS THE ONLY ONE

- `RS2002/RPMC-L2` (HuggingFace, CC-BY-4.0, ~699 clips) — that's the whole public corpus.
- GDTF / MVR are **fixture and rig-geometry formats, not behavioural corpora** — a common
  misconception. grandMA3 imports them; they contain no show timeline.
- No public show-file corpus from grandMA, Chamsys, Resolume, WYSIWYG, Capture, or from any
  lighting-design competition. Searched, none found.

## 5. Simulators — BlenderDMX is the only viable RL / eval environment

| Tool | Open? | Scriptable? | Verdict |
|---|---|---|---|
| **BlenderDMX** (`open-stage/blender-dmx`) | GPLv3 | Python + Art-Net/sACN, Eevee realtime | **best candidate**, wrap in Gym |
| OLA | yes | C++/Python API | transport only, not a renderer |
| OSC2DMX | yes | yes | bridge + toy simulator |
| Capture / WYSIWYG / Depence / ETC Vision | no | limited | commercial, not usable as reward env |

## 6. Dance motion as a lighting / visual control signal — NO PUBLICATIONS

Exists only as **VJ artistic practice** — Kinect + TouchDesigner, projection mapping onto
dancers. Zero papers with code or architecture.

The bridge between dance generation (AIST++ / FACT / Bailando / EDGE / Lodge / POPDG) and
Automatic Stage Lighting Control (Skip-BART / SeqLight) **has not been built by anyone**.

⚠️ We independently tested the *naive* version of this bridge and it failed — see
`probe1`–`probe4` and the rule-out in `/root/lightsem-vault`. The gap being open is not
evidence that it is fertile.

## 7. Evaluation — essentially unsolved

- What exists: RMSE / MAE / correlation on HSV channels, plus a 6-dimension human rubric
  (Erdmann et al.).
- **No FID analog. No Beat Alignment Score analog for light. No perceptual light↔music
  synchrony metric of any kind.**
- Skip-BART's headline result is a *human* evaluation: p < 0.001 vs rule-based,
  p = 0.72 vs human engineers (i.e. indistinguishable).

## 8. Open problems sized for one GPU

All are bounded by assembly effort, not compute:

1. **EDM fine-tune** of Skip-BART, measure domain gap vs rock/metal.
2. **Lookahead layer** on top of an online beat tracker → anticipatory cueing.
3. **Causal Skip-BART** + the first real audio→DMX latency benchmark.
4. **Port Beat Alignment Score to light** — supply the missing objective metric.
5. **Gym wrapper over BlenderDMX** — supply the missing RL environment.

---

## Primary sources

- Skip-BART — arXiv `2506.01482`, ICLR 2026, OpenReview `a4Got6azjF`
- SeqLight — arXiv `2605.03660`
- EDMFormer + EDM-98 — arXiv `2603.08759`
- RPMC-L2 — `huggingface.co/datasets/RS2002/RPMC-L2`
- EDM-CUE — `huggingface.co/datasets/disco-eth/edm-cue`
- BlenderDMX — `github.com/open-stage/blender-dmx`
- OLA — `docs.openlighting.org/ola/doc/0.9.6/`
- BeatNet — ISMIR 2021
- Zero-latency beat tracking — TISMIR 2024, `10.5334/tismir.189`
- EDM drop detection — ISMIR 2014
- EDM tension/anticipation — *Music Perception* 42(3)

## Products claiming "AI" without evidence

SoundSwitch Autoscript (beatgrid + phrase heuristics over preset templates — inMusic does not
actually claim ML), MaestroDMX, AI LIGHTSHOW, AULIOS, Y-Link. No technical detail published by
any of them.
