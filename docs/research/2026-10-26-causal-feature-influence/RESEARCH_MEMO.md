# Causal feature influence in the old LightSem pipeline

Date: 2026-10-26
Status: preregistered; no new model run or checkpoint promotion

## Goal

Identify which declared inputs causally change both LightSem neural networks:

```text
song -> frozen MuQ/Base92/descriptors -> native C -> raw 30 Hz body
     -> fixed mapper / LightLayers / BodyToDMX -> decoded 40 Hz DMX
```

The test must hold weights, seed/noise, clock, non-audio inputs, mapping and
render settings fixed. It must measure the raw body before mapper, interpolation,
DMX or video, and keep motion, RGB, brightness and event readouts separate.
Shpongle and AURORA — Cure For Me are recomputed from their own audio paths.

## Existing evidence to preserve

- The 2026-10-15 fixed 8-song diagnostic found full MuQ+Base92 nearly matched
  Base92-only on aim/RGB/brightness and cross-song correlation. MuQ-only raised
  sensitivity but produced unsafe raw jerk. Removing clock features did not remove
  the shared trajectory. These are diagnostic-only results.
- Removing all 1,024 MuQ channels changed root-relative body RMS much more than
  small source/mixture/event groups, but wide input and a large RMS delta do not
  establish useful music conditioning.
- Prior body-to-DMX deltas were diagnostic only. World-space RMS was confounded by
  root drift, so the new body test uses root-relative limb patches and decoded
  angular pan/tilt rather than 16-bit coarse/fine byte differences.
- The rejected stereo studies show that visually indistinguishable outputs,
  different hashes, or nonzero world-space RMS are insufficient evidence.

## Preregistered mechanism arms

1. **Native dance / MuQ masking** (`h-dance-muq-vs-base92-20261026`): masking
   MuQ with a matched mean/local replacement is predicted to produce at least
   0.03 m root-relative articulation change in each song, without a positive
   phrase-timing margin. A jerk increase above 2x baseline is a safety failure.
2. **Body-to-DMX source retention** (`h-light-body-to-dmx-retention-20261026`):
   patching an assigned source limb is predicted to move assigned fixture angles
   by at least 0.5 degrees and more than unassigned fixtures, with duplicate
   streams remaining below 0.10.
3. **Lighting audio-group causality** (`h-light-audio-group-causality-20261026`):
   rhythm/energy groups are predicted to affect brightness/events, timbre/chroma
   groups RGB, and own-song audio to beat time-shift/donor controls by at least
   0.05 standardized effect. Audio-only changes do not count as body-motion
   evidence.

The fixed basis and queued runbooks live in Vibescience problem
`lightsem-causal-feature-influence-20261026`:

- `runbook-causal-noise-floor-20261027`
- `runbook-dance-feature-causality-20261026`
- `runbook-light-feature-causality-20261026`

## Required intervention matrix

Dance arms isolate MuQ blocks/layers, Base92, drums/bass/other/vocals,
mixture scalars, beat/downbeat/section channels and D8 planner channels. Use
conditional train-mean or matched local replacement beside zeroing to detect
out-of-distribution masking. Add same-BPM donor, time-shift and beat-jitter
controls.

Lighting arms isolate the 94-D bus groups (mel, energy/onset, timbre/chroma,
rhythm phase/accent, BeatThis and palette history) and the BodyToDMX groups
(body position, velocity, contacts, Laban, audio groups, fusion and assigned
limbs). Activation patching is a screening tool only; strongest candidates must
be verified by a full forward intervention.

## Required readouts and gates

- Raw finite values, speed, acceleration, jerk, root/support and inactive frames.
- Latent/sample diversity, effective rank and exact duplicate output streams.
- Root-relative articulation and beat/phrase/motif timing for dance.
- Decoded angular DMX, RGB, brightness, event timing and saturation separately.
- Beat-light concurrence, loudness-brightness correlation and structural-part
  correspondence as downstream diagnostics, not substitutes for source transfer.
- Two own-song recomputations, with all source-body, soundtrack, checkpoint,
  noise, mapper and renderer provenance hashes recorded.

Unsafe, nonfinite or collapsed native bodies are quarantined before downstream
rendering. A passing downstream video or DMX decode cannot repair a failed raw
body. No result from this plan promotes a checkpoint or claims choreography.

## Literature-derived approaches

The following primary sources were added to the Vibescience record and mapped to
explicit interventions:

- Ahmad et al., **Causal Analysis for Robust Interpretability of Neural Networks**
  (WACV 2024): path interventions and causal explanatory graphs; adapt to rank
  feature blocks and hidden fusion paths.
- Zhang & Nanda, **Towards Best Practices of Activation Patching in Language
  Models** (ICLR 2024): corruption choice and metrics can change localization;
  require matched controls and restoration metrics.
- Tseng et al., **EDGE: Editable Dance Generation From Music** (CVPR 2023):
  Jukebox conditioning, joint-wise editing and foot-contact metrics.
- You et al., **MoMu-Diffusion** (NeurIPS 2024): motion/music latent alignment
  and long-term beat/kinematic correspondence.
- Alexanderson et al., **Listen, Denoise, Action!** (ACM TOG 2023): separate
  audio timing from style prior with null-audio and style controls.
- Zhou et al., **CausalMM** (ICLR 2025), and Zha et al., **Implicit
  Counterfactual Learning for Audio-Visual Segmentation** (ICCV 2025): modality
  counterfactuals and orthogonal replacements to expose modality-prior use.
- Kohl et al., **Cross-Modal Metrics for Capturing Correspondences Between Music
  Audio and Stage Lighting Signals** (ACM Multimedia 2025): beat/light,
  loudness/brightness and structural-part metrics.
- Zhao et al., **Automatic Stage Lighting Control: Is it a Rule-Driven Process or
  Generative Task?** (ICLR 2026, Skip-BART): music-conditioned lighting tokens,
  temporal alignment and constrained sampling.
- **LightingGen** (IEEE TMM 2026): latent DMX representation and
  music-conditioned sampling as an external lighting baseline.
- Aristidou et al., **Rhythm is a Dancer** (TVCG 2023): hierarchical pose,
  motif and choreography structure; use phrase recurrence and holds beyond beat
  alignment.
- Yang et al., **Keyframe Control of Music-Driven 3D Dance Generation** (TVCG
  2024): sparse key-pose/time interventions for local gesture-retention tests.

### Directly read on 2026-10-27

- **When Reconstruction Is Not Enough** (arXiv:2610.30220): reconstruction or
  contrastive quality cannot certify that a representation preserves causal
  information. Adaptation: compare held-out nuisance/predictive loss using the
  learned representation against the same learner with the raw feature group
  appended; select candidates on a screening split and confirm them on a locked
  split.
- **Admissible Diffusion for Multimodal Interventional Trajectories**
  (arXiv:2609.34433): plausible trajectories do not establish intervention
  validity. Adaptation: use fixed-plan contrasts, keep the history prefix causal,
  and report residual/constraint violations separately from output quality.
- **MotionSpaceFlow** (arXiv:2609.34190): direct full-resolution motion keeps
  joint/frame access and supports inference-time any-joint constraints. The
  useful LightSem transfer is a root-relative joint patch, while retaining the
  warning that representation-specific temporal attention changes what a suffix
  intervention means.
- **Architecture-Dependent Fusion Pathways** (arXiv:2610.03289): CKA and
  attention routing identify candidate fusion transitions, but causal masking or
  hidden-state noise is still required to show functional importance. For
  LightSem, layer-wise routing/CKA/intrinsic-dimension plots become screens for
  the MuQ/audio fusion blocks, never final attribution.
- **MCD: Causal Distillation of Multimodal In-Context Learning**
  (arXiv:2609.39920): gradient ranking is a candidate screen; necessity and
  sufficiency are tested with attribute-matched retain-evidence and
  remove-evidence replacements using a bounded output divergence. This is the
  direct design for the lighting audio-group matrix.
- **An Informational Curse of Horizon** (arXiv:2610.09247): conditional mutual
  information and input Jacobians expose a shift from conditioning inputs to
  state/history. Adaptation: rank MuQ/Base92 and lighting bus groups by local
  Jacobian/conditional-information scores, then verify with paired forward
  interventions.
- **Causally-Constrained Probabilistic Forecasting** (arXiv:2604.17998):
  time-lagged structural priors, per-output attribution and input clamping can
  separate propagated effects from root causes. Adaptation: clamp feature groups
  and score each body joint/DMX channel, while keeping any residual correlational
  path separate.
- **Identifiable Multimodal Causal Representation Learning under Partial Latent
  Sharing** (arXiv:2605.19135): shared and modality-specific latent factors need
  explicit mixing assumptions. Adaptation: label MuQ/Base92/source/body factors
  as shared or specific before interpreting intervention deltas.
- **Query-Conditioned Evidential Keyframe Sampling** (arXiv:2604.01002):
  conditional information can prioritize temporally diverse evidence windows.
  Adaptation: use phrase-bin scores to prioritize windows, then verify actual
  feature masks and output deltas rather than treating the score as causality.

These readings changed the protocol: Jacobian/CMI/CKA screens are now explicitly
non-evidence; matched replacement and full forward retain/remove tests are
required. The v2 dance hypothesis aligns the problem target (standardized effect
>=0.05) with the absolute root-relative gate (>=0.03 m). Noise floors must be
measured by repeat runs before a directional result is treated as support.

## Deep search results and implementation consequences

The completed deep search adds a stronger five-stage evidence stack:

1. Frozen external counterfactuals establish sensitivity of the computation.
2. Conditional dependence and lag screens identify unique information after
   competing inputs and output history are controlled.
3. Internal interchange/path interventions test whether a hidden route mediates
   the external effect.
4. Necessity/sufficiency and redundancy checks prevent one interchangeable basis
   from being mislabeled as the unique cause.
5. Structured output metrics and raw-safety gates reject destructive changes.

The most actionable additions are:

- **RAVEL/MDAS:** use donor/factual interchange patches and report target
  `Cause` plus off-target `Isolate` for phrase timing, limb articulation, RGB,
  brightness and angular DMX. Compare with norm-matched random subspaces.
- **HyperDAS:** automate the search over layer, time, channel group and low-rank
  subspace, but keep held-out donor pairs and random-location controls so the
  search network cannot inject the answer.
- **DIME:** estimate conditional mutual information for feature-group screening;
  CMI is not a causal verdict and must be followed by the fixed forward matrix.
- **CaRiNG:** treat body and DMX decoding as non-invertible observation processes;
  use temporal context to screen latent factors but verify at the raw body/DMX
  seams.
- **DanceMeld, BADM, DanceFusion and X-Dancer:** add phrase/movement versus pose,
  past/future context, and global/local/body-part interventions. These target the
  missing phrase and limb attribution evidence beyond beat correlation.
- **MOSPA:** an optional spatial-audio extension for separating content/loudness
  from audio geometry; it is not a prerequisite for the old pipeline.

For the lighting path, decompose fixed-body and regenerated-body effects:

```text
direct      = L(body, audio_counterfactual) - L(body, audio_factual)
mediated    = L(body_counterfactual, audio) - L(body_factual, audio)
total       = L(body_counterfactual, audio_counterfactual) - L(body_factual, audio_factual)
interaction = total - direct - mediated
```

These are controlled computational effects, not automatically natural direct or
indirect effects. Report angular pan/tilt, RGB, brightness and events separately.
The search found no strong 2024–2026 fixture-level DMX benchmark with released
causal evaluation code, so LightSem's decoded angular-DMX/source-limb tests remain
an open measurement contribution.

## Cross-check against the attached deep-research PDF

The PDF is consistent with the Perplexity report and adds several useful
distinctions:

- **RAVEL/MDAS, Same Task–Different Circuits, and Circuit Hypothesis Testing**
  turn patching into target-preservation, functional-interchange and minimality
  tests. LightSem should report target `Cause`, off-target `Isolate`,
  retained-versus-removed evidence and a random-path reference distribution.
- **FANS** adds a necessity/sufficiency score over conditional neighborhoods;
  use it for MuQ groups, assigned limbs and BodyToDMX channels rather than one
  RMS response.
- **CoMM and MCR** distinguish redundant, unique and synergistic modality
  information. The direct LightSem test is a factorial matrix: MuQ-only,
  descriptor-only, coherent joint replacement, and conditional permutation.
- **TempoMoE, DanceMeld and Lodge** motivate separate global tempo, local accent,
  phrase/movement and pose/primitive readouts. They do not validate arbitrary
  zero masks; replacements must preserve the retained feature distribution.
- **MOSPA** is an optional stereo-control template. It suggests correct stereo,
  mono duplicate, left/right swap, mismatched channel and location-only arms,
  but LightSem should use angular DMX and root-relative body metrics.
- **DanceEditor** is less directly transferable than its summary suggests: its
  edits are open-vocabulary text operations and its reported metrics do not prove
  preservation of unmentioned joints or time regions.

The attached PDF confirms the claim boundary: these are transferable tests or
architectural ideas; none establishes causal attribution for this LightSem
checkpoint. The active goal remains an executed, one-seed frozen-model audit,
not a paper-based promotion.

The incremental search also surfaced two 2026 primary sources that I read
directly. **DiscoForcing** (arXiv:2605.28491) factorizes strictly causal
streaming audio into discrete rhythmic RVQ codes and continuous periodic-phase
features, then uses root-relative incremental motion and diffusion forcing. Its
ablation reports a learned VQ-PAE encoder improving realism over Librosa while
the causal/latency protocol controls lookahead and stale-history confounds. The
transferable LightSem test is a matched rhythmic-code versus phase-feature
retain/remove matrix measured on raw body timing and safety, not a wholesale
architecture swap. **SeqLight / Stage Light is Sequence^2** (arXiv:2605.03660)
predicts global hue/value distributions and then decomposes them into
venue-specific per-light controls with BC, AIRL/GRPO and HER-style goals. Its
BC/AUX ablations show policy dependence, but the model is hue/value-only and
does not establish pan/tilt or DMX motion attribution. It supports keeping
palette/RGB and angular DMX as separate endpoints in the LightSem audit.

The methods are novel relative to the old LightSem evidence when they add
root-relative source patches, angular-DMX readouts, orthogonal modality controls,
or causal hidden-path verification. MuQ/Base92 zeroing, donor/time-shift controls,
raw-safety checks and separate RGB/brightness diagnostics are already covered by
prior LightSem work and are retained as controls.

## Vault record

Problem: `lightsem-causal-feature-influence-20261026`

Synthesis: current state recorded in Vibescience on 2026-10-27
The cached dance, mapper, and palette-state diagnostics have executed; full-waveform
confirmation and decoded-DMX audio-group retention remain queued.

## Executed cache-backed diagnostic (2026-10-27)

The new CPU runner executed 60-second cached-feature windows for both songs using
the selected fixed-C checkpoint, seed 7001, authenticated MuQ/Base92/timeline/
stereo caches, exact deterministic repeats, and no mapper/DMX/video. The original
waveform files were unavailable, so the receipts are explicitly
`diagnostic_only`/`unqualified_missing_raw_waveform`.

- Zero-MuQ root-relative articulation RMS: **0.0114 m** (Shpongle) and
  **0.0337 m** (Cure For Me); standardized effects **0.0379** and **0.1106**.
- Zero-Base92 root-relative articulation RMS: **0.0258 m** and **0.0585 m**;
  standardized effects **0.0858** and **0.1918**.
- All repeated forwards were byte-identical. Latent effective rank stayed 139.
- Raw p99 safety ratios stayed below 2x for these short windows; no arm was
  rendered or mapped.

This is mechanistic cache evidence only. It suggests Base92 may carry more direct
body sensitivity than MuQ in this recipe, but it does not establish full-song
music conditioning, phrase timing, or product qualification. The follow-up
hypothesis `h-dance-base92-role-20261027` and runbook
`runbook-dance-base92-causal-20261027` commit the Base92 claim for a proper
full-waveform test.

The paired cached-body mapper diagnostic also patched the left arm before
mapper-01. Decoded assigned left-wrist/elbow/shoulder fixtures moved **3.91° RMS**
on both songs versus **1.88°** for unassigned moving fixtures (ratio **2.08x**);
RGB stayed fixed and brightness remained a separate small effect. This is
mechanical source-transfer evidence only. The experiment closed
`directional_only` because exact duplicate-rate evidence was missing and the raw
waveforms were unavailable.

The corrected duplicate audit measured exact duplicate fixture rate **0.0** on
both mapper receipts. A paired 2-second block bootstrap over the saved 60-second
body windows measured root-relative RMS sigma **0.00351 m** and standardized-effect
sigma approximately **0.01157**; these measured floors are registered in
Vibescience while full-waveform confirmation remains pending.

A two-song cached LightLayers screen ran on sealed 94-D buses with the same frozen
P2/Rich96 checkpoint and isolated mean masks. On Shpongle, zeroing mel, energy,
timbre or rhythm changed palette IDs by **0.567**, **0.402**, **0.013** and
**0.834** of 10 Hz states; on Cure For Me the corresponding changes were
**0.334**, **0.992**, **0.663** and **0.987**. All ten arms were finite, and each
song exceeded the preregistered 0.20 screen MDE. This supports audio-group
dependence at the palette-state seam, with rhythm dominant for Shpongle and
energy/rhythm dominant for Cure For Me. It remains a diagnostic-only proxy: no
decoded DMX, RGB, brightness, body, choreography, or full-waveform own-audio
claim follows. The next admissible lighting run must carry matched group
retain/remove interventions through the mapper and emitted DMX while keeping the
body fixed.

Perplexity searches were not silently stopped: the authenticated account still
had web/scholar search capacity, while the deep-research quota had reached zero.
The earlier research sweep is complete; subsequent sources were searched with
the search endpoint and then read from their primary arXiv pages directly.
