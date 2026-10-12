# Body-to-DMX multimodal foundation model (2026-10-13)

This research goal is to learn a separate body-to-light model that consumes the
LightSem dance trajectory and audio context, then emits a rig-independent light
description that can be compiled to arbitrary DMX fixtures. The intended output
includes movement/aim, brightness, colour, and timed accents. The model is a
research branch; it does not replace the selected native-C motion baseline or
the accepted P2 colour baseline.

The first implementation follows the local Hermes Agent `multimodal-ai` skill:

1. write the modality profile and state the required invariances;
2. measure body-only and audio-only baselines before fusion;
3. use multiplicative/attention fusion only where body and music are expected
   to interact, and compare it with the nearest additive projection;
4. use masked multi-target learning because public corpora expose different
   parts of the lighting state;
5. audit source dependence, collapse, safety, and held-out-song behaviour before
   any visual or product claim.

The existing public inputs are complementary rather than one homogeneous DMX
corpus. BeatSaver maps provide authored event timing and virtual light groups
paired with the map audio. RPMC-L2 provides 699 songs with aggregated HSV light
histograms and frozen audio embeddings, but no fixture geometry or raw DMX
universes. FineDance/AIST/Just Dance provide motion priors and evaluation
controls, not lighting ground truth. The model therefore predicts a common
group-level representation and uses an explicit observation mask; it must not
pretend that a histogram target supervises pan/tilt.

Current preparation status is recorded in [PLAN.md](PLAN.md) and
[DATA_SOURCES.md](DATA_SOURCES.md). The BeatSaver converter now creates its
worker output directories, completed an 8-map smoke, and completed the full
300-map conversion. The canonical preparation receipt is
`/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013/DATASET_RECEIPT.json`.

The required past-experiment trace is preserved in the project vault as
[`body-to-dmx-experience-trace-20261013.md`](/root/lightsem-vault/records/body-to-dmx-experience-trace-20261013.md).
It records the earlier H-I decoder refutation, BeatSaver admission failure,
Laban/readability results, stereo perceptual collapse and the limitations of
the preliminary four-song pilot.

The current active native-C baseline requires an authenticated stereo context
for only the two showcase songs. For unseen BeatSaver songs, the first corpus
build will use the preserved unconditioned RTMPose fixed-C checkpoint as an
explicit diagnostic source generator, or stop if raw safety fails. This keeps
the source lineage honest and prevents silently calling a new-song body an
active dual-attention baseline.

No checkpoint is qualified by this document. A qualification decision requires
the machine audit, paired controls, and the two-song LightSem delivery checks.

The latest diagnostic checkpoint is `pilot_seed0_v6_laban_angle_8songs`: it adds a
trainable six-head pan/tilt interface with weak body-geometry supervision. Its
DMX output is still quarantined because the unseen-song active-baseline safety
pair and physical authored-angle targets are missing. The eight-song branch uses
two development songs rather than a one-song validation split.
