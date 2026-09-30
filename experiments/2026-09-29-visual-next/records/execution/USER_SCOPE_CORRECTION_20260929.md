# User correction: automate the source analysis

The user rejected making manual annotation of43clips/430stills part of their
workflow and asked about pretrained models that already perform these tasks.
The practical objective remains testing all four methods and producing Shpongle
demos for the user's visual ratings. Full manual source annotation is OPTIONAL,
not a prerequisite for those experiments, deliveries, or the user's work.

The earlier annotation workspace remains an optional audit facility. Do not
spend further task time expanding it or ask the user to draw all masks. Prioritize
automatic model-derived observations, fixed controls, factual uncertainty and
actual rendered comparisons. A small targeted inspection can be proposed later
if a specific disagreement would change a decision; do not impose another bulk
annotation requirement.

Version6 was already trained on automatically extracted authored-video targets.
The purpose of human source labels was independent validation of the new
observers, not supplying every training target. That distinction was not made
clear enough in the earlier delivery and the validation workload was excessive
for the requested prototype campaign.

Preserve original preregistrations and report their independent-human endpoints
as unmeasured wherever they remain unavailable. Model agreement is not perfect
ground truth. This limits formal confirmation claims; it must not block the
authorized automated comparisons and demos. Do not silently replace a registered
model family or call automatic predictions human labels.

Current public primary-source check: SAM3 supports text/concept prompted object
detection, segmentation and tracking (for example the category person), and Meta
explicitly documents follow-up corrections for missed objects/mistakes. It is a
relevant future candidate for removing the separate YOLO prompt stage, not an
already verified replacement on this corpus or a perfect reference. The current
Cutie/SAM2/TransNet/DINO comparisons continue under their existing bindings.

Sources checked2026-09-29:
- https://ai.meta.com/research/sam3/
- https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- https://github.com/soCzech/TransNetV2
