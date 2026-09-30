# Image supervision after the baseline06 selection

Research and preregistration are complete. See [the successor study package](../../2026-09-29-visual-next/README.md) for the verified papers, four native hypotheses and exact prospective protocol. The original research runbook below is complete; separate queued runbooks carry the future experiments. Baseline C includes history and300ms fades.

User direction: retain06, add color transitions, repeat palette-history A/B on06,
and investigate the picture as well as movement. Native queue:
`rb-baseline06-image-supervision-20260929` in `/root/lightsem-vault`.

## What06 already learned

The fixed independent architecture has three audio encoders for brightness
accent, color-change timing and palette. `P0` means no auxiliary pose target.
Its Rich96 auxiliary target contains72 hue/saturation bins and24 quadrant color
and mask summaries. Accent targets come from image brightness observations;
color targets come from image changes and a32-token palette vocabulary.
Its training inventory was91 TRAIN groups and31 historical DEV groups, with
three numerical seeds per recipe; the demo uses selected seed0.

The current delivery then maps each selected palette token to one global RGB
summary and applies that same color to all color-enabled fixtures. It does not
learn a different color assignment for each fixture or reconstruct the image's
spatial color composition. Keep this mapping limitation distinct from the
quality of the upstream image representation.

Cached person boxes from the pose extraction path prompt SAM2 on decoded frames.
People and UI are excluded to measure the retained image. That is the role of
YOLO in this chain. The light model receives audio at inference. Spatial beam
movement in the demo is still the separately accepted rhythm companion and saved
internal motion style.

Rich96 describes low-level image color and coarse layout; it is not a learned
semantic representation of scene composition, nor a complete account of visual
motion. Therefore the new direction starts from an existing image-supervised
model. Replacing the person detector would primarily change masking, and must be
evaluated separately from a new image representation.

## Next controlled investigation

1. Inspect a bounded sample of existing TRAIN/DEV image targets next to original
   frames, person/UI masks, shot boundaries and audio. Annotate whether measured
   changes come from light/background, people, UI or editing. The old observer
   contains unknown shot cuts and mask nuisance.
2. Define the useful image signal: spatial color composition, brightness dynamics,
   and temporal scene changes. Compare the current96 descriptors with a richer
   image/temporal representation while retaining the same audio and split.
   Evaluate fixture-wise color composition as a separate output-mapping change;
   a richer image encoder alone cannot express spatial colors through one sharedRGB.
3. Include shifted-audio and scene/mask controls, check leakage and event support,
   freeze a bounded pilot and measurable prediction before fitting. Measure
   output lighting with the selected06 baseline and multi-track human review.

No specific replacement detector/encoder, new dataset, training run or claimed
gain is selected by this document. Architecture and representation benefits are
open hypotheses. Existing source-coverage and null-validity failures remain in
the scientific record. User enthusiasm about the Shpongle06 result is recorded
as direct qualitative feedback, without invented numerical scores.

Source references: `lightsem/model/light_layers/{model,targets,training,visual}.py`,
the pinned checkpoint metadata, and September22 observer reports. No external
literature or current model leaderboard was consulted for these local findings.
