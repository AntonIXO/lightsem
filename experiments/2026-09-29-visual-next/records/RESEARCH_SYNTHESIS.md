# Research synthesis and next experiment decisions

Baseline C stays fixed: independent06, rhythm companion02, generated palette
history and300ms fades. Poses still shape beams; the new questions concern the
picture used as supervision and the color-output representation.

Perplexity Deep Research completed on the second attempt with the same full
query after the desktop transport timed out at300s. The installed MCP was called
with a900s client limit; the successful response is preserved, including498 source
results and its backend reference. All10 recommended primary papers were fetched
and their relevant full-text sections read. Seven additional primary papers were
checked to cover current dense encoders, positional bias, palette transport and
perceptual metrics. The source-note file names exactly what was read. Downloaded
PDFs total259302591bytes; no weights or new datasets were downloaded.

## What the literature changes

1. **Do not start by replacing YOLO.** It currently supplies person boxes to the
   offline observer. TransNetV2, DEVA and Cutie suggest testable temporal/target
   interventions while retaining those boxes. A stable wrong mask is still wrong;
   primary evaluation must include independent retained-background color error.
2. **Preserve explicit color channels.** DINOv2 is the smallest practical first
   patch-feature baseline. DINOv3 improves dense features but adds gated weight
   access and has reported positional artifacts in low-semantic scenes (INSID3).
   VideoMAE and V-JEPA2.1 motivate temporal-order controls. VideoPrism is relevant
   but currently adds a JAX/Flax environment. None proves chromatic or LIGHT transfer.
3. **Test output capacity separately.** A single shared RGB cannot reproduce
   several spatial colors. Regularized transport and palette recoloring motivate
   a constrained fixture oracle, but the comparator must have the same privileged
   target information. Human design quality is a different question.
4. **Keep evaluation independent.** DAVIS separates region/boundary/stability
   errors. DreamSim emphasizes foreground objects; Synchformer was trained for
   natural audiovisual synchronization; ColorVideoVDP needs matched reference
   content and a declared display model. These cannot become automatic stage-light
   quality or music-fit judges. Use them, if qualified, as secondary diagnostics.

## Corrections to the generated Deep Research advice

- Its incremental-bibliography warning was resolved against the full current vault:
  the17 identities are new; official titles/IDs and fetched PDF hashes are logged.
- Some inline citation numbers in the generated answer are reused inconsistently.
  Native paper records use independently verified primary URLs, not those numbers.
- Reconstructing color statistics already included in a representation can be
  tautological. We use that only as a retention sanity check. H3's primary endpoint
  is independent human-annotated color-event AP of an audio-only student.
- A multicolor oracle with access to reference images must beat an equally
  privileged GLOBAL-color oracle. Beating the audio-only C anchor alone does not
  identify an output-capacity benefit.
- Removing all edit-adjacent samples would manufacture easy support. H1 vetoes
  predictions while keeping the independent scoring denominator, including genuine
  color changes coincident with cuts. Recall and false-removal guards can refute it.
- The failed historical31/40 coverage gate is not replaced by an easier cohort.
  New findings remain conditional pilots on explicitly enrolled historical groups;
  old qualification failures remain. Fresh confirmation needs its own enrollment.
- Different studies have different dependencies. Mask intervention failure does
  not prevent an oracle study with reliable manual masks; a temporal teacher failure
  does not refute every explicit spatial-color representation.
- Magnitudes are design targets, not published LightSem effect estimates. No
  measured sigma, power calculation or MDE exists yet; none is invented.

## Committed next studies

| Study | Primary comparison | Primary direction and engineering target |
| --- | --- | --- |
| H1 edit-aware targets | TransNetV2 veto vs current observer, same masks | content-event F1 +.05; recall loss <=.03 |
| H2 temporal masks | anchored Cutie vs stateless SAM2 | independent background-color JS distance -15% and -.02 absolute |
| H3 picture-over-time supervision | fixed color/spatial/temporal auxiliary vs Rich96, matched audio student | independent content-event AP +.03; must beat static, random and order-destroyed controls |
| H4 fixture oracle | four-group color oracle vs equally privileged global oracle | spatial color error -20%; beat permutation and donor by10% |

Exact definitions, controls, analysis, resources and stop rules are in
PREREGISTRATION_V1.md and the machine-readable native payloads. Each claim has
one primary endpoint and separate blocking guards. No experiment is run now.

**First executable work:** freeze the annotation instrument and perform the
12TRAIN/31historicalDEV audit already enumerated in ANNOTATION_ENROLLMENT.json.
There are43 existing source windows; independent annotations are not yet present.
Use TRAIN for instrument calibration, freeze assessment labels before predictions,
then run the isolated cut/mask comparisons. Feature qualification precedes H3
student fits. No sealed TEST access, Shpongle selection, baseline replacement,
new motion generation or broad hardware/product qualification follows here.
