# Pose-stage holds: stable holds, more pose-specific grouping, more overlap

**Outcome:** In these four frozen reference windows, candidate `pose_stage` generally preserves sampled hold stability. It gives asymmetric and crossed poses more distinct beam arrangements than matched same-body `stage02`. The cost is more projected shaft crossing and reduced separation, especially in 078 and 083. This is not visual or music-fit acceptance. More motion in a hold is not a success criterion.

## Coverage and method

Independently and silently inspected all 4 entries: **16/16 receipt-listed sheets, 140/140 unique decoded sample frames via those sheets, and 60/60 fixed-fast frames additionally at native panel resolution**. Each entry has 21 chronological 1 Hz-plus-last frames and 15 fixed-fast frames, with frame 270 shared (35 unique). Fixed-fast frames are 270, 274, 278, 282, 286, 290, 294, 298, 302, 306, 310, 314, 318, 322, 326.

The original comparison PNGs are 1920×360: shared saved mocap body (front plus oblique), baseline, candidate. Native preparation crops x=[0,640), [640,1280), [1280,1920), y=[0,360), then stacks them without resizing into 640×1080 PNGs. Every fixed-fast stack was viewed. Overview sheets are 1920×1212 and were displayed by attachment at 1200×758 JPEG quality 82; that overview is not counted as native inspection. No raw RGB/PCM cache or new render was made.

The comparison changes the mapper only. Both arms use the same saved body, music, optics and controls. No new generator and no legacy lighting comparison. Numeric gates were not used as visual acceptance.

## holds_005_0_reference_planned

**Source body:** Held wide-foot stance with elbows bent outward through sampled 0–12 s, then turns/steps and raised-arm poses from 13–19.967 s. Fast 9–10.867 s is visibly near-static in both body views.

**Mapping and stability:** Both preserve quiet geometry in all fifteen fast hold samples; candidate does not visibly add busy scanning there. Candidate shifts the outer cyan shafts inward and crosses/merges their projected lower portions with adjacent blue shafts, while keeping the two central blue shafts separate. Baseline retains a more rigid separated upright fan; candidate shows stronger regrouping/crossing when the source turns and raises arms late in the window. The fan remains a fan, not a readable literal body drawing.

**Visibility:** Body and lit shafts fit the sampled panels. Much of the stage is very dark, especially early; no obvious broad white clipping. Candidate side intersections look brighter/denser, so stronger visibility is not an isolated geometry preference result. Fixed source bulbs remain in their row; these are not moving body joints.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/holds_005_0_reference_planned/chronological_1hz_last_{000,001}.png`, `fixed_fast_{000,001}.png`; native fast evidence `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_005_0_reference_planned/native_decoded_*.png`. Raw observation: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_005_0_reference_planned/observation.json`. Fast evidence is source 9–10.867 s (source video frames 270–326).

## holds_067_0_reference_planned

**Source body:** Upright near-still stance initially. By 7–10 s the screen-left arm extends while the other is lower; all fifteen fast samples show small elbow/wrist changes, then the screen-right arm opens at 10.467–10.867 s. Wider arm poses and leaning/stepping appear later.

**Mapping and stability:** Baseline stays close to its symmetric repeated fan despite the asymmetric body. Candidate carries clearer asymmetry: the screen-right cyan/blue projected pair overlaps strongly around 9–10.333 s, then separates as the lower arm opens at 10.467–10.867 s. Opposite outer pair is more separated and central shafts remain distinct. This is limited pose correspondence, not a literal joint-to-shaft reconstruction. No large gratuitous sweep in the fast held stance; modest extra motion aligns with visible arm changes. Later sampled bends produce stronger candidate convergence/crossing while baseline changes less. Persistent fixed fixture row is not a defect by itself.

**Visibility:** Full body and six principal shafts visible in fast samples, with legible native captions. Early samples are dim; later cyan/purple beams become much brighter in both arms. Right-pair overlap is visible as superposed luminous shafts, not inferred from an endpoint statistic. No obvious large clipped white beam region; brightness is unmatched across geometry arms.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/holds_067_0_reference_planned/chronological_1hz_last_{000,001}.png`, `fixed_fast_{000,001}.png`; native fast evidence `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_067_0_reference_planned/native_decoded_*.png`. Raw observation: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_067_0_reference_planned/observation.json`. Fast evidence is source 9–10.867 s (source video frames 270–326).

## holds_078_0_reference_planned

**Source body:** After initial arm-down stance, source settles into a crossed-leg asymmetric held pose with screen-left arm extended and other arm folded near chest. That pose persists through all fifteen fast 9–10.867 s samples with small hand/elbow changes, and most 2–17 s chronological samples. It bends and changes arms at 18–19.967 s.

**Mapping and stability:** Candidate distinguishes this held asymmetric/crossed pose from neutral: lower blue shafts form a persistent convergent V and screen-right cyan shaft cuts diagonally through the central blue group; screen-left cyan remains separate. Baseline largely retains its symmetric upright fan despite crossed legs and asymmetric arms. This makes candidate grouping more pose-specific, but the center is visually tangled and some individual shafts lose separation. The V and crossing remain stable in every fast sample rather than adding busy scanning. Stronger changes late coincide with source pose changes; increased motion itself is not credited.

**Visibility:** No sampled body cropping or obvious broad white clipping. Beams are dark for much of this clip. Native fast views confirm real projected luminous superposition through the central shafts and at the lower V, not an equivalent-circle assertion; this does not mean all six beams collapse. Central overlap and brightness make individual shaft tracking harder. Fixed bulbs stay separate on the upper row.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/holds_078_0_reference_planned/chronological_1hz_last_{000,001}.png`, `fixed_fast_{000,001}.png`; native fast evidence `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_078_0_reference_planned/native_decoded_*.png`. Raw observation: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_078_0_reference_planned/observation.json`. Fast evidence is source 9–10.867 s (source video frames 270–326).

## holds_083_1200_reference_planned

**Source body:** Saved mocap source interval 40–59.967 s. Feet remain close and torso stays upright with modest sway. One arm is held bent overhead while the other makes repeated elbow/forearm extensions and raises. Fast source 49–50.867 s is not completely frozen: the screen-right arm extends almost horizontally then lifts and bends back.

**Mapping and stability:** Candidate keeps a pose-specific asymmetric crossed arrangement, unlike baseline’s mostly upright symmetric fan. Its screen-left cyan shaft traverses both central blue shafts, the lower blue pair is a narrow V, and the right cyan stays separate. In all fifteen fast samples the left held-arm geometry stays approximately steady while the right cyan aim shifts with the changing right-arm pose. Some response is also visible in baseline, but candidate has a larger offset and more regrouping. This is not indiscriminate large scanning during a hold. However the recurrent center crossing makes the result cluttered rather than a clearly readable body; no general visual acceptance follows.

**Visibility:** Body and shafts stay inside the reviewed views; source bulbs are fixed fixture positions. All native fast captions visible. Both arms remain dark blue/cyan with brightness pulses across chronological samples; no obvious broad white clipping. Candidate projected central shaft overlaps are directly visible and degrade separation, although several beams remain distinct. No brightness-matched reference control separates overlap brightness from visual preference.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/holds_083_1200_reference_planned/chronological_1hz_last_{000,001}.png`, `fixed_fast_{000,001}.png`; native fast evidence `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_083_1200_reference_planned/native_decoded_*.png`. Raw observation: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/holds_083_1200_reference_planned/observation.json`. Local fast 9–10.867 s corresponds to source 49–50.867 s (source video frames 1470–1526).

## Limits and provenance

- Silent sampled image audit only: chronological 1 Hz plus last, and fixed 7.5 Hz samples at local 9–10.867 s. Not continuous playback, AV synchrony, music fit, or human acceptance.
- Same saved reference mocap body, music, optics and controls in both mapper arms; not a generator comparison. No hidden legacy lighting comparator.
- Reference pairs are not display-brightness matched. Common non-angle controls do not imply equal emitted image brightness after changing geometry.
- Projected luminous crossings are assessed from decoded pixels; they do not prove physical 3D beam intersection or an overlap area. No inference from equivalent-circle scalars or raw endpoint distances.
- Chronological and fixed-fast overview sheets were attached at 1200x758 from 1920x1212 (JPEG quality 82). All fifteen fixed-fast samples per entry were additionally viewed as 640x1080 lossless panel stacks, with each body/stage panel at native 640x360 and no resize.
- No claim about unsampled frames, hardware, pipeline qualification, default promotion, or general visual preference. Camera crop alone would not establish anatomical invalidity.

`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/holds/evidence_manifest.json` records receipt/source PNG/sheet/derived PNG SHA-256 hashes, dimensions, exact samples, source times and source video frames, crop coordinates, stack positions and viewing downsampling. Source PNG and sheet hashes were checked against each render receipt. `prepare_native.py` records the standalone CPU preparation. Source receipts and images were not modified. No main review was submitted by this worker.
