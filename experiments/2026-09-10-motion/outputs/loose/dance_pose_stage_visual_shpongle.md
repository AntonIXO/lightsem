# Shpongle frozen pose-stage: independent silent audit

## Outcome

**Candidate geometry is visibly more expressive than matched stage02, including after global dimming. Overall visual acceptance is not established.** The change increases outer-shaft sweeps, asymmetric grouping and projected crossings. It does not remove the dominant six-fixture fan or the relatively steady central pair. More motion is not, by itself, better lighting or better music fit.

Both entries use the same saved `prior_native_flow` body, music, optics and controls. Only the mapper differs (`articulated_stage` versus `pose_stage`). This is not a new generator comparison. Shpongle has no ground-truth body here.

## What is visible

- **Across the track:** baseline repeatedly shows a near-upright central pair and two outer paired arrangements. Candidate outer shafts sweep farther inward and change their grouping, often crossing neighboring shafts in the rendered image. Both remain recognizably fan-like. The fixed row of bulbs is the physical source arrangement, not evidence that shafts cannot move.
- **0–1.867 s:** changing body arms coexist with nearly held beams in both arms. Candidate has a more inward V arrangement. Neither visibly adds a large sweep in these quiet samples. This is not evidence of faithful instantaneous arm tracking.
- **80–81.867 s:** body arms move through bent/asymmetric/open positions into an overhead shape. Candidate outer blue shafts move through broader inward trajectories. At 81.467–81.867 s they visibly merge/cross near the lower central region while baseline outer ends remain farther apart. Candidate still has a comparatively steady magenta core. This is coarse pose correspondence, not a literal readable skeleton or measured per-joint fidelity.
- **196–197.867 s:** candidate starts with an inward crossing arrangement and opens as the body lowers/opens its arms. Baseline changes less. At 196–196.267 s the body configuration and candidate layout both change relatively little; no gross unrelated jump is apparent in those samples. Continuous hold stability and lag remain unassessed.
- **270–271.867 s:** restrained changes in both arms. Candidate retains an inward V and changes outer-shaft direction, rather than making a large uncontrolled sweep.
- **Visibility:** no obvious white wash or stage-edge truncation of the six main shafts in the inspected samples. Early and late blue shafts are dim and harder to read. Middle-track magenta shafts are much clearer. Broad cone bodies sometimes merge visually. These are image-plane shaft observations, not equivalent-circle overlap statistics or claims of actual 3D beam intersection.
- **Body framing:** the front view remains usable in these samples. The oblique view drifts toward and beyond its right edge in many later samples, including 196 s. This limits visual correspondence checks. Camera cropping does not establish anatomical invalidity. No caption defect is asserted.

## Brightness control

Original outside-bulb mean: baseline **2.9368334504**, candidate **3.1405769729** (+6.94%). That entry alone has a brightness confound.

The globally matched entry uses one full-record scalar per arm, **[1, 0.9351254485075525]**, not per-frame normalization. Its means are **2.9368334504 versus 2.9620804781**. The +0.0252470278 (+0.8597%) residual comes from quantization, per the receipt. Both original raw streams have identical repeated-frame and whole-stream hashes before dimming, per the matched receipt.

The larger candidate trajectories and crossings remain visible after dimming. Thus the geometric difference is not explained solely by the original global mean-brightness gap. Local luminance, spatial support and perceptual salience are not matched. This control does not establish preference or music acceptance.

## Coverage and evidence

Completed **2/2 entries**. For **each** entry I viewed:

- All 24 `chronological_1hz_last_000.png`–`023.png` sheets: 279 frames, 0–277.2667 s.
- All five `fixed_fast_000.png`–`004.png` sheets: 60 frames, stride 4 at 30 fps (7.5 Hz), from intervals [0,2), [80,82), [196,198), [270,272).
- All 60 fixed-fast frames again as **native 640×360 panels**, stacked body/baseline/candidate into 640×1080 PNGs with no resampling.

The exact union is **335 distinct sampled frames per entry**, 670 entry-frame views total. This is sampled coverage over the whole 277.3 s, not continuous viewing.

Evidence directories:

- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/shpongle_prior_flow0_planned/`
- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/shpongle_prior_flow0_global_matched/`
- Native stacks: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/shpongle/<entry_id>/native_<frame>.png`

Provenance:

- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/shpongle/preparation_manifest.json`: receipt/source/derived hashes, native crop coordinates and output dimensions.
- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/shpongle/viewing_provenance.json`: all 335 source PNG hashes verified per entry, exact frame/time lists, sheet lists and source paths.
- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/shpongle/shpongle_prior_flow0_planned_observation.json`
- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/shpongle/shpongle_prior_flow0_global_matched_observation.json`

Native crop boxes in each decoded 1920×360 image are [0,0,640,360], [640,0,1280,360], [1280,0,1920,360]. Destination origins are [0,0], [0,360], [0,720]. Existing sheets use half-width composite thumbnails. `attach_image` displayed each 1920×1212 sheet at1200×758 JPEG quality82. Native stacks produced no attachment resize notices. No new RGB/PCM stream cache was made.

## Limits and status

Silent still-sequence inspection cannot establish continuous smoothness, jitter, exact phase lag, beat fit, held-pose noninferiority or human music-fit acceptance. Numeric gates are separate evidence and do not grant visual acceptance. There is no hidden legacy-lighting comparator, no hardware/pipeline qualification and no default promotion.

Only standalone CPU image preparation and raw audit files were written. Originals and frozen receipts were preserved. **No main review was submitted.**
