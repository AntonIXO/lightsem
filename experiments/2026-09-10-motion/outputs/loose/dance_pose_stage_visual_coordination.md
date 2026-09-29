# Coordination pose-stage silent visual audit
## Outcome

The candidate shows larger and more varied shaft-angle changes than the same-body stage02 baseline. This is **not visual or music-fit acceptance**. The recurring inward fan and nearly vertical central pair remain prominent. Large body changes, especially low poses and knee lifts, are not always equally clear in the light shapes.

No persistent whole-group visual collapse was apparent in the inspected samples. Local shaft crossings do occur. That sampled finding does not exclude events between samples.
## Coverage and evidence

- Completed **4/4 entries**, **16/16 listed sheets**, **140/140 unique scheduled decoded samples**.
- Additionally inspected **60/60 fixed-fast frames at native panel resolution**, not whole-image thumbnails.
- Each reference is 600 video frames at 30 fps. Chronological coverage: local frames 0,30,...,570,599. Fixed-fast coverage: 270,274,...,326 (9–10.867 s local, 7.5 Hz). Their union has 35 frames per entry.
- These are true saved mocap reference bodies shared by both mapping arms. This is not generator testing.
- Frozen source images, sheet hashes, receipt hashes, dimensions, source frame/time and crop coordinates are recorded in [/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/evidence_manifest.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/evidence_manifest.json). Every listed source PNG and sheet hash was checked against its render receipt during CPU preparation.
- Original sheets: 1920×1212, with decoded 1920×360 samples resized to 960×180 tiles. Inspection attachment: 1200×758 JPEG quality 82.
- Native inspection: exact crops `(0,0,640,360)`, `(640,0,1280,360)`, `(1280,0,1920,360)` stacked at y=0,360,720 in a 640×1080 PNG. No resizing. Top/body, middle/baseline, bottom/candidate.
## coordination_057_1200_reference_planned

Source interval: 40–59.967 s.

- **Body:** Saved mocap shows upright bent-knee stepping with changing elbow/hand height and torso lean at source 40-54 s, then a marked descent into crouched/floor-level shapes at 55-59.967 s. The fast 49-50.867 s sequence changes arms from out/bent to lifted and back while weight and leg placement change.
- **Mapper result:** Baseline retains a mostly separated, near-vertical central pair with outer shafts moving in narrow lanes. Candidate makes a stronger inward fan with larger outer-shaft sweeps and distinct changes of crossing/grouping: native frames 282-306 show outer shafts visibly crossing other shafts, then reopening at 310-326. This is not six-head collapse: separate central pair and an outlying shaft remain visible. Candidate is more responsive in geometry but compresses many distinct body configurations into a similar inward V. The low-pose transition at 450-599 is not an equally clear global downward/body-shape cue in the light fan. More motion alone does not establish better pose readability.
- **Visibility:** All six fixture sources stay fixed. Visible cone shafts and end regions stay inside the stage panels in the inspected samples; no gross white washout hides them. Overlapping magenta shafts make local intersections brighter, and the candidate often looks denser/brighter centrally. No brightness-matched reference control exists, so stronger apparent activity is not a brightness-independent preference result.

Evidence: [coordination_057_1200_reference_planned.observation.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_057_1200_reference_planned.observation.json); [render.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_057_1200_reference_planned/render.json); all four sheets in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_057_1200_reference_planned`; all 15 `native_*.png` in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_057_1200_reference_planned`.

## coordination_076_2400_reference_planned

Source interval: 80–99.967 s.

- **Body:** Saved mocap remains mostly upright with repeated broad arm raises, asymmetric reaches, torso turns and modest knee/weight changes across source 80-99.967 s. Native fast frames 270-282 show arms spreading and lifting overhead; 286-310 lower them, and 314-326 return to an asymmetric raised-arm shape. Legs and trunk are relatively quiet during this large arm arc, but are not exactly held.
- **Mapper result:** Both mappings reflect the arm-height arc by raising/lowering the outer beam ends. Baseline retains narrow side lanes and a nearly stationary central pair. Candidate carries the changing asymmetry and lower-body/torso shift more visibly: a left-inner shaft sweeps diagonally across the central group at 286-306 while the outer ends move up and down. At 278-282 similar overhead poses produce fairly similar beam shapes, rather than obvious arbitrary jitter. Candidate shows stronger coordinated change, but it still repeatedly returns to an inward fan with two central vertical shafts. Pair crossings are visible; separate end regions remain, so this is not persistent whole-group collapse.
- **Visibility:** Fixed bulbs do not move. All six shafts are discernible at native resolution, including the near-horizontal outer shaft during arm raises; no gross viewport loss or white washout is apparent in inspected samples. Candidate crossings increase local brightness and central density. Reference pair is not brightness-matched, so visual salience cannot be assigned to motion alone.

Evidence: [coordination_076_2400_reference_planned.observation.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_076_2400_reference_planned.observation.json); [render.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_076_2400_reference_planned/render.json); all four sheets in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_076_2400_reference_planned`; all 15 `native_*.png` in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_076_2400_reference_planned`.

## coordination_085_0_reference_planned

Source interval: 0–19.967 s.

- **Body:** Source 0-3 s is a quiet upright preparation. Later samples show stepping, arm gestures and raised knees, then deep forward/low poses at 14-18 s and a large raised-leg shape at 19 s. Fast 9-10.867 s contains a relatively held asymmetric arm shape with small knee bends/weight changes, followed by a clearer arm/leg change at 326.
- **Mapper result:** Candidate shifts the side shafts inward and varies their relative angles more than baseline, which preserves a near-row of separated ends. During the fairly held fast passage, both remain stable without conspicuous invented sweeping; candidate makes small coordinated angle changes while retaining a central pair and separated outer ends. The candidate low-pose sequence at 420-540 reorganizes crossings and arm-shaft height more than baseline, but does not literally reproduce the body silhouette or make the deep torso descent unambiguous. Candidate repeatedly returns to a V/fan. Some pair intersections occur; no persistent all-six visual pile-up is established by these samples.
- **Visibility:** The first four chronological samples are dark in both arms, which limits shaft visibility there; this is not evidence of missing geometry. Fast native images expose all six shafts and distinct ends without apparent gross clipping or white washout. Cyan and purple channels make outer-versus-inner grouping easier to read than in the monochrome reference clips. Bulbs remain fixed. No matched-brightness control was supplied for this reference.

Evidence: [coordination_085_0_reference_planned.observation.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_085_0_reference_planned.observation.json); [render.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_085_0_reference_planned/render.json); all four sheets in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_085_0_reference_planned`; all 15 `native_*.png` in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_085_0_reference_planned`.

## coordination_102_600_reference_planned

Source interval: 20–39.967 s.

- **Body:** Source 20-39.967 s has repeated upright arm gestures, knee lifts and side steps with modest torso lean. Native 29-30.867 s includes clear alternating lifted/bent legs at 274-278 and 306-314, arm openings at 290-294 and 326, and narrower bent-elbow poses between them. There is no sustained still hold in the fast passage.
- **Mapper result:** Candidate shifts the outer/inner-side shafts inward and makes their relative changes easier to see than the baseline narrow side lanes. But the fast passage is still dominated by a recurring six-shaft fan: the conspicuous knee lifts do not turn into comparably conspicuous separate light gestures, and the central pair remains nearly vertical. Candidate motion is visibly more differentiated at some moments but is not a literal limb-by-limb silhouette. Side pair intersections occur in both arms; the native samples retain a separated middle pair and lateral end groups, not persistent whole-group collapse.
- **Visibility:** All fixed bulbs remain in place. Six shafts remain visible with no obvious panel-edge loss or gross white washout in inspected coverage. The candidate has denser brighter regions where shafts intersect, while baseline often leaves larger dark gaps. No matched-brightness reference control exists, so this does not settle visual preference independently of brightness.

Evidence: [coordination_102_600_reference_planned.observation.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_102_600_reference_planned.observation.json); [render.json](/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_102_600_reference_planned/render.json); all four sheets in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/coordination_102_600_reference_planned`; all 15 `native_*.png` in `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/coordination/coordination_102_600_reference_planned`.

## Limits and status

- Silent sampled review only: every listed chronological 1 Hz plus last sample and every fixed-fast 7.5 Hz sample was viewed. This is not continuous video/audio playback or human music-fit acceptance.
- Chronological and fixed-fast sheets were viewed at 1200x758 after attach-image resized their 1920x1212 originals. Original sheet tiles reduce decoded 1920x360 images to 960x180. Every fixed-fast frame was additionally viewed as a lossless 640x1080 vertical stack of three native 640x360 panels, with no resampling.
- These are fixed saved reference mocap bodies mapped by baseline stage02 articulated_stage and candidate pose_stage with the same source and controls, not a new generator comparison. No hidden legacy lighting comparator was used.
- No brightness-matched control is supplied for these four reference clips. Geometry, apparent shaft area and local intersection brightness can affect salience despite shared controls.
- Visual pair intersections mean rendered camera-plane shaft intersections, not measured physical beam-overlap area. Equivalent-circle ratios and raw target endpoints were not used to infer actual overlap.
- No full-frame pixel clipping measurement, continuous hold/jitter bound, detailed anatomical validity verdict, hardware verdict or pipeline/default qualification is made here. Camera cropping by itself would not establish anatomical invalidity.

No missing-caption claim is made. Source bulbs are fixed fixtures; motion descriptions refer to shafts and their visible end regions. No numerical gate was rerun. No observations were submitted to the main reviewer, no vault was accessed, and no source, frozen file, render receipt or default was changed.
