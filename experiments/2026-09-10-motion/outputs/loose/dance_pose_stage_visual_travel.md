# Travel: sampled pose-to-beam gains, not visual/music acceptance

**Outcome:** In all four frozen reference clips, `pose_stage` makes shaft grouping and angles more pose-responsive than the same-body `articulated_stage` baseline. This does **not** resolve the persistent fan structure, bounded root travel, or projected beam merges. Brightness is not matched here. No continuous AV or human music-fit acceptance was performed.

This is a silent comparison of two lighting mappings on **the same saved true mocap**, music, optics and controls. It is not a generator comparison. Bulbs are fixed fixtures; the moving features are shafts and their rendered ends.

## Per-entry findings

### travel_027_4200_reference_planned

Source 140.000–159.967s. Fast native evidence: source 149.000–150.867s.

- **Body:** The saved mocap steps and turns while raising/lowering arms and lifting a leg. In native frames 270-326 (source149.000-150.867s), a bent-knee stance becomes a high leg lift at286-294 and then a turn/step with arms spread. World root shifts are distinct from this articulation; pose_stage does not demonstrate unbounded travel transfer.
- **Baseline versus candidate:** Candidate has visibly larger asymmetric shaft sweeps and central regrouping than the baseline persistent separated fan. At270-274 shafts cross in the candidate lower-center cluster; at286-298 the raised leg coincides with an elevated diagonal right shaft while other shafts remain low, a readable but not literal body correspondence. At306-314 candidate lower shafts converge and visibly overlap in projection, strongest310, despite baseline remaining separated. The candidate therefore gains pose-linked shape change but does not eliminate fan structure or overlap. Fixed bulb positions are unchanged. Similar poses286-294 produce relatively stable forms, not a proven static-hold test.
- **Visibility:** All six shafts remain visible in the native fast samples and do not hit panel borders; body is visible there. Candidate longer crossing shafts make a denser/brighter central region than baseline, with no brightness-matched control. This can exaggerate apparent motion/impact. No caption defect is asserted.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/travel_027_4200_reference_planned/` (all four receipt-listed sheets); `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_027_4200_reference_planned/native_*.png` (all 15 frames); raw `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_027_4200_reference_planned/observation.json`.

### travel_035_4800_reference_planned

Source 160.000–179.967s. Fast native evidence: source 169.000–170.867s.

- **Body:** Saved mocap contains stepping, weight shifts, repeated turns and asymmetric overhead/side arm poses. Native270-298 show turns with an approximately maintained one-arm-up/one-arm-out pose, then302-310 curved overhead arms, then314-326 lowered arms and a wider stepping stance. Root position changes and facing rotation are not the same as local articulation.
- **Baseline versus candidate:** Candidate visibly changes shaft grouping and diagonal angles more than the baseline lane-like fan. Native270-290 retains a similar candidate open fan through the repeated arm pose despite the turns, rather than arbitrary large swings. At294-310 the candidate right outer shaft sweeps inward and crosses other shafts as the arms curl; at306-310 the lower-middle beam image is visibly merged. At318-326 the candidate separates again toward a baseline-like fan. This is readable pose-linked regrouping, not literal travel reproduction or uniform overlap improvement. The bulb row stays fixed; a fan skeleton remains in both arms.
- **Visibility:** Native fast panels retain visible bodies and all six lit shafts inside the stage bounds. Candidate central convergence is perceptually denser and brighter; no matched-brightness reference control supports an equal-brightness preference claim. The cropped/small body presentation limits detailed anatomy judgment; it does not invalidate the mocap. No label defect asserted.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/travel_035_4800_reference_planned/` (all four receipt-listed sheets); `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_035_4800_reference_planned/native_*.png` (all 15 frames); raw `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_035_4800_reference_planned/observation.json`.

### travel_127_1200_reference_planned

Source 40.000–59.967s. Fast native evidence: source 49.000–50.867s.

- **Body:** The shared saved mocap steps and travels laterally/depthwise with repeated bent-arm poses, arm lifts and turns. Native270-326 holds roughly one bent arm near the head while stepping and changing facing; small articulation remains. Root world shifts are visible in the fixed-world display, not a different body between lighting arms.
- **Baseline versus candidate:** Candidate gives substantially larger outer-shaft sweeps and lower-center regrouping across the chronological samples than the baseline persistent lane fan. Native270-282 shows the left outer shaft sweeping from center-right back to left as the body rises;286-298 is relatively stable in both despite small foot/body changes. Candidate322-326 draws the two lower shafts inward as the body steps, without erasing the six-head fan. Correspondence is general stance/gesture regrouping, not a reliable visual reconstruction of limbs or root travel. Shaft crossings and projected pair merges remain, including left278 and right318; these are visible rendered beam images, not inferred endpoint overlap.
- **Visibility:** Native fast bodies and shafts remain inside panels. Early chronological samples are visibly darker/bluer than later ones, limiting shape visibility; this common lighting evolution is not evidence of candidate failure. Candidate longer diagonals and crossings change perceived brightness relative to baseline; no matched control here. No caption defect asserted.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/travel_127_1200_reference_planned/` (all four receipt-listed sheets); `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_127_1200_reference_planned/native_*.png` (all 15 frames); raw `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_127_1200_reference_planned/observation.json`.

### travel_135_3600_reference_planned

Source 120.000–139.967s. Fast native evidence: source 129.000–130.867s.

- **Body:** Saved mocap travels and turns with a fairly maintained low-spread arm pose through much of the clip. Native270-326 shows alternating steps/knee lifts while the arms remain comparatively steady. Root x moves from about -1.02 near286 back to -0.14 at322: world travel is distinct from the smaller local upper-body articulation.
- **Baseline versus candidate:** Candidate produces stronger lower-shaft convergence for the leg-lift at282-290, with visible lower-tip contact/merge at290, while upper/outer shafts stay relatively steady with the held arm posture. It reopens by294-302, stays broadly stable through the walking samples306-318, then converges again at326 with the next knee lift. This is a useful local-pose response rather than gratuitous whole-rig movement. Baseline is more lane-bound, sometimes merging outer shafts with adjacent shafts instead. Candidate still looks like a bounded six-shaft fan and does not track the large root displacement as a whole moving stage pattern; pose_stage does not solve bounded root translation. Fixed source bulbs never move.
- **Visibility:** All native fast bodies and shafts remain visible within panels. Much of the first chronological half is dark blue and dim, so thumbnail beam visibility is weaker than the later purple samples. No missing shaft or caption is inferred from those thumbnails. Candidate longer lower shafts and occasional contact alter perceived brightness without a brightness-matched control. Camera/body cropping would be a display limitation, not invalid anatomy.

Evidence: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/renders/travel_135_3600_reference_planned/` (all four receipt-listed sheets); `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_135_3600_reference_planned/native_*.png` (all 15 frames); raw `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/travel_135_3600_reference_planned/observation.json`.

## Coverage and provenance

- Viewed all 16 receipt-listed sheets: two chronological `1hz_last` sheets and two `fixed_fast` sheets per entry.
- Viewed 35 unique frozen samples per entry, 140 total. Exact sorted frame unions and sheet filenames are in each raw observation JSON.
- Additionally inspected **every** fixed-fast frame at native panel size: 15 per entry, 60 total. No selected-best-still substitution.
- Each decoded source is 1920×360. Native evidence stacks unscaled crops `(0,0,640,360)`, `(640,0,1280,360)`, `(1280,0,1920,360)` vertically into 640×1080. Source body / baseline / candidate order is preserved. No raw RGB or PCM caches were made.
- Frozen sheets first resize each 1920×360 composite with `image.resize((960,180))` (no explicit resample argument; Pillow default), then arrange 2 columns × 6 rows with 22px headers into 1920×1212. Provenance: `outputs/pose_stage_sheet_provenance.md`. `attach_image` separately displayed them at 1200×758, JPEG quality82; these were overview views, **not** native evidence. Native stacks were attached unchanged at 640×1080.
- Source video frame `f` is source mocap frame `start+f`, source time `(start+f)/30` seconds. This is recorded per native crop. All source PNG and sheet hashes were checked against each render receipt.

Full source paths, SHA256 hashes, source frames/times, dimensions and crop coordinates: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/evidence_manifest.json`. Viewing accounting: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/viewing_summary.json`. Preparation script: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/travel/prepare_native.py`. The initial sheet attach returned a post-delivery path-type error; its delivered images were viewed and the failure is preserved in the transcript.

## Limits

- Silent sampled visual audit only: not continuous AV playback, human music-fit acceptance, pipeline qualification or default promotion.
- Chronological samples are 1 Hz plus last; fixed-fast samples are frames270..326 every4 frames (7.5Hz) from the frozen30fps movies. Unsampled transients, temporal jitter and exact controller lag are not established.
- One shared saved reference mocap body per entry, same saved music/optics/controls for baseline articulated_stage and candidate pose_stage. No new generator comparison or hidden legacy-lighting comparator.
- No brightness-matched reference control: changed shaft length, grouping and projected beam overlap can change apparent brightness.
- Rendered 2D shaft/beam overlap observations are not 3D intersection tests, scalar equivalent-circle overlap or raw-target endpoint measurements.
- Bounded root translation remains; stronger local pose articulation is not proof of faithful world-travel transfer.
- Body display and two projections limit anatomical interpretation. Cropping alone is not invalid mocap. Brief similar-pose samples are not a formal static-hold stability test.

Raw observations are prepared for the parent. **No main review submission was made by this worker.**
