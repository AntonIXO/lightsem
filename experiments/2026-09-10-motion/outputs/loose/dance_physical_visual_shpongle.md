# Full Shpongle physical visual audit

## Finding

Physical reduces some large pose excursions and delays the early masked-arm drift. It does **not** solve full-track body drift. The oblique physical body clips from about118s and is absent for much of the later track. In the270s FAST window, prior oblique is still visible near the edge while both new obliques are absent. The frontal bodies remain articulated. Lighting moves, but stays close to one repetitive six-shaft fan. This is not visual acceptance or a promotion recommendation.

## Instrument and limits

AI visual observer; all279 frozen1Hz-plus-last overviews were attached at1200x405 from3200x1080 (automatic resize), then180 native640x720 body+closer crops were inspected from four fixed FAST windows. FAST provenance label8Hz means actual every4 frames at30fps =7.5Hz. CPU full video+audio decode passed; no listening and no continuous audiovisual playback. No music-fit, perceptual AV-sync, hardware safety, conditioning, human acceptance or default-promotion claim. Direct audio dynamics enabled; arms/rows/windows not brightness matched. Main119-window conditioning failure was not a visual-quality scoring criterion. This is sampled diagnostic evidence only.

No SOURCE ground-truth motion exists. The historical incumbent contains only2D body decodes; its lighting is explicitly UNAVAILABLE. That lighting comparison remains unmet. Compare world arms within each optical row, not between rows. No TEST, vault, new inference, GPU renderer or review stage was called by this worker. Source stages were read-only.

## Integrity and full software decode

- Authenticated receipt payload, movie and every frozen sample SHA256 verified.
- Full software FFmpeg video **and audio** decode,2threads: exit0. This checks decodability, not listening.
- FFprobe decoded video:8319frames,30fps,3200x1080,H.264,277.300000s.
- Audio:AAC,stereo,44100Hz,277.300000s. Container277.300000s.
- Frozen samples279:every30frames from0 through8310,plus8318. All actually viewed.
- Native FAST:60source frames ×3arms =180crops. All viewed at640x720, no spatial resize. Every4frames gives7.5Hz, not literally8Hz.
- Movie SHA256:`cc5261d337c9fb2dde409a88ea1b9168220405076a6644eb6987840798cf4062`.
- Render receipt SHA256:`8ea0c5440cef73a21809242e9b0e6b0df2d84fd35f47f583a4562be7426e33df`.

## Within-row comparisons

### body

Prior, masked_native and physical remain articulated, with repeated upright/bent-knee poses and arm gestures. Physical usually reduces large leans and overhead/knee excursions relative to prior/masked; this looks more restrained, not proven more lifelike. It delays major oblique drift versus masked, but is not drift-free: physical starts clipping near118s, often loses the oblique body later, and has22 off-panel anchors at270s. Prior oblique is visible at the late FAST window when both new obliques are absent. Front projections remain recognizable. Old incumbent is frontal2D only at another scale; its motion differs, but no calibrated physical superiority comparison is available. No SOURCE motion exists for Shpongle.

### original_legacy

Within this row, all three native arms retain a small blue/purple fan, initially very dark and more magenta after~140s. Changes of shaft direction are visible, notably166s and199-201s, but the family is repetitive. Physical does not visibly supply a new layout family. Broad dim light masses hide detail; no pixel clipping or brightness score is claimed. Old incumbent lighting and SOURCE lighting are unavailable.

### closer_zero_angular_flux

Native640x360 optical panels show fixed fixture bulbs but moving beam shafts/endpoints. The six-shaft family persists: two short inner shafts, two longer near-vertical shafts and outer diagonals, with small skew/length changes. Prior/masked fan is less convergent than physical in196-197.867s; central/outer angles differ.80-81.867s outer blue endpoints sweep around long inner shafts.270-271.867s returns to dim blue/purple fan. Motion is not literally static, but change is mostly constrained modulation of one layout, not diverse staging. Brighter magenta and closer presentation are not independent evidence of better optical quality. Old incumbent and SOURCE lighting unavailable.

## Frozen chronology

- 0-39s: all three world arms vary recognizable upright/lean/crouch and overhead/lateral arms; physical stays closer to centre/upright; masked accumulates depth displacement. Early lights very dim blue fan; static fixture points not evidence of static beams.
- 40-54s: masked oblique body reaches/crosses right panel edge around45s; off-panel warning increases, while physical remains contained. Legacy row very dark; closer shows blue inner vertical/outer diagonal shafts, same broad family in all world arms.
- 55-99s: masked oblique repeatedly disappears/crops while front remains recognizable; physical retains both projections. Around60-70s lights brighten from dim blue into purple-centre/cyan-outer fan. Within-row small shaft angle/length differences; no new layout family. Old 2D silhouette changes but incomparable calibration.
- 100-129s: prior oblique reaches/crops right edge by117s; physical oblique near edge then partial cropping from118s onward; physical not drift-free. Masked oblique already absent; frontal bodies remain visible. Lighting dims113-117s then brightens118s, preserves six-shaft fan with small asymmetric angle changes.
- 130-159s: body drift remains visible in oblique: physical partly/mostly clipped135-136s then returns near edge144-158s, prior oblique reappears intermittently. Fronts show small steps, bent knees and recurrent shoulder-height bent arms. Brighter magenta wash from140s makes beam shafts clearer but does not diversify layout.
- 160-189s: deep knee flexion160-163s followed by upright/raised-arm returns. Physical front generally less extreme, but oblique disappears again160-174 and184-189s. At166/172/179s prior and masked fans skew shafts more visibly than physical; fixture points stay fixed, shafts vary. Still same six-light fan family.
- 190-219s: front poses continue; physical smaller leans versus prior/masked around193-200. Old2D crouches201 when native fronts upright; no correspondence GT. Prior/masked rightmost shaft becomes short/upward199-201s, physical follows less persistently; returns to familiar fan by202s. All oblique views problematic by212-219.
- 220-249s: mostly upright alternating narrow/wide stance and shoulder-arm gestures. Physical frontal position drifts modestly right and stays visible, but oblique now consistently absent; masked oblique absent, prior edge fragments/reappearances. Lights highly repetitive purple fan with modest outer shaft angle/endpoint changes.
- 250-end: crouch/forward fold256-259s then upright/reach sequences. Physical less deep flexion than prior/masked257s. Prior oblique returns267-274s; physical oblique absent through end, frontal right-shift remains. Lighting fades262-266s, briefly returns267-269s then dim blue-purple ending. Last277.267s valid nonblank body and fan.

## Fixed native FAST inspection

- FAST0-1.867s (45 native crops): all fronts/obliques fully visible. Prior/masked larger overhead extension, physical often shoulder-height bend and reduced lean; coordinated knee lift and arm transitions remain articulated, not frozen. Very dim blue/cyan six-beam fan; shaft endpoints/angles change little despite body gestures. Thin angular sticks do not establish weight/support realism.
- FAST80-81.867s: masked oblique progressively clips14 to22 anchors while prior/physical0; front masked offset left. Feet/legs and bent-arm rises progress coherently across samples; physical restrained elbow elevation vs prior. Blue outer shafts cross toward long inner blue shafts; ends subtly sweep/shorten while bulbs fixed. Purple central pair persistent, no new layout family.
- FAST196-197.867s: masked oblique absent22 anchors, prior partial8-18, physical partial15-21. Physical front more upright with reduced side lean/knee lift vs prior/masked196.533 and197.467-197.733. Prior/masked shafts straighter/less convergent than physical; central pair shorter, outer pair angles differ visibly. Within-segment beam motion modest, not frozen, same six-shaft family. Native optics expose grainy shaft edges, bright rounded/flat endpoints on dark wash.
- FAST270-271.867s: prior oblique visible at extreme right0-2 off-panel anchors; masked/physical absent22 (physical23 once). Physical front farther right than prior, restrained shoulders/knee lift; prior/masked show stronger whole-body lean and one-leg lift270.4-270.933. Some sampled feet sit above floor line; not enough temporal/3D support evidence to certify contact or physical realism. Dim blue outer/purple inner shafts brighten slightly at end; endpoints visibly shift but still same fan.

## Artifacts

- Source:`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review_support_revision_01/renders/full_shpongle_draw7001`.
- Observer JSON:`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/physical_visual_audits_01/shpongle/observer_full_shpongle_draw7001.json`.
- Observer SHA256:`b16eb62ff5fff8f06e9cc15391d831367fce101d1c6c5c19b183babd65e10e48`.
- FAST provenance, exact frames, crop boxes and all crop SHA256s:`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/physical_visual_audits_01/shpongle/fast8hz_inspection_provenance.json`.
- FAST provenance SHA256:`672bf69cae42d42a0b9caf0de183dd939f83aedec660e74f0d040a7a317a5068`.
- CPU logs/probe/integrity:`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/physical_visual_audits_01/shpongle/{decode.log,ffprobe.json,ffprobe.log,integrity.json}`.
- Extraction command, script, raw frames and180native crops:`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/physical_visual_audits_01/shpongle/fast8hz_native`.

Observer coverage is an AI attestation, not machine proof of viewing. Parent can submit this observer to the frozen reviewer; this worker did not do so.
