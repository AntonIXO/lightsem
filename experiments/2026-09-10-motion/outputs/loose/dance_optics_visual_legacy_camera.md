# LightSem optics visual audit — legacy camera

## Scope and result
AI sampled-image inspection of six `legacy_*` arms. **Not visual, human, music, or hardware acceptance.** No rerender, GPU work, vault write, frozen-output edit, or TEST access.

Positive: beam-body direction and endpoint changes are visible below stationary bulbs. Angular edges separate the central cones. Zero zoom plus angular flux exposes thin shafts and small tips.

Negative: the observed display remains a small centered six-source row and repeated paired fan. Sharper/brighter beams do not demonstrate a new spatial arrangement. Zero-angular-peak is very dim; bulbs can dominate while the moving bodies nearly disappear. No quality or preference ranking is valid because these arms are not brightness matched.

## Exact viewed coverage
- All 10 chronological review sheets in each of six arms: 60 sheets, 278 samples/arm, 1,668 samples total. Seconds 0 through 277 inclusive at 1 Hz. Page names `review_000.jpg` through `review_270.jpg` in increments of 30. First nine pages have 30 samples; last has 8, with black padding not counted.
- Sheets start from 640×360 raw frames, reduced to 320×180 tiles using INTER_AREA. Each 960×1800 JPEG sheet was automatically attached as 640×1200 JPEG, quality 82. Effective tile size is about 213×120. These are not native-resolution inspections.
- All 45 fastframes/arm viewed: 270 native 640×360 PNG attachments. No attachment resize was reported. Frames were decoded from the existing MP4, not rerendered or exposure-enhanced.
- Fixed intervals [80,82), [196,198), [270,272), every fourth frame at 30 fps = 7.5 Hz. Fifteen images/interval/arm, ending at 81.866667, 197.866667, 271.866667 seconds.
- Exact zero-based video indexes: 2400,2404,…,2456; 5880,5884,…,5936; 8100,8104,…,8156. Files `fastframes/<arm>/sample_001.png`…`sample_045.png` map in that order. All exact paths, indexes, times, dimensions, SHA-256 hashes are in `fastframes_manifest.json`.
- This is chronological static-image inspection, not continuous AV playback or listening. No music-fit, flicker-free playback, or full temporal motion claim.

## Controlled distinctions
- **Camera:** every reviewed arm has the same legacy camera, scale 0.85, position [0,2.6,11], target [0,3.2,-3], orbit 0. Camera-only effect requires the separate closer-camera audit; it is not inferred here. The fixed source row occupies roughly x232–407 at y129 in the native frame.
- **Zoom only:** native versus zero under each identical edge/flux setting. Zero produces narrower bodies without moving bulb origins. Under legacy edges the change is modest and diffuse overlap remains. Under angular constant-peak, zero greatly reduces visible bodies. Under angular flux, zero yields thin, conspicuous shafts.
- **Edge only:** angular peak versus legacy at a fixed zoom. Angular creates straighter, narrower boundaries and darker gaps. Native angular separates the central round-ended cones that merge in the legacy wash. It is also darker; this is not an isolated preference test.
- **Constant peak versus flux:** angular flux uses reference 14° instead of 0°. It restores body brightness while retaining angular width and direction, especially at zero zoom. Source glows remain visually fixed. This is renderer solid-angle normalization, not lumen or hardware calibration.
- **Motion versus layout:** source glows stay in a horizontal row. Lower shaft/footpoint changes are separate from bulb brightness. At [196,198), the second/fifth lower ends drift left then right; the last sampled rightmost ending lifts. These are modest local movements, not relocation of the rig or a wide stage sweep. Thin bright flux shafts also expose visible speckled texture; its cause is not assigned.
- **Frozen source:** all six receipts report angle hash `3148e7fde21b4d91e73884e1f0c8fe6597e88eb701d2d368f2d42ffc2026a8fe`. Native arms keep source DMX; zero arms alter only channels 9,21,33,45,57,69 (zero based). No new learned motion was generated.
- The supplied raw geometry proxy passes **2.3958× versus legacy_native_legacy** for the combined closer/zero/angular-flux arm. It is not a perceptual acceptance measure and is not a legacy-camera-only result.

## Arm observations

### `legacy_native_legacy`
- 1 Hz: All ten pages viewed: dim blue small central cluster 0-59; rises blue/violet 60-89; violet/magenta 90-260 with modest alternating cone lean and footprint separation; dips 114-117 and264-266; dimmer late270-277. Six source dots remain horizontal; broad cone fan changes, not whole rig relocation.
- [80,82): 15 native frames80-81.8667: broad blurred side pairs and merged central purple pair; small left/right rocking at footpoints; fixed six bulbs. No whole-array motion.
- [196,198): 15frames: magenta paired cones; second/fifth footprints drift left then right slightly, central pair stays merged. At last sample rightmost termination lifts. Fixed bulb row; no large travel.
- [270,272): 15frames: much dimmer blue/violet; same fan, slight side footprint shifts; intensity increases toward end. Very little visible layout change.

### `legacy_native_angular_peak`
- 1 Hz: All ten: same small centered fan and temporal color/intensity progression. Angular peak has narrower, more separated triangles and less diffuse fill than legacy edge; often darker. Early 0-59 nearly hidden. Lean changes visible late, horizontal source row remains. No large moving layout.
- [80,82): 15frames: sharper narrower triangles, central pair separated by dark gap vs legacy. Small side pair tilt changes; some overlapping side triangles remain. Same bulbs.
- [196,198): 15frames: central ends are rounded separate blobs rather than merged wash; side footprints slide slightly horizontally and terminal rightmost lifts at197.8667. No broad layout travel.
- [270,272): 15frames: very dark but central pair separately visible; small footprint drift and final intensity rise, fixed source row.

### `legacy_native_angular_flux`
- 1 Hz: All ten: angular boundaries remain narrower than legacy; flux makes triangles more conspicuous than peak without changing six-point source row or apparent directional sequence. Clearer separated blue wedges66-74 and magenta wedges150-261; low early and late samples still occupy same small central patch. No spatial-layout transformation.
- [80,82): 15frames: same angular shapes and small shifts as peak, visibly brighter body. Central teardrop terminations separated; side cones can still overlap.
- [196,198): 15frames: magenta footprints distinct, second/fifth move left then right under fixed bulbs. Central round-ended cones and rightmost lifted ending visible. No new layout.
- [270,272): 15frames: dim blue/magenta pairwise arrangement persists with subtle shifts and final brightness rise; brighter than peak but no direction change attributable to flux.

### `legacy_zero_legacy`
- 1 Hz: All ten: zero zoom gives slimmer, somewhat dimmer fan than native legacy, not changed source positions. Same blue-to-magenta progression, same dim breaks114-117,264-266 and late270-277. Sampled left/right leaning of cone bodies persists, but visual organization remains repeated central row/fan.
- [80,82): 15frames: slimmer than native legacy but still diffuse side overlap; center partly merged. Slight left/right movement at lower ends, fixed row.
- [196,198): 15frames: narrower magenta cones with less diffuse merger than native legacy. Lower side spots drift and rightmost ending lifts; repeated paired pattern.
- [270,272): 15frames: extremely dim blue/violet, narrower faint cones below unchanged bulbs, slight shifts and end brightening. No large motion becomes evident.

### `legacy_zero_angular_peak`
- 1 Hz: All ten: sharply reduced visible beam area; narrow faint filaments below conspicuous fixed source row. Early0-59 effectively black at sheet scale. Magenta lines around150-261 do change lean, but floor glow and bulbs are prominent relative to thin bodies; no broad rearranged layout.
- [80,82): 15frames: faint thin shafts; outer/inner left pair touch/cross near bottoms; center now narrow separate slivers. Bulbs brighter than bodies. Small shifts visible but low confidence in fine edges.
- [196,198): 15frames: narrow magenta lines with small round endpoints central and thin side footpoints; easier to distinguish endpoint positions than in diffuse legacy, but low brightness. Fixed source row and limited side sway.
- [270,272): 15frames: shafts almost disappear, bulbs dominate. Same weak central fan, slight changes visible but very low confidence in detailed motion; no strong visibility restoration from zero zoom alone.

### `legacy_zero_angular_flux`
- 1 Hz: All ten: thin bright colored shafts and detached small spots are legible; fixed bulbs unchanged. Fine angular zero-width beams have larger dark gaps than native flux. Directional lean varies beneath same row; overall centered fan shape persists. Most evident late magenta shaft separations; early blue still faint.
- [80,82): 15frames: narrow shafts bright enough to expose speckled/grainy beam texture, central round-ended magenta spots, blue side intersections. Lower ends shift slightly; bulbs fixed. Source-to-tip bodies are not fully solid near tops.
- [196,198): 15frames: bright pink narrow shafts and round endpoints; second/fifth drift left then right, central ends change height modestly, rightmost endpoint lifts at197.8667. Six fixtures remain in same small centered fan; no wide sweep.
- [270,272): 15frames: blue/purple shafts remain legible despite dimmer envelope; separate tips, slight side changes and end brightening. Same width/layout class, not new movement. Grain visible in shafts.

## Provenance and limits
All six render receipt hashes match their `.sha256` sidecars. All 60 review-page hashes match both receipt references, and all six MP4 hashes match their render receipts. Existing external media-audit receipts were hash-checked and report exact checks passed. Their full decoder audit was not repeated. `render.json` still contains its historical PENDING fields; this audit does not edit them.

Software extraction used ffmpeg `-hwaccel none -threads 2`, frame-index `select`, `-an`, `-fps_mode vfr`, PNG output with `-threads 2`. Each process exited 0; logs are empty. Per-arm command JSON and logs are retained. MP4 compression remains in the native PNGs.

One labeled AI observer. No blind study or human rating. The 1 Hz sheets cannot establish motion smoothness or rapid event behavior. Native 7.5 Hz coverage is restricted to three short intervals and skips three of four frames. Dim edges are uncertain. Brightness differences are descriptive only. No music qualification or acceptance is granted.

Audit directory: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/legacy_camera/`
- `hash_verification.json`: exact input receipt/page/video hashes.
- `fastframes_manifest.json`: every viewed fastframe path, index, timestamp and hash.
- `coverage.json`: exact viewed sample schedule and scaling disclosure.
- `observer_0001.json`: create-once observer record; `observer.jsonl`: append-only observer log.
- `fastframes/<arm>/command.json`, `extract.log`, `sample_*.png`: CPU extraction provenance.
- `ffmpeg_version.txt`: decoder build information.
