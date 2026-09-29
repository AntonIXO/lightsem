# Dance conditioning: read-only instrument-scope audit

## Status and boundary

This is source/metadata inspection for next-step planning, not an experiment. The closed physical study remains `directional_only`; its frozen conditioning failures and 5% gates are not changed or challenged here. The current pose-stage numeric/media status and pending independent visual reviews are parent-supplied context, not results rechecked by this audit. No music property is promoted by the pose-stage study.

**Instrument versus claim:** the old donor/shift gates compare conditional distributions of TRAIN-standardized, half-second native-motion summaries with one paired motion reference. They do not directly measure fine-timescale beat synchronization of emitted lighting.

No project code was executed or imported. No weights, waveform, motion, DMX, or image/video data arrays were loaded; no scoring, inference, evaluation, render, GPU/hardware access, registration, vault write, commit, or deletion occurred. Only this report was written. Cache inspection used writer source, JSON metadata, and ZIP central-directory member names, not NPZ member contents. Shapes below are writer-contract deductions, not array validation. Existing outcome prose was read, but no numerical outcome is recomputed or tabulated here.

### Reference aliases

All repo-relative paths below resolve under `/root/lightsem`.

- **P** = `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01`
- **E** = `P/evaluation/evaluator.py` (the frozen evaluator actually inspected)
- **S** = `P/source/lightsem/experiments/motion_study`
- **N** = `/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910`
- **Q** = `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/frozen`
- **M** = `lightsem/experiments/motion_study/lighting_map.py`
- **R** = `outputs/dance_summary_probe_20260910T213429Z.py`

`CLAUDE.md` and `docs/research/2026-09-10-motion/PLAN.md` were read. Existing closed-study outcome and limits are recorded in `outputs/dance_physical_final_outcome.md:3–19,31`. This document is not a replacement record or a new closure.

## 1. Exact frozen energy-score measurement

### Representation, source reference, and time support

Native features are 139 channels at 30 Hz: horizontal root velocities `(vx,vz)`, root height, 22 local 6D rotations, four contact probabilities. FK uses a fixed canonical skeleton in metres with Y up, not a subject-specific mesh. Root velocity at t encodes displacement t to t+1; root x/z starts at zero and integrates once within each evaluated sequence. Sources: `S/geometry.py:1–17,30–32,65–81`.

Base92 audio is extracted causally at 40 Hz and linearly interpolated to native frame timestamps. Waveform t=0 maps to frame 0, with no fitted offset/warp. The scorer receives standardized 92-channel audio at 30 Hz and a paired `(T,139)` reference slice. It does not consult detected beats. Sources: `S/audio_prepare.py:29–46,63–71`; `E:335–360`.

The reference is the original target-window motion under **all** correct/donor/shift conditions. Changing input audio does not substitute a donor-motion target. Both target and prediction are converted to FK joints and the same summary. Physical quality is measured separately for the correct-audio generation; donor/shift ES still gets full FK. Sources: `E:350–360,367–405`.

### The 18 descriptor channels

Let J be world FK joints, v be backward finite differences of J multiplied by 30, with zero v at the first frame, and s be the Euclidean norm of v. Upper set U is joints 12–21; lower set L is `[1,2,4,5,7,8,10,11]`. Names are descriptive aliases: “hand” here uses wrists and “foot speed” averages ankle plus foot. Exact definitions are `S/music_data.py:13–48`; joint identities are `S/geometry.py:13–17`.

| Channel | Per-frame quantity before any log transform |
|---|---|
| root_speed | norm of native `(vx,vz)` (not the FK backward-difference root speed) |
| root_height | native height |
| body_speed | mean s over all 22 joints |
| upper_speed / lower_speed | mean s over U / L |
| left_hand_speed / right_hand_speed | s at joint 20 / 21 (wrists) |
| left_foot_speed / right_foot_speed | mean s over `[7,10]` / `[8,11]` |
| hand_spread | norm `J20-J21` |
| hand_height | mean world Y of joints 20 and 21 |
| foot_spread | norm `J10-J11` |
| torso_lean | horizontal norm of `J12-J0`, divided by its full norm (denominator floor 1e-6) |
| arm_extension | mean of norms `J20-J16` and `J21-J17` |
| contact_fraction | mean of four native contacts after clipping to [0,1], not FK-derived contact event timing |
| vertical_speed | mean absolute Y velocity over all joints |
| upper_energy / lower_energy | mean squared s over U / L |

All speed channels, root_speed, vertical_speed and both energy channels receive `log1p(max(value,0))` **per frame before averaging**. Heights/spreads/extensions are metres; lean and contact fraction are dimensionless; transformed speed/energy coordinates are log1p(m/s) and log1p((m/s)^2).

The transformed sequence is reduced by **nonoverlapping 15-frame arithmetic means**: 0.5 s bins, 2 Hz. A 960-frame/32 s primary crop yields 64 ordered bins × 18 channels. A final incomplete bin is omitted; the 960-frame primary crop has none. Supplemental physical metrics retain every frame. The first FK-derived velocity frame is zero on each crop, while native root_speed is not forced to zero. Sources: `S/music_data.py:27–48`; `E:250,416–417`.

Each summary coordinate is standardized with the same fixed TRAIN mean and SD for reference and all generated conditions. The normalizer gives equal canonical-song weight, equal recording weight within song, and equal half-second-bin weight within recording, over audio-eligible TRAIN. SD has floor .01. No prediction-dependent normalization or record demeaning is used in this generative ES. Sources: `S/music_data.py:62–91`; `E:372–381,402`.

### Score, averaging, and gate

Flatten the **ordered** 64×18 standardized summary into a vector of dimension D=1152. Define `d(a,b)=||a-b||₂/sqrt(D)`. For six generated draws x_i and paired reference y, the actual score is:

`ES = (1/6) Σ_i d(x_i,y) − (1/[2·6·5]) Σ_(i≠j) d(x_i,x_j)`.

The implementation computes distances in float64; generation/FK use FP32. Pairwise diagonal distances are zero, with off-diagonal denominator K(K−1). This is an unbiased sample energy-score estimator. It is **not** an ES of the mean trajectory, an average of independent per-channel errors, or a per-bin beat score. Bins retain order; there is no time-warp or lag optimization. The second term measures sample dispersion in this same reduced space. Sources: `S/music_data.py:51–59`; `E:249–254,377–405`.

For ES, the six draws enter this one window-level score. For separate physical-quality values, `physical()` averages per-draw values (`S/music_evaluation.py:65–85`). Then all window metrics are averaged equally within recording, recordings equally within canonical song, and songs equally (`E:420–438`). Thus the protocol shorthand “mean draws per window” must not be read as averaging the six motion trajectories before ES.

For each control, the conditioning contrast is `(mean ES_control − mean ES_correct)/max(mean ES_control,1e-12)`. Uncertainty is a paired canonical-song percentile bootstrap: 10,000 song resamples, seed 20260910, preserving control/correct pairing after window/record reduction. Each bootstrap computes a ratio of its song means, not a mean of songwise relative percentages. Frozen gates require ≥5% improvement against **both** donor and shift, and both 95% interval lower bounds >0. Sources: `E:441–453,464–468`.

These are the unchanged gates. Their archived failures remain failures; a narrower description of what they measure neither invalidates them nor turns any unmeasured beat property into a pass.

## 2. Controls, cohort, and stochastic protocol

The physical revision imports the existing frozen 119 primary windows from `N/music/protocol.json`; it does not refit or choose them after outcomes (`E:201–222`). Original construction is `S/music_training.py:21–35,75–80`:

- Eligible records: VAL entries with at least 960 frames, sorted by recording key; canonical IDs define song groups.
- Target starts: 0, 960, 1920, … while a full 960-frame crop fits. Trailing partial crops are not primary windows.
- Donor: next recording in sorted cyclic order with a **different canonical song**, not a tempo/genre/energy-matched donor. Donor start is deterministic hashed modulo its valid 960-frame start support. The donor is a contiguous slice, without time resampling.
- Shift: choose the farther of start 0 and `n_frames−960` within the same recording; ties select the later endpoint. Displacement is at least 150 frames/5 s. It is **not circular shifting**, beat-phase perturbation, or a small lag sweep. Target and shifted windows can overlap if the recording permits only limited displacement. No wrap/pad/repeat is introduced.
- Token: integer from first 12 hex digits of SHA256 of `key:start`. Draw i uses `(token+7001+i·104729) mod 2^31`, i=0…5. All six distinct seeds are fixed. Support and canonical distinction are validated (`E:179–198`).

For each target window, six full-length noise fields are generated once, then cloned identically into correct/donor/shift. The same per-window seeds apply across both arms and initial/final checkpoints; frozen report code requires identical noise/audio content hashes and RNG devices across cells. Each condition generates all six 139-channel trajectories with 32 FP32 Euler steps, chunked vector-field calculation but one full sequence state. No draw/window rejection or best-sample selection is allowed; nonfinite generation stops the cell. Sources: `E:26–27,367–417,489–555,630–637`.

Noise is independent across native channels and temporally correlated by a fixed Gaussian kernel: sigma 3 native frames, support 25, unit pointwise variance, stationary random padding. This is a single field per whole sequence, not chunkwise reseeding. Euler updates the full normalized state at each step, then decodes native features. Sources: `S/music_model.py:46–49,70–84,109–125`; `S/music_revision_model.py:49` inherits this generator/noise implementation.

Short records are a separate supplemental cohort, with full real-frame support and available donor but **no same-record contiguous shift**. They are not quietly folded into the primary two-control gate. Full review uses seed 7001 once per complete fixed recording; Shpongle has six fixed full-sequence seeds `7001+i·104729`, each saved as a separate item. These review archives are not a six-draw correct/donor/shift primary cohort. Sources: `E:157–175,245–250,505–537,564–604,638–654`.

## 3. Archived predictability probe is a different instrument

Two archives must not be conflated:

1. `P/evaluation/primary_analysis.json`, JSON pointer `/payload/scope`, explicitly says “Complete119window final-checkpoint primary cohorts only”; it was an intermediate summary, not a new predictability model or final pipeline report. Its writer, `outputs/summarize_physical_primary.py:3–10`, reads the frozen primary summaries and invokes the existing comparison/quality helpers. This audit read its structure/scope, not rerun that script.
2. `outputs/dance_summary_predictability.md` and **R** describe the earlier exploratory deterministic ridge/lag diagnostic. It is not the frozen generative energy-score evaluation.

The ridge probe takes the same physical-summary function at 2 Hz, plus matching 15-frame Base92 means. All offsets share support with eight bins removed at both ends. Its primary model demeans input/target per recording, fits a fixed TRAIN ridge with penalty .1 and TRAIN coordinate scales, and scores standardized MSE averaged over channels/bins. VAL target means are scoring-only/oracle constants, not fitted deployment inputs. A complementary raw zero-lag fit tests global levels against TRAIN-mean and audio-track-constant controls. Sources: `R:17–55,57–68,71–110`; `outputs/dance_summary_predictability.md:17–28`.

Its donor is the next same-split canonical-distinct recording, whose centered audio features are **linearly resampled to target length**. Its shift control circularly rolls the correct predicted summary by 25%, 50%, 75% of support and averages the three **errors**, not predictions. It has no stochastic motion draws, no ODE sampling, and no donor motion synthesis. Equal song/record/bin weighting and paired 10,000-song bootstrap apply; intervals are exploratory. Sources: `R:71–91,112–121`.

The predeclared lags range −4 to +4 seconds with the finest nonzero step 0.5 s. Positive lag means motion t is predicted from audio t+lag. Each lag refits the same TRAIN ridge. It is not independent source A/V synchronization verification, not a fine-phase beat test, and not an information ceiling for nonlinear/context models. Its archived weak-signal/lag conclusions remain scoped to that probe; no scores are reproduced here. Sources: `outputs/dance_summary_predictability.md:79–101`; `R:42–55,71–89`.

## 4. What this does and does not establish about beat synchronization

**It measures:** whether correct audio gives a better distributional match to the paired, temporally ordered 2 Hz native-summary trajectory than these two prescribed wrong-audio inputs, with fixed stochastic sampling and song-level uncertainty. Slow activity, shape, and energy changes can contribute to the contrast. This is a legitimate conditioning instrument in its stated representation.

**It does not isolate:** beat onset/phase, sub-beat latency/jitter, accent choice, or motion-to-music event alignment at 30/40 Hz. Half-second averaging is many-to-one: distinct within-bin descriptor timing can produce the same bin mean. Euclidean norms and regional means also discard directions and joint detail. Keeping 64 ordered bins is not the same as retaining 960-frame phase detail. This is not an assertion that all rhythmic information is erased or that the score cannot respond to a beat-related change; it is an identification limit, not a new sensitivity measurement.

Long contiguous shifts disturb section/envelope correspondence as well as local timing. Different-song donors disturb identity, style, tempo and dynamics together. Neither isolates beat phase while holding other music properties fixed. Conversely, missing the frozen ES gates does not prove zero audio influence, universal absence of beat locking, or failure of every future music model. See the narrower existing wording in `outputs/dance_physical_final_outcome.md:7,31`.

**It does not measure final emitted lighting:** E calls no lighting mapper, DMX encoder/decoder, beat detector, renderer, projected-beam tracker or luminance event matcher. Lighting dynamics are explicitly disabled/not part of the numerical experiment (`E:255,555,651–654`). Downstream resampling, geometry, smoothing, motion limits, quantization and photometry intervene before emission (`M:178–198,245–295`). Native summaries, actual emitted angles, projected spots and visible luminance need separate scope. Their timing cannot be inferred as equivalent from these ES results.

The pose-stage frozen plan itself limits its claim to saved-body geometry/mechanics and excludes new music conditioning (`Q/plan.json:6–9`). It uses twelve reference-mocap windows and saved **prior_native_flow** on Shpongle, not a fresh physical-model control study. Its passed numeric/media checks, as reported by the parent, therefore cannot upgrade this closed conditioning result or establish learned beat synchronization. Visual review remains separate.

## 5. Cache schemas and available temporal support

No array contents or NPY headers were read. The following four example member lists were checked from ZIP central directories at `P/evaluation/physical/final/{stage}/000.npz`, with their `.json` `/payload/spec`, `/payload/arrays_path`, `/payload/effective_draws` metadata. Writer code determines dimensions and sampling rate.

| Archive | Saved contract | Limitation for a later timing question |
|---|---|---|
| `primary/000.npz` | `reference` (64,18); `correct`, `donor`, `shift` (6,64,18) | Summary only. No per-frame predicted features/joints. |
| `short/000.npz` | Same available-condition summaries plus `reference_features`, `reference_joints`, `correct_features`, `correct_joints`, `donor_features`, `donor_joints`; six generated real-frame trajectories | Per-frame data exist, but no shift control for the full short recording. Summary tail is shorter than native support where T is not divisible by 15. |
| `full/000.npz` | `features` (T,139), `joints` (T,22,3), `root`, `contacts`, `times`; matching reference fields | One seed-7001 correct-audio full recording, not six paired wrong-audio conditions. |
| `shpongle/000.npz` | `features`, `joints`, `root`, `contacts`, `times` | One of six separately saved full draws; no paired source motion, donor or shift in these archives. |

Sources: `E:385–410,505–541,580–600`. `stage_inference` passes `save_motion=False` for primary and True only for short. The primary JSON's `prediction_content_sha256` is an identity digest, **not reconstructible native motion**. Noise/audio hashes and seeds do not supply absent trajectories without prohibited regeneration. Other cells share this writer contract; this audit inspected one example per stage, not every archive's contents.

The older native music primary writer also saves only `reference/correct/donor/shift` summaries plus prediction hashes (`S/music_evaluation.py:127–160`), not framewise generations. The ridge diagnostic saves full-record 2 Hz `x` (92-channel averaged audio) and `y` (18-channel summary) plus `models.json` coefficients, not native generated motion (`R:24–31,129–130`). A ZIP-directory-only spot check of `N/diagnostics/dance_summary_probe_20260910T213429Z/001.npz` confirmed only `x.npy` and `y.npy`. Later rescoring of either summary-only archive cannot recover sub-bin timing.

Current pose-stage mapping additionally writes full world joints/times and baseline/candidate DMX/time streams (`Q/eval_pose_stage_study.py:406–420`). These offer distinct pre-/post-mapper temporal traces in principle, but are the fixed geometry-review cohort, not a matched learned-conditioning intervention. They were not loaded here. Any future use requires a separate authorization and an explicit source/control scope.

## 6. Direct-audio paths in the current mapper

The inspected live **M** text exactly matches `Q/source/lightsem/experiments/motion_study/lighting_map.py`. Both frozen pose-stage baseline and candidate call `compose_world_joints(..., dynamics=True)` (`Q/eval_pose_stage_study.py:411–415`). The mode flag changes geometry, not the audio-dynamics path.

The direct signal is measured EBU loudness, **not a learned motion output or an explicit beat clock**: 400 ms momentary and 3 s short-term LUFS, emitted in 100 ms blocks with timestamps at block end (`lightsem/experiments/motion_study/audio_dynamics.py:25–44,61–65`). In M it is interpolated to 40 Hz; a whole-track valid-short-term 90th-percentile anchor calibrates a 70/30 momentary/short-term blend, then smoothstep and a .30 s EMA produce activity (`M:207–214`). The following paths bypass the learned music-to-motion relation:

- **Brightness/activity:** activity sets the dimmer level and moving-head brightness; wash/projector level and blinder/strobe motion-accent gain also depend on it (`M:215,284–307`). Quiet-to-entry brightness can therefore follow music without learned motion timing.
- **Palette/beam parameter:** a slow activity EMA changes hue and the moving-head channel at offset +9, which the fixture schema defines as zoom (`M:216–219,287–295`; `lightsem/dmx/fixtures.py:85–97`).
- **Motion amplitude:** activity scales the legacy and surface targets (`M:147,245–257`). In pose-stage specifically, `a=.25+.75*activity` scales signed lean and vertical/depth articulation terms (`lightsem/experiments/motion_study/pose_stage.py:39–64`). It does not uniformly scale every pose-chart coordinate; anatomical x remains a separate term.
- **Emitted timing:** activity sets the local angular speed budget in the chase controller (`M:95–100`) and the current frozen offline-projection controller (`lightsem/experiments/motion_study/offline_angles.py:31–35`). Thus music can alter emitted path timing even with the same saved body.
- **Interaction with motion:** the accent starts from distal joint speed, but it is multiplied by activity for brightness (`M:228–229,284–285,307`). “Motion-based accent” therefore is not an audio-isolated measurement by itself.

`dynamics=False` holds activity/palette at .7 and removes these direct loudness dependencies from emitted fields in this mapper, though it does not erase any learned dependence already in the supplied body (`M:213–219`). This describes an existing code seam, **not an instruction to rerun an ablation or change the frozen study**.

The direct paths are permitted design features under the current plan. They are not evidence of a bug or leakage in the old ES, which never maps lighting. They confound only an attribution such as “observed audio-synchronous lighting proves learned music conditioning” unless the learned-motion and direct-audio routes are distinguished.

## 7. Narrow future questions, only after pose visual review and parent decision

These are measurement questions, not preregistrations, thresholds, training proposals, or execution authorization:

1. Which time-local musical event is meant by “beat synchronized”: onset, beat phase, downbeat, accent, or section entry? Which observable carries it: native motion event, emitted DMX turn, projected beam movement, or visible luminance?
2. On a separately authorized fixed support, does that specific correspondence survive from full-rate native motion to emitted angles and visible output? Which part is contributed by direct loudness rather than the saved body?
3. Can a control isolate fine timing while preserving the slow envelope/section structure that the existing long-shift and donor controls also perturb? What independent source A/V timing anchor is available before interpreting a sub-second lag?
4. Which existing full-rate archive has the necessary reference and controls? Primary summary caches do not. Full/short/pose archives have narrower, different control support; their existence alone does not authorize reuse or inference to fill gaps.

No answer to these questions is assumed. The closed physical result, frozen 5% requirements, and pending pose visual decision remain unchanged.
