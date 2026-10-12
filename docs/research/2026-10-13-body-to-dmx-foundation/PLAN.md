# Experimental plan

## Falsifiable question

Given the same generated body and song, does a learned multimodal decoder
produce a more source-dependent, temporally music-aligned, geometrically
diverse and capability-valid light stream than body-only, audio-only,
nearest-additive, and fixed-mapper controls, without raw-body or emitted-stream
safety failures?

The minimum claim is a health result on held-out songs. It is not a claim that
the model matches a professional lighting designer or that the public data has
complete DMX supervision.

## Modality profile

| Modality | Atomic element | Native clock/structure | Information available | Main missing or noisy parts |
|---|---|---|---|---|
| Native dance body | 22 joints × 3D position plus native latent/contact channels | 30 Hz sequence, root-relative and world geometry | body gesture, support and spatial intent | generated-body bias; no light labels; raw safety must be checked |
| Audio context | 92-channel causal bus plus frozen MuQ layer 9 | 40 Hz bus / 25 Hz MuQ, aligned to source time | beat, spectrum, source/section cues and music semantics | stereo position is not physical DOA; encoders are frozen inputs |
| Laban/effort features | effort factors, phase, phrase and accent descriptors | derived per-frame/beat sequence | interpretable weight/time/space/flow cues | shared vocabulary can be unconditioned; never an autonomous motion source |
| BeatSaver light | virtual groups, cue events, intensity and RGB | 40 Hz rendered event grid | authored timing, group participation, colour and effects | no real fixture geometry; map dialect and mapper variation |
| RPMC-L2 light | HSV histograms and reduced scene vector | dataset light-frame grid | global hue, saturation, value and concentration | no pan/tilt, fixture identity, or raw DMX; target is masked |
| Rig/capabilities | fixture type, position, channel mask and count | static set plus decoded 40 Hz DMX | maps group intent to available actuators | arbitrary patches can make channels unidentifiable |

## Architecture

The MVP has a shared causal temporal trunk for body, audio and effort streams;
separate modality projections; low-rank multiplicative cross-modal fusion; and a
set decoder with group/capability tokens. The decoder emits:

- group aim/effect parameters only when the target and rig can observe them;
- brightness/value and hue/saturation heads;
- cue/event logits and effect rate/phase parameters;
- uncertainty and observation masks used by the loss and audit.

A deterministic capability-aware renderer expands group effects to any number of
fixtures, then encodes and decodes DMX. Fixture permutations within a type must
be equivariant; mirror augmentation is allowed only when the rig metadata is
mirrored too. A nearest-additive projection is trained/evaluated as a diagnostic
so a complex fusion block cannot receive credit for an additive shortcut.

## Data and split

The first corpus uses the existing 300 usable BeatSaver maps and the 699-song
RPMC-L2 extraction. Split by song identity before windows, then audit artist,
mapper, map hash and duplicate-song groups. BeatSaver target windows require a
real map audio hash and a parseable authored-light file. RPMC windows use only
the histogram target mask. No window from a song may appear in train and test.

For each BeatSaver song, the saved waveform is passed through the pinned native
audio extraction and an explicitly identified fixed-C body generator. Every raw
body is safety-audited before any mapper, interpolation, DMX, or video step. A
failed body is retained as a quarantined diagnostic and is not used for training.

## Training stages

1. **Data audit:** finite values, clocks, duplicate songs/maps, target masks,
   event rates, palette occupancy and angular observability.
2. **Controls:** body-only, audio-only, constant/mean target, shallow MLP or
   random forest where dimensionality permits, and fixed LightSem mapper.
3. **BeatSaver pilot:** one seed, small causal model, masked group/event/value
   losses, with an overfit-one-window smoke before longer training.
4. **Masked joint fit:** add RPMC scene-vector loss and keep pan/tilt losses
   disabled where RPMC cannot observe them; monitor per-source gradients.
5. **Held-out evaluation:** beat-aligned timing, target reconstruction, angular
   coverage, source interventions, duplicate streams and raw safety.
6. **LightSem transfer:** run only after the machine gates pass; preserve P2,
   mapper 01, restored brightness, limits-off simulation and both own-song
   trajectories.

The first run uses one seed, as requested. A second seed is a future
confirmation requirement, not a hidden promotion criterion.

## Metrics and gates

**Safety and provenance:** all finite; native 30 Hz speed/acceleration/jerk,
root/support and high-frequency residual within the selected-baseline ratio;
body, audio, checkpoint, noise, mapper, rig and code hashes recorded. A raw
explosion or invalid range is `rejected_raw_safety` and cannot be rendered.

**Music and source dependence:** beat-aligned cue AP/F1 and phase error;
cross-song and same-BPM controls; shuffled/donor/time-shift audio controls;
frozen-body and limb perturbations. A new soundtrack on reused DMX is invalid.

**Geometry and activity:** decoded pan/tilt angular range, circular coverage and
occupancy entropy per fixture/group; active-frame fraction, inactive runs,
brightness/value distribution, colour transition count, palette occupancy and
exact/near-duplicate output streams. RGB and brightness are reported separately.

**Collapse and fusion:** sample effective rank and pairwise cosine diversity;
event inactivity and saturation; nearest-additive projection gap; modality
ablations and gradient reach. Zero event count alone is not called
representation collapse.

**Generalisation:** held-out song/map/artist groups, unseen fixture counts and
capability masks. A passing downstream video smoke cannot override a failed
raw-body gate, and nonzero motion or differing hashes cannot establish
choreography.

## Historical controls carried into this protocol

The earlier H-I group decoder is a refuted, recipe-specific control, not a
qualified baseline: its recorded FID/BAS gates failed and its regression head
collapsed within-clip intensity. The BeatSaver-only H-H admission test was
directional but missed its shifted-target and beat-trigger margins. These
results remain in the vault trace and must be reproduced as explicit controls
when a new target or body-conditioned mechanism is claimed. Laban method 15 is
available as a fixed feature transform from `body_transfer_v2`; its previous
crossing reduction passed engineering gates, while the later 3.29°/3.42°
movement-readability result is a mandatory amplitude/angle control. Stereo
variants must be evaluated root-relative and in decoded angular DMX because
nonzero world RMS and different hashes previously produced perceptually
indistinguishable output.

## Stop conditions

Stop and retain the artifacts if the public lighting target cannot be aligned to
the generated body, if the model is additive in disguise, if angular outputs
collapse or duplicate, if source interventions do not alter assigned outputs,
or if either native safety or provenance is incomplete. Thresholds are not
lowered after observing a result.
