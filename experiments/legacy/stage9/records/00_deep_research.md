# Deep research — datasets, audio decomposition, models, recommendation (Perplexity deep-research pass, 2026-09-04)

Project brief attached: `lightsem_brief.md`. Answer verbatim; 475 sources in `00_deep_research_sources.txt`.

LightSem’s main bottleneck is **not the Conformer, causality, or lack of dance corpora; it is the hand-written conductor and the absence of licensed, high-resolution examples of real show programming**. The most credible path to “human designer” quality is a hierarchical audio-to-show planner trained on authored lighting events, followed by fixture-aware rendering and a learned human-preference critic—not a larger motion predictor.

# A. Datasets

## Evidence standard

Three fundamentally different resources are often conflated:

1. **Control-ground-truth:** DMX streams, console cue stacks, show files, or authored game lighting events. These directly describe what a programmer commanded.
2. **Rendered-ground-truth:** concert video or extracted HSV/brightness. This captures the visible result but loses fixture identity and is contaminated by cameras, LED walls, scenery and performers.
3. **Adjacent choreography:** dance, gameplay, fireworks or fountains. These teach temporal composition but not necessarily lighting semantics.

“Commercially trainable” below means that both the annotations and synchronized audio have an explicit commercial licence or have been separately cleared. Public downloadability alone is not permission to train a commercial model.

## Lighting-specific sources

| Source | Verifiable size | Format and annotation | Acquisition | Licence | Commercial training |
|---|---:|---|---|---|---|
| **RPMC-L² / Skip-BART** | 699 samples from 35 live performances; 20 seconds to roughly 5 minutes; about 40 GB processed | HDF5; audio embeddings aligned at 10 Hz with video-derived hue and value. Original Skip-BART predicts one dominant hue and intensity; SeqLight uses 360-bin hue and 100-bin value distributions. No fixture IDs or DMX | Skip-BART repository; processed features rather than source videos | No explicit dataset licence was located; videos are withheld partly for copyright reasons | **No, not safely as released.** Obtain written rights from authors, venues and relevant music rightsholders |
| **SeqLight** | Reuses RPMC-L² rather than adding a new music-lighting corpus | Modified Skip-BART predicts full-light HSV distributions; a separately trained decomposition policy maps distributions to eight simulated point lights | Anonymous code repository linked from the paper | The arXiv licence covers paper distribution, not commercial reuse of code, data or weights | **No clear grant** |
| **BeatSaver / Beat Saber maps** | Historical work reported more than 20,000 maps and selected about 8,000 above a 70% rating; the current total was not exposed by the retrieved API documentation | ZIP containing audio, metadata and JSON-like `.dat` difficulty files. Lighting annotations include beat-positioned light events; Chroma maps can add RGB colour and richer environment-specific events | BeatSaver website/API; download by map key, hash, uploader or playlist | No collection-wide permissive licence. Some individual maps expressly restrict use to personal, non-commercial enjoyment; BeatSaver operates a DMCA process | **No for bulk commercial training.** Potentially yes for an opt-in, mapper-cleared subset with separately licensed audio |
| **Rock Band / Guitar Hero VENUE tracks** | Thousands of official and community song packages exist, but no cleanly licensed canonical corpus was verified | Standard MIDI `notes.mid` can contain `BEAT`, `EVENTS`, `CAMERA` and `VENUE` tracks; venue events describe lighting, camera cuts and stage animation. Some games also use proprietary MILO venue assets | Official game extraction, RBN/community packages, RhythmVerse/Clone Hero archives, Onyx conversion tools | Official assets are proprietary; community packages frequently redistribute copyrighted recordings | **No**, unless directly licensed. A legally clean corpus could be commissioned from former RBN authors using original or cleared music |
| **Audio Trip** | No central count verified | Game-specific choreography packages created with the same in-VR authoring tool used by the developers; annotations primarily encode gems, ribbons, drums and obstacles rather than a transferable lighting rig | In-game/community distribution | No corpus-level ML or commercial licence located | **Conditional only**, through developer or author agreements |
| **Synth Riders** | No reliable collection-wide count verified | `.synth` custom-song files; separate `.stage` and `.spinstage` environments. Song maps encode notes and rails; stages are not necessarily song-specific lighting decisions | Community editor and CustomSongs/CustomStages directories | No collection-wide commercial licence located; synchronized audio commonly carries separate copyright | **No by default** |
| **Audica / Maudica** | No reliable current count verified | `.audica` packages authored with Reaper/NotReaper; primarily target-shot timing and gameplay events. Custom packages are downloaded from Maudica or community channels | Maudica and community repositories | No corpus-wide training grant located; audio rights remain separate | **No by default** |
| **xLights community sequences** | Large but decentralised; no authoritative count | Typically xLights sequence/project files plus model definitions and media references. These can contain per-element effects, colours, intensities and timing tracks for pixels and props | Sequence-sharing groups, vendors, competitions and direct author acquisition | Varies per author/vendor; many downloads grant performance use, not model training | **Only with per-author permission and cleared audio** |
| **Light-O-Rama sequences** | Large commercial and community ecosystem; no consolidated count verified | Sequence files with channel/prop timing, intensity, colour and effect commands, often synchronized to a specific edited audio master | Sequence stores, forums and direct creator partnerships | Predominantly proprietary or author-controlled | **Potentially valuable but requires licensing** |
| **Vixen sequences** | Decentralised; no consolidated count verified | Project and sequence data for fixtures/elements, effects, colour gradients and timing marks | Community forums, Git repositories and direct exports | Project software licensing does not automatically license user sequences or music | **Only opt-in/licensed sequences** |
| **QLC+ projects and DMX dumps** | No public curated corpus verified | QLC+ supports projects, multitrack shows, audio functions, chasers and snapshots of current output-universe values; a DMX universe contains 512 values in the 0–255 range | Public projects, direct exports or purpose-built recording | Software availability does not imply rights to user-authored shows | **Conditional** |
| **grandMA show files/exports** | No public corpus verified | grandMA2/3 can export cues, sequences, effects, presets, fixtures, timecodes and other objects as XML; MVR exports patch geometry, fixture files and stage organization | Partnerships with designers, venues or rental houses; XML export is easier to normalize than opaque complete show files | Show programming is normally owned or controlled by creators/productions; no public training licence | **Best acquisition target, but only contractually** |
| **Laser show files** | Laserworld advertises more than 250 free shows, but “free” does not mean ML-licensed | ILDA stores 2D/3D vector points, blanking and palette or 24-bit colour per point. Timeline project formats add synchronization, fixture routing and effects | Laserworld, Pangolin, Showcontroller, LaserShowGen and direct creator partnerships | Pangolin show elements have product- and hardware-specific restrictions; vendor shows may have export protection | **Generally no without explicit permission** |
| **Pyromusical scripts** | Competitions provide templates and collect entries, but no reusable corpus size was verified | COBRA uses CSV event rows with hundredth-second timing, channel, cue and optional audio filename; Finale 3D `.fin` files can embed music, effects, positions, models and cue timing | PyroJam/Finale competitions, COBRA authors, display companies | Competition submission is not necessarily a commercial ML licence; music and effect assets have separate rights | **Potentially excellent under a contest opt-in or partnership** |
| **Musical fountains** | No public machine-learning corpus located | Commercial systems associate audio with nozzle, projection and lighting operations; some systems store a scenario file alongside each MP3/WAV | Fountain operators and Syncronorm-style vendors | Commercial show files are commissioned assets; no open licence identified | **Acquire by partnership only** |

RPMC-L² is currently the only verified downloadable corpus that is explicitly built around real stage-lighting outcomes. Its authors report 699 cleaned samples from performances recorded between December 2020 and July 2024, but release processed features rather than original videos, and no explicit commercial dataset licence is stated.[1][2]

Beat Saber is the largest readily machine-readable authoring source in this inventory: difficulty `.dat` files contain mapper-authored lighting alongside beat-synchronous gameplay, and BeatSaver provides downloadable ZIPs and API access. Legally, it is radioactive for a commercial model: BeatSaver has a DMCA process, and at least some current maps expressly grant only personal, non-commercial use.[3][4][5][6]

Rock Band is semantically richer than Beat Saber because its standard MIDI packages can contain dedicated `VENUE`, `CAMERA`, `BEAT` and structural tracks; the authoring process explicitly included lighting and camera moves created through text events. The problem is rights, not format: official game packages and most downloadable customs cannot responsibly become commercial training data without negotiated licences.[7][8]

Professional console exports are the highest-quality obtainable target. grandMA can export cues, sequences, effects, presets, fixtures and timecodes as XML, while MVR carries rig geometry and patch structure. There is no verified public corpus, so LightSem should treat console data as a **business-development acquisition problem**, not a web-scraping problem.[9][10]

## Video proxies

| Corpus | Size | Useful signal | Main defect | Commercial position |
|---|---:|---|---|---|
| **Music Video Dataset, MVD-MIX** | 1,600 videos across 16 classes | Downloadable frame-wise brightness, saturation, hue, colourfulness, colour-name and contrast features, alongside audio descriptors | Music-video editing, scenery and grading are mixed with lighting; no DMX or fixture actions | Media rights are unclear; derived-feature use still needs a licence audit |
| **WVU Fixation Database** | Large browsable collection; exact count not verified | Frame grids expose colour rhythm, strobing, section changes and climax structure | Primarily an artistic visual study, not aligned machine-readable show control | Not a commercially cleared training corpus |
| **MUSIC-AVQA** | 9,288 videos, more than 150 hours | Music-performance video useful for performer/instrument grounding | Focuses on audiovisual QA, not lighting annotation; many clips will not contain professional concert lighting | YouTube-derived; not commercially safe by default |
| **Saraga Audiovisual** | 42 concerts, more than 60 hours | Multitrack audio, video and pose estimates | Carnatic concert recording rather than authored spectacle lighting | Must follow its own data terms; not a substitute for lighting logs |
| **AIOZ-GDANCE** | 16.7 hours, more than 4,000 group-dance clips | Music, group timing and 3D motion | Dance target, not lighting; internet-video provenance | Research utility, not automatically commercially viable |

MVD-MIX is the strongest generic colour-rhythm proxy because it publishes aligned frame-wise visual descriptors rather than only URLs; its 1,600 videos include brightness, saturation, hue and multiple contrast features. It should be used for representation pretraining or a weak visual critic, not as the main LightSem target.[11]

MUSIC-AVQA provides 9,288 music-performance videos and more than 150 hours, but its annotations concern audiovisual reasoning rather than designed lighting. Saraga similarly offers more than 60 hours of multitrack concert audiovisual material, but there is no indication that its camera-visible lighting represents concert-show programming.[12][13]

## What actually exists

The hard truth is that there is **no verified, commercially clean, public corpus of full-song professional DMX programming at useful scale**. The public web contains many downloadable sequences, game maps and show files, but each is a distribution channel with heterogeneous rights—not a legally coherent dataset.

The highest-value dataset LightSem could create would therefore be:

- 300–1,000 complete shows from 10–30 consenting lighting programmers.
- Cleared or commissioned music, ideally with stems.
- Console XML/show exports plus raw Art-Net or sACN capture at 40 Hz.
- Fixture patch, GDTF/MVR geometry and fixture types.
- Cue names, executors, palettes, effects and timecode markers.
- A rendered canonical-rig version for model comparison.
- Explicit rights for commercial ML training, generated outputs and weight distribution.

# B. Audio decomposition

## Evidence hierarchy

LightSem’s own experiments sharply constrain what richer audio can plausibly fix. The present dance-trained descriptor model has negative held-out $$R^2=-0.22$$, inter-coach agreement on the same song is only approximately $$r=0.22$$–$$0.42$$, and offline look-ahead did not improve descriptor prediction or Light-BAS. This means the current target contains large amounts of human choice that audio alone cannot identify.[14]

Conversely, the RPMC-L² operator head produced the project’s strongest measured direct audio-to-light relationship, with an event-average-precision lift of 0.31. Better audio representations should therefore be tested first on **look changes, intensity envelopes, palette changes and effect-class decisions**, not on reconstructing a dancer’s exact frame-level motion.[14]

## Representation priorities

| Representation | Expected lighting information | Evidence status for LightSem | Consumer-GPU cost | Main leakage risk | Recommendation |
|---|---|---|---|---|---|
| **Drums stem** | Kick/snare accents, fills, half-time/double-time transitions, drop attacks | Strong mechanistic fit, but no verified lighting paper here isolates its gain | Moderate one-time separation; cheap cached onset features | Separation artefacts can identify songs; duplicate masters across splits | **Test first** |
| **Bass stem** | Groove density, low-frequency impact, sustained build energy | Plausible for intensity and movement scale; current mixed low-band flux is a crude substitute | Moderate | Bass bleed and mastering fingerprints | **Test with drums** |
| **Vocal stem/activity** | Verse/chorus contrast, entrances, pauses, front-person emphasis | Useful principally at section level; exact lyric meaning is less likely to control frame-level lighting | Moderate separation; transcription adds cost | Lyrics and artist identity can leak song/genre; copyrighted text storage | **Test binary activity before lyrics** |
| **Other/harmonic stem** | Pads, guitars, synth texture, spectral density | Likely more useful for palette and fixture family than timing | Moderate | Separator errors differ by genre | Second wave |
| **Per-stem onset and RMS** | Sparse event grid and independent energy envelopes | Cheap, interpretable and easy to ablate | Very low after separation | Poor normalization can reveal track mastering | **Highest-value representation test** |
| **Per-stem beat/downbeat** | Which instrument establishes pulse and meter | Better aligned with lighting phrase grammar than a single kick-flux phase | Low to moderate | Beat tracker may silently fail on breakdowns and rubato | High priority |
| **Chord/key features** | Harmonic stability, modulation and long-term colour context | Early rule systems used chord categories, but Skip-BART argues that coarse category-to-colour mappings are insufficient [1] | Low | Key/chord labels may simply encode genre or song identity | Use only in planner |
| **Harmonic tension** | Build/release, dominant preparation, chromatic instability | More relevant to anticipation and palette pressure than literal hue | Low to moderate | Pseudo-label models may embed corpus bias | Promising planner input |
| **Section/drop labels** | Intro, verse, chorus, build, drop, breakdown and outro | LightSem’s direct per-frame injection worsened BAS and improved FID only partly beyond a shuffled donor; a section-level planner remains untested [14] | Low | Target-derived boundaries under symmetric context caused severe leakage in the project | **Planner only; do not concatenate blindly** |
| **Vocal lyrics/semantics** | Narrative themes and special cue opportunities | Sparse high-level influence; unlikely to recover routine cue timing | ASR plus text encoder is moderate | Exact lyrics may identify the song and introduce rights/storage issues | Low priority until activity helps |
| **MERT/MuQ/MusicFM** | Learned timbre, rhythm, instrumentation and longer-context musical state | Candidate replacement for OpenL3; no verified direct LightSem ablation yet | Moderate if frozen and cached | Memorization of benchmark songs; overlapping pretraining repertoire | Test frozen, layer-pooled features |
| **CLAP** | Audio-text semantic and genre information | Better suited to global style and palette than 40 Hz event timing | Relatively cheap when cached | Text-label and artist leakage | Planner/style token only |
| **Jukebox features** | Rich hierarchical musical representation | EDGE-style dance systems motivate this family, but it is expensive and awkward on a 12 GB GPU | High preprocessing and storage | Strong song memorization and unknown pretraining overlap | Defer |
| **Raw separator embeddings** | Instrument-aware latent state without hand-designed onset extraction | Could outperform hand features but makes attribution difficult | High storage and decoder cost | Separator/model fingerprints; unstable across versions | Only after simple stem features win |

## Separators

Demucs or HTDemucs is the practical first implementation because the product is offline and separation can be cached. BS-RoFormer and Mel-RoFormer are reasonable quality challengers, but LightSem does not need to win source-separation leaderboards; it needs representations that improve held-out show decisions under a fixed compute budget.

The correct experiment is therefore not “which separator has the best SDR?” It is:

1. Separate only train/validation/test audio after song-level deduplication.
2. Compute drums, bass, vocals and residual RMS, flux and onset channels.
3. Freeze the current lighting model.
4. Add the channels through a small gated adapter.
5. Measure held-out event AP, section contrast, FIDlight and human A/B preference.
6. Repeat using deliberately degraded stems or simple frequency-band proxies.

If expensive separation cannot beat a four-band HPSS/onset baseline, discard it. A visually convincing product does not care whether its intermediate stems sound clean.

## Foundation embeddings

Skip-BART uses OpenL3 audio embeddings projected into a BART-derived model, and its ablations show that architecture and pretraining improve hue/value errors over the rule baseline. That supports replacing LightSem’s hand-built 92-channel bus with, or augmenting it by, a frozen music encoder—but it does not establish that any particular foundation model contains better lighting information.[1]

For MERT, MuQ, MusicFM and CLAP, evaluate several frozen layers rather than immediately fine-tuning the encoder. Fine-tuning a large encoder on 699 lighting samples is likely to overfit song, venue and genre identity; cached frozen features also fit the “hours, not days” training constraint.

A sensible multiscale bus is:

- 40 Hz: mixed and per-stem onsets, flux and energy.
- Beat level: beat/downbeat confidence and instrument dominance.
- Bar level: harmonic tension, vocal state and rhythmic density.
- Section level: learned embedding, section role, novelty and time to boundary.
- Song level: style token, dynamic range and instrumentation.

## Leakage controls

The brief already demonstrates the worst failure mode: target-derived inputs under symmetric context produced an apparent tenfold lift. Every richer representation should therefore pass the following controls:[14]

- Split by song and near-duplicate recording, never by clip.
- Detect remasters, radio edits, stems and live versions before splitting.
- Report offline and causal results together.
- Cache features independently of lighting targets.
- Ban target-derived section boundaries from the audio encoder.
- Compare real features against time-shifted, song-shuffled and same-genre donor features.
- Train a song-ID probe; if it is highly accurate, rerun with stricter deduplication.
- Evaluate out-of-domain genres and unseen artists.
- For pretrained embeddings, audit overlap with public benchmark songs where possible.

# C. Models

## Lighting-specific frontier

Skip-BART is the first verified end-to-end lighting-generation model in this review. It maps OpenL3 audio features to discretized hue and value at 10 Hz using a BART-derived encoder-decoder, aligned skip connection, masked audio pretraining and restricted stochastic sampling.[1]

Its reported model has 240 million parameters but only 19 million trainable parameters under LoRA, uses about 18 GB of GPU memory, and was trained with two RTX 4090s and an A100; reported pretraining and fine-tuning took approximately 15 hours and 1.5 hours respectively. That exact configuration does not fit LightSem’s single 12 GB GPU constraint, but a smaller frozen encoder plus adapter is feasible.[1]

SeqLight, released in May 2026, extends the system to multiple lights through two stages:

1. Skip-BART predicts complete hue/value distributions.
2. A goal-conditioned imitation/RL policy decomposes those distributions into individual fixture controls.

The policy is trained using behavioural cloning, adversarial inverse reinforcement learning and PPO or GRPO, while temporal constraints prevent large frame-to-frame jumps. Its human evaluation reports an in-domain overall score of 4.54 versus 3.90 for Skip-BART and 2.70 for a rule system, but this is a 30-participant study from the proposing authors, not an independently reproduced benchmark.[2]

SeqLight is not yet truly rig-agnostic. It assumes a fixed order of lights, a simulated eight-light rig, constant saturation and only hue/value output; it does not model pan, tilt, zoom, gobos, shutters, pixels or fixture-specific capability masks.[2]

## Dance-model lessons

| Model family | Useful idea for LightSem | What not to copy |
|---|---|---|
| **FACT-style autoregression** | Cross-modal attention and long-context sequence generation | Predicting exact full-resolution motion or DMX autoregressively for an entire song |
| **Bailando-style discrete motion tokens** | Learn a vocabulary of reusable lighting gestures, scenes and transitions | Assuming one global token stream adequately represents heterogeneous fixtures |
| **EDGE-style diffusion** | Generate multiple candidate sequences, then use constraints or a critic to select among them | Expensive frame-level diffusion across all 512 channels |
| **Lodge-style coarse-to-fine generation** | First plan sparse characteristic moments or sections, then fill local detail | Treating local filler as independent of rig physics |
| **POPDG-style phrase conditioning** | Model choreography at phrase rather than raw-frame scale | Using dance similarity as the final objective |
| **MotionStreamer-style streaming generation** | Useful future path for live mode and causal comparison | Optimizing the pre-rendered product around streaming constraints |
| **Skip-BART** | Direct audio-to-light imitation, aligned skip path, masked pretraining | Single dominant colour and large autoregressive backbone |
| **SeqLight** | Separate artistic intent from venue realization | Fixed fixture order, simulated eight-light decomposition and HSV-only controls |

The central lesson is hierarchical timescale separation. Human programmers do not independently choose 512 channel values every 25 ms; they choose scenes, palettes, fixture groups, movement effects and cue transitions, while the console expands these objects into DMX.

## Proposed architecture

### Show planner

Operate at beats, bars and sections rather than 40 Hz. Input the multiscale audio bus and emit:

- Section scene family.
- Palette and palette-transition type.
- Global intensity envelope.
- Active fixture groups.
- Movement family and movement scale.
- Density and visual-complexity budget.
- Special events: blackout, impact, strobe burst, blinder hit, solo emphasis and drop reveal.

Use a bidirectional transformer because LightSem is pre-rendered, while retaining a causal mask as an ablation. The planner should be small—roughly tens of millions of parameters, not hundreds.

### Event detailer

Conditioned on the plan, generate beat/sub-beat events and continuous control curves:

- Intensity accents.
- Pan/tilt phase and amplitude.
- Colour changes.
- Beam and zoom changes.
- Strobe/blinder event durations.
- Pixel chase direction, speed and density.

A masked-token or denoising model is preferable to one-pass regression because multiple valid realizations exist. Predicting absolute levels is explicitly ruled out by the brief; predict events, deltas, curve parameters and persistence states instead.[14]

### Fixture-set decoder

Represent each fixture or logical fixture group as a token:

$$
z_i = e_{\text{fixture type}} + e_{\text{capabilities}} + e_{\text{position}} + e_{\text{orientation}} + e_{\text{group}}.
$$

Cross-attend fixture tokens to the shared show plan, then apply type-specific heads:

- Wash expert: dimmer, RGB/CMY, zoom.
- Moving-head expert: dimmer, colour, pan, tilt, zoom, gobo, prism.
- Strobe/blinder expert: intensity, duration, rate.
- Pixel expert: low-rank spatial pattern parameters.
- Laser expert: abstract cue/class only; expansion should remain safety-constrained.

A mixture-of-experts router should be masked by actual fixture capabilities. Outputs should be normalized semantic parameters first and converted to DMX only through a deterministic fixture-profile layer.

### Set equivariance

The decoder should not depend on arbitrary patch order. Permuting fixture tokens should permute outputs correspondingly, while global plan outputs remain unchanged.

Training augmentations should include:

- Random fixture ordering.
- Random fixture deletion.
- Duplicate fixtures with perturbed positions.
- Random rig scale and symmetry.
- Capability masking.
- Group-level supervision when per-fixture intent is ambiguous.
- Conversion of one authored show into several compatible sampled rigs.

Random rig sampling cannot invent how a professional would redesign a show for a radically different venue. It can teach robustness to modest patch variation, but genuine transfer still requires either paired redesigns or a strong rig-realization prior.

## Critics and reranking

A cross-modal critic is likely more valuable than making the generator substantially larger. Train it to score:

- Audio–plan compatibility.
- Audio–light event compatibility.
- Plan–fixture realization consistency.
- Temporal coherence.
- Human preference.
- Safety and hardware validity.

Generate 8–32 inexpensive candidates by varying masked tokens, flow noise or scene choices, then rerank. The critic must compare candidates for the **same song and rig**; otherwise it will learn that certain genres, brightness levels or fixture counts are intrinsically “better.”

Hard-negative construction matters:

- Correct show shifted by half a beat.
- Correct sections in the wrong order.
- Same show with palette randomly reassigned.
- Beat-trigger baseline.
- Over-smoothed and over-strobed variants.
- Candidate from another song with similar tempo.
- Correct global histogram but fixture groups scrambled.

## Human preference tuning

A small A/B dataset should tune the critic before tuning the generator. Collect comparisons on five separate questions:

- Fits the music.
- Feels intentionally structured.
- Has appropriate impact.
- Avoids repetition/metronomic behaviour.
- Looks professionally programmed.

Do not collapse these into one label initially. LightSem’s ten-pair pilot produced an 8–1–1 result favouring dancer-derived light over beat trigger on “looks like a lighting engineer did it,” but tied on musical fit; that indicates the two dimensions are not interchangeable.[14]

Once the critic predicts held-out A/B choices, use conservative preference optimization:

- Freeze the audio encoder.
- Tune only planner/detailer adapters.
- Keep a KL penalty to the imitation model.
- Mix preference pairs with supervised authored examples.
- Reject improvements that lose to persistence, beat trigger or the original model on objective controls.

With a genuinely small dataset, critic-based reranking is safer than RLHF. Direct preference optimization becomes attractive only after collecting enough repeated judgements to estimate annotator consistency.

## Masked lighting pretraining

Masked pretraining is a strong fit for unlabeled lighting logs because it does not require audio:

- Mask random fixture channels.
- Mask complete fixture groups.
- Mask beat- or bar-length spans.
- Mask semantic attributes such as colour or movement.
- Predict scene boundaries and event classes.
- Reconstruct after random fixture permutation or removal.

This teaches cue grammar, persistence and inter-fixture coordination. Fine-tuning then learns which of those valid lighting continuations fits a particular song.

The crucial split is by production or show, not by random time interval. Otherwise adjacent cues, repeated songs and copied palettes make reconstruction scores meaningless.

# D. Recommendation

## Ranked interventions

| Rank | Intervention | Why it is likely to matter | Cheapest falsifying test | Trivial baseline it must beat |
|---:|---|---|---|---|
| **1** | Replace the hand-written conductor with a learned beat/bar/section show planner | The current network produces signals, but the conductor decides almost all artistic semantics. Current output is more metronomic than human motion despite acceptable diversity | Train a tiny planner on RPMC-derived events or manually labelled existing LightSem shows: scene, palette, intensity class and transition at bar resolution. Render through the unchanged conductor/rig layer | Existing conductor, beat trigger and section-shuffled planner |
| **2** | Acquire 50–100 licensed console-authored shows | More dance data cannot teach decisions absent from dance targets. Console data contains actual cue hierarchy, fixture grouping and effect intent | Sign one designer; collect ten cleared tracks with XML/show export and rendered output. Leave two whole songs out and train only an event/scene predictor | Persistence, beat trigger and RPMC operator head |
| **3** | Train an audio–show critic and rerank candidates | Multiple shows can fit the same audio; reconstruction loss punishes valid alternatives. Reranking improves quality without a huge generator | Generate eight variants from the current system for 20 songs; gather 300–500 blinded A/B comparisons; test critic accuracy on unseen songs and people | RMS/beat-alignment heuristic and random candidate selection |
| **4** | Add per-stem onset/energy and beat/downbeat features | This is the cheapest plausible way to separate kick hits, fills, bass pressure and vocal/section entrances | Add cached Demucs/HTDemucs stem RMS, flux and onsets through a gated adapter to the RPMC event head; keep everything else frozen | Existing 92-channel bus and a cheap four-band HPSS proxy |
| **5** | Build the hierarchical detailer around events and deltas | Absolute levels and raw kinetic energy already failed; human shows are persistent state plus sparse changes | Predict only look-change, blackout, hit, colour-change and movement-change events, then reconstruct with deterministic persistence | Last-value persistence and current direct parameter head |
| **6** | Section planner with explicit anticipation | Direct section-channel injection failed, but the untested planner can make decisions before a boundary and sustain them afterward | Manually annotate 30 tracks with section roles; let a small planner select scene/palette one bar ahead while the frame model remains unchanged | Same planner with shuffled sections, causal planner and current no-planner model |
| **7** | Masked pretraining on lighting-only logs | Teaches persistence, cue grammar and fixture coordination before scarce audio pairing | Pretrain a small transformer on current generated/logged DMX plus any licensed sequences; mask bar spans and fixture groups, then fine-tune the event head | Same model from random initialization |
| **8** | Fixture-token, type-aware decoder | Required for product transfer across venues, but it will not itself create artistic intent | Replace the fixed canonical-rig mapper with tokens for only washes, moving heads and strobes; randomize ordering and remove 20% of fixtures during training | Existing fixed rig and nearest-fixture remapping |
| **9** | Frozen music-foundation embeddings | May improve timbre and section representation but can also become a song-ID shortcut | Cache selected layers from one compact encoder, add a low-rank adapter and run song-level splits plus song-ID probes | OpenL3/current bus at equal parameter count |
| **10** | More dance datasets | Useful only for generic movement priors; the present failure is target ambiguity and domain mismatch | Train an identical small model on one additional corpus and test only Oslo transfer and human-light preference | Just Dance model and AIST model |
| **11** | Lyrics and text semantics | Expensive legal/engineering complexity with weak expected effect on 40 Hz control | Binary vocal activity first; only add ASR/lyrics if vocal activity produces a clear planner gain | Vocal RMS/activity and section labels |
| **12** | End-to-end RL for DMX | Highest instability and easiest route to reward hacking; premature without a validated critic | Do not attempt until the critic predicts unseen human A/B choices substantially above chance | Critic-only reranking |

## Immediate experiment

The fastest decisive experiment is a **three-level ablation on the existing RPMC operator task**:

- Baseline A: current 92-channel audio bus.
- Baseline B: four manually designed frequency bands with RMS/onsets.
- Candidate C: separated drums, bass, vocals and other with RMS/onsets.
- Candidate D: Candidate C plus bar-level section/tension state.
- Candidate E: Candidate C feeding a bar-level planner instead of direct frame concatenation.

Hold the generator, renderer, split and smoothing constant. Report causal and offline versions, event AP against shifted and shuffled controls, FIDlight, Light-BAS against the beat-trigger baseline, and a small blinded human A/B.

Kill source separation if Candidate C cannot beat both A and B. Kill direct section features again if D loses to C; continue with E only if it beats a section-shuffled planner.

## Acquisition strategy

The best commercial data strategy is not “scrape more YouTube.” It is:

1. Offer lighting programmers a free console backup/analysis tool that exports normalized cues, fixture metadata and anonymized DMX.
2. Pay for an explicit commercial-training licence.
3. Use production-owned, commissioned or public-domain music whenever possible.
4. Preserve cue hierarchy and semantic fixture groups rather than recording only raw 512-channel frames.
5. Give contributors attribution, royalties or model access.
6. Collect paired redesigns of a few songs across multiple rigs; these are disproportionately valuable for rig transfer.
7. Render all shows onto the same canonical rig for evaluation while retaining original-rig commands.

Professional show files carry much more information than video-derived HSV. grandMA exports can preserve cues, effects, presets, sequences, fixtures and timecodes, while MVR supplies patch and geometry information. Even 50 properly licensed shows of this kind could be more useful than tens of thousands of legally dubious videos.[10][9]

## Success gates

Do not call an intervention successful unless it passes all relevant gates:

- **Musicality:** beats the beat-trigger baseline, not merely a shifted model.
- **Structure:** beats a section-shuffled donor.
- **Dynamics:** beats persistence and fixed-intensity envelopes.
- **Human quality:** wins on “looks intentionally programmed,” not just “moves with the beat.”
- **Generalization:** split by complete song and near-duplicate family.
- **Offline value:** reports the causal counterpart.
- **Diversity:** preserves multiple plausible shows without random incoherence.
- **Rig transfer:** survives fixture reordering, deletion and capability changes.
- **Commercial viability:** every training asset has documented provenance and rights.

The painful conclusion is that another sophisticated music-to-motion transformer is unlikely to close the gap. LightSem already has enough temporal machinery; it needs **authored lighting intent, a learned hierarchical planner, and a preference model that can distinguish deliberate show composition from technically synchronized motion**.
