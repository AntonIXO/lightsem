# Direction 2b — non-game authored-light datasets (Perplexity reasoning pass, 2026-09-04)

Project brief attached: `lightsem_brief.md` (copied below the answer). Answer verbatim.

Pull **xLights** free `.xsq` first: it is the only large, public, human-authored music-to-light corpus with effect events plus per-model RGB at 25 ms. Everything else is either one mixed HSV stream from concert video, layout-baked playback, auto-generated DJ scripts, or an empty archive.[1][2][3]

## Ranked sources

| Rank | Source | Paired hours (est.) | Resolution | Encodes | Audio-function at beat scale | Research | Commercial |
|---|---|---|---|---|---|---|---|
| 1 | Free xLights `.xsq` / `.xsqz` (xlightsseq.com + community) | ~25–40 h (≈556 free songs; more on other boards) | 25 ms default (40 fps); 50 ms common | Per-model effects (bars, shockwave, strobe, twinkle, wash, faces) with palettes/params; not raw channels until render | Yes, if you keep effect onsets; no if you only regress RGB | Weak: sequence copyright + uncleared music | No without sequencer + music licenses; vendor files are home-use only |
| 2 | RPMC-L2 (Skip-BART / SeqLight) | ~15–30 h (699 clips, 20 s–5 min, 35 shows) | 10 Hz labels; video-derived | Per-frame mixed-light HSV: hue mode + value; S fixed 255; also `(F,3,256)` HSV histograms | Partially: strobes/beats yes, hue often song/venue prior | Yes on the released HDF5 (CC-BY-4.0) | No for the underlying concert audio/video |
| 3 | Rendered Falcon `.fseq` from those sequences | Same hours as (1) | 20–40 fps, 8-bit/channel | Per-channel intensity/RGB only; effects destroyed | Weak: envelopes follow music, events smeared | Same as (1) | Same as (1) |
| 4 | Paid/vendor xLights + LOR `.lms`/`.lcs` + Vixen `.tim` | 50–150 h catalog, not free | 25–50 ms | Same as (1); LOR often AC channels + RGB | Yes | No scraping | Home-display license; audio almost never included |
| 5 | QLC+ `.qxw` GitHub demos | &lt;2 h | Cue/chase, not frame grid | Fixture functions, chasers, audio/MIDI triggers | Sometimes | GPL software; tiny data | Tiny and not a corpus |
| 6 | ILDA `.ild` + Pangolin Cloud / Laserworld | Hours of clips, almost 0 timed-to-music public shows | Laser frames ~10–30 fps | XY + color + blanking vectors, not DMX | No unless a proprietary timeline is attached | Clip files sometimes shared; music shows are not | Pangolin/Laserworld marketplaces; no train-and-sell |
| 7 | SoundSwitch / Lightkey / Engine Lighting | Large internally, ~0 public | Beatgrid | Autoscript is **machine** lighting; custom scripts are private metadata in the audio file | Autoscript is a beat-trigger baseline, not a target | No | Subscription product; do not scrape |
| 8 | grandMA / MagicQ / Onyx / Avolites show files | Manufacturer demos only | Cue lists + timecode | Full console state; not published as music pairs | Unknown | No public archive | No |
| 9 | Raveform, VJ packs, MV colour stats, fountains/fireworks | 0 lighting hours | N/A | Structure / loops / clip colour / product catalogs | Raveform is useful as a **condition**, not a label | Raveform yes | Music still uncleared |

SeqLight does not add a new corpus. It retrains on RPMC-L2 and then decomposes mixed HSV onto simulated fixtures with imitation learning, not human multi-light desks.[4][5]

## Holiday files

xlightsseq.com currently lists 336 free Christmas, 98 free Halloween, 89 free music, 33 free misc (≈556 free), plus ~1,000 paid/vendor listings. Sequences are typically 3–4 min songs, so free hours land around a day of audio, not a season. Other catalogs (xlights.info, vendor shops) duplicate a lot of the same titles.[1]

Shared files almost never include the commercial MP3. The `.xsq` stores a media path; you buy the track separately. Original/self-made audio is the exception and sometimes ships inside `.xsqz` (zip of sequence + images/video/shaders). Vendor terms are explicit: labour fee for the sequence, single-home licence, no share/resale, no commercial playback, audio not included.[6][7][8]

What the formats actually hold:

- `.xsq` (XML) / older `.xml`: `&lt;head&gt;` duration + timing (usually 25 ms), palettes, effect DB, per-model layers with start/end and params (bars, shockwave, strobe, etc.). This is the authored object.[2]
- `.xsqz`: packaged sequence + assets.
- `.fseq`: Falcon Player playback dump. Per-universe 8-bit channel frames. No effect names, no groups, no palettes. Useless for event learning unless you only want envelopes.[9][10]
- Light-O-Rama `.lms`/`.lcs`: channel timelines, sold without the song. Smaller, older, more AC-channel than pixel-model.
- Vixen 3 `.tim`: timeline effects; export often 25 ms CSV. Sharing volume is far below xLights.
- `.fseq` is layout-specific. An `.xsq` is remappable onto your models; an `.fseq` is not.

Licence: xLights the program is GPL-3.0. Sequence files are copyright of the sequencer. Music is a third copyright. For a paper, cite and do not redistribute audio; for a product, you cannot train-and-ship on Mariah Carey sequences you downloaded.[6][11]

**Conversion** to 40 fps event stream + group envelopes: parse `.xsq` `ElementEffects`; emit `{t_start, t_end, model, effect, params, palette}`; map models → fixture groups (roofline, matrix, tree, megatree, …); sample each effect at 25 ms or headless-render to `.fseq` and reshape `(T, C)` → `(T, G, 3)` mean RGB → HSV; hold at 40 fps (already native if timing is 25 ms). Drop singing-face visemes unless you want a lyrics head.

## Stage, academic, consoles

RPMC-L2 is the only public stage-domain set. 699 HDF5 clips, ~40 GB, CC-BY-4.0 on Hugging Face, from 2020–2024 livehouse video. Music features (OpenL3, mel, CQT, …) aligned to `light/threshold` of shape `(F, 3, 256)` HSV histograms. Skip-BART then uses per-frame hue + value with S=255. Audio/light grid is 10 Hz, not 40 fps. It is **observed** mixed light through haze and cameras, not a desk file. SeqLight uses the same split and does not publish per-fixture DMX.[3][5][12][13]

Legal: research on the released features is the intended use. Commercial models trained on those concerts still sit on uncleared performance audio and venue video. Do not scrape the original YouTube/livehouse files “to improve labels.”

Raveform (TISMIR 2026) is 4,902 DJ mixes + 1,423 expert structure tracks. Zero lights. Use it as section/energy conditioning, not as a target.[14]

Concert-video lighting annotation corpora besides RPMC-L2: none that are public. VJ/Resolume packs are unsynced loops. Music-video “colour statistics” datasets are clip-level aesthetics, not fixture envelopes. SoundSwitch Autoscript writes beat-aligned looks into audio tags; that is a rule engine, not a human show library. Lightkey/DasLight/QLC+ have no public timecoded music-show archive. grandMA2/3 `.show`, MagicQ `.shw`, Onyx, Avolites: vendor demo desks exist; music-locked public dumps do not.[15][16][17]

**Conversion (RPMC-L2):** take hue mode + value (or full 256-bin HV) per frame; upsample 10 Hz → 40 fps with hold or circular interpolation on H; emit events where circular ΔH or ΔV exceeds a threshold; treat the whole stage as one fixture group until you run a SeqLight-style decomposer. Do not pretend this is per-head DMX.

## Lasers, pyro, leftover

ILDA `.ild` is a frame of vector points (XY, colour, blanking). Public clip packs exist; public **music-timed** shows do not. Pangolin QuickShow/BEYOND and LaserOS keep timelines in proprietary workspaces. Pangolin Cloud is a store. Laserworld advertises 250+ free shows inside their paid controller, not as an open train set. Treat lasers as a later output head, not a pretraining corpus.[18][19]

Fountains and fireworks: Finale 3D / Finale DB share **product** simulations, not public music choreographies. Visual Show Director, Pyrodigital, and musical-fountain vendor files are closed. Hours of paired public data: ≈0.[20]

## Pull first and audit

Download the free xLights `.xsq` set (not `.fseq`), buy or skip audio per track, keep only sequences with a legal audio path you actually possess. Render a 40 fps group-HSV + effect-onset dump. Then run this gate on a **show-disjoint** 10% holdout before any Seq2Seq training:

1. Events = effect onsets plus |ΔV| or circular |ΔH| jumps on the group envelopes.
2. Time-shift null: same audio, lights shifted by a random offset in {±2, ±4, ±8} s. Event AP should collapse.
3. Beat-trigger baseline: madmom/librosa beats + bar onsets fire a flash/colour-jump. This is SoundSwitch Autoscript / “Move Fast” strobe. It will look decent on metal and Christmas drops.
4. Pass: AP_music − AP_shift ≥ 0.15 **and** AP_music ≥ AP_beat + 0.05 on both onset-AP and envelope-R². Fail either and you are fitting tempo, not lighting authorship.

If xLights passes the gate, it is your pretrain (events + 40 fps envelopes). Fine-tune the stage head on RPMC-L2, knowing you only supervise one mixed wash at 10 Hz. If xLights fails the gate, stop; a generative model will not beat a beat-flash baseline, and Skip-BART already showed the rule-based floor is weak on hue but the data is tiny and genre-narrow. Do not touch vendor sequences, SoundSwitch libraries, or concert rips for a product until you have written licences.

## Sources

[1] Free Christmas Sequences | https://xlightsseq.com/sequences/categories/free-christmas-sequences.5/
[2] xLights MCP Server by JohnBreault | https://glama.ai/mcp/servers/JohnBreault/xlights-mcp-server
[3] RS2002/RPMC-L2 · Datasets at Hugging Face | https://huggingface.co/datasets/RS2002/RPMC-L2
[4] Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning | https://arxiv.org/abs/2605.03660v1
[5] Stage Light is Sequence2: Multi-Light Control via Imitation Learning | https://arxiv.org/html/2605.03660
[6] Oh Santa! – Visionary Light Shows, LLC | https://visionarylightshows.com/products/oh-santa
[7] Sequence - Minions Intro - Illumination | https://xlightsseq.com/sequences/minions-intro-illumination.1910/
[8] Mary Did You Know - Pentatonix - xLights Sequence - lightlink | https://www.golightlink.com/product-page/mary-did-you-know-pentatonix-xlights-sequence
[9] Reverse FSQ to XSQ | https://www.reddit.com/r/xlights/comments/1o95e8e/reverse_fsq_to_xsq/
[10] What Is a Holiday Light Show Sequence? A Clear Guide - ezrgb | https://ezrgb.com/blog/what-is-a-holiday-light-show-script-a-clear-guide
[11] GitHub - xLightsSequencer/xLights: xLights is a sequencer for Lights. xLights has usb and E1.31 drivers. You c | https://github.com/xLightsSequencer/xLights
[12] [Paper Note] Automatic Stage Lighting Control: Is it a Rule ... | https://en.papernotes.org/ICLR2026/audio_speech/automatic_stage_lighting_control_is_it_a_rule-driven_process_or_generative_task/
[13] Automatic Stage Lighting Control: Is it a Rule-Driven Process ... | https://nanless.github.io/audio-paper-digest-blog/posts/2026-05-04-automatic-stage-lighting-control-is-it-a-rule/
[14] Raveform: A Dataset of Metrical and Functional Structure ... | https://transactions.ismir.net/articles/10.5334/tismir.288
[15] SoundSwitch | Autoscripting Audio Files, Playlists and Crates | https://support.inmusicstore.com/en/support/solutions/articles/69000847098-soundswitch-autoscripting-audio-files-playlists-and-crates
[16] NOBODY EXPECTED QLC+ TO DO THIS (ART-NET TIMECODE FROM REKORDBOX) | https://www.youtube.com/watch?v=1NPjqm9zbDI
[17] Free VJ Loops, Clips &amp; Plugins — 18 Trusted Sources | VJ Academy | https://vjacademy.info/free-vj-content
[18] BEYOND - The World's Leading Professional Laser Show Software | https://pangolin.com/pages/beyond
[19] 💡 Laserworld Laser Controllers | https://www.thomann.de/be/laserworld_laser_controllers.html
[20] Introducing Finale DB — Online Fireworks Database | https://finale3d.com/groups/english/forum/topic/%F0%9F%94%A5-introducing-finale-db/
[21] Automatic Stage Lighting Control: Is it a Rule-Driven ... | https://iclr.cc/virtual/2026/poster/10008746
[22] Datasets – Hugging Face | https://huggingface.co/datasets?other=electron-microscopy
[23] davidheineman/iclr-2026 · Datasets at Hugging Face | https://huggingface.co/datasets/davidheineman/iclr-2026
[24] Datasets | https://huggingface.co/datasets?other=polymarket
[25] ICLR 2026 Papers with Code &amp; Data | https://www.paperdigest.org/2026/04/iclr-2026-papers-with-code-data/
[26] Is it a rule-driven process or generative task? | Xiaoyu Zhang | https://www.linkedin.com/posts/xiaoyu-zhang-vis_iclr-2026-automatic-stage-lighting-control-activity-7452181807841812480-PvlJ
[27] Paid Sequence &amp; Lyrics - Kpop Demon Hunters - Golden (Moving Heads) | https://xlightsseq.com/sequences/kpop-demon-hunters-golden-moving-heads.1374/
[28] Halloween Sequence - Halloween Star Wars compilation - feat Mandalorian, Imperial March and others | https://xlightsseq.com/threads/halloween-star-wars-compilation-feat-mandalorian-imperial-march-and-others.4768/
[29] Build software better, together | https://ithub.global.ssl.fastly.net/topics/light-control
[30] You're a Mean One, Mr. Grinch by Lindsey Stirling | https://xlightsseq.com/sequences/youre-a-mean-one-mr-grinch-by-lindsey-stirling.1925/
[31] NVIDIA Open Model Agreement | https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
[32] TestCategory | Test Agreement v1 | https://www.nvidia.com/en-us/agreements/test-category/test-agreement-v1/
[33] Multi-agent system for automated bioinformatics analysis workflow | https://www.cell.com/patterns/fulltext/S2666-3899(26)00120-0
[34] xLights - App Store | https://apps.apple.com/tr/app/xlights/id1137297689?platform=mac
[35] LiteRT-LM (google-ai-edge/litert-lm) | https://context7.com/google-ai-edge/litert-lm
[36] README.md - bobbyfriday/xlight-autosequencer | https://github.com/bobbyfriday/xlight-autosequencer/blob/main/README.md
[37] v4.2 Copying fseq to remotes? - Falcon Christmas | https://falconchristmas.com/forum/index.php?topic=13089.0
[38] Sequence - Six Seven Christmas | https://xlightsseq.com/sequences/six-seven-christmas.1537/
[39] Beaglebone setup question with sd card | https://falconchristmas.com/forum/index.php?topic=9969.0
[40] Playing sequence using schedule no light output | https://auschristmaslighting.com/threads/playing-sequence-using-schedule-no-light-output.17186/
[41] 2026.13 Released | https://xlights.org/2026-13-released/
[42] Christmas Light Show Software Free Options To Create Pro Level ... | https://www.alibaba.com/product-insights/christmas-light-show-software-free-options-to-create-pro-level-displays-at-home.html
[43] Probably a stupid question but can I download premade sequences timed to songs? | https://www.reddit.com/r/xlights/comments/1o4c0tq/probably_a_stupid_question_but_can_i_download/
[44] Santa Tell Me by Ariana Grande Lightorama Singing Face Quartet ... | https://www.etsy.com/listing/1063434345/santa-tell-me-by-ariana-grande
[45] Drag sequence onto another sequence without audio track? | https://community.adobe.com/questions-729/drag-sequence-onto-another-sequence-without-audio-track-1336243
[46] Sequences | Page 4 | https://xlightsseq.com/sequences/?page=4
[47] Sequence &amp; Lyrics - Thriller (Steve Aoki Midnight Hour Remix) | https://xlightsseq.com/sequences/thriller-steve-aoki-midnight-hour-remix.1892/
[48] Sequence &amp; Lyrics - burn my Tongue | https://xlightsseq.com/sequences/burn-my-tongue.1871/
[49] Paid Sequence - Send me an Angel | https://xlightsseq.com/sequences/send-me-an-angel.2003/
[50] I need to convert subsequences to Multicam clips whilst retaining the audio (multiple tracks) from the subsequ | https://www.reddit.com/r/premiere/comments/1nbjs83/i_need_to_convert_subsequences_to_multicam_clips/
[51] Paid Sequence - Christmas Without You - Ava Max | https://xlightsseq.com/sequences/christmas-without-you-ava-max.1545/
[52] Sequence Catalog | https://pixelsequencepros.com/collections
[53] Vendor Sequence - New Year's Animation | https://xlightsseq.com/sequences/new-years-animation.1755/
[54] Lights on Abby Sequences | https://xlightsseq.com/members/lights-on-abby-sequences.21833/
[55] Raveform: A Dataset of Metrical and Functional Structure... | https://reference-global.com/article/10.5334/tismir.288
[56] Volume 9 - Issue 1 - 2026 | https://transactions.ismir.net/11/volume/9/issue/1
[57] RWC Revisited: Towards a Community‑Driven MIR Corpus | https://transactions.ismir.net/articles/10.5334/tismir.326
[58] Multimodal Datasets for Studying Expert Performances of ... | https://transactions.ismir.net/articles/10.5334/tismir.230
[59] RS2002 (Zijian Zhao) | https://huggingface.co/RS2002/datasets
[60] telegramtrading/telegram-trading-bots · Datasets at Hugging Face | https://huggingface.co/datasets/telegramtrading/telegram-trading-bots
[61] trentmkelly/polymarket_crypto_derivatives · Datasets at ... | https://huggingface.co/datasets/trentmkelly/polymarket_crypto_derivatives
[62] Madhurananda/PROCESS-2 · Datasets at Hugging Face | https://huggingface.co/datasets/Madhurananda/PROCESS-2
[63] DiffusionGemma - a google Collection | https://huggingface.co/collections/google/diffusiongemma
[64] Datasets - Hugging Face | https://huggingface.co/datasets?other=gpt-5
[65] obadiaha/polymarket-crypto-5m-15m at main - Hugging Face | https://huggingface.co/datasets/obadiaha/polymarket-crypto-5m-15m/tree/main
[66] NeMo Gym - a nvidia Collection | https://huggingface.co/collections/nvidia/nemo-gym
[67] phanerozoic/qiskit-calibration-drift · Datasets at Hugging Face | https://huggingface.co/datasets/phanerozoic/qiskit-calibration-drift
[68] [2506.01482] Automatic Stage Lighting Control: Is it a Rule-Driven ... | https://arxiv.org/abs/2506.01482
[69] ILD Files: How to Open ILDA Laser Show Image Files | https://www.youtube.com/watch?v=8RlxO3uNzks
[70] Download Center [Complete Help Docs] - The Pangolin Wiki | https://wiki.pangolin.com/doku.php?id
[71] Artnet and Lightkey - DMX over Wi-Fi | https://www.youtube.com/watch?v=YgQ8v__hJY0
[72] Unity Lasers documentation [Complete Help Docs] - The Pangolin Wiki | https://wiki.pangolin.com/doku.php?id=laserprojectors:unity_manuals
[73] 2026.16 Released | https://xlights.org/2026-16-released/
[74] Pangolin laser software en ILDA interface en ILDA kabel | https://www.rfsystems.nl/pangolin-laser-software-en-ilda-interface-en-ilda-kabel-quickshow/
[75] QLC+ 5 - Inputs &amp; Outputs basics | https://www.youtube.com/watch?v=nkPnY70_CEs
[76] Generating sequences | https://xlightsseq.com/threads/generating-sequences.5961/
[77] 2026.12 Released – xLights | https://xlights.org/2026-12-released/
[78] Do people actually use QLC+? | https://www.reddit.com/r/lightingdesign/comments/1orevfn/do_people_actually_use_qlc/
[79] Table 1. | https://pmc.ncbi.nlm.nih.gov/articles/PMC12847014/table/T1/
[80] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric ... | https://arxiv.org/html/2511.11002v1
[81] ABC–MusicXML–MIDI Correspondence Dataset: A Multi- ... | https://zenodo.org/records/19861807
[82] Changing Colorspaces - OpenCV | https://docs.opencv.org/4.13.0/df/d9d/tutorial_py_colorspaces.html
[83] maxLWSv2/motionatlas-data · Datasets at Hugging Face | https://huggingface.co/datasets/maxLWSv2/motionatlas-data
[84] URMP - Multi Modal Music Performance - Kaggle | https://www.kaggle.com/datasets/alonhaviv/multi-modal-music-performance-urmp/data
[85] Dance Motion Capture Dataset (DMCD) - Kaggle | https://www.kaggle.com/datasets/zara2099/aist-dance-motion-capture-dataset
[86] Multimodal Musical Performance Assessment Dataset | https://www.kaggle.com/datasets/programmer3/multimodal-musical-performance-assessment-dataset/data
[87] Dance Performance Dataset - Kaggle | https://www.kaggle.com/datasets/freshersstaff/dance-heritage-visual-dataset/code
[88] Datasets | https://mozilladatacollective.com/datasets
[89] PANGOLIN FB3-QS QUICKSHOW Lasershow Designer Software Program | https://lightingelstore.com/pangolin-fb3-qs-quickshow-lasershow-designer-software-program/
[90] SoundSwitch | Utilizing the Effects Generator | https://support.inmusicstore.com/en/support/solutions/articles/69000847595-soundswitch-utilizing-the-effects-generator
[91] SoundSwitch | My Fixture is Missing from the Fixture Library.. Help! | https://support.inmusicstore.com/en/support/solutions/articles/69000845696-soundswitch-my-fixture-is-missing-from-the-fixture-library-help-
[92] xLights - App Store | https://apps.apple.com/nl/app/xlights/id1137297689?l=en-GB&amp;platform=mac
[93] 2026.01 Released | https://xlights.org/2026-01-released/
[94] Rane SYSTEM ONE | Introduction to Engine Lighting | https://inmusicsupport.freshdesk.com/en/support/solutions/articles/69000875520-rane-system-one-introduction-to-engine-lighting
[95] Controller / Xlights settings | https://auschristmaslighting.com/threads/controller-xlights-settings.17529/
[96] 2025.12 Released - xLights | https://xlights.org/2025-12-released/
[97] SoundSwitch | Getting Started with Engine Lighting | https://inmusicsupport.freshdesk.com/en/support/solutions/articles/69000844854-soundswitch-getting-started-with-engine-lighting
[98] SoundSwitch | Saving Projects and Light Shows | https://inmusicsupport.freshdesk.com/en/support/solutions/articles/69000853039-soundswitch-saving-projects-and-light-shows
[99] xLights - App Store - Apple | https://apps.apple.com/us/app/xlights/id1137297689?platform=iphone
[100] How to Upload Lighting Sequences to LED Suits and Props — Step-by-Step | https://lightdancepro.com/blog/upload-lighting-sequence-led-suits-props/
[101] Vixen Lights | PDF | Digital Technology - Scribd | https://www.scribd.com/document/967239940/VixenLights
[102] 2026.15 Released | https://xlights.org/2026-15-released/
[103] 2026.14 Released | https://xlights.org/2026-14-released/
[104] How Do I Use Other People's Sequences for My Christmas Lights? | https://www.learnchristmaslighting.com/post/how-do-i-use-other-people-s-sequences-for-my-christmas-lights
[105] Guía de Inicio Rápido - Xlights | PDF | https://www.scribd.com/document/552912695/Guia-de-inicio-rapido-xLights
[106] Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? | https://arxiv.org/pdf/2506.01482.pdf
[107] Pixel LED Animator 3 with xLights Video Effect Guide | https://pixel-led-animator.com/posts/pixel-led-animator-3-xlights-video-effects/
[108] How to Make a Holiday Light and Sound Show Efficiently (2024) | https://lifetips.alibaba.com/tech-efficiency/make-a-holiday-light-and-sound-show
[109] Watch Folders.... - Resolume Forum | https://resolume.com/forum/viewtopic.php?p=107867
[110] How to make a video based on an audio click track - Resolume Forum | https://www.resolume.com/forum/viewtopic.php?t=32821
[111] Getting The Sources From Git | https://github.com/mcallegari/qlcplus/wiki/macOS-build-(Qt6-and-cmake)
[112] Sending Art-Net with QLC+: From Windows Installation to Simple Desk | https://zenn.dev/toshikun/articles/qlcplus-artnet-windows-simple-desk?locale=en
[113] Resolume Arena / Avenue 7 — OSC Reference | https://onshworks-vj.github.io/resolume-dev-docs/osc/osc-reference.en.html
[114] Sync Music &amp; Visuals - Sound Reactive Visuals in Resolume - 5 Pro tips tutorial | https://www.youtube.com/watch?v=D710Df9mYQM
[115] Resolume Avenue Tutorial - Creating a Vintage Live Show | https://www.youtube.com/watch?v=oJGjuZeJlcM
[116] Resolume Avenue — Curated by Alec Johnson - Liminary | https://on.liminary.io/c/resolume-documentation
[117] 1/3 ways to work with hard cuts and loop a clip where the ... | https://www.instagram.com/reel/DbnAlHItnDR/
[118] [QLC+ - HOWTO] Download the latest GitHub build | https://www.youtube.com/watch?v=Ke7EmQZaUZ0
[119] Resolume Development Docs | https://onshworks-vj.github.io/resolume-dev-docs/
[120] Need help triggering clips in Resolume | https://www.reddit.com/r/lightingdesign/comments/1qpoz30/need_help_triggering_clips_in_resolume/
[121] Vj’ing as a performer | https://www.reddit.com/r/vjing/comments/1se16c5/vjing_as_a_performer/
[122] Grand Finale Stock Footage: Royalty-Free Video Clips | https://www.storyblocks.com/video/search/grand-finale
[123] facebook/show3d-dataset · Datasets at Hugging Face | https://huggingface.co/datasets/facebook/show3d-dataset
[124] ToxSam/open-source-3D-assets - GitHub | https://github.com/toxsam/open-source-3D-assets
[125] 5,467 Pyros Stock Video Footage - 4K and HD Video Clips | Shutterstock | https://www.shutterstock.com/video/search/pyros
[126] Grand Finale: 3D Fireworks Simulator - Chris Pirillo's Arcade | https://arcade.pirillo.com/grand-finale.html
[127] Fireworks that are not in Finale 3D's catalog of effects | https://finale3d.com/groups/english/forum/topic/fireworks-that-are-not-in-finale-3ds-catalog-of-effects/
[128] Pyrogenie - Finale 3D | https://finale3d.com/partners/pyrogenie/
[129] Conveyor Kit Asset Pack (Free CC0) | https://www.summerengine.com/asset-store/pack/conveyor-kit
[130] English - Search All Supplier Catalogs – Possible? Feature Request? | https://finale3d.com/groups/english/forum/topic/search-all-supplier-catalogs-possible-feature-request/
[131] Built a Subscription and User Portal for Fireworks Design ... | https://www.bytestechnolab.com/case-study/finale3d-fireworks-design-software/
[132] Forum – English – Finale 3D | https://finale3d.com/groups/english/forum/topic/pyromax-help/
[133] Finale 3D Fireworks Software Invoice | PDF | https://www.scribd.com/document/520172998/Invoice-1421
[134] Updated tool suite released - Forum – English – Finale 3D | https://finale3d.com/groups/english/forum/topic/updated-tool-suite-released/
[135] Identity and Similarity Calculator - OmicsTools | https://www.omicstools.com/tools/sequence_analysis/ident_sim
