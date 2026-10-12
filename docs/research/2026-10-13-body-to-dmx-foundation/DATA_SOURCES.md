# Data-source inventory (2026-10-13)

This inventory records what is actually present locally and what it can support.
It is a provenance index, not a claim that all sources are license-compatible
for product deployment.

| Source | Local evidence | Paired modalities | Usable target | Limitations |
|---|---|---|---|---|
| BeatSaver lightshow maps | `/mnt/lightsem-data/beatsaver/manifest.json`, 317 inspected / 300 usable, 300 map WAVs | map audio + authored light events | 40 Hz virtual groups, cue events, RGB/intensity/effect trains | mapper dialect, virtual groups, no body and no physical fixture patch |
| RPMC-L2 | `/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5`, `/root/lightsem/data/cache/rpmc_train`, 699 extracted songs | frozen OpenL3 audio embedding + HSV light histogram | reduced scene vector (hue cos/sin, saturation, value, p90, concentration) | no raw waveform in the compact extraction, no geometry or per-fixture DMX |
| FineDance | repository manifests and existing native-dynamics feature receipts | music + paired 3D body | motion prior / body validation | no authored DMX; research data terms apply |
| AIST++ | `/mnt/lightsem-data/aistpp` | music + motion | motion prior / mapper control | reconstructed motion, no authored DMX |
| Just Dance | `/mnt/lightsem-data` native BF16/full-fit receipts | music + masked 2D observations / generated native body | motion prior and safety/development control | no 3D ground truth for most observations, no authored DMX |
| LightSem showcase songs | pinned local Shpongle and AURORA — Cure For Me feature/body/DMX receipts | exact own-song audio + generated body + emitted DMX | end-to-end source-dependence and delivery controls | only two songs; cannot train a foundation model |

## Internet research leads

The search found no openly downloadable, large, license-clean corpus of raw
professional console DMX paired with commercial song audio. The closest
research leads are:

- **LightingGen** reports a 2.62-hour virtual-stage DMX sequence corpus and a
  DULA representation, but the dataset is not currently exposed as a verified
  public download in this workspace.
- **SeqLight** publishes code and describes multi-light HSV control, but its
  public repository does not establish a general raw-DMX corpus for this
  project.
- **BeatSaver** is the practical public source because maps contain authored
  timing and colours, although map audio and licences vary.
- **xLights sequences** and **Beat Saber/Vivify** maps are useful event/effect
  sources when their assets are parseable, but they are not interchangeable with
  a physical DMX universe.

The external references used for this inventory are the official or primary
pages returned by the web search: [FineDance paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_FineDance_A_Fine-grained_Choreography_Dataset_for_3D_Full_Body_Dance_ICCV_2023_paper.pdf),
[OpenDance paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_OpenDance_Multimodal_Controllable_3D_Dance_Generation_with_Large-scale_Internet_Data_CVPR_2026_paper.pdf),
[LightingGen record](https://www.researchgate.net/publication/400853675_LightingGen_A_DMX_based_Generation_Method_for_Entertainment_Stage_Lighting),
[SeqLight repository](https://github.com/RS2002/SeqLight), and
[SeqLight paper](https://arxiv.org/abs/2605.03660).

Before using any source for a released model, record its exact revision,
licence, audio redistribution rights, and map/asset hash. The present MVP is
research-only until that audit is complete.
