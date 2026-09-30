# LightSem

LightSem turns audio into stage lighting. The current development baseline is **C: variant06 independent Rich96 (`I_VR_P0`, selected seed0), rhythm companion, palette history on, and300ms smooth color transitions**, explicitly chosen by the user on September29. Start with the [baseline and A/B review](experiments/2026-09-29-baseline/README.md); its exact checkpoint, behavior and accepted video are pinned in [the baseline configuration](configs/light_layers/baseline.json). Reviews use compact native1080p AV1 with AAC128k; the four full-track Shpongle videos are43–44MB each.

The [September22 campaign](experiments/2026-09-22-color-beat/README.md) contains the training implementation and earlier measurements. The user selection is a product-development decision from Shpongle review; previous scientific results retain their recorded scope. The older causal `LightModel` API in `lightsem/model/infer.py` is a separate implementation, not a drop-in loader for the independent light-layer model.

The [visual-learning studies](experiments/2026-09-29-visual-next/README.md) are complete under the automated exploratory scope: [19 Shpongle review videos](experiments/2026-09-29-visual-next/records/REVIEW.md) and [measured results](experiments/2026-09-29-visual-next/records/RESULTS_EXPLORATORY.md). Source annotation is optional; original independent-label hypotheses remain unconfirmed. Baseline C remains unchanged.

## Where things live

The [festival scene](docs/FESTIVAL_SCENE.md) adds a complete Blender venue and a
configurable multi-universe DMX patch, offline music-video rendering, and a live
Art-Net receiver. It expands the accepted controller into arbitrary fixture
counts without changing the selected model or the existing diagnostic renderer.

| Path | Purpose |
| --- | --- |
| `lightsem/model/light_layers/` | September 22 model, targets, training, evaluation, and overlay algorithms |
| `lightsem/audio/`, `conductor/`, `dmx/`, `motion/`, `sim/` | Reusable audio-to-light pipeline |
| `lightsem/experiments/` | Earlier model studies, grouped by campaign |
| `scripts/` | Maintained launchers, data preparation, diagnostics, and vault tooling |
| `experiments/<campaign>/records/` | Protocols, reports, receipts, and preserved research history |
| `experiments/<campaign>/outputs/` | Existing repository-local results and deliveries |
| `runs/` | New transient local outputs; ignored by Git |
| `/mnt/lightsem-data/` | Datasets, caches, checkpoints, and large experiment artifacts; existing paths are unchanged |

Use `bash run.sh <script>` for the CPU environment and `bash run_torch.sh <script>` for the dedicated Torch environment. Launchers now live under `scripts/`; see [the script index](scripts/README.md). The research [campaign index](experiments/README.md) and [relocation map](experiments/RELOCATION_MAP.json) locate historical files by their old path.

The historical records keep their original contents and absolute path claims. Those paths are documentary after this move; use the relocation map to find the files. Replaying a frozen experiment from its original binding requires a new, versioned binding for the new source paths. No new training, scoring, or model promotion occurred in this reorganization.
