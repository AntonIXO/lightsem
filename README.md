# LightSem

LightSem turns audio into stage lighting. Start with the [September 22 light-layer delivery](experiments/2026-09-22-color-beat/README.md): it connects the trained Rich96 model, the Shpongle overlay, its source code, and the measured results. That delivery is the current **reference for navigation**, not a promoted runtime default. The normal causal `LightModel` path remains in `lightsem/model/infer.py`.

## Where things live

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
