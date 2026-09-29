# Launchers and tools

Run these from the repository root through `run.sh` or `run_torch.sh`. The two wrappers remain at the root so they still set the expected working directory and Python path.

| Directory | Contents |
| --- | --- |
| `light_layers/` | September 22 cohort, feature, training, evaluation, and campaign drivers |
| `motion/` | Motion-study and rhythm-companion drivers and audits |
| `legacy_stages/` | Earlier numbered stage and Oslo campaign launchers |
| `data/` | Download, extraction, and cache builders |
| `training/` | Earlier model trainers and calibration |
| `rendering/` | Demo and figure renderers |
| `diagnostics/` | Probes, checks, and one-off audits |
| `vault/` | Vault registration and result logging |
| `maintenance/` | Environment setup helpers |

Example: `TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh scripts/light_layers/run_light_layers_train.py --help`. The wrapper's designated `/mnt` Torch environment currently cannot be executed on this host; the existing local Torch access environment is used for tests. Historical commands recorded in `experiments/*/records/` are not rewritten.
