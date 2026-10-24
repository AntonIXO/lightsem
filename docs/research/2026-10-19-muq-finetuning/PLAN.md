# MuQ fine-tuning implementation

Protocol: `muq-peft-pilot-20261019-v1`

The implementation preserves the authenticated MuQ layer-9 + Base92 contract.
The first arms train only downstream temporal adapters and light heads over
cached features. Their receipts explicitly record `muq_finetuned: false`.
Actual final-block or full-MuQ arms require a live MuQ module and a new receipt;
cached features cannot be described as backbone fine-tuning.

Implemented components:

- `foundation_audio_light/peft.py`: zero-initialized local convolutional,
  causal-transformer, and hierarchical adapters; generic zero-initialized
  LoRA replacement; trainable-parameter reports.
- `foundation_audio_light/layer_selection.py`: finite-contract checks,
  effective-rank/constant-channel geometry, grouped blocked ridge probes with
  train-only PCA, and layer-family ranking.
- `foundation_audio_light/muq.py`: pinned MuQ metadata validation, cached
  layer loading, lazy live-MuQ loading, and explicit frozen/last-block/full
  live-backbone scopes.
- `scripts/experiments/run_muq_layer_selection.py`: CPU layer-selection CLI.
- `scripts/experiments/run_muq_peft_pilot.py`: one-seed downstream PEFT pilot
  with raw aim/control diagnostics and promotion-forbidden receipts.
- `scripts/experiments/extract_muq_foundation_v2.py`: optional multi-layer
  extraction; the default remains layer 9 and preserves the existing cache
  format.

The layer-selection probe runs through `bash run.sh`; Torch adapter pilots run
through `bash run_torch.sh`. Every output path must be new and must preserve
the input, checkpoint, feature, split, and code hashes.

The current implementation is an experiment harness and does not promote a
checkpoint or emit native DMX. The existing raw-body and Music–Motion Causal
Audit remain mandatory before any mapper, renderer, or show delivery.
