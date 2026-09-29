# September 22 light-layer delivery

For new development, use the [September29 user-selected independent baseline06](../2026-09-29-baseline/README.md). This page describes the earlier shared-model delivery and its training campaign.

**Reference model:** fixed seed-0 `S_VR_P0` Rich96 selected checkpoint at `/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z/observer_fits_v1/03_seed0_S_VR_P0/best.pt`. Its Python implementation is in `lightsem/model/light_layers/`; maintained drivers are in `scripts/light_layers/`.

**Reference delivery:** [Shpongle demo report](outputs/shpongle_layers_demo_20260922T161917Z/README.md), with its [frozen one-song inference helper](records/source/shpongle_layers_demo_20260922T161917Z/infer_overlay.py) preserved separately from the media. The overlay builds on the saved rhythm-companion lighting stream; its checkpoint and source assets remain under `/mnt/lightsem-data/`.

The [observer report](records/OBSERVER_RESULTS_V1.md) says the proxy study completed but missed its null-validity requirement. The [palette report](records/PALETTE_SECONDARY_RESULTS_V1.md) also did not qualify the proposed benefit. This delivery is experimental and does not change `lightsem.model.infer.LightModel` or the default runtime pipeline. Protocols, receipts, and research scripts are preserved in `records/`.
