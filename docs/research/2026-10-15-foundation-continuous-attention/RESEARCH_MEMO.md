# Continuous attention foundation decoder — 2026-10-15

This is a diagnostic one-seed decoder pilot. It consumes an already saved and
audited native-C body plus pinned audio features and predicts a rig-independent
light representation. It does not retrain or replace native C, and it does
not produce a product demo or a promoted checkpoint.

## External research

Perplexity Deep Research was run on 2026-10-15 with the attached LightSem
brief, the pinned MuQ revision, and the repository safety contract. Its main
recommendation was a frozen-MuQ, audited-body-conditioned, stateful fixture
decoder with local attention, recurrent state, persistent Perceiver latents,
and index-free fixture queries. It explicitly stated that eight songs and one
seed can falsify a design but cannot establish broad generalisation.

The report also corrected several claims that had been treated as facts:

- MuQ's exact output rate must be measured on the pinned checkpoint; the
  external sweep did not verify a universal 25 Hz claim.
- Third-party parameter counts and Skip-BART/RPMC storage figures are not used
  as protocol facts here.
- BeatSaver rights are map-specific and must be audited per asset.
- All numeric pilot thresholds are engineering hypotheses, not literature
  constants; they are recorded before the run and cannot be changed after a
  result.

Primary references returned by the sweep are [Transformer-XL](https://arxiv.org/abs/1901.02860),
[S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752),
[Perceiver IO](https://arxiv.org/abs/2107.14795), [Deep Sets](https://arxiv.org/abs/1703.06114),
[Set Transformer](https://arxiv.org/abs/1810.00825), the [MuQ repository](https://github.com/tencent-ailab/MuQ),
and the [Skip-BART/RPMC-L2 paper](https://arxiv.org/abs/2506.01482). These support
architecture motifs; none establishes LightSem choreography or safety.

## Local decision

The first implementation is intentionally smaller than a foundation model:

```text
52-D frozen feature bus @ 10 Hz
  + saved root-relative native body @ 10 Hz
  -> local causal attention (32-frame windows)
  -> GRU recurrent state across chunks
  -> 8 persistent latent slots with cross-attention
  -> shared arbitrary-N fixture queries
  -> separate motion, OKLab color, brightness, scene and event heads
```

The body source is audited before the decoder is called. Motion proposals are
body-effort gated and passed through a bounded jerk/acceleration/velocity
projection. That projection is an output constraint, not a post-hoc repair of
native motion. The pilot does not interpolate, compile, render or emit DMX.

The first run exposed two implementation failures and preserved both receipts.
Unbounded body/audio attention caused a `0.316656` full-vs-chunk scene error;
after localizing that attention, the fused GRU still produced `3.63e-5` error.
Replacing the fused kernel with an explicit GRUCell recurrence yielded
`6.56e-7` max error on the v3 receipt. The v3 smoke also measured zero fixture
permutation error, zero frozen-body angle, and a nonzero assigned-limb response
of `0.002523`. These are architecture checks, not choreography evidence.

The v3 result was then re-run with the actual pinned 1024-D MuQ layer-9 tokens
from each row's authenticated timeline. That v4 receipt has chunk error
`5.96e-7`, permutation error `0`, frozen-body angle `0`, limb response
`0.005693`, and validation total `0.467880`. The earlier v3 52-D visual-bus
run remains a control and is not counted as the MuQ experiment.

The data are the eight existing diagnostic BeatSaver body/light rows (`1023`,
`10dcc`, `1181f`, `11b01`, `1672d`, `210e3`, `21c3a`, `3b4ab`). Six rows are
training windows and two are held out by song. Every row keeps its body,
feature, visual target and light target hashes in the external receipt.

## Interpretation boundary

Passing a loss or chunk-continuity check does not establish music-conditioned
choreography, raw native-body quality, decoded-DMX safety, or generalisation.
The pilot is useful if it exposes a state/fixture/source failure early. A
future promotion would still require both own-song trajectories, decoded 40 Hz
DMX safety, source counterfactuals, arbitrary-N checks, and an explicit user
selection.
