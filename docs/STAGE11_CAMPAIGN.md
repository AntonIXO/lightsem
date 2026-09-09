# Stage 11 — expanded data and temporal scene generation

The training and numerical evaluation have finished; see [results and limitations](STAGE11_RESULTS_20260908.md). This historical protocol is retained for reproducibility. The subsequent [dance-trained product requirement](DANCE_DRIVEN_CONTRACT.md) rejects programmed beat/pulse effects in the product path; Stage11's fixed scene renderer does not satisfy it.

Approved 2026-09-08. Execute unattended on the RTX 5070, with CPU work overlapping GPU work, one accelerator experiment at a time, at least ten planned GPU hours and a 24-hour overall ceiling. Existing product defaults, historical checkpoints and the 39 original ratings remain unchanged.

## Scientific comparisons

| Family | Arms | Seeds | Updates and batch |
| --- | --- | --- | --- |
| Added JD data | D_old, D_full | 0–4 | 12,000 × 48 four-second windows |
| Scene generation | S_pointwise, S_temporal, S_swapped | 0–4 | 32,000 × 32 sixty-second windows |

The descriptor recipe retains its four deterministic, three event and seventeen stochastic outputs. The two scene denoisers have approximately 3.73M parameters, matched within 0.08%. The temporal head uses six dilated convolutions; the pointwise head has matched active MLP parameters. S_swapped replaces the first 48 audio features with another training song while retaining four recipient beat features. All deployment paths use audio only.

Vibescience hypotheses: `h11-d-expanded-jd-20260908`, `h11-t-temporal-scene-flow-20260908`, `h11-a-nonbeat-audio-dependence-20260908`, under `dancing-light-generative-20260903` in `/root/lightsem-vault`.

Primary gates: motion R² gain ≥0.02; temporal multi-lag Wasserstein reduction ≥15% and ≥0.02 normalized units; minimum paired novelty-AP advantage ≥0.02 against beat, donor, shuffled-training and mean circular-shift controls. Uncertainty resamples canonical songs and training seeds. Additional fidelity, variation, frequency and eligibility gates are committed in `protocol.json`. Hypothesis confirmation additionally requires the same gates on the once-locked test.

## Data and integrity

669 usable source videos and pose caches; 617 recordings meet the common 60-second/80%-coverage core. A versioned preparation preserves raw masked HSV histograms and excludes menus via active spans. Original-pixel pose boxes are rescaled to the decoded image; one fixed padding policy per recording prevents masking changes from creating events. Visible-pixel counts remain available for quality diagnostics.

Identity checks use title/known links plus offset-aligned active-span audio fingerprints. All 15 known duplicate pairs are recovered. The frozen map has 531 canonical families, with 120 verified new eligible families supplying 40 validation and 40 locked-test groups. Core training has 419 song families / 515 recordings / 798 coaches; validation 40 / 52 / 85; test 40 / 50 / 80. All 307 old eligible recordings / 453 coaches belong to training; expanded training adds 208 recordings / 345 coaches. RPMC is excluded because source-performance grouping is absent locally.

New statistics use only permitted training data and live inside checkpoints. Recording-relative scene value targets are computed before cropping. AP uses symmetric ±0.2-second alignment labels and correct ties; no generated percentile quota creates artificial events. Donor and matched generation share input length and explicit initial noise. Pending validation and best-checkpoint publication are recoverable without selecting a new seed or redrawing a finished evaluation.

## Execution

Root: `/mnt/lightsem-data/experiments/lightsem/stage11_20260908/`.

The CLI is `bash run_torch.sh -m lightsem.experiments.stage11 --root ROOT COMMAND` with `init`, `prepare`, `freeze`, `verify`, `train`, `evaluate`, `aggregate`, `unlock-test`, `render`, and `supervise`. The persistent supervisor dispatches PATH-based Python from the read-only source snapshot; it never borrows another project's environment. It authenticates every completed run and stops at `awaiting_primary_review` after validation. Only an explicit, authenticated primary review can open test.

The source/config/data snapshot is immutable during the campaign. Resume requires matching optimizer/scheduler/RNG state and explicit recovery authorization. Logs and status files are under the campaign root; no remote messages or model uploads are sent. GPU-resident training caches remove per-window disk reads. Short engineering profiling reached 9.1 GiB at batch 128 but reduced pointwise-model throughput about 5% relative to batch 32; batches 64 and 96 were also slower. These measurements preceded the final deterministic settings and do not establish the optimal batch for that exact configuration or its convergence. The original batch sizes and optimizer-update counts were retained for the matched campaign. A strict-settings integration smoke measured 99% GPU activity; this indicates kernel activity, not 99% of peak compute throughput. Core training is now estimated at 16–17 hours rather than the initial 10.2 hours.

At the end, verify all artifacts and record computed Vibescience verdicts separately from numerical gates and deployment eligibility. Produce full-song examples and 40 primary blind pairs plus four repeated and four sham pairs, all with fixed renderer configuration and a final encoded-video image-luma gate. These are image measurements, not calibrated physical luminance. Human preference remains pending.

## Validation

CPU tests cover models, temporal dependence, chunk equivalence, metric nulls/ties, identity grouping, offset/normalization, artifact authentication, interruption recovery, supervisor deadlines and rendering provenance. Real GPU engineering smoke uses two old training songs only and is excluded from scientific results. The actual renderer smoke passed with a1.31% encoded-video luma gap. Completed run artifacts, rather than these engineering checks, decide the hypotheses.

The ancillary CPU ridge baseline runs concurrently after freeze, using equally weighted songs and a fixed mean-squared-loss +0.001L2 objective (effective normal-equation alpha=0.001×training songs), with no locked-test access or neural-gate override. New feedback uses `stage11 rate --rater anton`; ties and inability to judge are distinct.
