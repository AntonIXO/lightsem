# Row65 latency instrument scope addendum V1

**Original learned-path latency qualification is unestablished. This is not a measured65ms failure.**

The earlier repair limits future-audio causality and real-fixture evidence. It does not explicitly limit “historical software timing result” to the partial calculated budget found here. This is a second original row65 instrument gap, not a new target or execution authorization.

## Original boundary

Hypothesis: `the-deployed-path-meets-its-latency-budget-and-b0aff5`.
Experiment: `exp-006-the-deployed-path-meets-its-eb0fc6`.

The Statement claims learned-path latency below65ms, audio-in→DMX-out. `G-lat-learned` requires learned-path p99≤65ms. The Plan says to time each stage over many blocks and report p50/p99: analysis hop, primed model forward, conductor and DMX packing.

**Stagewise measurement is allowed. A single monolithic timer is not a new required gate.** However, a full-path claim needs complete critical-path accounting and defensible tail/overlap composition.

## Available candidate source

Commit `1656ffde47a26bbd9655b5ceef8ddaebd80e4895` is a saved historical snapshot, **not an authenticated executed checkout**. Its exclusions below are source facts, not proof of which bytes exp-006 ran. Exact blob IDs and SHA256s of the read source text are in the JSON.

| Source | Lines | Boundary |
|---|---|---|
| `run_stage4.py` |65–79|Times `lm.model(x)` with preconstructed synthetic GPU inputs.|
| `lightsem/model/light_trunk.py` |221–223|Forward returns hidden state,A and B logits; it does not call C sampling.|
| `lightsem/model/infer.py` |68,86–96,100–124|Feature assembly, transfer/normalization,C sampling, CPU copies,unscaling and event processing are separate descriptor work.|
| `lightsem/model/light_trunk.py` |137–143;185–200|C sampling has separate integration loops; actual loaded head/config remains unpinned.|
| `run_stage4.py` |85–89;119|`extract_bus` and `descriptors` run outside these timers. The audio-hop term is a sample-count/rate constant, not measured feature computation.|
| `run_stage4.py` |88–92|One clip-level conductor/pack duration is divided by frame count and copied100times for percentiles; this is not per-frame p99.|
| `run_stage4.py` |119–129|“learned_path_p99_ms” adds hop,model p99,conductor/pack p50,and assumed5msnetwork/30msfixture.|

The saved62.5ms versus65ms is therefore retained as a **partial calculated budget**, not an established complete audio-in→DMX-out p99. p99 of one term plus medians of other terms is not generally p99 of their sum. The stored endpoint assumptions must remain visible.

No excluded operation's cost was measured here. Nor is it established that every operation adds serial delay; a scheduling/overlap bound was not supplied. Disk loading need not be charged to a live audio-in endpoint. Do not infer an overrun, a lower bound, or a failed gate.

## Recommended record disposition

Add the software-latency qualification beside the already missing future-AUDIO causality instrument. Retain all component observations, numbers, gates, verdicts and Notes. Exact-version reconstruction remains blocked by the prior provenance gaps. A current-version test cannot retrospectively complete this result.

Proposed dated clarification:

> Retain the reported62.5ms calculation versus the65ms budget and the separately recorded component timings. The available,execution-unpinned historical source times trunk+A/B forward but excludes feature extraction,C sampling and descriptor processing from that timer,and combines model p99 with amortized conductor/packing summaries and assumed network/fixture terms. The saved sum is a partial budget estimate,not an established full audio-in→DMX-out p99. No latency overrun is measured;preserve historical numbers,gates,verdict and Notes,and do not substitute a current-version test.

No native write was made by this child. No further payload/model/runtime work was done. Existing reports and sources remain unchanged. This pair is capped at16KiB within the262,144-byte independent row65 metadata reserve; the global cap and12MiBpreparation allowance are unchanged. See `ROW65_METADATA_FOLLOWUP_BUDGET_V1.json`.
