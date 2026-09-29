# GPU execution ledger readiness V1

**Status: UNRECONCILED. No positive GPU admission credit. Exhaustion is not established.**

This is an accounting-only review. No run, test, model call, payload read or vault call was made. The JSON has exact source paths, SHA-256 hashes, timer scopes and completion checks.

## Authority and start

- `CAMPAIGN_DESIGN.json` sets **36 GPU hours** for `lighting-cue-model-20260922t094347z`, with one GPU and 60 minutes per fit. The nine native hypothesis receipts repeat this campaign ceiling. This is not authority for a project-lifetime ledger.
- Design creation: **2026-09-22 09:43:47 UTC**. Nine hypothesis registrations: **10:06:05–10:06:10 UTC**. First located execution registration: rhythm features at **10:09:34 UTC**.
- These are creation/registration times, **not actual GPU job starts**. The first actual GPU launch time and full launch inventory are not established.
- End-to-end `EXECUTION_PLAN_V1.json` retains the 36-hour limit unless revised. It does not grant a reset or define whether every later demo/legacy task falls into the original campaign.
- CONTROL means policy/authorization/execution metadata, not a required filename.

## Located completed work

All times below are seconds. Different timer scopes must not be added together.

| Work | Completion evidence | Available time and missing coverage |
|---|---|---|
| 15 full fits | All 15 nonsmoke receipt hashes match the queue; all child exits 0 | Training inner sum **3199.489**; full child sum **3246.234**; native queue **3252.367**; published outer wrapper **3254.131**. Nested, not additive. |
| Rhythm features | 516 train/demo and 52 historical-DEV records; completion logs | **75.226 + 7.313** after CUDA model construction. Loading/outer job time missing. The 13-record pilot receipts are prefixes of these jobs, not extra executions. |
| Visual engineering V1/V2/V3 | All three attempts retain 12 record outcomes; V3 has six successes/six failures | Per-record sums **0.004 / 107.710 / 503.340**. V1/V2 failed before inference. Outer initialization/probe durations remain missing; failures are not silently deleted. |
| Observer capture | **131 unique** completed records, all receipt hashes match aggregate; original120 + shard5 + shard5 + shard1 | Recorded process-segment sum **9851.970**. Shards overlap on one GPU. This is neither device-busy time nor full worker walltime. Initial loading and two preserved partial-record attempts remain uncovered. |
| Student smokes | One initial complete receipt; two batch4 child exits0 | Initial inner **1.545**; two whole child jobs **8.628 / 12.759**. Initial outer job missing. These engineering jobs are separate from 15 scientific fits. |
| 30 raw predictions | All 30 completion events and receipt hashes verified | Native queue **37.611**, published outer **39.646**, overlapping child sum **109.947**. These scopes are compatible, not contradictory. Three processes do not mean three GPUs. |
| Palette CUDA diagnostics | One completed failed-fixture replay and completed 84-cell numerical preflight | Internal diagnostic timers **0.488 / 3.855**. CUDA head replay occurred in both; full outer lifetimes are missing. CPU scientific palette prediction does not erase these GPU jobs. |
| Original Shpongle anchor | Complete inference log and preview/full render receipts | Rounded forward/rollout **0.188**; render inner **14.694 / 193.529**. Full native wrapper timing is unavailable under the required retained-handle condition. |
| Four comparison arms | Five directly retained completed native handles, all exit0 | Whole inference job **6.804** and four renders total **800.635**: **807.439** whole-job subtotal. Inner timers are subsets, not added. Original anchor was not rerun. |

Palette scientific prediction/continuation are CPU-only: the failed initial invocation and all101 retained +85 new cells remain recorded. V5/V6 row8 preparation also used CPU with GPU devices blocked. Their GPU charge is zero **only within that enforced scope**. CPU controller waits overlap capture and are not extra GPU jobs. Published scientific metrics occurred incidentally in compound receipts; none were recomputed.

## Limits do not fill gaps

- Fit controller enforces each child under `min(3600s, supplied residual)`. Its supplied **30 hours** is not a reconciled balance or evidence that six hours were spent.
- Capture policy says original **14400s** deadline and **3600s per shard**. No complete external worker-lifetime/enforcement record was located. Hash-matched visual source has no global wall-budget enforcement. Policy is not actual elapsed time.
- Prediction timeout **900s applies per child**, not to the entire queue. Parallel scheduling amended the old one-process rule but did not define a new GPU-hour accounting unit.
- Comparison inference has a native `timeout 1800s` command and completed result. Four render handles also completed; no matching original-demo handle was supplied.
- GPU utilization, allocated memory, video duration and file mtime cannot supply missing walltime or budget credit.

## Admission decision

**Actual cumulative spend: unknown. Verified total upper bound: unknown. Safe remainder: unknown. Positive admission credit: 0.** This is not an exhausted-budget verdict.

Root must fix scope/charge convention and recover extant complete launch/exit/outer-timing metadata for the named gaps. If unavailable, request an explicit prospective budget decision. Do not rerun work, open TEST, reconstruct hidden jobs from data, or silently reset the ledger. Only the parent may make native vault writes.

The two report files stay within 48 KiB. They and the parent handle receipt use the existing 64 KiB GPU-ledger metadata reservation. The scientific cap and row8 packet reservation are unchanged.
