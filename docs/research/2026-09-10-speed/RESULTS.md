# RTX 5070 training efficiency results

Subsequent adoption decision: the user explicitly requested the fast training
profile as the default after reviewing these results. New Night protocols use
it; the measurements and scientific verdict below remain unchanged.

The implemented BF16 + compilation + prefetch profile cut complete training
time by 5.18–5.35× across all four fixed runs. It did not establish quality
retention within the agreed 3% margin, and MuQ+FiLM lost a previously passing
visibility gate in both seeds. At the pilot's conclusion no fast profile was
promoted and FP32 remained the default; the later user-directed adoption is
recorded separately above.

Vibescience computed `directional_only` for
`exp-night-iteration-speed-20260910`: the speed prediction passed, but the
quality and preserved-output gates failed. This is a result for the combined
runtime recipe, not a rejection of BF16 alone.

## Complete training performance

All runs used batch 32, 640-frame crops, 15,200 updates, 500 warmup updates,
learning rate 3e-4, ordinary AdamW, and 486,400 crops. Parameters and optimizer
states remained FP32. Data, initial weights, final sampling/flow RNG positions,
and evaluation inputs matched the corresponding archived reference.

| Variant | Seed | Archived FP32, minutes | Candidate, minutes | Speedup |
|---|---:|---:|---:|---:|
| Base once | 0 | 36.75 | 6.88 | 5.346× |
| MuQ FiLM | 0 | 40.96 | 7.91 | 5.180× |
| Base once | 1 | 36.71 | 6.87 | 5.346× |
| MuQ FiLM | 1 | 40.98 | 7.90 | 5.188× |

Total: **155.40 → 29.55 minutes**, or **5.26×**. Candidate times include data
loading, required checks, actual compiler startup and checkpoints. The archived
baseline excludes loading, making this comparison conservative. Compiler caches
were populated by the engineering screen; separate cold compilation measured
about 26 seconds per model. These numbers measure training, not the entire
feature-extraction/rendering campaign.

Peak allocated GPU memory was 4.25 GiB for Base and 4.51 GiB for MuQ FiLM.
Batches 64 and 128 fit but processed fewer crops per second. Fused AdamW changed
timing by less than 1%; it was not selected. An 8-bit optimizer would save only
about 22 MiB for these models and was not installed.

## Quality and uncertainty

Each checkpoint was evaluated in FP32 on all 40 fixed development songs, with
six draws, eight Euler steps, nine controls, and the eight fixed visual audit
clips. Relative degradation below is signed so **negative is improvement for
every metric**, including AP. Brackets are paired canonical-song bootstrap
95% intervals, using 10,000 resamples and seed 20260910.

| Variant / seed | Energy degradation % [95% CI] | A-MSE degradation % [95% CI] | Event-AP degradation % [95% CI] |
|---|---:|---:|---:|
| Base / 0 | +0.270 [−0.617, +1.246] | −1.076 [−5.372, +3.508] | −1.022 [−6.875, +4.503] |
| MuQ / 0 | −1.599 [−3.156, −0.222] | −3.387 [−9.613, +2.551] | +2.954 [−2.141, +8.257] |
| Base / 1 | +0.297 [−0.616, +1.235] | −0.468 [−5.233, +4.099] | −0.175 [−5.831, +5.512] |
| MuQ / 1 | −0.201 [−1.585, +1.016] | −2.136 [−6.297, +2.196] | +3.111 [−2.848, +8.756] |

Energy retention passed for all four cells; A-MSE retention passed for both
MuQ cells. Base A-MSE and all AP comparisons failed to exclude degradation
beyond 3%. The MuQ seed-1 AP point estimate also exceeded the margin. Wide
intervals are inconclusive evidence, not proof that Base became worse: its
mean A-MSE and AP improved in both seeds.

The visibility gate requires at least six of eight clips to meet all physical
and screen checks. Qualifying clip counts were:

| Variant / seed | Before | After | Previously passing gate retained? |
|---|---:|---:|---|
| Base / 0 | 5/8 | 4/8 | Already failed; remains failed |
| Base / 1 | 7/8 | 7/8 | Yes |
| MuQ / 0 | 7/8 | 4/8 | No |
| MuQ / 1 | 7/8 | 5/8 | No |

The new MuQ failures came from the `screen_light` subcheck (brightness/support),
not newly constant native channels. No new constant channels were found.
Preserving existing musical-conditioning failures does not establish musical
control or aesthetic quality.

## Verification and reproducibility

- 52 runtime/model/training/evaluation tests passed, including exact compiled
  checkpoint/resume replay. Two additional clean-setup tests passed afterward.
- A real two-song engineering smoke exercised all nine controls, compact
  retention and the renderer before scientific training.
- Independent audit authenticated all 2,986 pinned inputs, initial tensors,
  FP32 optimizer states, final sample exposure and sampling/flow RNG positions.
- Metrics were independently recomputed for 288 retained control arrays.
  Maximum native-energy discrepancy was 1.45e-15; A-MSE and AP also matched.
- All nine controls for one non-retained song per cell were regenerated: all
  36 outputs matched standardized-trajectory, event and DMX hashes exactly.
- No scientific retries, substitutions, OOMs, nonfinite failures or TEST access
  occurred. All four planned cells were retained.

The initial batch-128 engineering measurements hit a Dynamo recompile limit
and fell back to eager execution. Those measurements were preserved and
excluded; only those affected engineering cases were repeated with isolated
compiler state. Batch 32 still won. Runtime now fails explicitly on that limit.

After the scientific audits, the setup helper was corrected so clean future
benchmarks do not require a compiler-repair archive. An AST/hash audit confirmed
that only the freeze/setup logic changed; numerical, training and evaluation
functions are unchanged. The original protocol and source snapshot remain
immutable. See [usage and archived replay](../../ITERATION_SPEED.md).

The next informative comparison would isolate BF16 from compilation and its
different stochastic dropout execution under a new protocol. Repeating this
bundle to select a favorable seed would not resolve the failed gates.

## Artifacts

- [Complete comparisons and gates](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/summary.json)
- [Engineering screen](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/engineering.json)
- [Frozen protocol](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/protocol.json)
- [Independent metric/RNG audit](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/independent_audit.json)
- [Exact regeneration audit](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/regeneration_audit.json)
- [Post-pilot setup audit](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/release_audit.json)
- [Computed scientific verdict](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/vibescience_result.json)
- [Recorded compute scope](/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910/cost_ledger.json)
