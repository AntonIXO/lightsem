# Full-track dynamic dance combinations

Completed 5 October 2026. The [synchronized gallery](http://127.0.0.1:8776)
contains four full-song comparisons: C3 and C4 on Shpongle and AURORA —
Cure For Me. The [final report, including all four videos and dancing data](/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/reports/FINAL_REPORT.md)
and [machine-readable report](/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/reports/FINAL_REPORT.json)
contain the complete evidence. No combination was promoted to the product
baseline, and no model was retrained.

## Actual composition

- **C3 = D4(D7 + bounded D8 global-minus-local residual).** D7 generates the
  trajectory. At the same flow state, the D8 global-plan response minus its
  local-plan response contributes a residual capped at 0.25 per-frame
  normalized-native RMS, without a nonzero floor. D4 then recomposes
  contact-compatible fragments from that song's generated body.
- **C4 = D6(exact saved C3).** The learned local effort/duration planner retimes
  that same C3 trajectory. Disabling D6 removes this extra retiming stage;
  D8 still retains its global planner and local-plan reference.

This is experimental inference composition, not a jointly trained model.
Existing seed 0 checkpoints remain frozen. Both songs supply their own audio
and generated bodies, with the same explicit initial noise for reference and
candidate. The left-hand visual reference is freshly generated full-track D7
in every video; C4's incremental numerical comparator is C3 with D6 off.

## Restored brightness and measured results

The historical audio loudness controller and modest body accents are restored.
The six-head reference replay reproduces all 66,552 original baseline dimmer
bytes; current fixture roles and bulbs keep their anatomical assignments.
No autonomous motion controller is restored. P2's color recipe remains fixed;
full-song Cure inference can differ from its earlier 180-second crop.

| Measurement | Shpongle | Cure For Me |
| --- | --- | --- |
| C3 emitted recurrence increment versus D7 | −0.03313 | −0.04085 |
| C4 emitted release increment versus C3 | +0.00928 | −0.22304 |
| C4 supported whole-body holds below 0.05 m/s | 31.7 s | 10.0 s |

Both registered primaries fail the required +0.02 increment on each song.
C3 passes physical nonregression. C4 creates supported rests but regresses jerk
and emitted-motion retention on both songs, plus foot sliding on Shpongle.
Pause placement is not independently qualified; more holds and source
sensitivity do not establish musical superiority. All videos remain labeled
diagnostics, with artistic acceptance pending user review.

## Delivery and verification

The complete source songs last 277.3067 seconds and 201.64 seconds. Videos are
1920×1080 at 30 fps, with 8 moving beams, 5 simulated laser beams and 50 bulbs.
Complete software decoding verified 28,740 frames across the four movies.
Encoded durations are 277.3333 and 201.6667 seconds: 26.7 ms of endpoint/audio
padding, with original audio speed preserved. Audio and live-browser checks
passed.

Two actual-noise generation repeats and compiled-output repeats were exact.
Frozen-body, limb and added-fixture checks passed; common controls remain
exact across 1, 63 and 117 outputs. These tests do not imply additional
independent dance information.

The campaign used 0.2552044165 GPU-stage hours; the cumulative total including
the preceding campaign is 5.1085025054 hours. Accounting includes conservative
lock-owned overhead. The 20 GiB storage reserve was maintained. Original
artifacts, failed attempts and narrow engineering corrections remain recorded.

## Restart the gallery

After the server stops, restart it from the frozen execution source with its
Python import path explicitly set:

```bash
cd /mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/sources/execution_v4
PYTHONPATH=/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/sources/execution_v4 \
  /root/lightsem/.venv/torch-access/bin/python -m scripts.rendering.serve_body_portfolio \
  --root /mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/review \
  --port 8776
```
