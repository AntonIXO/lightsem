# Dynamic musical dance campaign

User-approved implementation, 5 October 2026. Primary artifact root:
`/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1`.
Scientific records belong to `/root/lightsem-vault`, problem
`dynamic-musical-dance-20261005`.

## Objective and scope

Eight distinct upstream interventions produce each song's own body and feed
the same V3/01 mapper, 8 beams, 5 simulated lasers, 50 bulbs, and exact P2 RGB.
The display interval is 60–180 seconds of Shpongle and AURORA — Cure For Me,
with actual baseline/candidate bodies and actual decoded DMX visible together.
Musical dramaturgy, not lyrics or unvalidated emotion recognition, is in scope.

The existing dancer does not have literally constant speed. Its reviewed
windows show continuous movement and no mean joint-speed frames below the
exploratory 0.05 m/s reference after removing horizontal root travel. That is
not an aesthetic verdict. Emitted motion also loses fast articulation and
approaches actuator acceleration limits. Body, mapper and emission must be
measured separately; more stops or larger motion alone do not count as success.

## Implemented mechanisms

| ID | Intervention | Source of movement |
| --- | --- | --- |
| D1 | Sparse musical-event retiming | Same song's accepted body, monotone time map |
| D2 | Phrase preparation and release | Same body, source/texture/boundary-derived timing |
| D3 | Coordinated source-specific timing | Same body, shared support clock and regional clocks |
| D4 | Recurring motif composition | Explicitly identified same-song body fragments |
| D5 | Fixed-time expressive refinement | Bounded upper-body change; root/support chains preserved |
| D6 | Learned effort/duration planner | Predicted audio plan retimes same-song body |
| D7 | Repeated source conditioning | Warmstarted native139 flow generates new movement |
| D8 | Global phrase plan plus local flow | Predicted whole-song plan conditions native139 generation |

Pure retiming conserves endpoints, source order and the 180-second playback
duration with rates in [0, 1.5]. D4 is recomposition, not pure retiming; source
clock jumps are provenance, never displayed as playback speed. Geometry-aware
resampling applies symmetrically to the newly compiled comparison baseline and
candidates. Historical artifacts remain untouched.

## Reproducibility

`PLAN.json` binds the protected original inputs. Immutable phase snapshots,
command queues, dependency hashes, output manifests and process receipts bind
what actually executes. No completed outputs are overwritten or silently
replayed. Partial runs require an explicit preserved recovery identity.

The four music analyzers are real frozen models: MuQ, pinned HTDemucs,
Beat This and the official All-In-One eight-model ensemble. All-In-One uses an
isolated CPU-compatible build; no loudness heuristic substitutes its labels.
Its labels remain estimates. The failed historical ridge source-presence
threshold is not used. Full waveform counterfactuals recompute both native
Base92 and the new music features and bind them to one analyzed waveform hash.

TRAIN fitting uses 109 canonical FineDance songs (128 recordings); development
uses 31 canonical songs (36 recordings). Short recordings remain mask-aware.
No TEST motion or exposed MUSDB TEST evaluation is opened. Both showcase songs
and development data are explicitly exposed. Training seeds are 0 and 1;
displayed models use seed 0 regardless of scores, generation draw seed 7001.

## Execution and interpretation

`scripts/motion/dynamic_dance_guard.py` is the only GPU campaign launcher.
One sequential curator holds both shared GPU locks. The total cap is 24 hours:
features 6, training 12, evaluation 4, rendering 2. At least 20 GiB free space
is preserved. Elapsed lock-owned GPU stages are conservatively accounted,
including CPU work while the lock remains held; GPU-active time is not claimed.

The eight hypotheses and their fixed metrics/gates were registered before
scientific generation or training. Synthetic unit tests and throughput
preflights are engineering evidence. Direction, magnitude, physical validity,
actual emitted transfer and artistic acceptance remain separate conclusions.
Controller-derived labels cannot independently certify musical meaning.

Finite unsuccessful methods may be rendered as labeled failure diagnostics.
Exact visual duplicates are grouped. No autonomous oscillators, canned light
sweeps, new seeds, threshold relaxation or substitute motion may hide failure.
User artistic review is a separate final step after verified delivery.

## Entrypoints

- `prepare_dynamic_dance_audio.py`: manifest, performance preflight, extraction,
  waveform counterfactuals and complete feature index.
- `train_dynamic_dance.py`: cache preparation, performance preflight, protocol
  freeze, paired training, predicted plans, inference and development evaluation.
- `run_dynamic_dance.py`: original bodies, audio-only intervals, D1–D5,
  frozen mapper source tests and emitted timelines.
- `render_dynamic_dance.py`: smoke, inspected full renders, media audit and gallery.

Always execute stages from their frozen source snapshot using their registered
queue. The workspace is for implementation; it is not an implicit execution pin.

## Completed results

The [synchronized gallery](http://127.0.0.1:8775) contains 16 comparisons: eight
approaches on both songs, each 120 seconds at 1920×1080 and 30 fps with original
audio speed. The campaign used 4.853 GPU-stage hours. No approach achieved a
fully qualified improvement; unsuccessful results remain labeled diagnostics.

Complete video decoding, source-audio verification and browser checks passed.
The pixel audit retains 3 thin-trace compression warnings among 64 sampled
frames; their thresholds and videos remain unchanged. See the
[final report](/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/reports/FINAL_REPORT.md)
and [machine-readable evidence](/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/reports/FINAL_REPORT.json).

To restart the gallery server after it stops, use the frozen delivery source:

```bash
cd /mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/sources/delivery_v2
/root/lightsem/.venv/torch-access/bin/python -m scripts.rendering.serve_body_portfolio \
  --root /mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1/review \
  --port 8775
```
