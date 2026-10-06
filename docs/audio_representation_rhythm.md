# Optional representation rhythm input and lit dance rendering

The new `lightsem.audio.representation_rhythm` interface applies authenticated frozen rhythm readouts to the existing native dancer's five timing inputs. It preserves the other 87 acoustic channels and creates no substitute beat clock. The default P2/Rich96 color policy and existing dancer weights are unchanged.

```python
from lightsem.audio.representation_rhythm import load_ensemble, score_ensemble, prepare_native_input

heads = load_ensemble(report_path, method, expected_sha256=report_sha256)
readout = score_ensemble(features40, times40, heads)
inputs = prepare_native_input(base92_40, times40, times30, train_normalizers,
                              beats=readout["events"]["beats"])
# inputs["normalized92"] enters the existing frozen native-flow generator.
# inputs["coverage"] marks support from real consecutive beat detections.
```

These are threshold-relative scores from five GuitarSet cross-validation heads, not calibrated probabilities or a validated full-mix beat detector. An empty beat array explicitly gives neutral timing; `beats=None` preserves the original baseline. The complete embedding sequence does not enter the dancer: this is a rhythm-mediated integration.

`lightsem.sim.lit_body_renderer.LitBodyStageRenderer` accepts the actual source `(22,3)` pose alongside decoded fixture directions, RGB and dimmers. It renders opaque body geometry in the light scene. A fixed, declared neutral fill keeps the source readable between beam crossings; its diagnostic direct-light mask excludes that fill. Cast shadows and interreflection are outside this renderer.

The matched nine-approach, two-song campaign runs from `/root/lightsem-audio-integration` using `run_integration.sh`. Source snapshots, immutable commands, receipts and videos are in `/mnt/lightsem-data/experiments/lightsem/audio_pipeline_integration_20261005`. The frozen campaign launcher intentionally refuses changed imports and existing unverified outputs. It should not be replaced by the live checkout's hard-coded legacy launcher.

Both modules have been added without replacing ongoing live pipeline fixes. The isolated campaign also contains the reusable feature, body-generation, mapping and rendering harnesses. Reports distinguish input propagation, mechanical loss, source visibility and artistic quality; differing hashes are not choreography evidence.
