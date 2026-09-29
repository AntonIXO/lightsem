# EXPERIMENTAL Shpongle learned-light overlay

Demonstration only. The historicalDEV observer study remains unqualified.
No claim of accuracy, LIGHT quality, generalization, preference, fine beat timing,
chunk equivalence, or calibrated photometry. No inference retuning after viewing.

## Renderer handoff

- Data: `/mnt/lightsem-data/experiments/lightsem/shpongle_layers_demo_20260922T161917Z/overlay_dmx.npz`
- `baseline`: exact saved accepted `companion` stream.
- `overlay`: learned-layer candidate.
- Both are uint8 `[11092,512]` on the canonical rig.
- `time`: float64 `arange(11092)/40`. Support `[0,277.3)` seconds.
- Audio: `/root/shpongle.flac`. No audio or dancer retiming.
- This worker did not render, retrain, or generate new motion.

## Fixed inference and mapping

- Fixed seed0 `S_VR_P0/best.pt`, primary selected role; not selected across seeds.
  Checkpoint SHA256: `1396e8177c8a936d8c6de41314fbf2419c6ddc0fd616ba98c2aebd7e49309fd7`.
- One full-song float32 forward, generated palette history only. No chunks.
- Frozen TRAIN normalizer and HS72 prototypes. No fitting on Shpongle.
- Exact original `SourceStore.audio`: source bus64mel+28scalars and cached
  BeatThis probabilities/interpolation. The first92 coordinates equal the
  authenticated original audio92 cache byte-for-byte.
- Historical source timestamps were float32. Both stored and nominal clocks
  are preserved in `audio94.npz`; maximum quantization error is12.2 microseconds.
- Palette generation stays unchanged: null initial state, own mark argmax,
  own strict probability>0.5 cues and frozen0.5s NMS. Internal start-state
  initialization is not a displayed cue. Baseline colors last until first
  actual cue36.3s. Generated token colors are held between cues.
- HS72 hue-major H=k/12 and S=j/5. Fixed visualization:
  `z=sum(p*S*exp(2*pi*i*H))`, hue=`arg(z)/(2*pi) mod1`, saturation=`abs(z)`, V=1.
  Exact zero z gives H0/S0. Python colorsys HSV→RGB; overlay byte quantization.
  One global color for MH/wash/projector RGB. This is not learned fixture
  assignment or calibrated photometry. No gamma or dimmer compensation.
- Accent: frozen evaluation interior peak/plateau extraction and0.2s score-NMS,
  then probability>=0.5. Fixed+0.35 MH/wash gain, raised-cosine full width0.25s.
  Overlap uses max. All baseline brightness roles remain, subject only to
  this bounded gain and byte saturation. Zero dimmers stay zero.
- Output:52 generated color cues; one accent at263.7s. No extra cues invented.
- Every protected byte is exact:5,202,148 comparisons over469 slots, including
  all pan/tilt, strobe/blinder bytes, and projector dimmers.

## Evidence

In the data directory:
- `model_outputs.npz`: all full logits, scores, palette probabilities/context,
  generated/previous token states, cue mask, final state, native40/10Hz clocks.
- `audio94.npz`: raw and frozen-normalized input plus both clock representations.
- `events.json`, `protected_byte_equality.json`, `provenance_manifest.json`.
- `readback_receipt.json`: independent saved-output RGB/accent/protected-byte checks.

Here:
- `inference_mapping_plan.json`: fixed pre-inference mapping and source bindings.
- `infer_overlay.py`: new helper only. Native command:
  `TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/infer_overlay.py`
- Native preflight and full logs. The first preflight failed only on an
  overstrict stored-clock equality assertion; the fix is logged in
  `preflight_clock_fix.json`. No model forward occurred before that fix.

Data produced by inference: approximately9.14MiB, below64MiB cap.
Frozen model/training/evaluation/overlay files and old artifacts were not edited.
No vault writes or commits.
