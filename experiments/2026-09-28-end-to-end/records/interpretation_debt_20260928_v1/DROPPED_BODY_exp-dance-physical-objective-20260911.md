---
artifacts: []
closed: true
config_note: '{"scope": "New seed0 matched native-flow physical-objective ablation;
  no claim that physical quality alone establishes music conditioning. Old runs immutable.
  Both new arms share inclusion of short recordings and valid masks; difference is
  physical objective only.", "arms": ["masked_native", "physical"], "seed": 0, "weights":
  {"position": 0.5, "velocity": 0.05, "planted_foot_velocity": 1.0, "root_displacement":
  0.25, "floor_penetration": 10.0}, "model": "unchanged native MusicConfig(flow,width256,8
  dilated blocks,32Euler inference); no extra audio fusion/dropout/schedule selection",
  "training": {"steps": 30000, "batch": 32, "frames": 960, "lr": 0.0003, "warmup":
  500, "checkpoint_interval": 1000, "runtime": {"precision": "bf16", "optimizer":
  "adamw", "compile": "vector_field_fullgraph_only", "prefetch": 2}, "physical_precision":
  "FP32 eager validated FK", "gradient_clip": 1.0, "sampling": "uniform canonical
  song then recording then crop; all128TRAIN recordings/109songs; short sequences
  zero-padded, valid masks; same paired RNG"}, "prediction": "Physical arm reduces
  canonical-weighted generated-contact foot-sliding excess by>=20% vs matched masked_native
  final model. Record source and generated sliding/coverage separately; mean distribution
  score alone does not pass.", "evaluation": {"main_windows": "exact119 native prior
  frozen VAL windows from motion_native_audio_20260910/music/protocol.json", "draws":
  6, "conditions": ["correct", "donor", "shift"], "same_draws_across_conditions":
  true, "steps": 32, "checkpoints": ["initial", "final"], "aggregation": "equal canonical
  song / recording / window", "bootstrap": {"draws": 10000, "seed": 20260910}, "short_val":
  "supplemental all3 omitted short VAL full recordings; score separately without merging
  into original119window primary; paired donor and contiguous shifts only if valid
  support exists", "test_access": false, "model_selection": "none: report both final
  scheduled arms; no checkpoint/weight/schedule selection"}, "gates": {"sliding":
  "physical excess foot sliding <=.8*matched_native excess, and absolute excess<=.20m/s",
  "conditioning": "physical correct-audio ES improves>=5% versus BOTH donor and within-track
  shift; song-bootstrap95%CI lower>0; direct lighting dynamics disabled", "energy":
  "canonical-weighted generated/reference kinetic ratio between.5 and2 inclusive",
  "travel": "canonical-weighted generated/reference root travel between.25 and2 inclusive;
  full sequence drift separately", "contacts": "generated/reference contact coverage
  ratio between.5 and1.5; evaluate sliding using GENERATED labels, not donor labels",
  "floor": "p95 floor penetration excess<=.05m", "high_frequency": "relative-joint>6Hz
  power fraction excess<=.05", "continuity": "FP32 full/chunk max difference<=5e-5
  on full12fixed review recordings; no repeated root integration", "short_holds":
  "all3shortTRAIN included and masks verified; report source short/hold counts and
  new full shortVAL motion; no hidden padding/repetition", "finite": "all losses,
  gradients, generated draws finite, no discarded samples", "visual": "all12 fixed
  review clips, incumbent and matched new baseline comparisons, plus fullShpongle
  diagnostic; sampled visual review distinct from human music-fit acceptance"}, "seed1":
  "blocked unless entire pipeline independently qualifies", "test_access": false,
  "defaults_unchanged": true, "hardware_output": false}'
cost: null
crash_reason: ''
created_at: '2026-09-10T21:58:04+00:00'
diagnostics_measured:
- after: 0.34312670003332923
  before: 0.8817441494835282
  diagnostic_id: motion-music-generated-sliding-excess
do_not_retry: []
external_run: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01
former_ids: []
gate_results:
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: sliding
  passed: false
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: conditioning
  passed: false
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: energy
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: travel
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: contacts
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: floor
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: high_frequency
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: continuity
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: short_holds
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
  gate_id: finite
  passed: true
- evidence: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review_support_revision_01/review_report.json;
    all13 scheduled renders/sample reviews complete, but matched incumbent lighting
    unavailable; no overall visual/music-fit acceptance established. See outputs/dance_physical_final_outcome.md
    for instrument scope.
  gate_id: visual
  passed: false
git_ref: master@5b4503a+dirty
hypothesis_id: h-dance-physical-objective-20260911
id: exp-dance-physical-objective-20260911
interpretation:
  at: '2026-09-28T23:13:21+00:00'
  failure_mode: recipe_conditional
  lesson: '2026-09-28 retrospective interpretation, not a new experiment: In the matched
    seed0 study, the physical objective reduced generated sliding excess to 0.343127
    m/s versus masked_native at 0.881744 m/s, but still failed the absolute 0.20 m/s
    gate. Correct-versus-donor/shift conditioning failed on the frozen summary-energy-score
    instrument, and visual qualification failed, with matched incumbent 3D lighting
    unavailable despite completed scheduled renders and sampled reviews. Other physical,
    consistency and coverage passes do not establish learned music-driven LIGHT quality
    or human music-fit acceptance, reverse the blocked qualification/seed1 decision,
    or refute physical objectives generally.'
  transfers_to: []
key_numbers: {}
observed_effects:
- delta: -0.53861744945
  diagnostic_id: motion-music-generated-sliding-excess
  direction: down
parent_experiment_id: null
prediction_match:
  overall: true
  per_diagnostic:
    motion-music-generated-sliding-excess: true
provenance: {}
recipe_fingerprint:
  git_ref: master@5b4503a+dirty
verdict: directional_only
---

## Summary
Tests hypothesis [[h-dance-physical-objective-20260911]].
At commit `master@5b4503a+dirty`.
External run: /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01

## Measurements
- [[motion-music-generated-sliding-excess]]: before 0.8817441494835282 → after 0.34312670003332923  (Δ -0.538617, down ✓ predicted down)

## Gates
- `sliding`: **FAIL** — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `conditioning`: **FAIL** — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `energy`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `travel`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `contacts`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `floor`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `high_frequency`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `continuity`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `short_holds`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `finite`: PASS — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation/numerical_report.json
- `visual`: **FAIL** — /mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review_support_revision_01/review_report.json; all13 scheduled renders/sample reviews complete, but matched incumbent lighting unavailable; no overall visual/music-fit acceptance established. See outputs/dance_physical_final_outcome.md for instrument scope.

**Verdict:** directional_only

> Direction matched, but a preregistered blocking gate did NOT pass. This does not establish the claim.

## Interpretation
**Failure mode:** `recipe_conditional`

2026-09-28 retrospective interpretation, not a new experiment: In the matched seed0 study, the physical objective reduced generated sliding excess to 0.343127 m/s versus masked_native at 0.881744 m/s, but still failed the absolute 0.20 m/s gate. Correct-versus-donor/shift conditioning failed on the frozen summary-energy-score instrument, and visual qualification failed, with matched incumbent 3D lighting unavailable despite completed scheduled renders and sampled reviews. Other physical, consistency and coverage passes do not establish learned music-driven LIGHT quality or human music-fit acceptance, reverse the blocked qualification/seed1 decision, or refute physical objectives generally.

## Notes
# Physical-objective study: not a dancing-pipeline fix

The matched seed0, 30000-update intervention improves measured motion physics but fails qualification. No checkpoint selection, default promotion, TEST, seed1 or hardware output.

#audio-to-light #music-to-dance #motion-to-light #generative-modelling #real-time #evaluation-metric #cross-modal #pose #music-induced-movement #conditional-mean-collapse #metric-gaming #ill-posed-ground-truth #matched-baseline
