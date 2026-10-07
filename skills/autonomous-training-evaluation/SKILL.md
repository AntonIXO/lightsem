---
name: autonomous-training-evaluation
description: Evaluate completed model-training runs against a frozen baseline with deterministic provenance, collapse and input-sensitivity checks, paired quality gates, and machine-readable reports; use when deciding whether a training improved, regressed, or collapsed without relying on human review.
---

# Autonomous Training Evaluation

Use this skill for post-training evaluation of LightSem native models and related audio-to-motion/light experiments. It is a read-only audit: it must not retrain, select a replacement checkpoint, overwrite a sealed receipt, or turn a diagnostic into a product promotion.

Read [references/protocol.md](references/protocol.md) before constructing a run manifest, and read [references/metrics.md](references/metrics.md) when interpreting a metric or gate. Read [references/repo-pitfalls.md](references/repo-pitfalls.md) for native-C, P2, two-song, cache, and chart rules.

Run the CPU audit through the repository wrapper:

```bash
bash run.sh scripts/evaluation/audit_training.py \
  --manifest /path/to/audit_manifest.json \
  --output /path/to/audit_report.json
```

The manifest points to raw `.npz` arrays and a provenance receipt for one run. Generate a matched baseline report separately, then pass `--compare-baseline baseline_report.json` to obtain paired gate deltas. Keep weights, non-audio inputs, random noise, mapping, renderer settings, and song-specific source bodies explicit. The runner computes finite checks, raw 3D speed/acceleration/jerk, root range, representation diversity, output inactivity, integer-stream saturation when a range is known, exact duplicate streams, RGB and brightness separately, input sensitivity, intervention response, and provenance hashes. Its optional `--plot` is an inspection aid; comparison plots must use pair-scoped axes and a signed candidate-minus-reference panel while retaining every raw peak.

Interpret the machine verdict conservatively:

- `qualified_improvement` is the comparison `qualified_status` when complete paired gates show a candidate gate gain and no candidate safety loss; it is an engineering health result, not artistic acceptance.
- `qualified_regression` is the comparison `qualified_status` when the candidate fails a gate that the baseline passed.
- `collapsed` is reserved for representation/sample collapse or exact/near-duplicate output mechanisms; zero event count alone is not collapse.
- `failed_safety` covers nonfinite values, invalid ranges, raw-motion instability, or invalid stream data.
- `unqualified` means evidence is missing, coverage is insufficient, or musical/choreographic claims are not measured.
- `diagnostic_only` is for explicitly requested failure controls or historical/unqualified branches.

Missing evidence is reported as missing and makes the result unqualified; never replace it with zero or silently lower a threshold. Keep motion, color, brightness, and emitted DMX conclusions separate. A passing downstream renderer or post-hoc smoother cannot repair a failed native generator. The report is evidence for a decision, not a claim of artistic acceptance.
