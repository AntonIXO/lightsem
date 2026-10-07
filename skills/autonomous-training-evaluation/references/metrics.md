# Metrics and interpretation

The stable API is available from
`scripts.evaluation.training_audit` and `lightsem.eval.training_health`.

`motion_metrics(body, fps=30)` computes finite-frame coverage and raw,
unfiltered speed, acceleration, jerk, frame-level activity, position range,
root range, and a jerk-to-speed diagnostic. It does not certify support or
contact; supply a separate support mask/metric when one exists. Filtered 2D
profiles cannot certify raw native 3D quality. The CLI can enforce predeclared
p95 speed/acceleration/jerk and activity limits; without explicit limits these
quantities remain measurements rather than pass/fail thresholds.

`stream_metrics(streams, rgb_channels=..., brightness_channels=...)` hashes each
output stream, finds exact duplicate groups, measures activity and changed
fractions, and reports explicit RGB and brightness summaries. Channel semantics
are never guessed. In the generic 2-D form the array is interpreted as
`time × output`; use `time × output × channel` for multi-channel outputs.

`collapse_metrics(samples)` measures pairwise distance, exact/near duplicates,
and effective sample rank. It diagnoses sample/representation diversity only;
quiet output, no events, saturation, and failed input response are separate
mechanisms.

`input_sensitivity(reference, interventions)` compares a fixed reference with
declared audio, limb, noise, or other interventions. It reports shape/finite
coverage, RMS response, changed fraction, and exact equality. A response proves
dependency, not choreography or musical fit. Mark controls that are expected to
remain fixed explicitly and do not count them as required responsive inputs.

`provenance_report(receipt, required_fields)` checks required metadata and any
`*_path`/`*_sha256` pairs. `complete=false` or `status=unqualified` blocks
promotion; it is never converted into a zero score.

`compare_training_runs(baseline, candidate)` (also exported as
`compare_bundle`) compares explicit gate booleans. It reports candidate gate
losses and gains, but it cannot invent a quality metric or an artistic verdict.

The optional CLI plot is an inspection aid. Use a pair-scoped y-axis and a
signed candidate-minus-reference panel for comparisons; disclose every raw peak
and time/scale scope. Never let a broken outlier set the scale for a passing
reference trace, and never clip or smooth away a defect.
