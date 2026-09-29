# Independent palette-secondary audit V1

**Result: no hard arithmetic or saved-support issue found. Both role-specific benefit conjunctions fail. No promotion.**

Audit time: 2026-09-26T15:38:11.867519+00:00. This is a read-only audit, not a new experiment.

## What was checked

- Raw report SHA256: `d25c8edbf229869fd8b3b8a7e0428bba09326703216d5583c1554efafc0336f6`.
- Exact inventory:186 role/seed/record cells,744 paired draw rows,2976 four-arm scores,186 self/quantization floor pairs;101 original outputs plus85 continuation outputs. All91 TRAIN calibration records and positive frozen IQRs match.
- 62,486 arithmetic/metadata checks passed. Maximum floating-point difference was3.55e-15. The audit independently reconstructed component scaling, equal four-component means, undefined propagation, hierarchical song aggregation, both contrast inputs, Holm-2 correction, diversity ratios, support counts and gates.
- Saved prediction support audit passed186 cells and154,122 checks. Original NPZ hashes, masks, exact cues, common uniforms, cue-held outputs, persistence, lag/adjacency support and emitted-candidate counts agree.
- No inference, calibration, training, raw source inputs, AP/W1 scoring, bootstrap reruns, vault writes, or frozen source edits occurred. CI inputs and saved metadata were checked; bootstrap endpoints were not regenerated.

## Scientific interpretation

| Role | Own error | Trained-null error | Limited donor error | Persistence error |
|---|---:|---:|---:|---:|
| Selected primary |0.859022|0.830589|0.850720|1.119751|
| Final companion |0.750180|0.755757|0.754158|0.903743|

Lower error is better. Selected null-minus-own is−0.028433: own history is3.42% worse than null. Both selected97.5% benefit intervals are entirely negative. The Holm-2 significance is in the wrong direction. Final null-minus-own is only+0.005577 (0.738%), below0.02 absolute and15% relative; both intervals includezero. Beating persistence does not rescue either failure. Do not choose the final checkpoint after seeing these results.

Each role has30/31 common palette groups and30 timing-supported groups. AP uses17 event-bearing groups; false-cues/min uses all30, including13 zero-event groups. `8UTvmHbgc5w` retains92.656% full-window valid time but haszero complete8s pairs. The undefined lag component and aggregate are retained, not silently dropped.

Self comparison iszero where defined. Quantization temporal floors are0.465517 selected and0.538344 final; marginal floors are0.725234 and0.928224. Floors are separate, not subtracted. Observed residence runs remain gap/edge-censored lower bounds.

Selected own occupancy18.864 exceeds null16.931 and adapted reference4.833. Effective diversity13.475 exceeds11.546 and3.331. The0.70 lower-only guards prevent collapse; passing them does not establish realistic diversity. Identical intercue drift error in all arms reflects zero generated drift between held cues, not a history benefit. The saved component table and raw diversity summaries are in the audit JSON.

Cue inventories and adapted references differ by checkpoint role. Absolute selected-versus-final errors are not a controlled checkpoint-selection comparison. Emitted AP is not full-candidate head AP, beat accuracy or a quality measure. Passing its descriptive preservation check is not timing improvement.

## Scope and provenance

Original31/40 source coverage failure remains. The completed event-study29/30 paired-null failure also remains, but is not a palette prerequisite. The original shuffled-palette control is missing; limited tempo-donor input does not replace it. Historical DEV exposure, unknown observer/pretraining nuisances, unannotated cuts, full64s edge context and fixed stochastic draws limit inference. This does not establish LIGHT quality, unseen confirmation, human preference or improvement of the argmax demo.

**Retrospective V1 vault-order/execution deviation remains explicit.** Source/test and calibration bindings were locally frozen but unsuccessfully updated in the vault before the first101 outputs. The subsequent correction and authorized85-cell continuation do not establish fully prospective compliance. Six exact final-role head exceptions and101/85 producer lineage remain documented. No blanket replay tolerance change or palette GPU generation occurred.

Original palette qualification remainsundefined. Promotion remainsfalse. Do not auto-close the broad original hypothesis from this limited observer result.

## Parent-summary review

The parent MD and compact JSON agree with the inspected numerical results and scientific caveats. One minor prose correction was sent to the parent: distinguish17 AP groups from30 false-cue-rate groups. No numerical change is needed. The hashes in the audit JSON identify the draft reviewed; a later corrected summary will have a different hash.

## Artifacts

- Audit JSON: `/root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/PALETTE_SECONDARY_INDEPENDENT_AUDIT_V1.json`
- Saved support audit: `/root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/PALETTE_SAVED_SUPPORT_AUDIT_V1.json`
- Saved support audit source: `/root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/palette_secondary_v1/audit_saved_palette_support_v1.py`
- Raw report: `/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1/palette_secondary_report_v1.json`
