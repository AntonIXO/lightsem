# Claim-scope repair: independent integrity review V1

**Result: verified, with the disclosed five-field schema exception.**

This is a read-only record review. It is not a new measurement or product acceptance.

## Independent checks

- All 14 current experiment interpretations match the approved payloads and native receipts: lesson, failure mode and transfer fields.
- All pre-existing non-interpretation fields are unchanged. This includes measurements, predictions, gates, verdicts and status fields.
- Historical Notes and all body text outside Interpretation are preserved. Only outer blank-line framing is normalized for the body comparison.
- All 128 untargeted document hashes match the prior audit: 71 hypotheses and 57 experiments.
- Definition snapshots match the prior audit hashes. All current targeted hashes match the parent's integrity report.
- No unexpected mutation or new scope overclaim was found in the applied lessons.

## Do not call this strict metadata equality

Legacy row 6 received exactly these native serializer defaults:

```json
{"cost": null, "do_not_retry": [], "former_ids": [], "key_numbers": {}, "recipe_fingerprint": {}}
```

Strict non-interpretation equality is therefore **13/14**, not 14/14. Every old field is preserved in all 14. `CLAIM_SCOPE_METADATA_NORMALIZATION_V2.json` correctly includes the previously absent `cost` key. V1 missed that addition and remains retained as history.

The authorized Interpretation mode changes are also explicit in the JSON. For example, row 9's `test_invalid` limits broad qualification without erasing its one-rater pilot. Row 56 changes `mechanism_wrong` to `recipe_conditional`.

## Limits

- Immutable machine labels can still conflict with corrected scientific scope. This is not a failed repair.
- Historical unsupported Notes remain visible; the dated Interpretation supersedes them. Do not erase the history.
- The foreign-vault integrity assertion was not independently rehashed. This review accessed only the allowed LightSem record set and named reports.
- Root problem-note updates are outside this experiment/hypothesis comparison.
- No vault calls, raw-data/model access, scientific runs, or existing-file edits were made.

The JSON contains all 14 record comparisons, all 128 untargeted hash checks, source hashes and the exact exception. No integrity repair or rerun is needed.
