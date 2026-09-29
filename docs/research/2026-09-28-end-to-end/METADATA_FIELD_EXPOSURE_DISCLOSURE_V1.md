# Metadata field exposure disclosure V1

This audit used already-recorded transcripts only. No raw FineDance labels were reopened.

## What happened

- `new-frozen-legacy-eval` read and parsed the **entire** FineDance `data/split.json` at **2026-09-28 18:48:43.037 UTC**. The tool result arrived at18:48:43.063.
- It printed `entries[0]` at **18:48:52.300 UTC**. This was **key001, TRAIN**, canonical ID `fd-7a3e6b16cb75f48f`. The printed label object contained `frames`, `name`, `style1`, and `style2`.
- Full parsing included embedded label objects for **all partitions, including TEST**. Therefore a global assertion of zero TEST-label-metadata access is not valid. The record does not show TEST style values printed or used for selection or scientific metrics.
- The immediate parent, `legacy-remainder-readiness`, had also parsed the full split at **18:30:57.069**, with a prefix display at18:30:57.554 containing the same001 TRAIN label object.
- It later parsed the full FineDance manifest at **18:49:48.509**, displaying a prefix at18:49:48.548 with001 TRAIN labels and the beginning of stored TRAIN metric summaries. The partition is established from the already-recorded split, not from the truncated manifest print.

## Notification failure

The child sent disclosure message `agentmsg_1a9f472d-2c03-4c73-92a3-99674037a5ff` at18:54:52.204; it reached **its immediate parent** at18:54:53.848. That was not the root parent. I did not explicitly relay it. The root first found the disclosure in the readiness JSON. The statement “Parent was notified” was therefore ambiguous and misleading for the root.

The explicit root correction was sent as `agentmsg_b518614f-dd20-4282-bc09-3a8d588ee284` at19:09:38.443 UTC.

## Scope and effect

Parsing, printing, and using fields are different events. All-partition metadata was parsed. Confirmed style displays concern TRAIN001. The inspected recorded code used identity/fold fields, and later the explicitly permitted song-name identity field. It did not use style values to choose cohorts or compute new scientific metrics.

No numerical array, model forward, training, or new metric is established by these events. Do not broaden this into a claim of all TEST numeric exposure or void scientific results without evidence. Do not claim global zero TEST metadata-label access either.

Future identity lookups are frozen to `NEW_IDENTITY_EXCLUSION_METADATA_V1.json`: key, canonical ID, partition, and song identity name only. The name-only change does not undo the historical exposure. The JSON companion contains exact transcript paths, record IDs, times and field scope. No style values are re-exported in this disclosure.
