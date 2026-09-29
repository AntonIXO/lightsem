# Recovery and driver readiness — complete, 2026-09-29

This resolves the narrow handoff task: unfinished driver/fixture readiness and
the pending two-record memory repair. It does not complete the wider scientific
campaign or claim archive feasibility, model quality or GPU admission.

## Current authorization

The user explicitly corrected the inherited restriction: changes to memory do
not require separate permission, and restoration should proceed when needed.
The earlier approval-blocked reports are historical. This preference is also
recorded in `/root/lightsem/CLAUDE.md` and the native readiness runbook.

## Restored records

The isolated, allowlisted `restore_experiment_notes_snapshot` MCP tool was
called through the unchanged `log_to_vault.Client`. It applied the existing
exact candidates for:

- `exp-dance-physical-objective-20260911`: restored 3,907 bytes of Notes;
- `exp-dance-optics-full-20260911`: restored 2,714 bytes of Notes.

Both resulting SHA256 values match `RESTORATION_CANDIDATES_V1.json`. No
experiment was reclosed and no scientific state was recomputed. An independent
readback implementation verified all 20 original bodies outside Interpretation,
all current frontmatter bytes including timestamps, all original metadata
except the already-approved interpretation additions, and all 20 native lesson
payloads. The 406 other LightSem Markdown files and all 756 files in the foreign
vault were unchanged. The installed server and Client source were unchanged.
The live LightSem writer was briefly suspended during the atomic replacements
and its resumption was verified, with a detached recovery watchdog in place.

Authoritative repair evidence:
`repair_v1/REPAIR_AUDIT_V1.json`, `repair_v1/MCP_REPAIR_RECEIPT_V1.json`, and
`repair_v1/REPAIR_BEFORE_V1.json.gz`.

Eleven recovery checks passed on disposable copies: exact restoration,
idempotent no-write behavior, all-target preflight, metadata/interpretation and
boundary rejection, symlink/hardlink rejection, failed replacement preservation,
late drift rejection and MCP dispatch isolation. The first fixture attempt
failed before any vault write because the diff boundary shared `#` with the
footer. Its source and log are retained; the corrected second attempt passed.

## Driver readiness

The runtime-field and fixture changes were already complete. The frozen driver,
test source, templates and handoff matched the source review; no change to
those files was needed. The 67 native synthetic tests now passed with zero
failures, errors, skips, expected failures or unexpected successes.

The real pre-execution proof verified 512MiB memory, no swap, PrivateDevices,
hidden CUDA, a 50s process-tree limit and exact command/source binding. The
whole job took 0.361432273s; combined output was 12,910 bytes, below 32KiB.
Runtime: Python 3.12.13, NumPy 2.5.2, zlib 1.3.1. The cached `_lzma` path and
linked liblzma version were both unavailable/null; no binary/environment
identity or future compressed-size invariance is claimed. A changed runtime
requires fresh validation.

Authoritative test evidence under
`/root/lightsem/outputs/legacy_struct_intensity_20260928_v1/`:
`NATIVE_LOSSLESS_ARCHIVE_TEST_RESULT_V1.json`,
`NATIVE_LOSSLESS_ARCHIVE_TEST_ENVELOPE_V1.json`,
`NATIVE_LOSSLESS_ARCHIVE_TEST_ADMISSION_V1.json`, and
`native_lossless_archive_tests_v1.log`.

Original campaign and evaluation-workspace stat inventories were unchanged.
Fresh conservative global commitments were 1,001,013,996 bytes versus the
1,020,054,732-byte cap, retaining the full row8/comparison allocations and other
metadata reservations. Both filesystem free-space floors passed.

## Remaining separate work and known limitation

The installed generic experiment writer still loses custom Notes headings.
This repair restores the records; it does not patch that installed writer.
Avoid that lossy route for later writes to these records and verify complete
note preservation. The task's narrow recovery tool cannot serve as a general
editor for changed records.

The full40/two-codec feasibility measurement remains separately queued in
`rb-ls2-lossless-format-readiness-20260929`. A 67-test pass does not establish its
compressed size, runtime, full manifest fit or scientific admission. Existing
format-measurement authorization/envelope templates remain templates. No model,
GPU job, scientific payload evaluation or product behavior change occurred.
