# Readiness and proposed two-record repair — 2026-09-29

Status: source readiness confirmed; repair prepared and verified, not applied.
The goal is incomplete. No project/server imports, project tests, codec
measurements, model execution or vault writes ran during this review.
Verification used source/control text, saved snapshot data and live Markdown.

## Driver and fixtures

The handoff's unfinished-fixture status is stale. The current driver, tests,
authorization template, envelope template and both implementation handoffs
match every size and SHA256 in `STATIC_LOSSLESS_ARCHIVE_REVIEW_V1.json`.
The test source contains exactly 67 unique authored methods, numbered 01–67.
That count is not a native test result.

The runtime record contains Python/NumPy/zlib versions and the cached `_lzma`
path; `linked_liblzma_version` is null. No binary identity or future compressed
size guarantee follows. Runtime-version equality against the prospective test
attestation remains intentional; binary/reference fields and runtime equality
against the external envelope are absent. Runtime changes require validation.
The separate coverage clarification correctly limits the float payload tests
to finite patterns. No source or fixture edit was needed.

## Exact proposed repair

The live files still match the expected damaged-file hashes in
`RESTORATION_CANDIDATES_V1.json`. Both candidates match their saved hashes and
sizes. Each differs from its current record by one insertion in the Notes
tail. The whole frontmatter, preceding Interpretation and remaining body bytes
are identical. Removing only the Interpretation section from the candidate
reproduces the original snapshot body under its declared whitespace convention.

| Experiment | Restored bytes | Candidate SHA256 |
| --- | ---: | --- |
| exp-dance-physical-objective-20260911 | 3,907 | 588434c50b6fd566ad7df8a8efaf94b10e04af1b93dcbbc56f52664bf49169d1 |
| exp-dance-optics-full-20260911 | 2,714 | c594246f09808412167e4c38d2363cc8749fff20d07eaac981c5d7a0cbbb191f |

All 388 protected Markdown hashes match the before snapshot. Of the 20
interpretation targets, 18 retain the original body outside Interpretation;
only the two listed records still lack sections. This review does not claim
a new independent semantic audit of all 20 interpretations.

The installed MCP has no notes-only restoration tool. Source inspection
confirms that `import_run_report` changes provenance and uses the same lossy
writer, while `close_experiment` recomputes scientific state and writes the
hypothesis. Neither is a suitable repair path.

The reviewable proposal remains the narrow Client-compatible repair in
`NATIVE_NOTES_PRESERVATION_DIAGNOSIS_V1.json`: add and validate an isolated,
two-target, hash-bound MCP adapter, then apply these exact candidate bytes and
audit preservation. Keep the installed server and Client unchanged. Preserve
current metadata and timestamps; do not reclose experiments. Fresh hash checks,
exclusive access and resource accounting must precede actual writes. This
one-time restoration does not fix the installed writer's preservation defect.

## Approval and remaining validation

The supplied handoff explicitly says: “Vault changes require explicit approval.”
Approval for this repair is still pending. No adapter, fixture run or repair
was performed. After approval and successful repair validation, the 67 native
tests and two CPU codec measurements still need their separate recorded
admission, fresh accounting and external limits from the existing handoff.
No scientific or GPU permission follows from this review.

This additive report is charged to the existing 524,288-byte interpretation
control reservation. That directory held 324,366 regular-file bytes before
this report; no existing artifact or reservation was removed or rewritten.
