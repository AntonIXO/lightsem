# Parent pre-freeze engineering decisions

Registered plan is UNCHANGED. No real candidate execution or freeze is authorized yet.

Independent review found blockers: media failure currently prevents terminal report; map exceptions lack structured failure; storage has only ~3.07GiB free with no reserve/guard.

Before freeze:
- Record create-only structured operation failures (phase/item, exception, completed artifacts/gates, failed and unrun states separately). Preserve partial files and original exceptions. Do not turn missing/failed stages into scientific passes or inferred collapse.
- Terminal report must work for failed or incomplete media/map stages. Distinguish measured failed gates, technical failure, blocked/unrun coverage, and unknown values. Preserve completed numeric gates. No success-only verification prerequisite for negative reporting.
- Add a documented storage preflight and run-time reserve guard for streaming video/review outputs. Do not claim a CRF size estimate is a hard bound. Stop cleanly before exhausting disk; retain partial evidence and mark resource-blocked/incomplete, not a scientific refutation. No raw RGB cache, silent codec/quality changes, or evidence deletion.
- Synthetic tests must cover these failures and unchanged successful reporting. Send final file hashes and reports to independent reviewer for fresh validation.

These are failure-reporting/resource safety fixes, not changes to the registered numerical thresholds, bodies, source selection, optics or early-stop science.

## Explicit storage policy (parent)

- Leave at least **1 GiB free** on the output filesystem.
- Use **1.75 GiB total study-artifact bytes** as a stop threshold (including frozen sources, maps, partial/silent/muxed videos, PNGs and reports). This is a conservative operating budget, not a predicted size or scientific gate.
- Before each output stage, report current free space, current study bytes, remaining budget and a documented output estimate. Account for retained silent plus muxed movies and decoded sample/sheet files. Estimates are advisory: CRF compression is not a guaranteed byte bound.
- Check actual free space and study growth during streaming encoding, muxing, decoding and image/sheet creation, not only before launch. Allow for the next write and queued encoder data. Stop gracefully at reserve/budget risk; preserve partial files and an explicit resource-blocked receipt. State guard cadence and that concurrent external disk use is not controlled.
- If the planned workload cannot fit, stop and ask the parent. Do not lower quality/coverage, relocate registered outputs, delete evidence, or turn this into a scientific refutation.
- No preallocated full raw RGB cache. A safe bookkeeping margin for failure receipts must remain within the reserve.

Parent will recheck free space before launch. This policy is pre-freeze infrastructure only; the registered scientific plan remains byte-identical.

## Video planning-rate decision

Use **2 MiB/s as the combined advisory allowance for retained silent and muxed video files**, not 16 MiB/s. Report both retained-file components explicitly. This is a planning assumption only, not an encoder bitrate, measured size or guaranteed bound. Keep the exact 1GiB reserve and 1.75GiB actual artifact stop threshold.

Compute and report the metadata-only full-Shpongle and diagnostic-clip estimates, including samples/sheets. If a route still exceeds the preflight budget, report that explicitly and ask the parent; do not silently alter thresholds, codec, coverage or the registered plan. Do not describe 2MiB/s as exactly16Mbit/s (it is about16.78Mbit/s).

## Scheduling decision from metadata-only estimates

Use renderer **--budget 1** only. Do not request the original and dimmed full movies as one estimated batch. The reported full-entry estimate is1,536,058,266 bytes; it fits only when the actual current study bytes and filesystem free-space checks pass. Re-evaluate those checks before each subsequent entry, using actual retained bytes. No claim is made that all planned outputs will fit. If a later entry is resource-blocked, preserve it and ask the parent; do not change thresholds or coverage silently.

## Parent test/review checkpoint

Parent independently ran the39-test CPU suite against eval53c596ae4ca075cd361c55bd744512bb7b37a41de5c93158aef95ba7929ab212 / renderer3af281683b8a4b56f1a5b1da2e3962f2c2ee0f885719264893e3e962159ea46d. This test pass is NOT freeze clearance: independent review found a missing REAL-evaluator-lock low-space failure path in record_storage_block. A normal started.json write can fail before bookkeeping failed.json. Builder must fix and add the cross-module low-storage regression. Earlier FakeEval tests did not measure this path. Recheck final hashes and tests before freeze.
