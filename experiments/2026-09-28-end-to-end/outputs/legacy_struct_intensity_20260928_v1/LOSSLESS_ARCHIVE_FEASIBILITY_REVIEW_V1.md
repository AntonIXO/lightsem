# Lossless archive feasibility — source/control review

**A testable candidate, not a proven fit. No prototype or model work is authorized.**

## What the saved metadata determines

All40 rows retain371393 frames. The613 saved, ordered, nonoverlapping section intervals have `intersection_frames=end-start` and cover371390 frames. The other3 frames are one-frame invalid tails. Thus the intended integer outputs are fully specified:

- Local `int32` frame index0..T-1, **not i0+i**.
- Target `uint8` role from the saved section label;255 outside valid spans.
- Original `uint16` section ID, without renumbering;65535 outside valid spans.

These arrays must still be **stored**, with each per-clip shape(T,). Metadata determinacy is not permission to derive them away or claim an independent cache reread. All40 rows, including7 support-ineligible cases, remain. Four unknown float32 brightness vectors per clip require5942288 raw bytes. No value/compressibility assumption is allowed.

All613 sections for all4arms need2452 table records, including the short rejected section. Preserve every identity/count/reason field and float64 sum/mean computed later from the actual saved float32 values.

## Peak-safe layouts

Prefer160 write-once uncompressed NPY float files, one shared standard NPZ for120 DEFLATED integer arrays, and stored table/manifest artifacts. This supports reading actual saved floats for statistics without a second disk copy.

Alternatively, use one prospectively specified single-pass NPZ with160 STORED float members and120 DEFLATED integer members. Its saved-byte statistics/readback sequence must avoid rewriting a published archive or duplicating raw floats.

Raw floats followed by another archived float copy already need16023475bytes including retainedP, versus12582912 cap—before integers or other reserves. That layout is not allowed. Failed partials remain counted; no cleanup or destructive consolidation is assumed.

## Explicit candidate envelope

Before these reports:

| Item | Bytes |
|---|---:|
| Retained packet |4138899|
| Unknown float32 payload, uncompressed |5942288|
| New future code/control/metrics reserve, minimum |1048576|
| Separate GPU-ledger control charge inside packet |65536|
| Proposed NPY/ZIP header allowance |131072|
| Proposed stored table/manifest allowance |524288|
| Maximum compressed integer NPY streams |732253|
| Total cap |12582912|

The compressed integer total includes its compressed NPY headers. This report's bytes and other unreserved growth reduce that732253-byte limit; the JSON records the postpublication limit. The integer raw payload is2599751bytes. **No achieved compression ratio is asserted.**

A prospective fixed-width table can retain all listed fields in128bytes/row, or313856bytes. Three already-known metadata collections add128423 compact JSON bytes. Their442279-byte subtotal leaves82009 within the proposed512KiB table/manifest allowance. The final schema/manifest still needs exact measurement; this subtotal is not whole-archive proof.

## Required next gate—not executed here

1. Bind a new prospective lossless-format amendment, preserving V6 and all fields, dtypes, shapes, bits, roles and all40 rows. Pin the codec and exact schema.
2. Run a separately authorized CPU-only prototype using bounded in-memory BytesIO/high-water counting. Generate integers only from saved metadata. Never create a large dummy disk archive.
3. Count floats as STORED raw bytes plus headers. Generated float placeholders can test bytewise roundtrip, including signed zero, but their compressibility earns no discount. Unknown float64 section statistics also get no discount.
4. Measure complete serialization, safe readback and peak footprint. Enforce exact member names/counts, fixed nonobject dtypes/shapes, bounded headers and expanded bytes, CRC/EOF checks and `allow_pickle=False`. Do not weaken the existing input-cache loader to accept a different output schema.
5. Include immutable prior artifacts, all future code/controls/metrics,64KiB GPU-ledger charge, tables, headers and partial/temp writes. Preserve only small prototype receipts/logs.
6. If any byte, safety, field-equivalence or peak bound fails, **stop**. No cohort reduction, quantization, field omission, cap reset, relocation or deletion.

No NumPy/project import, integer-vector construction, compression, native command, scientific payload read or source implementation occurred. Only these review files were written. Compression feasibility and later scientific/GPU admission remain open.
