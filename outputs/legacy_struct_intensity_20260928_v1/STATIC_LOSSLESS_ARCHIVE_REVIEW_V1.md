# Static lossless archive review V1

**SCOPED PASS — source only. No execution is authorized.**

No blocking source defect was identified under the explicit root-enforced
contract. Driver/test/template and V6 control hashes match their bindings.
The JSON lists exact source/test line ranges and scope limits.

## Supported by source inspection

- Both fixed codecs independently process all120 integer arrays/all40 clips.
  Local indices, original IDs, sentinel gaps and rejected sections are retained.
- Strict bounded stream/header/schema/CRC/EOF checks precede actual whole-NPZ
  NumPy loading from BytesIO; every member is checked with allow_pickle=False.
  LZMA properties and its8MiB dictionary are bounded before decoder allocation.
- All160 unknown float32 arrays retain5942288 uncompressed payload bytes.
  Complete metadata and all2452 section rows must fit512KiB; unknown statistics
  receive conservative width bounds, not fabricated values or compression credit.
- Each candidate retains128KiB headers, >=1MiB still-future controls, distinct
  64KiB GPU ledger and96KiB prototype outputs. Old files, failures, partial/temp
  growth, eight protocol files and these review files remain charged to12MiB.
- Actual prototype reads fixed CONTROL files and writes only a small exclusive
  receipt. Archives stay in RAM. No scientific/model/GPU dispatcher exists.
- Exit0 requires both measurements, NOT either budget fitting. No selection,
  fallback, V6 amendment or subsequent execution authority follows.

## Coverage wording corrected

The handoff overstated nonfinite float32 payload tests: test_27 contains finite
zero/subnormal/normal bits only. The parent's additive
`LOSSLESS_ARCHIVE_COVERAGE_CLARIFICATION_V1.json` corrects this; its hash is bound
in the review JSON. No source version/test-count change is needed. The future
finite-intensity failure contract is unchanged.

## Still blocked

Root registration/note-repair approval, separate exact-command admission,
67/67 plain native passes, fresh complete accounting, and live external512MiB
process-tree RAM/no swap/private GPU/50s limits with <=32KiB logs and <=60s
whole-job acceptance remain required. Templates are not enforcement.
Runtime identity stays limited to versions and cached `_lzma` path, with linked
version null; no binary attestation is requested.

No import, compilation, tests, compression, payload reads, vault calls or old-file
changes occurred. No fit, runtime, scientific or LIGHT-quality result is claimed.
