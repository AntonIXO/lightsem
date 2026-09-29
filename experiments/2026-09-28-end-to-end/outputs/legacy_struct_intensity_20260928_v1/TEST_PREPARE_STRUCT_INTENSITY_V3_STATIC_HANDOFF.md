# V3 synthetic test source — static handoff

**Status: written, not executed. No PASS or readiness claim.**

- Test source: `test_prepare_struct_intensity_v3.py` (53250 bytes).
- Intended discovery count: **95 unittest methods**. Counted unique `def test_` source lines only; subtests do not add methods.
- Test SHA256: `5f801e85999f1ea6ef3b0d73b5f7babcbaf83bbd6a813ef2785ee0c604340b4e`.
- Driver text reviewed: `prepare_struct_intensity_v3.py`.
- Reviewed driver SHA256: `82ddd7ea4296ece2d9f5c70c421756a4208c6ebc4bc6dd3fe3de3aace0cd0fc2`.

Coverage: frozen synthetic plan checks; authorization and exact native no-skip test receipts; verbose-log rejection cases; CPU checkpoint metadata with shape-only fake tensors; bus and structure schemas, absolute/adjacent clocks, missing values, flags and float32 range; exact recipient slices; 95% role coverage; 80-frame section intersections; donor/target separation; packet and global resource bounds. Future AST checks cover prepare-only dispatch, authorization/log gate order, capped stdout/stderr nesting, pre-publication status printing, restricted CPU checkpoint loading, and pickle-disabled cache loading.

All numerical fixtures are generated inside future tests. Runtime mocks prohibit payload readers, checkpoint loading, main/preparation dispatch, filesystem reads inside test methods, and torch/model imports. No real plan, cache, raw manifest, weight or vault data is needed. Registration-receipt semantics remain a root-attested boundary; these tests do not prove vault semantic success or scientific validity.

During this writing task: **0 source imports, 0 compile/AST-parse operations, 0 test executions, 0 checkpoint deserializations, 0 model calls, 0 shell/project commands, and 0 numerical payload reads**. Only source/document text was read. Native test execution needs separate authorization. Any later driver or test edit requires updated hashes and a new native result; this handoff is not that result.
