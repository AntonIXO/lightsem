# V4 preparation-only handoff

**Source-only. No imports, compilation, tests, preparation, payload/checkpoint reads or model calls.** V3 is byte-preserved.

The final V3 static review reported F1–F3. V4 addresses them:

1. Cache duration is last centered STFT frame L, not waveform endpoint D. Only final Segment.end_s gets the source-derived512/22050s+1ns upper extension. Starts, intermediate ends, clocks, overlaps, role agreement and95%/80frame support stay unchanged. No cache/gap/tail filling. Scope is metadata consistency, not independent waveform-end verification.
2. Native receipt command must equal the exact published command, not merely be nonempty.
3. Bound schema-source refs now include `audio/features.py` and `audio/structure.py` plus existing22050/512 constants.

**104 unrun methods:** all95 prior names retained, plus9 sample-count-derived boundary/command cases. They cover exact-hop/511 residuals, both40Hz rounding directions, final/internal/start/overlap rejection, gap preservation and command drift. They do not test actual pickle compatibility or OS enforcement.

Full hashes, versioned paths, binding schema, preservation receipt and remaining blocks: `IMPLEMENTATION_HANDOFF_V4.json`. Template `AUTHORIZATION_SCHEMA_TEMPLATE_V4.json` is NOT authorization.

## Future native test command (root registration first; CWD/root/lightsem)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u -m unittest discover -s /root/lightsem/outputs/legacy_struct_intensity_20260928_v1 -p test_prepare_struct_intensity_v4.py -v > /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/native_synthetic_tests_v4.log 2>&1
```

## Future preparation (additional root registration/bindings/OS limits first)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/prepare_struct_intensity_v4.py prepare --plan /root/lightsem/docs/research/2026-09-28-end-to-end/PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V4.json --authorization /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/PARENT_BOUND_PREPARATION_AUTHORIZATION_V4.json
```

Root must actually enforce900s/4GiB/noGPU process-tree limits. The CLI does not do OS/cgroup enforcement. Its exclusive internal64KiB log is `preparation_v4.log`; no uncounted redirected preparation log. New output workspace is `.../end_to_end_20260928_v1/legacy_struct_intensity_v4`, with `preparation_readiness_v4.json`. Native exit0 plus all40/source/input/resource receipt validation remains mandatory. No automatic next stage.

Remaining: native104-pass receipt; semantic root verification of preparation registration;62 explicit hash-bound TRAIN inputs/donors and knownTEST exclusions; honest unknown-alias limits; actual restricted-reader/cache qualification; unchanged shared/storage/free-floor accounting; external limits. All V3/V4 source-dir files plus both document pairs count toward12MiB total. No actual readiness, empirical eligibility, scientific result, model/GPU authorization or original-gate completion is claimed.
