# V6 preparation-only handoff

**Frozen source only;112 authored tests, unrun.** No V6 imports, compilation, tests, data/checkpoint/model or preparation execution. V5 source/auth/failure/log/failedpacket and rootenvelopes are preserved, not resumed.

Confirmed producer source uses `role=np.int8`. V6 replaces `np.where(valid,int8roles,255)` with assignment only at validated observed positions into prefilled uint8/uint16 output buffers. Gaps/tails retain255/65535. No source-array mutation, clock/label/support/gate change or broad error catch.

All105 tests remain; representative fixtures nowuseproducerint8. Seven added cases cover all-valid int8, gap/tail/sentinels/sourceimmutability, unsignedcompatibility, invalidlabels andstaticprogress/fsync structure. No realpickle, cacheeligibility orOS/fsync result is claimed by this unrun suite.

New progress records occur aftersuccessful checkpointmetadata checks and eachrecordedclipoutcome. Each<=2KiB, uses existing64KiB CappedLog/packet guards, and is flushed+fsynced. Progress is explicitly notreadiness. It does not retrospectively establish individualV5qualification: actualV5 payloadaccessbegan, but its log/packet wereempty and noindividualreadiness receipts werepublished.

Full hashes/delta/preservation/evidence/binding schema: `IMPLEMENTATION_HANDOFF_V6.json`. `AUTHORIZATION_SCHEMA_TEMPLATE_V6.json` is NOTauthorization.

## Future rootnative112 command (CWD/root/lightsem; separately boundfirst)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u -m unittest discover -s /root/lightsem/outputs/legacy_struct_intensity_20260928_v1 -p test_prepare_struct_intensity_v6.py -v > /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/native_synthetic_tests_v6.log 2>&1
```

## Future preparation component (requires newroot registration/externaltarget)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/prepare_struct_intensity_v6.py prepare --plan /root/lightsem/docs/research/2026-09-28-end-to-end/PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V6.json --authorization /root/lightsem/outputs/legacy_struct_intensity_20260928_v1/PARENT_BOUND_PREPARATION_AUTHORIZATION_V6.json
```

Root must bind a V6-target external actual-exec guard and actually enforce900s/4GiB/noGPU limits. OldV5target/failedpacket is not authority. Newnames are PARENT_BOUND_PREPARATION_AUTHORIZATION_V6.json, native_synthetic_tests_v6.log, preparation_v6.log, and legacy_struct_intensity_v6/preparation_readiness_v6.json.

All8 V3/V4/V5/V6 protocol JSON/MD files and entiresharedsource directory, includingnative/auth/failure/progresslogs andtemp, count under12MiB. Entireworkspace includes preservedV5faileddirectory. Sharedglobalcap/32MiBcomparisonreservation/1GiBfloors unchanged.

Remaining: native112pass, semanticroot registration,62TRAIN hash/identitybindings andexposurelimits, original/resource checks, newexternalguard, actualCPUqualification, nativeexit0+parentall40receiptaudit. No model/scientific/smoke code or originalrendered/handmarkedgateclosure.
