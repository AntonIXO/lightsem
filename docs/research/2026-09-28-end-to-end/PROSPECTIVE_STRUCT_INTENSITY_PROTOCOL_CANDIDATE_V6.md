# Structure-intensity protocol candidate V6

**Source-only revision. No V6 imports, compilation, tests, preparation, data/checkpoint reads or model calls have run.** Preserve every V3/V4/V5 source, protocol, authorization, native receipt/log and failed packet. Do not rerun or resume V5. Root external-envelope files are untouched.

## Observed V5 failure and scope

The authorized actual CPU V5 preparation exited2 with `OverflowError: Python integer255 out of bounds for int8`. Root recorded507,879,424bytes peak RAM against the4,294,967,296-byte cap and1.167552s service time against the900s ceiling. The actual envelope proof passed for that failed V5 invocation. These are a short failed-prefix resource record, not evidence that a full qualification/study fits.

Actual approved payload access began. The internal log and new packet were empty; no readiness or individual checkpoint/clip receipt was published. **Do not infer audited successful checkpoint or clip qualification from control-flow order.** No model construction/forward, scientific score or GPU job was reported. Exact failure/native bindings remain in `PREPARATION_FAILURE_V5.json` and V6 provenance references.

## Minimal derived-buffer correction

Frozen `build_justdance_structure.py` allocates `role=np.zeros(n,dtype=np.int8)` (line45) and saves it unchanged in the NPZ (line69). V5 used `np.where(valid, int8_role_slice, 255)` before assignment into a uint8 output. NumPy weak scalar promotion can reject255 in signedint8 even when every mask entry is true.

V6 retains the existing0..4 source-role validation and prefilled output arrays:

- Output roles: uint8, initially255.
- Output section IDs: uint16, initially65535.
- Only validated observed positions are assigned through output views.

No source cache or source array is cast, rewritten or mutated. No label is repaired, clock changed, role gap/tail filled or support extended. Unsigned source compatibility remains. There is no broad catch/skip/unsafe fallback for the OverflowError; unexpected errors still stop.

The fixed40 inventory, original donors, TRAIN/no-knownTEST identity rules, all clocks/finite checks, final-only STFT-center/waveform-tail metadata bound,95% role coverage and80-frame section/drop/breakdown criteria remain unchanged. The endpoint check is still metadata consistency only, not waveform verification. No model/scientific/smoke dispatcher is added.

##112 source-authored tests, unrun

All105 V5 test names and guards remain, including eager NumPy assertion initialization. The representative structure fixture now uses the actual producer's int8 dtype, rather than int16. Seven new methods cover:

1. Representative producer-int8 fixture declaration.
2. All-valid int8 roles copied to typed output buffers without any source-array mutation.
3. int8 gaps and target tails retain255/65535 sentinels and unchanged labels/support.
4. Equivalent unsigned-source compatibility.
5. Invalid int8 labels remain rejected, not cast into validity.
6. Static progress ordering after successful checkpoint metadata checks and recorded clip outcomes.
7. Static bounded-log/fsync call structure.

Tests instrument only synthetic in-memory buffers. None ran in this task. They do not prove actual checkpoint/pickle compatibility, real cache eligibility, durable filesystem behavior or OS limit enforcement.

## Bounded durable progress, not readiness

After each successfully validated CPU checkpoint metadata block, V6 writes a `CHECKPOINT_METADATA_VERIFIED` line with the arm and exact checkpoint hash. After each recorded clip outcome, it writes `CLIP_QUALIFICATION_RECORDED` with clip/status/eligibility, reason count and at most128characters of the first reason.

Each JSON line is limited to2KiB, explicitly marks `progress_only_not_readiness`, and goes through the same authenticated64KiB CappedLog and packet/free-space guards. Each progress write is flushed and `os.fsync` is called. Log or fsync failure remains fatal, with no fallback. Full reasons remain in the final readiness receipt if the stage completes.

Thus a later fatal error can leave a bounded record of checks actually completed. Absence of a line is not proof that an earlier operation never accessed data. Progress is not a successful all40 readiness result, scientific score, execution grant or replacement for nativeexit0 and parent audit.

## Versioned bindings and cumulative storage

V6 names: `prepare_struct_intensity_v6.py`, `test_prepare_struct_intensity_v6.py`, `PARENT_BOUND_PREPARATION_AUTHORIZATION_V6.json`, `native_synthetic_tests_v6.log`, exclusive internal `preparation_v6.log`.

New packet: `/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/legacy_struct_intensity_v6/preparation_readiness_v6.json`.

Old V5 authorization/envelope/failed packet are not reused, overwritten, resumed or deleted. Root must bind a new V6 external target and actually enforce900s/4GiB/noGPU process-tree limits. This source and the synthetic suite do not enforce or prove OS/cgroup limits.

The12MiB TOTAL prospective packet includes the entire sharedsource directory recursively, all preserved failure/authorization/native/progress logs, new source/tests/templates/handoffs, all **eight V3/V4/V5/V6 JSON/MD protocol files**, authorizations/native receipts, readiness/traces and temporary files. The entire scientific workspace, including the failedV5 directory, remains counted under the unchanged1,020,054,732-byte combined cap,32MiB comparison reservation and1GiB free floors. No location/alias bypass.

## Root gates remain mandatory

- Static review and separately bound full112 native test run. Exact command/source/plan/test/log hashes; all plain passes; no skips/expected failures.
- Root semantic validation of new preparation registration; exact62 TRAIN input hash/identity bindings and exposure limitations. No TEST/rawmanifest access.
- Fresh original-inventory, cumulative packet/global reservation and free-space checks; new V6 external resource binding.
- Actual restricted CPU loader/cache qualification, still with no unsafe fallback.
- Nativeexit0 plus parent validation of all40/source/input/resource readiness semantics. Progress logs alone do not pass this gate.

Whole-rig emitted-proxy scope remains unchanged. No blinder/C-null substitute and no completion claim for the original rendered/independent-handmarked gate. Any future model/GPU/scientific outputs need separate implementation, source review, preregistered predictions and native registration.
