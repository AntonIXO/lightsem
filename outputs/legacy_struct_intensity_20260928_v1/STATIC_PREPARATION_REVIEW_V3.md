# Static preparation review — V3

**HOLD tests and actual preparation pending a prospective revision. Preserve V3.**

This review read source/protocol text and checked source hashes only. No imports, compilation, tests, payloads, checkpoints, raw manifests or vault calls were performed. No numeric eligibility or empirical runtime is claimed.

## F1 — Producer duration semantics do not match validator (blocking)

`validate_structure` (preparer309–347, especially338) treats structure metadata `duration_s` as the upper bound of every segment. The frozen producer does not use that meaning:

- `features.extract` stores waveform duration `D=len(y)/sr` separately from its frame-center times (`features.py111–113`).
- `build_justdance_structure.py42–44` stores `L=frame_times[-1]` as metadata duration and creates `N=round(L*40)` cached samples.
- `audio/structure.py106,116` closes segmentation at waveform end `D`, not `L`.
- For the ordinary centered512/22050 grid, `D-L` can approach23.22ms. It is not a100µs timestamp error.

Therefore a correctly produced terminal segment may be rejected as `MATERIAL_QUALIFICATION_FAILED`. This is a source-grounded incompatibility, not an observed failure count or corpus ineligibility result. Current synthetic fixtures set segment end equal to metadata duration and miss it.

### Narrow prospective fix

Keep three quantities separate: frame-center extent `L`, cached role samples `k/40` for `k=0..N-1`, and waveform/terminal endpoint `D`. `N/40` is a rounded grid stop, not proof of waveform duration.

Validate the terminal-end producer relationship separately. Prefer the already-approved matching AudioBus waveform duration when same-recording producer provenance is established (`online.py218–221`). Otherwise state only the weaker metadata-consistency claim. Any legitimate exception beyond `L` must be terminal-only and source-derived. Do not enlarge all segment tolerances, manufacture waveform truth from arbitrary segment maxima, or relax overlaps/role agreement.

Do not clamp/rewrite stored bounds, extend the cache to waveform end, alter `T`, resample or synthesize labels. Gaps/tails remain invalid255 and cannot become scoring support.

Add hypothetical sample-count fixtures that derive `L`, `D`, `N` and saved roles as the producer does. Cover zero/nonzero final-hop residuals and both40Hz rounding directions. A correct terminal `D>L` case must pass; internal overruns, overlaps, wrong labels and an end beyond the producer-consistent boundary must fail. Keep95%/80-frame gates unchanged; infer no real eligibility from these tests.

## F2 — Native command is not compared exactly

Preparer158 only requires a nonempty native test command, although argv/source/test/log/count checks are otherwise bound. A different nonempty command is accepted by that check.

In a new version, bind the exact prospective native command in the protocol and compare it to native evidence/authorization. Bind the revised suite's full method count, not an inherited95 if methods change. Add wrong-nonempty-command rejection. Root must independently enforce command/environment identity until that guard exists.

## F3 — Missing endpoint-defining source hashes

`preparation_v3.source_refs` omits `lightsem/audio/features.py` and `lightsem/audio/structure.py`. These define the producer semantics above. Add their hashes as schema-evidence dependencies in the revision. This does not authorize importing them, regenerating features or reading new data.

## Constraints, not additional measured failures

- The CPU checkpoint reader legitimately addresses NumPy `mu`/`sd` from `train_light.py223–225` with restricted `weights_only=True` and a narrow NumPy reconstruction/dtype allowlist. No unsafe fallback or project model class is present. Actual pickle/version compatibility is untested and must fail closed;95 pure tests cannot certify it.
-900s/4GiB are exact root-attestation fields, not an implemented process-tree limiter. Parent must bind and actually enforce the external wall/memory limits. Internal checks between operations cannot stop a blocked import/read/deserialization.
-95 methods exist but were not run. Their operational helpers are mocked or inspected as AST; a future pass does not prove actual cache compatibility, budget enforcement or eligibility.

## Static checks that agree

Prepare-only stage; exact declared62-file/40-row inventory; TRAIN/no-knownTEST gate and honest identity limits; recipient rounded offset versus donor0; exact bus support; target-role masks and retained80-frame intersections; no replacement after failure; no model/scoring/GPU path; original+wholeworkspace shared cap; global12MiB packet and32MiB comparison reservations; bounded log; exclusive packet/receipt; parent native-exit acceptance required.

All supplied handoff source hashes match. Full references, findings and limits are in `STATIC_PREPARATION_REVIEW_V3.json`. No V3 source/protocol file was edited.
