# Pose-stage renderer implementation (synthetic validation only)

Files: `outputs/render_pose_stage_study.py`, `tests/test_pose_stage_renderer.py`.
No freeze, real study mapping/render, GPU allocation, learned model, or vault call was made.

## Interface

Use the same frozen runtime as the evaluator:

```
bash run_torch.sh FROZEN_DIR/frozen/render_pose_stage_study.py --output FROZEN_DIR --budget 1
bash run_torch.sh FROZEN_DIR/frozen/render_pose_stage_study.py --output FROZEN_DIR --stage review --observation OBSERVATION.json
bash run_torch.sh FROZEN_DIR/frozen/render_pose_stage_study.py --output FROZEN_DIR --stage report
```

The adjacent evaluator verifies frozen source, helpers, saved inputs and runtime. The renderer verifies its own frozen digest. Only the authenticated evaluator schedule is accepted. Every render binds the exact schedule, consumed numerical stage receipt hashes, freeze, map, full world, full DMX and audio. No item/source/exposure selection CLI exists.

One shared EGL context per render invocation is enclosed by the explicit project-exclusive GPU lock. Encoding is CPU `libx264`. The fixed native panels are one cached world body (front + oblique), baseline, candidate. Only the common frozen zero-zoom override is applied. No pose interpolation, origin subtraction, recomposition or endpoint padding occurs in the renderer.

Raw uncaptioned pixel series include outside-bulb, bulb-only and outside-caption means, any-channel clipping, white clipping and support. Candidate outside-bulb any-channel clipping gates use strict mean < .01 and maximum < .05. Metrics, masks, per-frame hashes, aggregate hashes and caption provenance are saved. A failed gate remains false.

The full-Shpongle dim control uses one full-record RGB scalar per arm. It rerenders original pixels and checks every frame hash plus each complete original stream hash before claiming a match. Scalar values, channel rounding error, outside-bulb luma residuals and final mean difference are retained. No per-frame adjustment exists.

Every delivered video frame is software-decoded. Every delivered audio sample is software-decoded to stereo 48kHz PCM. Frame count, video dimensions/rate/duration and audio duration are checked. Raw decoded RGB/PCM hashes are retained. PNG samples and chronological 1Hz+last / fixed-fast contact sheets come from the decoded delivery, not substitute pre-encode images. Audio trims use the exact whole-frame interval without padding.

`--stage review` accepts only the next missing scheduled observation. JSON must contain `entry_id`, nonempty `reviewer` and `notes`, `independent_of_renderer: true`, `coverage: "all_frozen_decoded_samples"`, the complete sorted `viewed_sample_frames` union, every `viewed_sheets` filename, nonempty `comparisons` values for `body`, `baseline_vs_candidate`, `clipping_and_visibility`, and an explicit `limitations` list. Coverage and independence are observer attestations, not machine proof. This never sets human music-fit acceptance or pipeline qualification.

Outputs are create-only under the evaluator lock. Completed entries are fully verified and skipped. Partial entries keep their failed receipt and files and block retry in place. Review and report receipts are also create-only. The terminal report lists completed, failed, resource-blocked, incomplete and unrun approved entries. It inventories current partial bytes without treating them as completed coverage. Authenticated failure receipts retain assessed false gates and null unassessed gates. A full software decode failure can therefore be reported without requiring missing movies to complete. Missing coverage is never promoted to a pass.

## CPU tests

`bash run_torch.sh -m unittest discover -s tests -p test_pose_stage_renderer.py -v`

Fourteen tests pass. A run-level routing regression test covers review/report with both native gate outcomes and rejects any project/GPU import on those CPU paths. It was added after parent static inspection caught and fixed a CLI-stage variable shadowing bug that the lower-level tests did not measure. The other tests cover fixed sampling, strict clipping boundaries, full-array validation/no padding, software encoder/decoder flags, global dim quantization, and an integrated six-frame synthetic CPU media workflow. The latter checks actual x264 encoding, complete software video/audio decode, decoded sheets, global dim repeated hashes, create-only outputs, repeat-hash failure retention, missing review coverage rejection, observer receipt scope and modified media rejection.

The same synthetic suite also passed under `run.sh` before the final review/provenance refinements. The full frozen runtime path is `run_torch.sh`.

## Parent-approved storage policy

The renderer leaves **1GiB filesystem free** and applies the **1.75GiB total study-artifact threshold**. Every check counts all files below the frozen study root, including sources, maps, partial/silent/muxed videos, samples, sheets and reports. Normal output writes leave an additional **4MiB bookkeeping margin** for failure/report receipts. The frozen evaluator independently guards its serialized writes with the same policy.

The CLI and run API enforce exactly `--budget 1`; any other value is rejected before the lock, context or GPU work. Before GPU allocation, that single approved render entry reports current filesystem free bytes, current study bytes, remaining budget and planned outputs. Movie planning uses **2MiB/s combined retained-video allowance (about16.78Mbit/s)**. This renderer retains no separate silent intermediate: its silent component is0 bytes and the muxed component receives the whole advisory allowance. Any existing silent files still count in actual study usage. Decoded samples/sheets use actual native dimensions times3 plus64KiB per PNG. Metadata allowance is32MiB per entry. These are advisory planning allowances, **not hard CRF compression bounds**. If the estimate cannot fit, the tool records a resource-blocked entry and asks the parent. It does not reduce quality or coverage or relocate outputs.

During rendering, checks run before **every encoded and decoded video frame**, each decoded audio1MiB block, every review-sheet write, and each raw-metric/hash/final-receipt group. Encoding and audio muxing share the same guarded ffmpeg stream; there is no retained silent intermediate in this renderer, but any existing silent files count toward the total. Each check allows **32MiB next-write/queued-encoder headroom**, plus the4MiB bookkeeping margin. This is a conservative operating allowance, not a formally proven bound on an external encoder. Concurrent external filesystem writes remain outside this process's control.

A resource stop terminates subprocesses, releases the shared context, retains partial files, and records `resource_blocked`. Unassessed media remains null; previously measured pixel gates stay unchanged. Failure-receipt errors are added as secondary notes without replacing the original exception. No raw RGB cache, codec/quality change or evidence deletion was added.

Synthetic tests cover exact reserve and artifact-budget boundaries, inclusion of all study file categories, preflight-block receipts, runtime decode guard cleanup, failure-receipt OSError preservation, and terminal resource-blocked reporting. Real available capacity must still be rechecked by the parent before launch.

Not measured: real EGL rendering, actual frozen study assets, real-stage scheduling, real audio alignment, clipping or brightness gates on the recordings, independent visual review, and GPU repeat determinism. No scientific result or human acceptance follows from these synthetic tests.


## Metadata-only storage estimates

Computed from `pose_stage_study_plan.json` frame counts and fixed sampling metadata using `run_torch.sh`. Filesystem free space and existing study size were mocked for calculation only. No frozen input arrays, GPU, candidate mapping or rendering were used. These are planning assumptions, not measured output sizes or CRF bounds.

| Component (bytes) | Full Shpongle,8319frames | Shpongle diagnostic,600frames (0–20s) |
|---|---:|---:|
| Retained silent movie | 0 | 0 |
| Retained muxed movie advisory allowance | 581,540,250 | 41,943,040 |
| Decoded native sample PNG allowance | 716,610,560 | 74,869,760 |
| Chronological/fixed-fast sheet PNG allowance | 204,353,024 | 28,186,624 |
| Metadata allowance | 33,554,432 | 33,554,432 |
| **Total new output estimate** | **1,536,058,266** | **178,553,856** |
| Minimum filesystem free, including1GiB reserve +4MiB bookkeeping | 2,613,994,394 | 1,256,489,984 |
| Maximum already-existing study bytes for this estimate to fit1.75GiB cap | 338,795,622 | 1,696,300,032 |

A single full-Shpongle entry is not inherently over the1.75GiB operating cap. It passes this advisory preflight only when existing study artifacts are at most338,795,622bytes and actual filesystem free is at least2,613,994,394bytes. These conditions have not been measured on a frozen real study. Parent must check before launch.

A batch containing **both full original and full dimmed Shpongle entries** has an advisory estimate of3,072,116,532bytes before existing files or the bookkeeping margin. That hypothetical batch is **over the1.75GiB cap even on an empty study**. It is retained here only as a rejected planning example: the parent now forbids combined batches, and CLI/run reject `--budget 2` before work. Sequential budget1 invocations do not override the cap: each counts all actual files already retained. No thresholds, quality, coverage or output locations were changed to make this route fit.


### Real-evaluator bookkeeping regression

Independent review found that the earlier FakeEval tests did not measure the real active-lock margin behavior: ordinary `started.json` could fail before bookkeeping `failed.json` was attempted. The recorder now attempts failed bookkeeping independently after a blocked start. A cross-module CPU test uses the actual evaluator lock, guarded writer and authenticated reader, with only synthetic map identity and mocked storage state. At both filesystem-free=1GiB+1MiB and study-bytes=1.75GiB−1MiB, it verifies that started is absent, resource-blocked failed.json is saved, the original error gains a secondary note, and a negative terminal report succeeds. The test also verifies lock-state cleanup. No real freeze or study assets are opened.
