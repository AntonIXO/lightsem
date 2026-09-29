# Row65 causality provenance readiness V1

**Status: exact historical reconstruction is blocked by missing provenance.**

Goal remains active. This report gives no new execution authority. No model, audio, checkpoint, numeric-report or calibration payload was read. No test or scientific script was run.

## What is authenticated?

The native record identifies:

- Hypothesis `the-deployed-path-meets-its-latency-budget-and-b0aff5`.
- Experiment `exp-006-the-deployed-path-meets-its-eb0fc6`.
- Registration time `2026-09-04T00:04:43+00:00`.
- The recorded CUDA/context account: 1039 frames / 25.975 s, cuts50/100/200/300 and reported difference0.
- The original intended gate: “Test causality by replacing all audio after a cut with noise and asserting the output before the cut is bit-identical.”

But `git_ref` and `external_run` are empty. `artifacts`, `provenance` and `recipe_fingerprint` are empty. The record does not pin a checkpoint, source tree/dirty patch, calibration bytes, invocation or environment. Registration time is not an attested execution-start time.

Git authenticates **saved source snapshots**, not which snapshot this experiment executed. Stat authenticates current path metadata, not historical file use or bytes.

## Recoverable source candidates—not an executed checkout

| Commit | Recorded time | Meaning |
|---|---|---|
| `1656ffde47a26bbd9655b5ceef8ddaebd80e4895` | 2026-09-04 09:11:10Z | Earliest located reachable stage4 source snapshot. It is9h06m27s after native registration. A retrospective candidate, not execution attestation. |
| `20322d60083000f04ec4c35390dc822f4ac9235e` | 2026-09-04 10:00:26Z | Later feature-causality and conductor/packing sampling changes. Cannot substitute for the original source. |
| `cab408a51f0a2ea6889a996f2e26f56ac19b00c9` | 2026-09-04 10:13:33Z | Later event calibration. Exp-009 names `master@cab408a+dirty`, not an exact clean tree. |
| `ea51f6054fcf5dd0b3c95e7a20f5a57047558ad7` | 2026-09-04 12:21:55Z | Changes the default checkpoint from `light_trunk_v1.pt` to `light_trunk_v3causal_smooth.pt`. |

The first candidate tree is `e84acd58ca2f01e25159cc92570caa1f6490e639`. Its key Git blob IDs are:

| Path | Candidate blob |
|---|---|
| `run_stage4.py` | `20ee2f4b3007601d702952d5cd68905846aebcff` |
| `lightsem/model/light_trunk.py` | `e425117774fd8921baa8ec6b38124ee4dda2ddeb` |
| `lightsem/model/infer.py` | `72546ea3628b4504e3f92532b7bf27581f1ab107` |
| `lightsem/audio/online.py` | `959929f5bed79927aa76275dcca519edc792db84` |
| `lightsem/audio/beatgrid.py` | `6350f0f4b623fd6d77e6bc792613d16a8dc97773` |
| `lightsem/conductor/scalers.json` | `aa0c949a40379cb01776d473854d6debfbd744d3` |

The scaler blob identity was read from Git tree metadata only. Its numeric content was not opened. More module/logger/runner blob identities are in the JSON.

## Exact boundary in the available source

At the first candidate snapshot, `run_stage4.py:49–52` creates `torch.randn(4,400,lm.cfg.in_dim)` and passes it to `causality_probe` at the four recorded cut frames.

`light_trunk.py:230–242` compares **only `model(x)["a"]`**, before the cut, after replacing later feature-tensor positions with random values. The helper equals the current helper.

Thus this source observes the regression A-head under synthetic model-input perturbation. It does not observe waveform→features, B hazards, C samples, descriptor postprocessing, conductor parameters or DMX under that perturbation. This is a source-snapshot boundary—not proof that these exact bytes executed historically. Do not quote it as an all-head or full audio-path test.

The source separately extracts20s from `/root/lightsem/data/wav/wav/mBR0.wav`, calls `descriptors(seed=0)`, and times conductor/packing. That audio branch is **not perturbed**. No global synthetic-probe seed is set in the runner.

## Assets: present does not mean authenticated

- The first candidate `infer.py` defaults to `/mnt/lightsem-data/ckpt/light_trunk_v1.pt`. `LightModel()` supplies no override. But configuration comes from the checkpoint payload, and exp-006 does not identify that payload.
- `v1`, `v2`, `v2b`, `v2smoke`, `v3causal`, `v3causal_smooth`, `v4offline` and `v4offline_smooth` paths exist by stat. None was opened, hashed or deserialized. Their names/sizes/mtimes cannot prove use in exp-006.
- `outputs/stage4/latency.json` exists:2223bytes, mtime `2026-09-04T00:03:50.532572+00:00`. It is absent from the candidate Git tree and current index. Its payload was not read. The plausible timestamp is not a run-to-assets attestation.
- The source logger reads that fixed overwriteable report path, records no source/checkpoint/report digest, and copies evidence with hardcoded passed flags. Matching Notes support candidate provenance only; this report does not claim the recorded numbers are false.
- The waveform and scalar-calibration file exist by stat. The calibration working-copy content was not compared with its index blob. No historical scalar/audio identity is run-pinned.
- The historical torch wrapper selects a venv path; it does not pin installed interpreter/package/driver versions. The current directory's presence is not historical runtime recovery.

The native1039-frame context is compatible with six blocks,160-frame attention and15-tap convolutions. It is not a unique checkpoint/config fingerprint. Neighboring Stage1's named v1 and later exp-009's named v3causal do not identify exp-006's weights or upstream version.

## Readiness decision

**Blocked:** exact executed source, checkpoint/config/normalization, calibration/upstream boundary, invocation/environment, and run-to-result linkage are not authenticated in the reviewed evidence.

This bounded search does not prove that no execution log or stronger archive exists. A separately authorized record search could look for those links. Choosing the nearest dated commit, current HEAD, current default or a plausible filename is not a substitute. A new hash would pin current bytes, not retroactively authenticate old bytes.

Source snapshots and extant paths could support a *candidate reconstruction* after additional permissions. They do not justify calling it an exact historical replay.

If the historical identity cannot be established, a newly pinned current-version controlled-synthetic audio→output test must be a **separate version-bounded study**. It would need explicit source/external-asset pins, full output boundary, timestamp/priming/RNG policy, inclusion of beat/oracle paths, and separate payload/runtime approval. No implementation, test, training, body retiming, or current-pipeline repair is authorized here.

## Review boundary

Only native/Markdown records, source text, read-only Git log/show/blame/tree/index metadata and file stat/path metadata were used. Numeric/audio/checkpoint payloads, raw manifests, models, tests and vault calls were excluded. Only this new exclusive report pair is written; historical sources/records remain unchanged. Exact metadata, hashes, gaps and candidate classifications are in the JSON.
