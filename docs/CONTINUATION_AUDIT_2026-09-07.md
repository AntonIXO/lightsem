# LightSem continuation audit — 2026-09-07

Resume with consistent evaluation splits and a preference study of the current wave/colour engine. The last Claude conversation stopped after Stage 10 and expected Anton's mapped demo feedback. The code contains useful work that can proceed before those ratings arrive; the claim that ratings are the only remaining input is too strong.

This audit inspected source code, git history, Claude's JSONL, saved result JSON, the project markdown vault, cache inventories, and live processes. It did not train models or change implementation code. The numerical experiment results below are historical results, not new experiments.

**Recovered conversation and stopping point**

The latest substantive session is [71c8c079-19a2-4ec4-bea5-2ce7db778b3e.jsonl](/root/.claude/projects/-root-lightsem/71c8c079-19a2-4ec4-bea5-2ce7db778b3e.jsonl), spanning September 3–5 UTC. It contains 7,695 records. The final record on September 5 at 08:30 UTC reports a session limit; the last substantive handoff was earlier that morning.

- [Readable conversation extraction](/root/lightsem/logs/claude_latest_conversation.md): 304 user/assistant text messages, with original source line numbers. Tool payloads, thinking, metadata, automated prompts and compaction dumps are excluded.
- [Final handoff excerpt](/root/lightsem/logs/claude_latest_handoff_excerpt.md): the September 5 continuation discussion and demo feedback.
- [Machine-readable audit evidence](/root/lightsem/logs/continuation_audit_evidence.json): source hash, split reconstruction, cache snapshot and verification scope.

At source line 7642 Claude closed Stage 10 at commit `c274c23`, which remains HEAD. At line 7645 Anton said: “next we gonna work to improve further and I ll send mapped results”. At line 7688 Claude reported no H-L ratings and waited. No Stage 9 `ratings.jsonl` or fitted `bradley_terry.json` is present in the inspected outputs today. Earlier Stage 0c score sheets exist and are a separate study.

The latest concrete product feedback, source line 7559: colour hardly changes; the wavy decoder variants fit the music better than the simultaneous on-beat variants; cue-held is decent but repeats its movement. Claude implemented travelling intensity waves, hue spread/drift and cue-triggered hue changes, then rendered [shpongle__cue_held_waves.mp4](/root/lightsem/outputs/stage9/demo_songs/shpongle__cue_held_waves.mp4). There is no subsequent evaluation of that variant in the recovered conversation.

**What the project currently executes**

The intended product is pre-rendered **audio → stage lighting**. Dance/video is training supervision; inference does not require or generate a dancer. The latest user direction prioritised access to future audio over live deployment.

| Path | Actual behaviour |
| --- | --- |
| [render_song_demo.py](/root/lightsem/render_song_demo.py:17) | Audio bus → JD/AIST LightModel → motion-descriptor bundle → hybrid conductor → DMX frames → MP4. This is the established learned demo path. |
| [lightsem/model/light_trunk.py](/root/lightsem/lightsem/model/light_trunk.py) | Conformer trunk with predictable-descriptor regression, event hazards and sampled style. Descriptors form the interface to the conductor. |
| [render_stage9_demo.py](/root/lightsem/render_stage9_demo.py:121) | Experimental group decoder, planner, critic and cue-held/wave arms. GroupEngine distributes wash/beam/strobe/blinder/back roles across fixtures. |
| [lightsem/cli.py](/root/lightsem/lightsem/cli.py:88) | Older structure/anticipation/rule renderer. The generic CLI does not select either learned pipeline. |

Stage 9/10 did not replace the established learned demo. The default Stage 9 `--arms` excludes both cue-held variants; they need explicit selection. The new wave/colour values are hardcoded in [render_stage9_demo.py:150](/root/lightsem/render_stage9_demo.py:150). `LightModel`'s default checkpoint is still AIST, while recent demos explicitly select JD. These entry points need clearer configuration before a user can reliably reproduce “the current model”.

**What was tried, and what remains supported**

[PLAN.md:1188](/root/lightsem/PLAN.md:1188) and [PLAN.md:1251](/root/lightsem/PLAN.md:1251) describe Stages 9 and 10. The local vault's [exp-024](/root/lightsem-vault/experiments/exp-024-a-look-sampler-that-holds-its-c32dbb.md) and [exp-025](/root/lightsem-vault/experiments/exp-025-feeding-the-group-decoder-the-6ba213.md) are closed with `directional_only` verdicts and failed blocking gates.

| Experiment family | Recorded outcome and implication |
| --- | --- |
| Beat Saber lighting supervision | Small residual over beat-trigger baseline; failed its gate and was excluded from target losses. Its mapper BPM remains useful timing reference data. |
| Stems / MERT audio enrichment | About +0.019 AP for ridge; no gain demonstrated by the neural head. Another feature expansion is not the first supported step. |
| Group decoder / bar planner | Regressed looks collapsed; sampled looks recovered variation but weak song matching; planner did not establish useful scene-level prediction. |
| Critic | Distinguished real from beat-trigger light largely through level statistics; reranking had little measured benefit. |
| Cue-held sampler, H-N | Transition distance 6.86 versus per-frame 8.87, but shuffled donor 5.66 was better. Failed the donor gate. Its reported BAS gain needs the split qualification below and conflicts with Anton's aesthetic ranking. |
| Self-hazard / size sweep, H-O | Tested causal models reached roughly 2.2–2.41× cue AP lift on JD. Operator-head own-cue feedback scored 1.83× versus audio-only 2.32×; true-history conditioning had overstated deployable performance. |
| H-L preference fitting | Materials exist; ratings and validated fitted settings do not. |

These findings discourage repeating the same recipes. They do not prove that audio-to-light aesthetics, future-context methods, better targets or more data cannot improve. The measured size sweep is particularly a result about the tested causal models, not the entire pre-rendered design space.

**Issues that change the continuation order**

1. **The comparative DMX evaluation uses incompatible model splits.** Decoder training hashes recording IDs through [train_operator_head.py:55](/root/lightsem/train_operator_head.py:55). The Stage 9/10 lighting evaluations select the motion model's independently shuffled song validation set ([run_stage9_decoder_eval.py:136](/root/lightsem/run_stage9_decoder_eval.py:136), [run_stage10_cue_eval.py:158](/root/lightsem/run_stage10_cue_eval.py:158)). Independently reconstructing these rules from the current tensor metadata gives:

   | Membership under decoder splitting | Clips in the 39-clip motion validation set | Distinct recordings |
   | --- | ---: | ---: |
   | Training | 30 | 22 |
   | Validation | 5 | 5 |
   | Test | 4 | 2 |

   “Held-out coach clips” therefore does not establish decoder generalisation. The separate 38-song transition evaluation uses the decoder's recording-level test split, but that split has a second issue: the 332-file visual corpus contains six normalized song groups crossing partitions. One, `automaton`, crosses train/test (`rWo5v_99zn0` / `vPu3WGIoyD4`). Negative results still do not justify promotion; positive comparisons require corrected evaluation. Save canonical song identities and per-song results, not just aggregate scores.

2. **The current H-L pairs do not measure the new engine.** [build_hl_pairs.py:35](/root/lightsem/build_hl_pairs.py:35) varies nine older ConductorHybrid knobs. Wave amplitude/period/spread and hue spread/drift belong to GroupEngine and are absent from these pairs. Preserve the old batch as its own study and create a separate immutable batch for the current engine.

3. **Rating collection and fitting have correctness defects.** [rate_hl_pairs.py:19](/root/lightsem/rate_hl_pairs.py:19) skips IDs rated by anyone, despite promising per-rater resume. [fit_bradley_terry.py:29](/root/lightsem/fit_bradley_terry.py:29) includes brightness foils in fitting before separately reporting them. Its random row folds at line 44 can place repeated ratings of one pair/song in both training and validation. Group the evaluation by the intended generalisation unit and keep foils separate. The `n >= 200` gate cannot be met by one rater completing 39 pairs; a small pilot can inform preferences but cannot satisfy that gate. A fitted-versus-default render comparison is also still required.

4. **Some measurement implementations differ from the plan.** The transition “within 100 ms” feature is actually ±10% of a beat ([run_stage10_cue_eval.py:75](/root/lightsem/run_stage10_cue_eval.py:75)). The H-O summary substitutes a gain threshold for its prescribed paired-confidence-interval gate because per-song CIs were not saved ([summarize_self_hazard_sweep.py:37](/root/lightsem/summarize_self_hazard_sweep.py:37)). That gate currently fails anyway. Correct the definitions before new evidence is collected.

5. **Expanded pose extraction has not expanded the training corpus yet.** At 20:24:55 UTC / 23:24:55 Moscow on September 7: manifest 675 recordings, 669 usable, 530 completed pose filenames, but audio/visual/structure caches remain 332 each and motion descriptors remain 453 coach clips. The song metadata has only 338 rows. An existing external job is running `extract_justdance.py run --model yolo26m-pose.pt --imgsz 640 --workers 6 --batch 32` (PID 7031); this audit did not launch or interrupt it. The pre-existing uncommitted extractor changes add manifest refresh, resumability and atomic writes. File counts do not establish pose quality. Refresh song identity mapping and dependent caches after quality checks. Freeze song partitions first: [make_jd_splits](/root/lightsem/lightsem/model/jd_data.py:51) reshuffles the full key set when the corpus grows.

6. **Documentation and queue status lag the artifacts.** CLAUDE.md still describes rev. 5; PLAN.md is rev. 10 and its opening text still says the completed sweep is running. The local Stage 9 and Stage 10 runbooks both remain `queued`, even though their individual experiments have results. Stage 9 also has the explicitly open H-L dependency. Reconcile the records without declaring unfinished human gates complete.

**Concrete continuation**

1. Freeze dataset identities, model-training memberships and a common evaluation manifest. For existing checkpoints, use material unseen by every compared model; the two decoder-test recordings inside the current motion-validation set are too few for strong conclusions. For a larger comparison, reserve new canonical songs or retrain comparators under a shared split. Repair the rating/CV defects and measurement-definition mismatches.
2. Expose the existing wave/colour settings as reproducible renderer configuration. Save audio segment, model/checkpoint, corpus embedding, cue selection policy, seeds, rig and settings with each render. The current full-song top-k cue policy is appropriate to label as an offline policy; it differs from the threshold policy evaluated in H-N.
3. Build a small new batch comparing the established conductor, cue-held baseline and controlled wave/colour changes. Hold audio, cue schedule, seed, rig and brightness constant where the comparison permits. Collect separate judgments for music fit, designed appearance and excessive metronomic flashing. Anton's mapped observations already guide variant selection; new ratings are needed to claim preference improvement.
4. Audit beat-grid octave and 2:3 mistakes against the existing mapper reference data. The recorded 62–65% BPM match rate makes this a concrete independent improvement candidate, but any correction needs held-out timing checks and render review rather than maximising BAS alone.
5. Complete the expanded corpus pipeline, audit pose coverage/alignment and duplicate-song identities, then register a controlled data-scaling comparison. More extracted poses alone are not a trained improvement. Prioritise the pre-rendered objective and preserve an L=0 comparator for any future-context claim.

Steps 1–2, variant preparation, timing audit and corpus bookkeeping can proceed autonomously. Human ratings are required for claims about human preference, not for all further engineering work.

**Verification and tool limits**

All 169 repository Python files parsed without syntax errors. Five existing CPU checks passed: synthetic tempo/phase recovery, noise tempo prior, DMX encode/decode round trip, slew limiting and brightness matching. Twelve checks were skipped by an explicit filter; no full GPU suite, training or visual-quality re-evaluation ran while extraction was active.

Honcho retries returned `Invalid API key`. The newly exposed Vibescience connector returned an index error on project lookup and then reported zero records via `doctor`; it did not provide this project's records, so this audit used `/root/lightsem-vault` markdown directly. Perplexity reasoning and a follow-up both returned a null answer, so no external research conclusion is attributed to it. No experiment or runbook records were modified.
