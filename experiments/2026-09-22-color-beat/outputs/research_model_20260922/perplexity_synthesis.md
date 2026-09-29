# Perplexity-method report and constrained synthesis — 2026-09-22

## Status and provenance

The requested `local_mcp.perplexity.perplexity_research` was called after reading its skill and discovering its schema. It DID NOT answer: the service reported **Deep Research has 0 queries remaining**. A clearly labelled `perplexity_ask` fallback also failed: **Pro Search has 0 queries remaining**. No Deep Research synthesis exists; neither failure may be represented as completed research.

Raw `perplexity_search` DOES work. Narrow searches were made only for LightFormer and RPMC. Those raw results, not a model-generated research answer, are stored in `perplexity_search_raw.json` and `perplexity_lightformer_searches.json`. The parent independently verifies the other originals and asked this worker to stop broad search.

## LightFormer exact-publication verification

Queries: `"LightFormer" lighting music publication`, `"LightFormer" "music"`, `"LightFormer" "RPMC"`.

**A music-to-stage-light publication named LightFormer remains unverified.** Do not cite it as the source for RPMC, a stage-light model, or a dataset. The search result alone does not prove no such publication exists.

The exact-name results instead include unrelated originals:
- *LightFormer: Light-weight Transformer Using SVD-based Weight Transfer and Parameter Sharing*, Lv et al., Findings of ACL 2023: https://aclanthology.org/2023.findings-acl.656/ (NLP compression).
- *LightFormer: An End-to-End Model for Intersection Right-of-Way Recognition Using Traffic Light Signals and an Attention Mechanism*: https://arxiv.org/abs/2307.07196 (traffic signals).
- *LightFormer: A lightweight and efficient decoder for remote sensing image segmentation*: https://arxiv.org/abs/2504.10834 (remote-sensing segmentation).
These are search-result identity checks, not a full-text review. None is evidence for music-to-stage-light generation.

## RPMC narrow-search lead

A direct original-paper lead surfaced: **Stage Light is Sequence2: Multi-Light Control via Imitation Learning**, https://arxiv.org/html/2605.03660 . The raw search snippet says it uses RPMC-L² recordings from 35 live performances, December 2020–July 2024, with 699 valid samples after discarding videos shorter than 20 seconds, covering rock/punk/metal/core. This worker has NOT opened or verified that full text. Parent notified for original-source verification. Treat the numbers as unverified search snippets, not accepted claims. Official dataset card: https://huggingface.co/datasets/RS2002/RPMC-L2 . Parent already independently inspected that card.

## Recommendation constrained by local evidence (reasoned synthesis, not Perplexity answer)

Prioritize **identifiable, genuine paired audio–light supervision with judged cue/scene alternatives**, not another generic hierarchical planner. First ask whether a target contains reproducible song-specific cue timing and whether alternatives can be judged against the incumbent. Use dance/motion supervision only as an auxiliary source of spatial/structural representation unless it improves emitted light in a held-out controlled comparison. A conditional distribution is appropriate for artistic alternatives, but stochasticity by itself does not establish audio dependence.

Keep **palette-change timing** separate from **exact hue selection**. Learned beats/downbeats and music sections can support candidate change times; they do not identify a hue. Emotion-related music–color associations support at most population- and context-conditioned color preferences, not a unique physical color target. Dance-video clothing/background colors and edited color grading are not recorded stage-light outputs and are not valid substitutes for paired light labels.

Local evidence supplied by parent must constrain any next proposal:
- The within-song group regressor collapsed; the sampled decoder diverged.
- Bar planner controls had R² approximately zero and worsened beat alignment.
- Cue-held sampling improved over per-frame sampling but failed the shuffled-donor null; this supports smoother statistics, not song-specific control.
- A cue hazard that improved with true past cues failed on its own emitted cues; teacher-forced gains are insufficient.
- Motion structure segmentation showed signal (F1 .462 versus .331 null), but depended on motion phase. It is incorrect to conclude that dance carries no structure.

A decisive comparison therefore needs the actual incumbent (lowpass motion sweeps plus deterministic 2/4/8-beat trajectories), audio-only and fixed/metronomic timing, shuffled/circularly shifted conditioning, held-out song/operator/style separation, and human preference judgments with brightness/activity matched. Evaluate on self-emitted histories. Palette-change timing and palette identity need separate outcomes. These are methodological recommendations, not claims validated by the retrieved papers.

## Files

- `perplexity_query.txt`: exact Deep Research prompt.
- `perplexity_schema.json`: discovered Deep Research schema.
- `perplexity_raw.json`: explicit failed-call status and error, not invented response.
- `perplexity_run.log`: raw server output / Deep Research quota error.
- `perplexity_fallback_query.txt`: exact ordinary-answer fallback prompt.
- `perplexity_fallback_run.log`: raw server output / Pro Search quota error.
- `perplexity_search_raw.json`: raw initial narrow search result.
- `perplexity_lightformer_searches.json`: raw narrow LightFormer/RPMC searches.
- `perplexity_synthesis.md`: this report.

No training, production-code changes, or vault operations were performed. Runner scripts are research orchestration artifacts only.
