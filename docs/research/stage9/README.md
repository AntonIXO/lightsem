# Stage 9 research provenance (PLAN.md rev. 8, section 6.9)

Five Perplexity passes run on 2026-09-04 with the full project brief (`lightsem_brief.md`) attached,
answers kept verbatim so the plan's claims can be traced:

| file | pass | note |
|---|---|---|
| `00_deep_research.md` | deep research (datasets, audio decomposition, models, ranked recommendation) | the one Deep Research call of the month; sources in `00_deep_research_sources.txt` |
| `01_audio_decomposition.md` | reasoning, direction 1 | ranked five feature families with the experiment that kills each |
| `02_datasets_non_game.md` | reasoning, direction 2b | xLights / RPMC-L2 / consoles / lasers / pyro; rhythm-game half (2a) returned an empty answer twice and is covered by the deep-research pass plus a direct BeatSaver API probe recorded in PLAN.md 6.9.1 |
| `03_multimodal_methods.md` | reasoning, direction 3 | factorised critic, two-level planner, what to borrow from dance and lighting models |
| `04_mixture_of_light.md` | reasoning, direction 4 | group-effects decoder vs type experts vs fixture tokens |

Papers read in full for the same pass: Skip-BART (arXiv:2506.01482) and SeqLight (arXiv:2605.03660);
their key claims are in the vault (`/root/lightsem-vault/papers/`). Corrections applied when writing
the plan: RPMC-L2 is CC-BY-4.0 on Hugging Face (the deep pass said no licence was located), and all
699 songs are cached locally as OpenL3 embeddings + scene vector, without waveform.
