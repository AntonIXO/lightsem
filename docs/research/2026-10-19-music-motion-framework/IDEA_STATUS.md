# Article idea status for the foundation-model audit

| Idea/source | Status | Decision |
|---|---|---|
| AI Choreographer BAS/BeatAlign | Implemented in `synchrony_report` | Keep, but only with metronome and circular-shift nulls. |
| Circular phase concentration | Implemented | Keep as timing diagnostic, not choreography proof. |
| Beat-bin accent/kinetic coupling | Implemented | Keep as simple musical evidence. |
| Within-bar profile | Implemented | Keep; it rejects some BPM-only controls. |
| Beat-synchronous phrase recurrence/novelty | Implemented in `phrase_structure_alignment` | Keep as secondary advanced gate; synthetic accent control exceeded null-z≈2.4, metronome was weaker. |
| MotionBeat beat-jitter negatives | Implemented as registered control | Keep for real-song counterfactuals. |
| MotionBeat Soft-DTW/OT loss | Not implemented as a training loss | Do not add to foundation training until the evaluation stack passes; current recurrence proxy is safer. |
| MDSC music-dance style encoder | Not implemented | Defer until a frozen encoder, code/checkpoint and song-disjoint calibration set exist. |
| Actor reference distribution | Implemented in `actor_calibration` | Keep as nearest-neighbour/real-floor calibration; never use framewise actor imitation loss. |
| Conditional mutual information | Not implemented as strict CMI | Use blocked predictive gain as an explicitly labelled proxy; scalar proxy remains secondary after synthetic instability. |
| MuQ hidden-layer comparison | Implemented as `feature_family_predictive_gain` | Run Base92, MuQ layer 9 and other layers on identical song folds; synthetic layer-like test did not justify assuming deeper is better. |
| Latent effective rank / projector bypass | Implemented | Keep native latent and learned projector as separate reports. |
| FID/diversity/physical motion | Existing health/diversity audits | Keep for safety/distribution checks; they do not establish music conditioning. |
| Body→decoded-DMX retention | Existing downstream tools, not wired into MMCA CLI | Required before delivery, but not a foundation musicality gate. |

The benchmark receipt is [METRIC_BENCHMARK_V5.json](METRIC_BENCHMARK_V5.json).
