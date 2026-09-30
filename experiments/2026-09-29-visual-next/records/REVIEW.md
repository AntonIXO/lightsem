# Shpongle experiment review

All 19 entries are ready: [open the comparison and rating page](http://127.0.0.1:8769/). Files below also open directly from this workspace.

Every video is 4:37, native 1080p30 AV1 with the same AAC soundtrack, and below 50 MB. C is still the baseline. Rate music fit and appearance separately; switching on the page keeps the same track position.

**Eight experiments produced exactly the same video.** R0, H1, H2, R2, R3, Rrand, R3-S and R3-R emitted no learned color cues. Their original companion colors still vary. Watch one of these to see their shared result; all filenames are retained.

| Version | Change | Actual Shpongle behavior | Video |
|---|---|---|---|
| C | Accepted baseline C | C model:262 cue opportunities,48 subsequent palette changes | [Open 43.6 MB](/root/lightsem/experiments/2026-09-29-baseline/outputs/shpongle_baseline_ab_v2/media/C/C_full_1080p_av1.mp4) |
| R0 | Fresh Rich96 control | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/02/R0_full_1080p_av1.mp4) |
| H1 | Camera-cut filtering | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/03/H1_full_1080p_av1.mp4) |
| H2 | Tracked person masks | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/04/H2_full_1080p_av1.mp4) |
| R1 | Spatial color grid | 10 cue opportunities; 2 subsequent palette changes | [Open 42.6 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/05/R1_full_1080p_av1.mp4) |
| R2 | Image-model features | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/06/R2_full_1080p_av1.mp4) |
| R3 | Image features through time | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/07/R3_full_1080p_av1.mp4) |
| ALL | Combined training changes | 3 cue opportunities; 0 subsequent palette changes | [Open 42.6 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/08/ALL_full_1080p_av1.mp4) |
| ALL4 | Combined + four color groups | 3 cue opportunities; 0 subsequent palette changes | [Open 42.6 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/09/ALL4_full_1080p_av1.mp4) |
| P2 | C with two color groups | C model:262 cue opportunities,48 subsequent palette changes | [Open 43.6 MB](/root/lightsem/runs/visual_next_20260929/shpongle_palette_previews_v1/media/P2/P2_full_1080p_av1.mp4) |
| P4 | C with four color groups | C model:262 cue opportunities,48 subsequent palette changes | [Open 43.5 MB](/root/lightsem/runs/visual_next_20260929/shpongle_palette_previews_v1/media/P4/P4_full_1080p_av1.mp4) |
| Rrand | Random image-feature control | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/12/Rrand_full_1080p_av1.mp4) |
| R3-S | Shuffled-history control | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/13/R3-S_full_1080p_av1.mp4) |
| R3-R | Reversed-coordinate control | No learned color activation; same pixels as R0 | [Open 42.4 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/14/R3-R_full_1080p_av1.mp4) |
| O0 | Reference-guided global color | Uses a separate reference; optimized only at1:50–2:54 | [Open 43.0 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/15/O0_full_1080p_av1.mp4) |
| O2 | Reference-guided two groups | Uses a separate reference; optimized only at1:50–2:54 | [Open 43.0 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/16/O2_full_1080p_av1.mp4) |
| O4 | Reference-guided four groups | Uses a separate reference; optimized only at1:50–2:54 | [Open 43.0 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/17/O4_full_1080p_av1.mp4) |
| O4-P | Permuted-group control | Uses a separate reference; optimized only at1:50–2:54 | [Open 43.0 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/18/O4-P_full_1080p_av1.mp4) |
| O4-D | Different-reference control | Uses a separate reference; optimized only at1:50–2:54 | [Open 43.0 MB](/root/lightsem/runs/visual_next_20260929/shpongle_full_review_v1/media/19/O4-D_full_1080p_av1.mp4) |

Cue opportunities do not necessarily change the palette. R1 has only two subsequent changes; ALL/ALL4 adopt a palette and do not subsequently change its ID. Eight inactive recipes share an authenticated encode through hard links, with a neutral caption.

For the reference-guided examples, compare O0/O2/O4 mainly around1:50–2:54. The first/last group assignments are carried outside that window while C continues supplying palette IDs and timing. [Open the separate silent source clip](/root/lightsem/runs/visual_next_20260929/annotations/jd-669b916cd15af69ae82c/source.mp4). These examples measure reference-guided color capacity, not learned audio-to-spatial allocation.

The temporal image-feature method did not meet its improvement/control targets. The spatial color grid had the highest observed AP, without a confirmed statistical win. [Full measured results](RESULTS_EXPLORATORY.md).

Integrity checks passed for all 19 media routes, byte-range seeking, downloads, frame/PTS/audio/size metadata and source-stream identities. All nine new unique soundtracks were independently redecoded. Browser playback/rating interactions were not automated; the page also provides downloads and JSON rating export.
