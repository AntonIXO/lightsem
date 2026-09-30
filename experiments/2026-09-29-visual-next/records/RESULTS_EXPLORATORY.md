# Visual-next exploratory results

Status: all authorized exploratory computations and 19-entry Shpongle delivery are complete and independently verified. These are measured automatic results, not confirmation of the four original independently annotated hypotheses. Source annotation is optional. Baseline C is unchanged.

## What has been measured

| Study | Actual result | Interpretation |
|---|---|---|
| H1: remove camera-edit events | Across 43 clips, 661 candidate color events became 587; 74 were vetoed. On the 31 historical DEV clips: 400 → 349. | The filter changes the training signal. Without independent labels, the number removed does not measure how many errors were corrected. |
| H2: track person masks | Cutie’s mean color-histogram distance from SAM2 was 0.00987, compared with 0.06181 for held masks, 0.08395 for lagged masks, and 0.11163 for boxes. | Cutie agrees closely with the incumbent color estimates. Agreement is not independent mask accuracy. |
| H3: retain color and layout in the temporal representation | R3 retention ratios were −1.626 for hue and −3.107 for layout, against a required 0.90. | The tested representation/projection/probe pipeline failed qualification substantially. This does not isolate the failure to DINOv2 itself. The fixed downstream fits continue as diagnostic comparisons. |
| H4: allow several fixture colors | Mean image-matching error: O0 0.87168 → O4 0.86429. Relative improvement 0.847%, against a 20% target. | There is a small measured direction of improvement, but the required effect size failed. |

H2’s changed-mask-pixel fraction was 0.01918 for SAM2 and 0.01870 for Cutie. A held mask scored an even smaller 0.00292, illustrating why smoothness alone cannot establish correctness.

For H4, the absolute error reduction was 0.00739, with a paired family bootstrap 95% interval of [0.00438, 0.01051] and a four-test-adjusted sign-flip p-value of 0.000400. The magnitude gate still failed. O4’s advantages over donor and permuted controls were 4.746% and 1.347%; both failed their 10% targets. The original independent-reference endpoint remains unmeasured.

## Downstream models and evaluation

The complete fixed matrix is seven auxiliary representations × three seeds, plus cut-only, mask-only, and combined seed-0 demonstration fits. Every fit uses the same 6,000-update schedule. All seven scientific comparison arms preserve the original main targets; the additional cut/mask/combined recipes deliberately change their specified training targets.

All 24 fits and all 27 evaluations are complete. The primary audit independently recomputed all 837 family scores and controls. On the 29 event-bearing families with sufficient source support, the three-seed mean color-event AP is:

| Representation | Mean AP |
|---|---:|
| R0: fresh Rich96 control | 0.10214 |
| R1: explicit spatial color grid | 0.12040 |
| Rrand: random image encoder control | 0.11021 |
| R2: pretrained image features | 0.10537 |
| R3: temporal image features | 0.10275 |
| R3-shuffle | 0.10538 |
| R3-reverse | 0.10711 |

The primary R3−R0 difference is **+0.000607**, 95% paired-family bootstrap interval **[−0.01181, +0.01404]**, four-test-adjusted p **1.0**. It fails the +0.03 target and every required representation/audio control. On the one eligible no-event family, mean false-cue rates were 0.3155/minute for R0 and 0.6310/minute for R3; this guard also fails and rests on very limited no-event coverage.

R1 has the highest observed mean, but its secondary R1−R0 sign-flip p-value is 0.07779 before correction (0.31117 after the reported four-test correction). Its percentile bootstrap interval is [0.00012, 0.04421]. These are different inferential procedures; the registered significance requirement is not met. Do not label R1 a confirmed winner.

The three untrained audio-model anchors have mean AP 0.10596. Additional single-seed recipe APs are cut-only 0.10371, mask-only 0.09131, and combined 0.11934. Their beat-only controls score 0.11255, 0.11601, and 0.12507 respectively. These descriptive numbers provide no evidence that the added recipes reliably exploit richer audio information.

The combined H3 proxy gate fails. The original representation qualification remains failed; the independently annotated endpoint remains unmeasured.

The full-window endpoint uses the original automatic observer labels. In particular, it cannot independently prove that the modified cut/mask labels are cleaner: those recipes are being compared against an incumbent-derived reference. Their practical lighting behavior is evaluated separately in the Shpongle demos.

The accepted C checkpoint’s previously saved predictions have been rescored on this full-64-second endpoint: macro color-event AP 0.11213; donor-audio AP 0.09948. Beat-only output was not saved for C and remains unmeasured. C is a previously selected descriptive anchor, not a member of the fresh paired-seed matrix. Its older center-window metrics are a different endpoint and remain unchanged.

## Limits that remain attached

- The historical DEV set and Shpongle have already been exposed. No sealed TEST data is used.
- Full-window source support is sufficient in 30 of 31 clips, but qualified circular-shift support is only 27 of the required 30. The earlier 31/40 source-coverage and 29/30 null-coverage failures are also preserved.
- The R3 retention failure is substantial. The diagnostic continuation was recorded prospectively; it does not convert that failure into a pass.
- R3-reverse is a fixed coordinate permutation and retains information. Its result cannot alone identify whether temporal order matters.
- Personal visual preference can select a future product baseline, but does not retrospectively confirm the original statistical hypotheses.

## Shpongle review delivery

The fixed inventory contains 19 displays: C; seven seed-0 representation models; cut-only, mask-only, and combined models; combined with four color groups; C with two/four color groups; and five reference-guided oracle examples. The oracle set is O0, O2, O4, permuted groups, and a donor-reference control.

All 19 entries are delivered at <http://127.0.0.1:8769/>; [direct video links and rating guide](REVIEW.md). All displays use the full 277.3-second track, the same camera, C’s beam movement and dimmer commands, native 1080p30 AV1, and the same decoded AAC soundtrack. Learned and oracle palette adoptions use 0.3-second RGB transitions. Before learned activation, original companion RGB is preserved. Each video must remain below 50 MB.

All 10 actual seed-0 Shpongle inferences and 11 learned display streams are independently verified. Eight models—R0, Rrand, R2, R3, shuffle, reverse, cut-only and mask-only—emit **zero color cues**. Their resulting DMX streams are identical and preserve the original companion colors. Those companion colors themselves vary; the videos do not demonstrate an active learned color contribution.

R1 emits 10 cue opportunities but only **two subsequent palette changes**. The combined model emits three cue opportunities but **zero subsequent palette changes** after its first adoption. For context, C has 262 cue opportunities and 48 subsequent palette changes. These are behavior counts on one exposed demonstration track, not accuracy or preference scores.

All 19 experiment entries and all 16 new filenames are retained. The eight identical inactive outputs share one authenticated encode through seven hard links, saving seven redundant renders. Each shared video has a truthful neutral caption. The complete delivery requires nine new encodes, seven exact-duplicate links, and the three already verified videos; no failed method is hidden or omitted.

All five oracle videos are rendered, independently audited, and visually inspected at common sampled times. They use a separate source-video reference during 110–174 seconds. Outside that interval, the first/last group choices are carried forward while C supplies the palette timeline. These examples show reference-guided color capacity, not learned audio-to-spatial behavior.

## Evidence

- [H1 results and trace bindings](execution/H1_EXPLORATORY_RESULTS_V1.json)
- [H2 results](execution/H2_EXPLORATORY_RESULTS_V1.json) and [primary audit](execution/H2_EXPLORATORY_PRIMARY_AUDIT_V1.json)
- [Retention primary audit](execution/RETENTION_PROBE_PRIMARY_AUDIT_V1.json) and [prospective diagnostic continuation](execution/DIAGNOSTIC_FIT_CONTINUATION_V1.md)
- [H4 primary audit](execution/H4_ORACLE_PRIMARY_AUDIT_V1.json)
- [Fixed training queue](execution/STUDENT_FIT_QUEUE_V1.json)
- [Fixed 19-display inventory](execution/REVIEW_INVENTORY_V1.json)
- [Oracle stream primary audit](execution/ORACLE_DEMO_PRIMARY_AUDIT_V1.json)
- [Current execution state](execution/STATE.json)

Additional completed evidence: [all24 fit audit](execution/STUDENT_FITS_PRIMARY_AUDIT_V1.json), [all27 evaluation audit](execution/EVALUATION_PRIMARY_AUDIT_V1.json), [independent aggregate audit](execution/H3_AGGREGATE_PRIMARY_AUDIT_V1.json), [all learned-stream audit](execution/LEARNED_DEMO_PRIMARY_AUDIT_V1.json), and [exact-output encoding policy](execution/EXACT_OUTPUT_RENDER_REUSE_V1.md).
