# Independent observer-score audit — read-only

Both completed `observer_selected_scores_roundoff_v2.json` and `observer_final_scores_roundoff_v2.json` were opened only after the parent confirmed native exit0. This audit reads stored outputs and checks their inventory/arithmetic/provenance. It does not rerun inference, AP, bootstrap scoring, or model selection. No frozen file, GPU state, or vault was modified.

## Conclusion

**Both reports correctly say `scored_interpretation_blocked`. The registered observer-effect conclusion is undefined (`null`), not a scientific refutation and not success.** No inconsistency was found in the checks below. The primary color point estimate misses its engineering minimum in each role; the failed null prerequisite independently blocks a qualified conclusion. Keep these statements separate.

## Support and qualification

- Usable DEV:31/31 enrolled; no capture-unavailable DEV groups. TRAIN:91 groups/91 recordings. Original cohort coverage remains31/40=.775: the original80% gate still fails.
- Primary support:30 common groups qualify at>=.8 coverage. Event-bearing groups are14 accent and17 color, exceeding the minimum10/head. Both primary support gates pass. AP is conditional on these14/17 event-bearing groups, not an AP over31 positive songs.
- `8UTvmHbgc5w` (`jd-d79d29aa3c0f14c7e4f8`) has4.7/8s=.5875 primary coverage in both heads and no reference events. It fails primary coverage. Its nonevent rates remain in the declared all-song/nonevent summaries.
- Null common support:29 groups, below required30. Event-bearing groups are13 accent/16 color. Both null gates fail due to common-group count, not insufficient event-bearing count.
- The extra null loss is `jkGH7JIHDSk` (`jd-e1da76a76ac38851fb95`). Primary support7.293/8s=.911625 passes. At40Hz both segments admit20 shifts. At10Hz the2.488s segment `[112.312,114.8)` is too short to admit20 unique offsets; retained4.805/7.293=.658850953 is below.8. The shared-head null population therefore excludes the song from both heads. This follows the frozen rule; do not weaken it after seeing results.
- All controls are present. All15 recipe/seed cells occur separately in each role. No role pooling, missing-cell imputation, or zero-AP substitution for undefined events was found.

## Descriptive AP, named comparators

Each row uses the same primary reference/support population:14 accent or17 color event-bearing songs. Seeds are averaged within songs. Baselines are identical across roles/recipes. Constant prevalence yields no candidate peaks, so its AP is0 rather than a useful learned timing baseline.

| Role | Head | Recipe | Model | Donor | Constant | Onset | Audio novelty | Logistic |
|---|---|---|---:|---:|---:|---:|---:|---:|
| selected | accent | S_none_P0 | 0.343294 | 0.374989 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| selected | accent | S_V6_P0 | 0.340917 | 0.387068 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| selected | accent | S_VR_P0 | 0.348218 | 0.382336 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| selected | accent | I_VR_P0 | 0.364996 | 0.386232 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| selected | accent | S_VR_P0_Ashuf | 0.310238 | 0.310238 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| selected | color | S_none_P0 | 0.300585 | 0.255905 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| selected | color | S_V6_P0 | 0.312085 | 0.261768 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| selected | color | S_VR_P0 | 0.297155 | 0.258255 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| selected | color | I_VR_P0 | 0.274437 | 0.202205 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| selected | color | S_VR_P0_Ashuf | 0.220517 | 0.220517 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| final | accent | S_none_P0 | 0.260327 | 0.335401 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| final | accent | S_V6_P0 | 0.268166 | 0.305651 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| final | accent | S_VR_P0 | 0.281931 | 0.321214 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| final | accent | I_VR_P0 | 0.309825 | 0.324554 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| final | accent | S_VR_P0_Ashuf | 0.297996 | 0.297996 | 0.000000 | 0.211531 | 0.242553 | 0.298806 |
| final | color | S_none_P0 | 0.247541 | 0.224761 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| final | color | S_V6_P0 | 0.320417 | 0.206556 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| final | color | S_VR_P0 | 0.290440 | 0.230155 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| final | color | I_VR_P0 | 0.283656 | 0.208557 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |
| final | color | S_VR_P0_Ashuf | 0.232057 | 0.232057 | 0.000000 | 0.197534 | 0.280549 | 0.329439 |

Rich shared color does not beat V6 or logistic in either role: selected `.297155` versus `.312085` / `.329439`; final `.290440` versus `.320417` / `.329439`. Selected rich accent `.348218` exceeds logistic `.298806`, but is below its donor control `.382336`; final rich accent `.281931` is below logistic `.298806` and donor `.321214`. These are descriptive comparisons, not newly tested contrasts.

## Fixed deployment false cues/minute

These outputs use frozen thresholds, not an optimized operating point. Lower false-cue rate alone does not establish recall or lighting quality. `All` uses all31 usable songs; nonevent uses17 accent/14 color no-reference songs, including the low-coverage nonevent song.

| Role | Head | Recipe | Model all | Model nonevent |
|---|---|---|---:|---:|
| selected | accent | S_none_P0 | 44.886884 | 27.947983 |
| selected | accent | S_V6_P0 | 44.644948 | 27.947983 |
| selected | accent | S_VR_P0 | 44.637131 | 28.095042 |
| selected | accent | I_VR_P0 | 38.851730 | 23.294743 |
| selected | accent | S_VR_P0_Ashuf | 45.080095 | 47.289650 |
| selected | color | S_none_P0 | 44.588899 | 40.189254 |
| selected | color | S_V6_P0 | 45.740647 | 42.182007 |
| selected | color | S_VR_P0 | 45.235081 | 41.082111 |
| selected | color | I_VR_P0 | 56.441479 | 52.141756 |
| selected | color | S_VR_P0_Ashuf | 29.015852 | 19.053951 |
| final | accent | S_none_P0 | 3.012646 | 2.846590 |
| final | accent | S_V6_P0 | 3.004545 | 2.229327 |
| final | accent | S_VR_P0 | 3.562264 | 3.995894 |
| final | accent | I_VR_P0 | 2.903226 | 2.500000 |
| final | accent | S_VR_P0_Ashuf | 2.178440 | 2.207744 |
| final | color | S_none_P0 | 2.397827 | 1.857903 |
| final | color | S_V6_P0 | 1.379806 | 0.714286 |
| final | color | S_VR_P0 | 2.020212 | 1.607143 |
| final | color | I_VR_P0 | 0.888118 | 0.535714 |
| final | color | S_VR_P0_Ashuf | 2.854201 | 1.607143 |

| Head | Frozen baseline | All false/min | Nonevent false/min |
|---|---|---:|---:|
| accent | prevalence | 0.000000 | 0.000000 |
| accent | audio_onset | 68.797611 | 94.144428 |
| accent | audio_novelty | 32.318480 | 46.096813 |
| accent | logistic | 129.541835 | 138.835863 |
| color | prevalence | 0.000000 | 0.000000 |
| color | audio_onset | 41.192977 | 52.487941 |
| color | audio_novelty | 25.511646 | 26.449385 |
| color | logistic | 116.712035 | 113.678472 |

## Paired circular-shift null (descriptive only)

Real and20-shift means below share exactly the same restricted support within each pair. This is NOT the primary AP support: null means use13 accent/16 color event-bearing songs after common29-group restriction. The20 shifts are not20 independent replicates. Their inference gate fails, and these comparisons are not part of Holm8. Do not compare these restricted null values to primary means as if support were identical.

| Role | Head | Recipe | Restricted real | Mean20 shifts |
|---|---|---|---:|---:|
| selected | accent | S_none_P0 | 0.339787 | 0.260122 |
| selected | accent | S_V6_P0 | 0.337227 | 0.260225 |
| selected | accent | S_VR_P0 | 0.340816 | 0.261031 |
| selected | accent | I_VR_P0 | 0.316149 | 0.265516 |
| selected | accent | S_VR_P0_Ashuf | 0.295641 | 0.259378 |
| selected | color | S_none_P0 | 0.312290 | 0.229599 |
| selected | color | S_V6_P0 | 0.324804 | 0.229405 |
| selected | color | S_VR_P0 | 0.308709 | 0.229878 |
| selected | color | I_VR_P0 | 0.281973 | 0.237305 |
| selected | color | S_VR_P0_Ashuf | 0.226832 | 0.240020 |
| final | accent | S_none_P0 | 0.273300 | 0.268983 |
| final | accent | S_V6_P0 | 0.281377 | 0.263001 |
| final | accent | S_VR_P0 | 0.299390 | 0.264395 |
| final | accent | I_VR_P0 | 0.321982 | 0.267610 |
| final | accent | S_VR_P0_Ashuf | 0.312838 | 0.263122 |
| final | color | S_none_P0 | 0.255130 | 0.240715 |
| final | color | S_V6_P0 | 0.328729 | 0.232410 |
| final | color | S_VR_P0 | 0.296687 | 0.240960 |
| final | color | I_VR_P0 | 0.292065 | 0.236795 |
| final | color | S_VR_P0_Ashuf | 0.240809 | 0.234255 |

## All eight frozen contrasts, separately by role

The primary support gate passed, so the frozen implementation reports all8 contrasts even though null qualification blocks interpretation. No contrast was manufactured. CIs are unadjusted95% percentile CIs from10000 canonical-song bootstrap draws; Holm corrects only the8 p-values within each role. All16 role-specific CIs contain0. Selected and final are not extra seeds or independent confirmations.

| Role | Contrast | AP difference | Unadjusted95% CI | Raw p | Holm8 p |
|---|---|---:|---|---:|---:|
| selected | S-I_accent | -0.016778 | [-0.114909, +0.054750] | 0.711029 | 1.000000 |
| selected | S-I_color | +0.022718 | [-0.026418, +0.081000] | 0.415958 | 1.000000 |
| selected | V6-none_accent | -0.002377 | [-0.009974, +0.004757] | 0.524648 | 1.000000 |
| selected | V6-none_color | +0.011500 | [-0.003461, +0.036293] | 0.338366 | 1.000000 |
| selected | VR-V6_accent | +0.007301 | [-0.001198, +0.017504] | 0.131287 | 1.000000 |
| selected | VR-V6_color | -0.014930 | [-0.040685, +0.001195] | 0.159184 | 1.000000 |
| selected | real-Ashuf_accent | +0.037980 | [-0.104934, +0.207513] | 0.643736 | 1.000000 |
| selected | real-Ashuf_color | +0.076638 | [-0.028783, +0.204407] | 0.199980 | 1.000000 |
| final | S-I_accent | -0.027894 | [-0.074864, +0.016612] | 0.239476 | 1.000000 |
| final | S-I_color | +0.006784 | [-0.101047, +0.117619] | 0.905109 | 1.000000 |
| final | V6-none_accent | +0.007839 | [-0.016301, +0.030467] | 0.517448 | 1.000000 |
| final | V6-none_color | +0.072876 | [-0.011470, +0.167739] | 0.113389 | 0.907109 |
| final | VR-V6_accent | +0.013765 | [-0.027580, +0.059218] | 0.528747 | 1.000000 |
| final | VR-V6_color | -0.029977 | [-0.112966, +0.047621] | 0.468953 | 1.000000 |
| final | real-Ashuf_accent | -0.016065 | [-0.189793, +0.136120] | 0.819718 | 1.000000 |
| final | real-Ashuf_color | +0.058383 | [-0.042003, +0.152814] | 0.246975 | 1.000000 |

## Exact registered conjunction

For **each role**, support/controls/inventory/statistics prerequisites pass but null qualification fails. Consequently `qualified_pass` is undefined, and `primary_gain_pass` is also undefined under the implementation’s interpretation requirement. Individual available components are:

- Color VR−V6 gain>=.01: **false** (selected−.014930; final−.029977).
- Color paired95% CI lower>0: **false** in both roles.
- Color Holm8 p<.05: **false** in both roles (p=1).
- Accent VR−V6 gain>=−.01: **true** (+.007301 selected; +.013765 final).
Neither a favorable accent point estimate nor a different contrast can replace this conjunction.

## Skepticism about the best-looking results

- Selected real−Ashuf color is+.076638, but its CI is[−.028783,+.204407] and Holm p=1. Two existing song differences are+.706999 and+.638889; the average is not a consistent guaranteed gain. No post-hoc exclusion or alternate bootstrap was run.
- Final V6−none color is+.072876, but CI[−.011470,+.167739], Holm p=.907109. Final V6 color AP.320417 still trails logistic.329439. Do not promote this companion result over the selected primary or relabel it as the registered rich-vs-V6 gain.
- Donor controls replace nonretained columns0..85 while keeping86..93. Mel/onset/flux may also carry rhythm. This is not a proof that all rhythm is held constant or that all audio dependence is isolated.

## Repair/provenance and consistency audit

- Selected803 / final858 candidate-draw values repaired across930 audited record/head cells each. Every stored repair moves exactly one inward binary64 step. Maximum displacement2.842170943040401e−14s,1ULP.
- Correction audit role/cell inventory exactly matches each report. Aggregate and per-draw repair counts agree. No original source mutation found; all8 bound original implementation files match.
- The correction protocol, scorer source, per-role correction registration, and CPU-test receipt hashes match the saved references. Report metadata explicitly distinguishes the amended scorer from original raw-prediction implementation bindings.
- Verified:930=5recipes×3seeds×31records×2heads in each role; every6-method score has identical reference/support metadata; each available20-draw null preserves count and support; stored macros equal stored per-song means; differences agree with comparator means and per-song differences; Holm arithmetic matches the full family.
- Selected/final target hashes, support, shift plans, and baseline score objects are identical. Roles remain separate. `companion_checkpoint_role_still_required` is an artifact-level reminder; both companion files now exist.

## Scope

These are historical-DEV conditional development estimates. Selected checkpoints use the same DEV common-main-loss selection data. The final companion is not fresh confirmation. Targets are SAM2-filtered authored-video observer events with unknown cuts, mask errors, UI and track nuisance included. The original coverage gate is still failed. There is no generalization, pure LIGHT-quality, segmentation, fine-beat, human-preference, or palette-quality acceptance claim.

## Input report hashes

- `/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z/observer_selected_scores_roundoff_v2.json` — `3268b61e9cc80745586077bd593f1f61ef8978d6665959c56f3d3487d3d77e10`
- `/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z/observer_final_scores_roundoff_v2.json` — `a82b4b09490b4a48658806a016d4bb4627cfc8cfc1d72a6edcf63a3f79519568`
