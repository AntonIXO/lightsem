# Overnight expression study

Completed September 10: [audited results and delivery links](RESULTS.md). No new configuration passed every acceptance gate; the delivered film uses the declared incumbent-plus-adapter fallback.

This directory preserves the reporting and audit source for the September 9–10 campaign. The core experiment implementation is in `lightsem/experiments/night_*.py`. Main training uses the immutable source snapshot at `/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/source`, with source digest `bc9eab5e730df1c8794ca55e9aa6e66799f1e9ca7f6ddb066bda6b17f8145c77` and implementation commit `40130f5`.

The executable study copies of these helpers are under the run root's `tooling/` directory. Their protocol hashes refer to those paths. The repository copies are retained for code review and reconstruction; the data, weights and generated movies remain outside Git.

The main comparison is Base/Fourier/MuQ × single-entry/repeated-FiLM conditioning × seeds 0/1, with 15,200 scheduled optimizer updates per cell. All final checkpoints use the same 40 development songs and eight rendering excerpts. The earlier Stage 11 TEST remains excluded.

The follow-through protocol fixes the pretrained Beat This! reference, the four-cell MuQ centering study, three pose movies, the full Shpongle comparison, and independent input/checkpoint/score/DMX/video audits before main candidate inference. It adds no training cells and does not change the main candidate-selection rule. The MuQ local-use interpretation operationalizes the plan's requirement to distinguish local information from an excerpt-mean explanation; the centering controls are explicitly described as distribution-shifting diagnostics.

Relevant entry points:

- `supplementary_statistics.py` and `diagnostic_quality_notes.py`: saved-prediction analyses, confidence support, event contexts, donor and persistence controls.
- `periodicity_diagnostics.py`: separate TRAIN Fourier, autocorrelation, cyclic and cross-band phase measurements.
- `independent_audit.py` and `video_audit.py`: scheduled training, artifact joins, independent energy-score calculation, emitted-byte reconstruction, actual encoded video and mirror checks.
- `post_pipeline_runner.py`: serial follow-through after the frozen main pipeline completes.
- `make_outcomes.py`: actual effect sizes and study gates; `scientific_close.py` lets Vibescience compute its own verdicts.
- `make_figures.py` and `make_report.py`: plots, source-linked spreadsheet/CSV data and the research PDF.
- `delivery_check.py`: final artifact hashes, completed stage manifests, PDF citation/page checks and workbook/CSV row counts.

Primary sources are listed in [source_ledger.json](source_ledger.json). Final artifacts are written to `/mnt/lightsem-data/experiments/lightsem/night_expression_20260909/delivery/`. During training, a diagnostic PDF may be marked as a draft; it does not represent final candidate or lighting acceptance.
