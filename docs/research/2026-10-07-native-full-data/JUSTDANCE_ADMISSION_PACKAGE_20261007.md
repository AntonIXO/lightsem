# Just Dance admission package v2

The package is a read-only curation result. It does not train, change the selected C/HF16 baseline, move protected groups, open TEST payloads, or overwrite historical artifacts.

Artifact root: [native_admission_package_20261007_v2](</mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2/README.md>).

The strict manifest contains **245 coach tracks, 181 recordings, and 153 canonical TRAIN groups**. It requires the finite-to-finite jump fix, all existing pose/audio gates, at least one masked 16-second window, and whole-record coverage of at least 80%. Its attached track duration is 930.58 minutes; one-best-coach group coverage is 583.58 minutes.

The maximum masked-window manifest contains **567 duplicate-safe coach tracks, 354 recordings, and 290 canonical TRAIN groups**. It retains native timestamp and mask contracts and does not forward-fill missing detections. Its attached track duration is 2,190.78 minutes; one-best-coach group coverage is 1,116.05 minutes. **322 tracks are window-only** and remain explicitly unqualified because they fail the whole-record gate.

The protected manifest contains **104 groups and 142 recording metadata rows**. It includes the 80 Stage11 VAL/TEST groups and 24 existing native VAL groups. The new leakage receipt reports no TRAIN/protected or TRAIN/TEST overlap, and TEST pose payloads were not opened during this package build.

The slow-target report is TRAIN-only. It reports per-region valid frames, observed quiet frames/seconds, complete one- and two-second bouts, censored bouts, track-weighted fractions, and canonical-group macro fractions. The quiet proxy is the existing projected-speed threshold after the 9-tap observation frontend; it is not a semantic rest label and cannot establish learned pauses.

Strict TRAIN has no missing pinned MuQ stream. The maximum manifest needs 13 recording streams. The verified cache receipt covers all existing streams, rejects the aggregate training matrix as a substitute, and records model/revision/layer/context/waveform identity. Missing extraction is **storage-blocked**: current `/mnt` free space is at 20 GiB and the output would cross the required floor. The exact next command, to run only after a storage gate passes, is in `COMMANDS.log`.

The package includes identity reconciliation, duplicate members and edges, coach IDs, source/pose/audio hashes, cross-corpus non-link policy, exclusion/coverage receipts, visual evidence, group-balanced sampling preview, and the exact command log. Thirty focused tests pass.
