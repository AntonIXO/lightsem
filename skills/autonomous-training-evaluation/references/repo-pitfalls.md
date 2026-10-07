# LightSem-specific pitfalls

The repository instructions in [`AGENTS.md`](../../../AGENTS.md) are
authoritative. The current native baseline is repaired C/HF16 seed 0 from
`configs/dance/fixed_c.json`; P2 color and restored historical LUFS brightness
remain fixed, and fixture speed caps are off in simulation. Frozen MuQ is
required. The full-data seed-1 branch is diagnostic only, and the selected
checkpoint predates the quiet-loss gradient correction.

For native-dynamics studies, reuse the paired estimands and raw-control
contracts in `lightsem/experiments/native_dynamics/evaluation.py` and
`scripts/motion/audit_dynamic_dance_training.py`. The lightweight evaluator in
this skill is the common safety/provenance front door; it does not replace
those domain-specific bootstrap, support, foot-stability, or masked-observation
audits.

For every current demo use both Shpongle and AURORA — Cure For Me. Preserve
source-body provenance and actual emitted DMX. Never substitute sine waves, beat
oscillators, canned sweeps, random motion, or a moving fallback controller for
failed transfer. Different hashes, nonzero motion, more stops, RGB changes, or
active lamps do not establish choreography.

Keep representation collapse, output inactivity, saturation, duplicate streams,
and fixed controls as different findings. Separate motion, color, and
brightness. Inspect raw native bodies before interpolation or rendering. Do not
claim that D4/D6, post-hoc smoothing, retiming, or re-composition repaired
training.

Preserve failed artifacts, verdicts, duplicate groups, quarantines, and cache
dependencies. Use the repository wrappers and existing GPU/data locks; keep
`/mnt/lightsem-data` above its 20 GiB free-space floor. New reports belong in a
new artifact root and must not overwrite sealed receipts.
