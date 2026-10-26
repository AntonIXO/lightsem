# LightSem causal feature map

This map records the executed one-seed evidence for the old native dance and
lighting networks. Cached-only results retain their waveform hashes and are not
full-song qualification. Raw safety and collapse gates were checked before any
downstream interpretation.

## Native dance

| Input intervention | Shpongle | Cure For Me | Interpretation |
| --- | ---: | ---: | --- |
| Zero MuQ | 0.01139 m root-relative RMS; standardized 0.03788 | 0.03371 m; 0.11064 | Mechanically changes the body; misses the preregistered per-song scale on Shpongle. Cached-only powered null at that scale. |
| Zero Base92 | 0.02580 m; 0.08584 | 0.05845 m; 0.19184 | Largest whole-family cached body effect; raw ratios remain below 2x. Full waveform/phrase qualification missing. |
| Zero all source descriptors | 0.000887 m | 0.001283 m | Weak, inside the measured cached noise floor. |
| Zero drums/bass/other/vocals separately | 0.000003–0.000697 m | 0.000400–0.000825 m | No individual source family reaches the 0.01 m screen MDE. |
| Zero mixture/timing/section groups | max 0.00293 m | max 0.00648 m | Timing block is mechanically present but below the 0.01 m / 0.05 standardized cached screen. |
| Opposite-song cached donor | 0.03767 m; standardized 0.1253 | 0.03767 m; 0.1236 | Strong audio-identity counterfactual beyond 15 s circular shifts (0.0372/0.0859), but Shpongle←Cure fails raw safety: speed 1.575x, jerk 2.033x. Directional-only. |
| Alpha=0.50 donor interpolation | 0.02146 m; standardized 0.07138 | 0.01914 m; 0.06282 | Both directions raw-safe; Shpongle exceeds its shift control, Cure remains below its shift control 0.08591. Directional-only. |
| Alpha=0.75 donor interpolation | 0.03139 m; standardized 0.10441 | 0.02927 m; 0.09605 | Both directions raw-safe and both exceed their circular-shift controls. Strong cached audio-identity evidence; waveform/phrase qualification remains missing. |
| Alpha=.50 donor interpolation | 0.02146 m; standardized 0.07138 | 0.01914 m; 0.06282 | Safe bounded audio-identity effect in both songs; all raw speed/acceleration/jerk ratios stay below 2x. Cached-only. |

All cached dance matrices used seed 7001, exact repeat forwards, rank 139 and
finite native bodies. The measured cached noise floor is root-relative RMS
σ = **0.003514 m** and standardized-effect σ ≈ **0.01157**.

## Body-to-light and lighting network

| Input intervention | Decoded output | Result |
| --- | --- | --- |
| Root-relative left-arm body patch | Angular DMX | Assigned fixtures 3.91° RMS versus 1.88° unassigned, ratio 2.08x, both songs. Mechanical mapper retention only. |
| Isolated LightLayers audio groups at palette seam | Palette state | Shpongle: mel 0.567, energy 0.402, timbre 0.013, rhythm 0.834. Cure For Me: mel 0.334, energy 0.992, timbre 0.663, rhythm 0.987. |
| Same audio masks through fixed P2 RGB delivery | Decoded RGB | Shpongle max 0.2694 8-bit RMS; Cure For Me max 0.4110. |
| Same audio masks through fixed P2 delivery | Emitted luma | Shpongle 0.00628–0.01603; Cure For Me 0.00016–0.00315. Luma is dimmer × max(RGB). |
| Same audio masks with body/mapper fixed | Angular DMX / dimmer bytes | Exactly zero effect. Audio masks do not create pan/tilt motion in this fixed-body path. |

Every lighting arm was finite, exact duplicate fixture rate was **0.0**, and
selected-body raw speed/acceleration/jerk receipts passed. These are cached
palette-to-RGB/luma endpoint findings, not learned full-waveform audio-to-DMX
qualification.

## Gates and remaining evidence

- Passed: finite values, deterministic repeats, rank/collapse checks, raw-body
  safety for accepted cached arms, two-song cached coverage, RGB/luma endpoint
  separation, and exact duplicate-stream checks.
- Failed or incomplete: authenticated raw waveform recomputation, full-song
  phrase timing, matched retain/remove plus donor/time-shift controls on original
  waveforms, and safe two-direction donor generation.
- The original waveform identities are still absent: Shpongle expects SHA256
  `95e438…`; Cure For Me expects `5363ce…`. Legacy `.m4a` and archived WAV files
  have different hashes and are rejected.

Authoritative detailed memo: [RESEARCH_MEMO.md](./RESEARCH_MEMO.md). The active
full-waveform runbook is recorded in Vibescience as
`runbook-light-audio-group-causality-20261028`.
