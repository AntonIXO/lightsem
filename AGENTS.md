# LightSem experiment and demo checks

The user explicitly requires collapse and input-sensitivity checks before model
comparisons or demo delivery (2026-09-30).

- Before a large render batch, inspect the actual generated outputs: finite
  values, motion range and speed, inactive time, color-cue count, actual palette
  changes, palette occupancy, and exact duplicate output streams.
- Test response to at least two distinct music inputs when judging music
  conditioning. Hold weights, non-audio inputs, random seed/noise, mapping and
  rendering settings fixed. Recompute the audio-dependent path for each song.
  Never put a new soundtrack over reused DMX and call it a new generation.
- Separate motion, color and brightness diagnostics. Color differences do not
  establish different motion. A moving fallback controller does not establish
  that a learned color head activated.
- Distinguish representation collapse, output inactivity, saturation, duplicate
  outputs, and intentionally fixed controls. Report the component that failed.
  Do not claim representation collapse from a zero event count alone.
- Say which components depend on music and which use a shared procedural
  vocabulary or unconditioned motion prior. Shared style is not evidence that
  audio changes generated body motion.
- Do not present failed/inactive or identical outputs as distinct baseline
  improvements. Keep full evidence, but show one representative of exact visual
  duplicates and identify the other runs as diagnostics.
- A user-requested failure diagnostic may still be rendered, clearly labeled.
  Never silently lower thresholds, inject events, change seeds, or replace a
  frozen checkpoint to make a demo look active.
- Store these checks and findings in the project Vibescience vault
  /root/lightsem-vault, not the unrelated default vault. Memory updates are
  already authorized by the user.
