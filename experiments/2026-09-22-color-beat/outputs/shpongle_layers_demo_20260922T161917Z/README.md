# Shpongle — experimental learned-light preview

- **20-second preview, track70–90s:** `shpongle_experimental_preview_70_90_1080p_av1.mp4` (complete; software AV/timestamp audit passed).
- **Full277.3s:** `shpongle_experimental_full_1080p_av1.mp4` (complete; all8319frames, continuous timestamps, and full software AV decode passed).
- Source: `/root/shpongle.flac`. Native1920×1080,30fps,AV1 + AAC source audio.
- Fixed primary Rich96 checkpoint, seed0. No training or seed/threshold selection on this song.
- Accepted companion pan/tilt and all469 protected DMX slots are byte-exact. No dance retiming or regeneration.

## What the model actually emitted

The unchanged threshold produced52 color-cue opportunities, but only **5 token changes after first adopting the model palette at36.3s**. Repeated selection of the same palette is not a visible color change. It produced one extra brightness accent, at263.7s. Existing companion brightness and rhythmic movement remain underneath this overlay. No extra cues were invented to make the demo look busier.

A fixed global HS-palette-to-RGB visualization applies the same palette summary to color-enabled fixtures. It is not learned fixture assignment, calibrated light output, or a validated aesthetic improvement. The historical-DEV observer study remains unqualified; this demo does not repair its failed gates. See `INFERENCE_HANDOFF.md`, `DEMO_PLAN.json`, and `/mnt/lightsem-data/experiments/lightsem/shpongle_layers_demo_20260922T161917Z/readback_receipt.json`.

The palette-history secondary test is complete. For the primary selected checkpoints, temporal error was 0.859 with history versus 0.831 for independent cue-held choices (3.4% worse). The final-checkpoint companion result did not meet the benefit criteria either. No model or default was promoted. This stochastic observer test is not a quality evaluation of this argmax demo. The report discloses the V1 vault-order deviation: see `docs/research/2026-09-22-color-beat/PALETTE_SECONDARY_RESULTS_V1.md` in the repository. No completed fits or event scores were rerun.
