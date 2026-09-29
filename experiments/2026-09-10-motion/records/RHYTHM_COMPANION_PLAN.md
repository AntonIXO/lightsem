# Rhythm-led companion lighting

User-authorized implementation. Dance remains a diagnostic, not the product.

```json
{
  "id": "rhythm-companion-20260912t193657z",
  "artifact_directory": "/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z",
  "source_world": "/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/mapped/shpongle_prior_flow0/full_world.npz",
  "source_dmx": "/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/mapped/shpongle_prior_flow0/full_dmx.npz",
  "loudness": "/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/data/shpongle/loudness.npz",
  "audio": "/root/shpongle.flac",
  "duration_s": 277.3,
  "frames": 8319,
  "source_arm": "fixed prior_native_flow0; saved body unchanged",
  "baseline": "pose_stage candidate from previous saved-body study; same optics, controller and photometry",
  "scope": "Engineering rhythmic lighting composition on one fixed full Shpongle track plus synthetic controls; not learned motion conditioning, universal music fit, human acceptance, or real-fixture certification.",
  "design": "Body controls low-pass root-local openness, height, lean and turn. Audio-derived continuous local beat index schedules bounded 2/4/8-beat paired sweeps. Confidence/activity suppress rhythmic motion in quiet/unsupported regions. No dance retiming or new generator.",
  "arms": [
    "baseline",
    "companion",
    "shifted_clock_plus_half_beat",
    "static_style_companion"
  ],
  "intervals": {
    "quiet": [
      0,
      20
    ],
    "active": [
      70,
      90
    ],
    "late": [
      190,
      210
    ]
  },
  "predictions": {
    "active_coordinate_speed_ratio": ">=1.25 candidate/baseline, fixed70-90; not itself music-fit",
    "quiet_coordinate_speed": "candidate<=max(3deg/s, baseline*1.2)",
    "phase_control": "half-beat shifted clock changes active decoded angles RMS>=1 degree, same source/levels",
    "style_control": "static first-frame style changes active raw targets RMS>=0.1m; not trained semantic validity"
  },
  "gates": {
    "mechanics": "All emitted candidate DMX speed<=90deg/s, acceleration<=360deg/s2 (simulator only)",
    "source": "Saved full body untouched; no new inference; old pose-stage DMX recomputation byte-exact",
    "photometry": "All non-angle DMX bytes equal across baseline and candidate; brightness-matched review reports image luma separately",
    "independence": "Synthetic stationary body + rhythmic audio moves; no clock/confidence produces no autonomous sweep; supplied 90/150bpm clocks change synthetic emitted timing",
    "quiet": "Predicted quiet limit above; full-track numeric and selected visual review, not unconditional no-motion requirement",
    "timing": "Synthetic known rhythmic audio beat timing median error<=60ms at90/120/150bpm and smooth100->140ramp; real detector timing diagnostic against onset+phase-shift null, no human beat annotations",
    "media": "Native1920x1080,30fps,AV1 video with source audio,8319frames; full software decode and timestamps; no upscale or dropped/padded video; clipping and spatial-collapse diagnostics reported"
  },
  "render": [
    "full_candidate_1080p_av1.mp4",
    "full_comparison_1080p_av1.mp4"
  ],
  "visual_review": "Decoded chronological sheets across full track and dense70-90/190-210 samples; user judges final music fit. Comparison 2x1920x1080 scenes downsampled to2x960x540 on1920x1080 canvas; standalone native1080p.",
  "storage": "Keep >=1GiB free, stream bounded-rate AV1, <=1.5GiB additional artifacts; no old artifact deletion",
  "defaults": "Existing mapper defaults unchanged; new companion entrypoint explicit",
  "no_commit": true
}
```
