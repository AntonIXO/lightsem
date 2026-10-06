# Current lighting baseline: C — variant06 with history and300ms fades

**Superseded September30:** the user selected P2 as the new **color** baseline
and rejected autonomous companion motion. See the [current body-transfer work](../2026-09-30-body-transfer/README.md).
The C selection and results below are historical records.

The user selected **06, independent Rich96 (`I_VR_P0`, selected seed0)** as the
most interesting and best Shpongle result on September29. They explicitly prefer
the rhythm companion02 to original motion mapping01. After reviewing the four
follow-up arms they explicitly selected **C: history enabled and300ms color
transitions** as the reference for new light-layer work and comparisons.

- [Exact checkpoint and accepted configuration](../../configs/light_layers/baseline.json)
- [Original user feedback](records/USER_FEEDBACK.json)
- [Accepted C selection](records/USER_BASELINE_C_SELECTION.json)
- [A/B review plan](records/AB_PLAN_V2.json)
- [Comparison videos and playback guidance](outputs/shpongle_baseline_ab_v2/README.md)
- [Next image-supervision investigation](records/IMAGE_SUPERVISION_NEXT.md)

The selected model is
`/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z/observer_fits_v1/04_seed0_I_VR_P0/best.pt`,
SHA256 `8de865906cfabb49290971eccf6016ab38edebe885eec7e6369f020fb5544b8a`.
It builds on saved rhythm-companion motion/brightness. It has separate audio
encoders for accents, chromatic changes and palette choice, Rich96 auxiliary
image supervision, and no pose auxiliary. YOLO-derived person boxes and SAM2
segmentation help construct training observations; neither runs inside its
audio-to-color inference.

The accepted C baseline uses palette history and a300ms causal
smoothstep in channel RGB, beginning at each supplied color change and reaching
the target300ms later. An interruption starts from the instantaneous interpolated
color. It does not advance cues, alter dimmers, or compensate perceived luma.
It is the user's accepted timing/appearance choice, not a trained transition policy.
The original hard-cut06 remains preserved as comparison A.

Four arms isolate the two changes:

| Arm | Previous palette history | Transition |
| --- | --- | --- |
| A | On | Original hard change; exact old06 DMX, rendered in the common review format |
| B | Off, trained null state | Hard change |
| **C — accepted baseline** | **On** | **300ms fade** |
| D | Off, trained null state | 300ms fade |

Compare A/B or C/D for history, A/C or B/D for transitions. C is the explicit
user selection. All arms share the same262 color
cue opportunities,30 model accents, movement, exposure and audio. All479 non-RGB
DMX slots remain byte-exact to06. The33 writable RGB slots are the sole change.

The current V2 delivery uses the concurrently updated compact review format:
all four arms are rendered with the same AV1/AAC settings, with a50,000,000-byte
maximum per video. Audio is encoded directly from the original lossless FLAC
at128kbps and must decode identically across these four arms. It does not match
the older192kbps decoded-audio fingerprint. The original06 video is preserved.
V1's interrupted render and failed restart receipt remain separate; no completed
V1 delivery is claimed.

Preparation replays the already saved GPU argmax table for every previous state;
its own-history palette and previous-state sequences must match the saved06
exactly. There are no model calls, threshold changes or training. Disabling history
changes palette IDs for178.4s of the277.3s track; this is not a perceptual-distance
score. See `PREPARATION.json` for intervals and channel-step descriptions.

Historical counting note: the previous comparison report counted49 palette-ID
changes including the transition from the initialized hidden token at0.3s.
The new preparation reports48 changes *after the learned overlay first becomes
active*, excluding that first adoption. Both refer to the same unchanged06.

The original event study remains unqualified and the shared-model palette-history
study remains negative in its own setup. Human preference on one known song
does not replace these measurements. The older causal `LightModel` interface is
separate from the selected independent light-layer architecture.

The bounded sequential renderer is
`scripts/light_layers/render_baseline_review.py --plan <new plan path>`, invoked
through `run.sh`; prepared outputs refuse overwrite. Frozen historical receipts
retain their original paths after the directory refactor. The current plan binds
the relocated source explicitly.

The frozen review plan still binds the pre-selection config. Its exact contents
are preserved in `records/BASELINE_BEFORE_C_SELECTION.json`; the current config
intentionally advances to C. `records/BASELINE_C_SELECTION_AUDIT.json` authenticates
the accepted media and selection. New executions require a fresh binding.
