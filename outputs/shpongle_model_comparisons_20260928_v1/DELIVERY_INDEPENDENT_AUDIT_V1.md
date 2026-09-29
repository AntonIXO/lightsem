# Independent delivery audit — V1

**PASS: no record, media-hash, inode-link, format-receipt or budget mismatch found.**

All four videos match their manifest/render hashes. Each unique home delivery is a hardlink to its intended video. The old `/root/shpongle_light_demo.mp4` is still the unchanged rich anchor.

Authenticated existing AV audit receipts agree on8319 frames,277.3s,1920×1080 AV1 at30fps, AAC stereo44100Hz, zero-origin uniform PTS and full software AV decode. All four decoded PCM fingerprints match. No new decode was run.

| Recipe | Current arm directory bytes | Token changes / accents |
|---|---:|---:|
| S_none_P0 | 227010196 | 5 / 1 |
| S_V6_P0 | 226925887 | 7 / 1 |
| I_VR_P0 | 228826085 | 49 / 30 |
| S_VR_P0_Ashuf | 230848503 | 2 / 247 |

Each arm is below536870912 bytes. Current media is913610671 bytes. The package before this audit is914599607 bytes; counting all home aliases again logically gives1814417301 bytes, below2214592512. Real hardlinks add no second video payload. The manifest package snapshot is102293 bytes smaller because it predates later records; this is not a mismatch. These two audit records are charged normally.

The original campaign is952976108 bytes; the whole new scientific workspace is7082491 bytes. Combined960058599 bytes, or986530540 with the remaining shared32MiB delivery reservation, is below1020054732. Both disk free-space floors pass. Original filename inventory matches; full original content-hash preservation relies on the authenticated parent receipt and was not repeated here.

## Limits

Counts are descriptive, not quality or timing scores. No images were re-inspected here. Parent-reported silent colors at120–131s do not prove music-fit acceptance. Ashuf used recipient audio, not its scientific donor route. No model/default promotion or scientific conclusion follows.

This audit used source/receipt/media hashes and filesystem metadata only. It made no model, raw-array, checkpoint, test, vault, render or decode calls. Sources/defaults were not edited. Full bindings and details are in `DELIVERY_INDEPENDENT_AUDIT_V1.json`.
