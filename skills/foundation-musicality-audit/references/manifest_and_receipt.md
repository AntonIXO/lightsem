# Manifest and receipt contract

The project CLI consumes `lightsem.music_motion.manifest.v1` as shown in
`docs/research/2026-10-19-music-motion-framework/manifest.example.json`.
Each pair needs:

```json
{
  "body": "raw native body .npz",
  "audio": "feature .npz with time/onset/beats",
  "latent": "optional saved foundation latent .npz",
  "reference_body": "optional single actor calibration body .npz",
  "reference_bodies": "optional list of >=2 actor calibration bodies .npz"
}
```

The audio NPZ clock and beat detector must be authenticated and shared by all
feature-family comparisons. Keep MuQ layer, revision, sample rate, extraction
window, Base92 layout, source WAV hash and timestamps in the provenance receipt.

An audit receipt must include:

* protocol id and resolved thresholds;
* checkpoint/code/feature/normalizer/split/seed hashes;
* own-song body and soundtrack hashes for both required songs;
* latent and output shapes/dtypes/finite coverage;
* raw safety report before mapper/DMX;
* simple and advanced feature-family metrics with train-only PCA metadata;
* all intervention labels and changed inputs;
* conditioning matrix and raw component matrices;
* status plus reasons and missing evidence.

Never overwrite a report. A changed checkpoint, feature cache, threshold,
layer, code revision or control requires a new artifact root and protocol id.
