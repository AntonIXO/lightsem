# Independent beat-test readiness — 2026-09-22

**BLOCKED. No new defensible comparison can run now.** The audit found **0 verified exact-audio/annotation pairs and 0 eligible canonical songs**, against the required **≥20 songs**. These are readiness counts, not accuracy results. No hypothesis was tested or closed.

## What is on disk

| Metadata check | Count / result |
|---|---:|
| Expert-subset `structures/beats/*.beat.csv` | 1,423 |
| `segments.json` records / unique YouTube IDs | 1,423 / 1,423 |
| Missing CSV-key or track-ID joins | 0 / 0 |
| Broader **estimated** beat files | 68,078: 63,038 tracks + 5,040 mixes |
| Media files inside the Raveform bundle | 0 |
| Media filenames inventoried | 3,393 under `/mnt/lightsem-data`; 62 under repo `data/` |
| Expert IDs matching these media filenames | 0 |
| Expert IDs matching Stage11 recording IDs | 0 |
| Exact source/timing-qualified local pairs | **0** |
| Candidate songs cleared against small0 training | **0** |

The local expert keys are a complete one-to-one filename join, not a canonical-song deduplication. Alternate uploads, versions and remixes still need grouping. Filenames and metadata cannot rule out renamed or archived audio. The finding is **no metadata-verifiable matching audio**, not proof that no possible alias exists anywhere.

Metadata inventories covered Just Dance/Stage11, Beat Saber, FineDance, Oslo, local AIST media and experiment media filenames. No audio, video, feature array, beat CSV or JAMS content was opened. RPMC’s cached features and AIST generated beat features are not independent annotations.

## Authoritative annotation provenance

Primary sources: [Raveform paper, DOI10.5334/tismir.288](https://transactions.ismir.net/articles/10.5334/tismir.288), [official project](https://mir-aidj.github.io/raveform/), [official dataset](https://huggingface.co/datasets/taejunkim/raveform).

- Paper Table1 distinguishes **estimated** `beats/{mixes,tracks}` from human-annotated `structures/{beats,segments.json}`. Only the latter is this candidate reference.
- Three experts rotated annotator, verifier and final-judge roles, with consensus meetings. They inspected and corrected **Rekordbox-assisted beat/downbeat grids**. Tracks with impractical grid correction were excluded. This is one expert-consensus reference, not three independent tap annotations and not BeatThis self-labels.
- The subset favors frequently played EDM and original/long versions. Short/radio edits, DJ-mix clips, fading starts and some ambiguous material were excluded. Do not generalize it to random music or all difficult rhythm.
- IDs are YouTube video IDs. A structure key adds the numeric prefix. The project defines CSV timestamps in seconds and bar position1 as downbeat. DJ-mix appearances are not independent songs and may use different versions.
- The project warns that **MP3 decoder offsets vary**. No quantitative annotation timing uncertainty, exact source-PCM checksum or universal decoder/offset contract was found. Serialized decimals and the paper’s **±70ms scoring tolerance are not annotation accuracy**.

Remote metadata exposes revision `f0fbae800603608c7d7a4f860524888313f92395` (2026-03-22), archive479,239,104bytes, LFS SHA256 `10c97fa9213fe4ca032195e73b6a9d068c0d5ca8a8f603615bb1bdbabffb34de`. No archive was fetched. No numbered dataset release was found; the project’s “v2” text concerns its model. **The extracted local tree was not authenticated against that archive.** Local `tracks.jsonl` has61,886 unique IDs versus56,873 unique tracks in the paper. Pin the actual tree, do not silently assert identical versions. Local metadata SHA256 bindings are in the JSON.

Dataset metadata says CC-BY-4.0. This does **not** supply rights to the underlying music.

## Local title leads — not eligible songs

There are four FineDance title-only leads with existing WAVs. Their filenames do not carry the Raveform source IDs. FineDance supplies no artist identity for these names. No offset, crop or version link exists in the inspected metadata.

| Raveform annotation reference | Local FineDance ID / canonical group / split | Local / annotation-source duration, seconds (metadata only) | Status |
|---|---|---:|---|
| `0739._Fdo4tO89uI` — Steve Aoki & Autoerotique vs Dimitri Vegas & Like Mike - Feedback [Dim Mak] | `001` / `fd-7a3e6b16cb75f48f` / TRAIN | 97.615 / 225.861 | Artist/source version absent; unresolved |
| `0833.8zz4lFBiL3Q` — Marco Faraone - Next Level (Santé Remix) [Moon Harbour] | `202` / `fd-c17edaae86e4016a` / VAL | 71.000 / 386.624 | Author-excluded; reject |
| `1662.wtEs1w4V2Ss` — Most Rated Track: Flashmob - Hot [Defected] | `197` / `fd-8bcbb4c131df56f7` / TRAIN | 92.500 / 435.943 | Artist/source version absent; unresolved |
| `1760.IO4PAvHV6OA` — Bladerunner - I Need You [Life] | `020` / `fd-c75f53bdcc638b0f` / TRAIN | 208.500 / 343.261 | Artist/source version absent; unresolved |

Exact WAV and annotation paths are in `BEAT_TEST_READINESS.json`. **Do not copy labels onto these audio files.** Duration differences could reflect a crop or a different recording; neither is established. Even if the three non-excluded leads later proved valid, they are fewer than20 songs.

Other joins:

- Stage11: 84 loose song-substring collisions across53 expert IDs; **0 full artist-and-title string joins** and0 exact source-ID joins. These are text collisions, not available pairs. Examples include Oxia’s “Domino” versus Jessie J’s “Domino”, and “Your Mind” versus “Blow Your Mind”.
- Beat Saber:317 manifest rows and300 existing WAVs. Eight equal-title rows, all with different artist metadata; seven have local WAVs. None is an exact pair.
- Oslo’s four named source tracks have no matching expert title. Local AIST uses corpus IDs, not matching Raveform IDs; there is no exact-source crosswalk. No generated beat features were treated as truth.

## TEST stays excluded

Joined Stage11 manifest, split and canonical map metadata. All669 recordings agree on split/canonical identity. The full manifest has564TRAIN /54VAL /**51TEST recordings**, with451 /40 /**40 canonical groups**. Exclusion uses **all51 TEST members**, not only the50 eligible-core recordings quoted in historical reports.

Zero exact Raveform ID matches were found. Five loose-text matches touched three sealed groups and were rejected without raw access:

- `jd-849690f6f73e4d04cfa4`: `Jz0guMrALKY`, `jmdIqYLflag` (“Your Mind” / “Blow Your Mind” collision).
- `jd-fbc288a7c9d4a30f0316`: `ta9bchvLUB8` (“Seeker” / “Heat Seeker” collision).
- `jd-7a762e2a3786fce79bc1`: `HbK4yFX9sc8`, `Pdo2eCGEEiU` (“Touch” / “Keep In Touch” collision).

These are **not confirmed shared songs**. All40 canonical IDs and51 recording IDs are listed in the JSON denylist. FineDance’s28TEST recordings /19 groups are also denied; none of the four title leads is TEST. Any later plausible identity with a sealed group stays excluded until resolved by permitted metadata, never by opening sealed audio/features. No new salt or split was created.

## Tracker and exposure readiness

Candidate: **BeatThis `small0`**, existing8,451,101-byte checkpoint. Frozen ledger SHA256:
`6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b`.
Only existence/stat and the prior metadata binding were read; weights were not loaded or rehashed.

Comparator: **`lightsem.audio.rhythm_clock.clock_from_audio`**, the current full-track companion clock. It uses broadband/kick global tactus plus adaptive local tracking and its fixed synthetic-onset correction. It is **not** the old standalone constant-grid GuitarSet comparator. Pin both this source and `beatgrid.py`; retain the fixed minimal small0 postprocessor. Do not retime the dancer.

The [official BeatThis README](https://github.com/CPJKU/beat_this) says small0 trains on all disclosed data except GTZAN. GuitarSet comping and Harmonix are included. Raveform is not named, but **shared songs/recordings remain unknown**, especially across commercial dance corpora. The local `OBSERVER_EXPOSURE_LEDGER.json` explicitly leaves song overlap uncleared. Neither it nor the cached research dossier contains a complete small0→Raveform crosswalk.

The [v1.0 annotation tree](https://api.github.com/repos/CPJKU/beat_this_annotations/git/trees/v1.0?recursive=1), commit `c3c47fd37d3074d9f8119f18bbf460f909609f22`, exposes IDs for all15 non-GTZAN directories, but many are opaque corpus codes. Its4,557 beat filenames differ from the paper’s4,556 training/validation tracks. **A raw filename inventory is not a checkpoint-bound eligible training list.** Missing exposure assets are:

1. small0’s eligible training+validation membership with that count reconciled;
2. original-corpus ID→artist/title/version/source-recording crosswalks, including opaque IDs;
3. an adjudicated Raveform song/version join, with known overlaps excluded and unknowns quarantined.

The earlier GuitarSet benchmark already has358 saved prediction NPZs. It was not rerun and cannot supply a newly untouched overlap-cleared test.

## Exact blockers and next action

Missing now: ≥20 canonical independent eligible pairs; exact source/version and timing binding; audio rights; small0 song-exposure clearance; and a prospectively frozen comparison adapter/manifest. The present `run_light_layers_rhythm.py` is a **CUDA-required feature extractor**, not an independent accuracy benchmark. Do not run it for this task.

**Minimal next action:** obtain a metadata package for ≥20 rights-cleared, non-TEST, exact-source audio/independent-annotation pairs plus small0 exposure clearance. If new audio cannot be acquired, commission independent annotations on ≥20 already-local non-TEST canonical songs **after identity/exposure clearance**. Chasing the three unresolved FineDance leads alone cannot meet the minimum. No downloads are authorized by this report.

Only after these gates pass: freeze source/weights/audio/labels/config hashes and grouping, verify synthetic instrument controls, run both trackers on identical full PCM, then apply the fixed first5s trim and70ms one-to-one F1/CMLt scoring. Preserve original paired song inference, **F1 gain≥.03 with positive paired95%CI**, CMLt decline≤.02, common-support requirements, and explicit octave/phase/ambiguity reporting. Do not tune offsets on test labels or lower the20-song minimum. Beat accuracy would still not establish LIGHT quality.

At audit time `/mnt` free space was1,269,473,280bytes, with a1GiB floor. No dataset/audio/model bytes were written. No inference, scoring, GPU work, vault access, source edits or TEST raw input access occurred. `segments.json` was parsed for identity/title/fold/duration; an initial schema preview incidentally displayed nested tempo/section examples. Those values were not used for measurement. Only the two requested readiness reports were written.
