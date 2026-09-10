# Reproduction and artifact map

All new data, checkpoints and movies live under `/mnt/lightsem-data/experiments/lightsem`.
The live implementation is `lightsem/experiments/motion_study`. Scientific runs use
read-only source snapshots with authenticated protocols and input hashes. Completed
artifacts are reused only after identity checks; a changed scientific recipe needs
a new output root and hypothesis. Historical snapshots are never updated from the
current checkout.

| Root suffix | Contents |
|---|---|
| `motion_representation_20260910` | FineDance split/source acceptance, three neural codecs, initial result log |
| `motion_representation_phasecodec_20260910` | Four linear capacities, full-recording and short-hold evaluation |
| `motion_native_audio_20260910` | Uncompressed reference, paired audio data, two native music models, optics and final diagnostic delivery |

Use `bash run_torch.sh` for torch, training, native geometry and combined rendering.
Use `bash run.sh` for NumPy/OpenGL-only tasks. Scientific vault calls use the
Vibescience environment with the vault explicitly pinned to `/root/lightsem-vault`.
Only one GPU task may run at a time. The package's GPU lock enforces this across
these launchers. The native mapper is CPU work.

Verify physical counterexamples, timestamp indexing, confidence visibility and
quantized mechanical limits:

```bash
bash run_torch.sh -m unittest discover -s tests -p test_motion_study.py -q
```

Existing immutable stages can be verified or resumed with their launchers:

```bash
bash run_torch.sh run_motion_study.py training --help
bash run_torch.sh run_phase_motion_study.py --help
bash run_torch.sh run_native_motion_study.py audit
bash run_torch.sh run_native_motion_review.py select
bash run_torch.sh run_native_music_study.py training train --arm regression --resume
bash run_torch.sh run_native_music_study.py training train --arm flow --resume
bash run_torch.sh run_native_music_study.py evaluation evaluate --arm regression --checkpoint initial
bash run_torch.sh run_native_music_study.py evaluation evaluate --arm regression --checkpoint final
bash run_torch.sh run_native_music_study.py evaluation evaluate --arm flow --checkpoint initial
bash run_torch.sh run_native_music_study.py evaluation evaluate --arm flow --checkpoint final
bash run_torch.sh run_native_music_study.py evaluation full --arm regression
bash run_torch.sh run_native_music_study.py evaluation full --arm flow
bash run_torch.sh run_native_music_review.py incumbent
bash run_torch.sh run_native_music_review.py render
bash run_torch.sh run_native_music_study.py selection --seed 0
bash run.sh run_motion_exposure_study.py run
bash run.sh run_motion_exposure_roles.py run
```

The final delivery is diagnostic because both native music models failed. It saves
all six full Shpongle flow draws, both native arms, incumbent 8/32-step poses, and
full generated sequences underlying the three historical clips. Lighting variants
keep source motion fixed for the brightness and renderer ablations:

```bash
bash run_torch.sh run_motion_delivery.py infer
bash run_torch.sh run_motion_delivery.py map
bash run_torch.sh run_motion_delivery.py skeletons
bash run_torch.sh run_motion_delivery.py render
bash run_torch.sh run_motion_delivery.py observe
bash run.sh -m lightsem.experiments.motion_study.delivery_plot
bash run.sh -m lightsem.experiments.motion_study.delivery_plot --comparison
bash run.sh -m lightsem.experiments.motion_study.delivery_plot --correct-labels
```

Start at `motion_native_audio_20260910/delivery/final_diagnostic/protocol.json` for
the full input list, parameters, fixed intervals and gate thresholds. Every video
has its original synchronized audio. `lighting/brightness_off` changes only dimmer
bytes; `lighting/hybrid_legacy` changes only renderer profile; `lighting/independent`
uses six generated bodies instead of one. None of these comparisons retroactively
changes the failed music conditioning experiment.

The labelled side-by-side movie is `final_diagnostic/shpongle_comparison.mp4`.
The three historical reviews in `skeletons/label_corrected` explicitly label the
source video as the same frontal view in both rows; only generated native panels
have front and oblique projections. Original review files are retained with a
record of the heading correction. The plot command produces PNG/PDF plus a CSV
of all lighting variants. Run the comparison and heading-correction commands
after other GPU work; their encoder also uses the GPU.

No confirmation lock or TEST model evaluation was produced. Seed 1 remains blocked
by qualification; six inference draws are not six training seeds.
