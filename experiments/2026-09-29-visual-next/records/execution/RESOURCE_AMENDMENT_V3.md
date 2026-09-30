# Prospective resource amendment for the full automated matrix

The full automated task now includes source observations for all91TRAIN and
31historicalDEV families, needed both for the seven-arm study and the cut/mask
downstream demonstration fits. No full shared-corpus extraction or student fit
has started when this amendment is recorded.

The frozen combined16-frame smoke passed with about1.055GiB reserved GPU memory
and2.44GiB process RSS. Its timers measured7.0473s for masks/cuts and0.7175s for
teacher features, including startup and decoding. CPU measurement on the first
frozen TRAIN frame found that reusing HSV preserves the original targets exactly
and reduces repeated image conversion work. These are engineering measurements,
not accuracy estimates or a claim that16frames establish a precise corpus runtime.

To retain the complete matrix rather than truncate families or comparisons,
increase the H1/H2/shared-source preparation allowance from2GPU-hours to4GPU-hours.
Keep H3 teacher features at1hour, H3 student fits at3hours and H4 at2hours. Thus
the total prospective cap becomes10GPU-hours instead of8. Account process wall
time while the experiment holds the accelerator, including source decoding and
interrupted work; record the shared extractor's separate stage timers. Waiting
for another task before acquiring the GPU is not experiment GPU time.

H1 already used a conservative1047s before the external-render interruption and
203.3933s for its five-family continuation. Add all model/shared-smoke timing
before computing remaining stage allowances. Completed source/teacher outputs
will be reused; no change to model families, predictions, thresholds, cohort,
seeds,6000updates, method controls or quality gates is made by this amendment.

Retain the4GiB artifact cap from IMPLEMENTATION_DETAILS_V2.md, the8GiB RAM and
10GiB reserved VRAM limits, at least1GiB free on both volumes, no scientific-cache
growth on /mnt, and one experiment GPU process at a time. The annotation UI
remains optional. This amendment is an explicit change to engineering limits;
do not claim the original8-hour/3GiB allocation was unchanged or passed.
