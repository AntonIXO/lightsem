# Fifteen body-to-light transfer algorithms

**V3 numerical comparison is complete: 30 verified timelines, 14 methods pass
the unchanged engineering gate on both songs, and method 03 fails on both.**
All 30 full dark-stage videos are rendered and independently delivery-verified.
[Open the review page](http://127.0.0.1:8773/?campaign=v3-complete): 15 methods,
both songs, 120 seconds each, 1080p30 AV1, with the source body visible and two
historical baseline references. Method 03 is
retained as an explicitly labeled negative diagnostic; no artistic winner has
been selected. See [RESULTS_NUMERIC.md](RESULTS_NUMERIC.md).

This campaign implements the user-approved 15-method portfolio on Shpongle and
AURORA — Cure For Me. Each has its own authenticated saved audio-conditioned
body. The comparison preserves P2 RGB and processes the full 180 seconds while
reviewing 60–180 seconds. These are original deterministic adaptations of paper
mechanisms, not reproductions of the authors' learned models.

Implementation: `lightsem/conductor/body_transfer_v2/`. The public interface is
`map_body(body, outputs, algorithm_id, fps=40)`, returning targets, dimmers,
anatomical field samples and attribution. `make_outputs` creates stable fixture
identities; explicit output records can also be supplied. `compile_intent` in
the original adapter applies the shared mechanical/DMX path. Existing V1 entry
points remain available for reproduction.

| ID | Method | Source |
|---|---|---|
|01|Bone directions|[Neural Kinematic Networks](https://arxiv.org/abs/1804.05653)|
|02|Bend planes and flexion|[Neural Kinematic Networks](https://arxiv.org/abs/1804.05653)|
|03|Articulated sculpture|[Skeleton-Aware Networks](https://arxiv.org/abs/2005.05732)|
|04|Pose-aware anatomical pooling|[PAN](https://arxiv.org/abs/2306.08006)|
|05|Support-relative transfer|[Contact-Aware Retargeting](https://arxiv.org/abs/2109.07431)|
|06|Distance constellation|[R2ET](https://arxiv.org/abs/2303.08658)|
|07|Near/far interaction field|[MeshRet](https://arxiv.org/abs/2410.20986)|
|08|Proximity bridges|[Contact-Aware Retargeting](https://arxiv.org/abs/2109.07431)|
|09|Differential coordinates|[OmniRetarget](https://arxiv.org/abs/2509.26633)|
|10|Coherent Gaussian deformation|[CPD](https://arxiv.org/abs/0905.2635)|
|11|Balanced anatomical transport|[Sinkhorn](https://arxiv.org/abs/1306.0895)|
|12|Graph-spectral deformation|[Graph signal processing](https://arxiv.org/abs/1211.0053)|
|13|Anatomical harmonic field|[Graph signal processing](https://arxiv.org/abs/1211.0053)|
|14|Bearing-preserving formation|[Bearing rigidity](https://arxiv.org/abs/1408.6552)|
|15|Laban shape and effort|[Effort/Shape](https://arxiv.org/abs/2006.06071), [LaMoGen](https://arxiv.org/abs/2509.24469)|

Each method's code and saved result metadata specify equations, fixed constants,
attribution and limitations. Twenty-two semantic samples are interpolated into
arbitrary counts; this does not create additional independent dance information.
Attribution records the source joints that can affect each output, including
dependencies introduced by a shared structural scale. Emitted six-region
interventions and frozen-body controls test that dependency after compilation.
The default 8 beams, 5 simulated rays and 50 bulbs preserve the previous physical placement.
The dark-stage renderer separately supports at most 16 moving outputs and 128 bulbs.

The controlled engineering comparison estimates the combined anatomical method
and capability mapping versus V1. Most methods share fixture anchors and spatial
bases, so they share a broad visual style. Their crossing reduction cannot be
attributed solely to an individual paper mechanism. Methods 10 and 11 are
strongly similar in measured motion; fifteen algorithms do not imply fifteen
independent visual discoveries. RGB remains the accepted P2 stream. Brightness
comes from the body-derived field; fixed-brightness controls expose geometry.

Uniqueness is component-specific. The 30 whole-installation DMX variants and
each installation's 13 moving-angle streams are distinct, but brightness and
individual fixture commands can duplicate. In method 03's review crop, 11
Shpongle edge bulbs and 12 Cure edge bulbs stay at dimmer byte 6 (2.35%);
same-side bulbs form identical complete dimmer/RGB groups. In method 08, beams
007–008 and all five lasers stay at dimmer byte 20 (7.84%) on both songs for
the full 180 seconds. Their angles still move and differ; there are no complete
fixture-command duplicates in 08. These inactive brightness components are
separate from motion, and the existing P2 color changes remain present. See the
precise groups and time-window scope in [RESULTS_NUMERIC.md](RESULTS_NUMERIC.md).

V3 corrects two semantic defects found after the first numerical pass. Method
03 now preserves its expanded absolute FK skeleton through one shared similarity
transform, rather than replacing skeletal relationships with separate fixture
offsets. Method 08 uses source-proximity graph consensus and a shared transform
for wrists, torso and head; convergence is retained in targets and the garland
field. Independent verification checks actual finite emitted rays as well as
the intermediate coordinates. Structure preservation and low ray clutter are
separate objectives: corrected 03 increases crossings and fails the original
gate; corrected 08 passes with remaining mechanical/contact errors reported.

Current artifacts live under
`/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v3`.
The original V2 campaign remains intact at the corresponding `_v2` root,
including its original 30 passing numerical comparisons and the evidence of
the semantic defects. V3 changes only methods 03/08 in the default installation;
the other 26 timeline files are byte-identical to V2. V2 measurements remain
historical for corrected 03/08. V3 now has its own full ancillary timing and
similarity reports, recomputed with the same methods and new input freezes. Each
execution uses its frozen source snapshot and checks source/data hashes.

`scripts/motion/run_body_portfolio.py` exposes `freeze`, `baseline`, `preflight`,
`generate`, and `summarize`. `scripts/rendering/render_body_portfolio.py` exposes
`smoke`, `render`, `audit`, and `review`. CPU schematic mode requires explicit
`--mode schematic --codec libx264`; dark mode requires authenticated GPU release
and both shared locks. CPU work can continue while the audio study owns the GPU.

The shared compiler fits the full trajectory to 90°/s and 360°/s² simulation
limits, quantizes 16-bit pan/tilt and packs the fixture patch into DMX universes.
The renderer reads those emitted bytes. This is an offline virtual-fixture
pipeline; numerical arbitrary-N support and simulated laser rays do not imply
hardware laser control or measured physical actuator capability.

Thirty preregistered comparisons in `/root/lightsem-vault` predict reduced
projected ray crossing, with an absolute 0.10 magnitude gate, non-regressing
six-region emitted source sensitivity, and hard validity gates. These numerical
gates do not approve choreography. V3 records 28 `supports` outcomes and two
`refutes` outcomes for method 03, without lowering the crossing threshold.
Actual output activity, duplicates, timing,
holds, recurrence, mechanical attenuation, visibility, brightness and color are
reported separately for both songs. Failed methods remain diagnostic results.

The independent V3 numeric and structural audits are
`records/PRIMARY_NUMERIC_VERIFICATION.json` and
`records/STRUCTURAL_READOUT_VERIFICATION.json` in the V3 artifact root.
V3 `records/ANCILLARY_TIMING_PROTOCOL.json` and `ANCILLARY_TIMING_REPORT.json`
measure the repaired outputs, including source-shift, reversal and constant
controls. V3 also has its own `records/ANCILLARY_SIMILARITY_REPORT.json`; 10/11
remain strongly similar, while corrected 03/08 differ from their first-pass
waveforms. Low signed similarity is not an artistic improvement metric.
P2 cue/palette provenance and both original ancillary analyses remain in V2.
Corrected 03/08 retain substantial
position recurrence while velocity recurrence and high-frequency amplitude
show remaining mechanical losses; these proxies do not override the 03 failure.
Cure For Me still has only one actual RGB transition in the 120-second review
crop and then 112.4 seconds of constant color. This separate color limitation
was preserved in V3. Dark smoke inspection admitted all 30 labeled comparisons
for full rendering. Final verification passed for all 30 videos: 3,600 frames,
uniform timestamps, full software A/V decode, exact reference-audio PCM, frozen
inputs, and HTTP Range/download identity. The independent delivery receipt is
`records/DELIVERY_INDEPENDENT_VERIFICATION_V1.json` in V3. It also checks 120
decoded frames against their smoke references. The primary inspected the
largest encoding differences; browser playback itself was not automated.

Research evidence is in [PERPLEXITY_RESEARCH.md](records/PERPLEXITY_RESEARCH.md)
and [RESEARCH_TRACE.json](records/RESEARCH_TRACE.json). Perplexity Reason returned
empty answers; Deep Research completed and the actual report was retrieved.
ArXiv methods were checked; CPD used the original PDF after the MCP text
conversion was unavailable. No claim that published retargeting results already
establish useful lighting transfer is made.
