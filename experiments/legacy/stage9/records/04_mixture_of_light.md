# Direction 4 — mixture of light (Perplexity reasoning pass, 2026-09-04)

Project brief attached: `lightsem_brief.md` (copied below the answer). Answer verbatim.

**(c) wins.** A group-level effect decoder plus a frozen engine is the only option that matches how shows are actually programmed, that is permutation- and mirror-safe, and that can be trained from the corpora you have — which almost never contain per-fixture DMX.[1][2][3]

## Architecture choice

A lighting programmer does not drive 512 DMX slots. They drive groups, palettes, an effects engine (waveform, rate in beats, phase spread), and cue/look changes. Beat Saber lighting is the same language: group lanes with On/Flash/Fade, laser speed, ring triggers, and event-box distribution across a group. xLights is effects-on-groups, not raw universes. SeqLight, the closest published multi-light system, also refuses to predict fixtures from mixed video: it predicts a global HSV distribution, then decomposes.[2][3][4][5]

| Design | Permute same-type fixtures | Mirror of the rig | Train from your data |
|---|---|---|---|
| (a) type-MoE heads | Broken if heads are slot-indexed; independent copies cannot chase/phase | Broken unless left/right is a feature, not a slot | Needs per-fixture DMX you do not have |
| (b) fixture-token transformer | Yes, if the only PE is type + 3D position, no patch index | Not free; needs flip-aug or even/odd spatial basis | Only via a renderer onto HSV/delta features; pan/tilt/gobo are almost invisible in those |
| (c) group effects + engine | Yes: the engine fans a waveform over a set sorted by `(type, x)` | Yes if phase is a function of $$x$$ or $$\lvert x\rvert$$, not DMX address | Yes: events, hue histograms, intensity envelopes, beat-rate are exactly RPMC-L2 / Just Dance visual / rhythm-game tracks |

**(a)** is a typed output vector pretending to be a mixture of experts. Six moving-head slots concatenated is a canonical-rig overfit. Shared per-type heads that ignore neighbors cannot implement a chase. Kill it.

**(b)** is the right geometry for a point cloud of lights, and a set transformer is permutation-equivariant if you do not add order encodings. It is the wrong **target**. RPMC-L2 is per-frame HSV of mixed stage light. Just Dance “visual” is a background histogram. Dancer descriptors are not DMX. A token decoder that emits pan/tilt/gobo is inventing degrees of freedom the loss cannot see. You would be training a color model with a free unsupervised choreography head.[6][7]

**(c)** is the professional interface and the observable interface. The network emits, per group role (wash, spot/profile, beam, strobe, blinder, pixel, laser, haze): relative intensity envelope, palette/hue, effect id, rate in beats, phase spread, participation density. A deterministic engine binds groups to whatever is patched and writes DMX. That engine is your current conductor, with the if-then table ripped out and the knobs exposed.

Brutal part: the current 30 motion descriptors are a dead bottleneck for this. Held-out block A $$R^2 = -0.22$$; inter-coach agreement is $$r = 0.22$$–$$0.42$$, which is the ceiling of any dance-trained map; the model is already more metronomic than the dancer (Light-BAS $$+0.135$$ vs $$+0.087$$). A learned decoder sitting on those channels will clone a bad clock. The decoder must read the audio trunk and the operator head (your strongest measured audio→light coupling), and use dance only as a training signal, which is what the product spec already claims.[1]

## Loss under random rigs

Ground truth is not a set of fixtures, so Hungarian/Chamfer matching is inapplicable. Random-rig sampling is only legal if the loss lives on a rig-invariant observable.

Use a cheap differentiable renderer, not EGL in the loop. Each active group is an additive colored blob (or a 2D Gaussian on a stage plane for spots). Output a per-frame HSV histogram and the same delta-light features you already use for FID_light. Train against RPMC-L2 histograms and Just Dance visual histograms. That is the SeqLight move: mixed light in, no professional per-fixture demos, venue config as a condition.[1][2]

Define the per-type loss on **groups**, not heads:

- Wash / LED: histogram hue+value, no absolute level (already a permanent rule-out).
- Strobe / blinder: onset times and duty cycle, not brightness.
- Spot / beam: contribution to the histogram plus an effect-rate term; do not backprop pan/tilt from HSV — those stay engine-procedural until you have console logs.
- Pixel bar: optional later; spatial patterns need a 1-D renderer.

Random rig: dropout groups, jitter group cardinality, shuffle fixture positions inside a type, occasionally drop the strobe. Because the loss is on the render, two washes swapping places is a zero; that is the invariance you want. If you instead L2 the conductor’s canonical DMX, you have distilled the bottleneck and random rigs become noise.

Do not use set-matching. Do not supervise fixture tokens from histograms and hope pan appears. Smoke and cameras eat gobos; RPMC-L2 even freezes saturation at 255 to fight scatter.[8]

## Type MoE is the wrong cut

The MAS.S60 caveat applies to **experts in the trunk**: hard-partition by modality or by human type, then watch them under-share. Intensity envelopes, palette changes, and beat-sync are common across washes, spots, and pixels. A “strobe expert” versus a “wash expert” in the Conformer is the failure mode. Learned routing, when you ever have the budget, should route by musical function (drop vs pad), not by fixture label. Modality-aware MoE literature still uses a router rather than a frozen type partition for that reason.[9]

Type is justified only as an output **mask**. A wash token has no gobo wheel; a strobe has no CMY. That is a parameter mask or a type-conditioned FiLM on a shared MLP, not an MoE. Different parameter spaces ≠ different experts.

## Eval without the cheap wins

Your model already beats a beat-trigger on Light-BAS by strobing more like a metronome than a dancer. Brighter and flashier will win naive A/B and BAS. Pre-register the confounds away.[1]

- Match mean Value per song to the reference (or to the conductor) before BAS, FID, and A/B. Absolute level is banned; enforce it.
- Match onset density, or cap strobe duty cycle to the human/GT band. If BAS only wins by flashing every beat, it is a fail.
- Split FID: hue-structure vs onset-structure vs full delta features. Quote the split-half floor $$8.1$$ every time. Any FID drop must beat a shuffled-donor control — H-C already fake-won $$36.5 \to 27.6$$ with shuffle at $$33.1$$.[1]
- Human A/B, brightness-matched, $$n \ge 20$$ song-split pairs, two questions kept separate: “fits the music” (you currently tie) and “looks like an LD” (real dancer motion won $$8$$–$$1$$–$$1$$ through the same conductor). Add a “too metronomic” rating so BAS-maxing is visible.
- Always report vs beat-trigger and vs the handwritten conductor, causal $$L=0$$ and offline.
- Rig-transfer: freeze the decoder, patch an unseen rig (different $$N$$, no strobe, extra LED bars). Score HSV-hist distance of the transferred show to the canonical render, and a brightness-matched A/B vs “copy channels / drop extras.” Architecture (c) gets this almost free if the engine binds by type+position. Architecture (b) is actually tested here; (a) will die.

## Build this, kill it fast

Recommended: **(c) with a shared trunk, type as a mask, engine as the only fixture expander.** Not type-MoE. Not a fixture transformer until you have console logs or a renderer that can see pan.

MVP, one week, one RTX 5070:

1. Freeze Conformer + operator head.
2. Tiny causal 1-D conv / MLP ($$\sim 0.5$$–$$2$$ M params) from trunk+operator features → group-effect vector at $$40$$ fps or on a beat grid.
3. Reuse the current expander; sort fixtures by `(type, x)`, not DMX address; keep the fitted one-pole (largest lever you have found).[1]
4. Train on RPMC-L2 + Just Dance visual via the cheap histogram renderer. Auxiliary: look-change CE. No conductor-DMX distillation. No block-A L2 as the main loss. Randomly dropout groups.
5. Skip the set transformer, skip EGL-in-the-loop, skip pan/tilt learning.

Kill gate, held-out songs, Value-matched and onset-matched: FID_light vs conductor must beat a shuffled-donor control by $$&gt;2\times$$; Light-BAS must beat conductor without raising onset density; brightness-matched A/B on “fits the music” vs conductor must have win-rate CI excluding $$50\%$$ at $$n\ge 20$$; unseen-patch hist distance must beat channel-copy. Miss any two: the conductor is not the bottleneck, stop decorating it, and go get stems or real light logs.

## Sources

[1] lightsem_brief.md | https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/16322705/34d4f2f3-8b2e-4455-8e95-b2091b9bbc5f/lightsem_brief.md?AWSAccessKeyId=ASIA2F3E
[2] Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning | https://arxiv.org/abs/2605.03660
[3] Basic Lighting | BSMG Wiki | https://bsmg.wiki/mapping/basic-lighting.html
[4] Lightshow | BSMG Wiki | https://bsmg.wiki/mapping/map-format/lightshow.html
[5] Getting Started with xLights: A Beginner's Guide | https://www.learnchristmaslighting.com/post/getting-started-with-xlights-a-beginner-s-guide
[6] [PDF] DISCOFORMER: PLUG-IN DENSITY AND SCORE ESTI | https://openreview.net/pdf?id=ZuqbPd7xTy
[7] Universal Set Transformer: A Scalable and Interpretable... | https://openreview.net/forum?id=gIpuW5Ekiw
[8] [Paper Note] Automatic Stage Lighting Control: Is it a Rule ... | https://en.papernotes.org/ICLR2026/audio_speech/automatic_stage_lighting_control_is_it_a_rule-driven_process_or_generative_task/
[9] Mixture of Modality-Aware Experts (MoMa) - Emergent Mind | https://www.emergentmind.com/topics/mixture-of-modality-aware-experts-moma
[10] Automatic Stage Lighting Control: Is it a Rule-Driven ... | https://iclr.cc/virtual/2026/poster/10008746
[11] xLights - App Store | https://apps.apple.com/nl/app/xlights/id1137297689?l=en-GB&amp;platform=mac
[12] xLights - App Store - Apple | https://apps.apple.com/us/app/xlights/id1137297689?platform=iphone
[13] Massively Multimodal Foundation Models: A Framework for Capturing Interactions with Specialized Mixture-of-Exp | https://arxiv.org/abs/2509.25678
[14] Paper page - Stage Light is Sequence^2: Multi-Light Control via Imitation Learning | https://huggingface.co/papers/2605.03660
[15] README.md - bobbyfriday/xlight-autosequencer | https://github.com/bobbyfriday/xlight-autosequencer/blob/main/README.md
[16] Automatic Stage Lighting Control: Is it a Rule-Driven Process ... | https://nanless.github.io/audio-paper-digest-blog/posts/2026-05-04-automatic-stage-lighting-control-is-it-a-rule/
[17] 2026.16 Released | https://xlights.org/2026-16-released/
[18] 2026.12 Released – xLights | https://xlights.org/2026-12-released/
[19] Page 2 – Light sequencer and Show scheduler - xLights | https://xlights.org/page/2/
[20] 2026.13 Released | https://xlights.org/2026-13-released/
[21] Confused on DMX PAR light modeling / capabilities | https://www.reddit.com/r/xlights/comments/1pbvls7/confused_on_dmx_par_light_modeling_capabilities/
[22] Quantitative Approximation Rates for Group Equivariant Learning | https://arxiv.org/abs/2602.20370
[23] [PDF] On the Expressive Power of Permutation-Equivariant Weight-Space ... | https://arxiv.org/pdf/2602.01083.pdf
[24] BeastSaber | https://bsaber.com/
[25] Self-Attentionは語順を知らない — 置換同変性と位置 ... | https://disassemble-channel.com/attention-permutation-equivariance/
[26] Extended Lighting - BSMG Wiki | https://bsmg.wiki/mapping/extended-lighting.html
[27] Beat Saber Update v1.43.0 with The Prodigy Music Pack - SteamDB | https://steamdb.info/patchnotes/22868537/
[28] Beat Saber Update v1.42.1 | https://steamdb.info/patchnotes/21650839/
[29] Map - [V3 Lights] Lightspark - Breaking Walls - BeatSaver | https://beatsaver.com/maps/414ae
[30] Set Interdependence Transformer: Set-to-Sequence Neural ... | https://liner.com/review/set-interdependence-transformer-settosequence-neural-networks-for-permutation-learning-and
[31] Commutativity: Why Transformers Need Positional Encodings | https://softwarebits.substack.com/p/commutativity-why-transformers-need
