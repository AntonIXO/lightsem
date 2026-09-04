#!/usr/bin/env python
"""Stage 9 addendum in the vault (PLAN.md rev. 8 section 6.9.9): the transformer-architecture pass,
five papers it rests on, and the corpora now on disk. Nothing is measured here.

    bash run.sh queue_stage9b_to_vault.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
VAULT = Path("/root/lightsem-vault")
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


PAPERS = [
    ("Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts", "arXiv:2311.13687",
     ["chart generation as sequence generation on a beat grid (hop 1/48 beat): time token + action token, 3-layer "
      "Transformer d=256; tempo-informed preprocessing is load-bearing (unaligned training collapses on 12th/24th/32nd notes)",
      "the closest published object to Beat Saber lighting event boxes: sparse authored events on an exact BPM grid"],
     ["audio-to-light", "generative-modelling"]),
    ("Tempo as the Stable Cue: Hierarchical Mixture of Tempo and Beat Experts for Music to 3D Dance Generation (TempoMoE)",
     "arXiv:2512.18804",
     ["experts organised by tempo range plus multi-scale beat experts with music-driven routing; no genre labels",
      "a legitimate routing axis for LightSem's decoder (rate, phase spread, density); fixture-type experts are not"],
     ["music-to-dance", "generative-modelling"]),
    ("Audio synthesizer inversion in symmetric parameter spaces with approximately equivariant flow matching",
     "arXiv:2506.07199",
     ["under permutation symmetry, regressing a point estimate fails even with an invariant loss; equivalent solutions "
      "are modes of a distribution, so a conditional generative model (and an approximately equivariant flow) is needed",
      "the diagnosis for the mixture-of-light problem (many fixture assignments render the same look); the prescription "
      "adopted is SeqLight's look-then-assign split, not an equivariant CNF over fixture permutations"],
     ["cross-modal", "generative-modelling"]),
    ("Automatic Music Mixing using a Generative Model of Effect Embeddings (MEGAMI)", "arXiv:2511.08040",
     ["a permutation-equivariant model generates per-track effect embeddings that drive a track-agnostic effects "
      "processor; handles arbitrary unlabeled tracks; listening tests approach human level",
      "structural analogue of the group-effects decoder + deterministic effects engine: generate the effect, let the "
      "engine realise it"],
     ["generative-modelling", "learned-conductor"]),
    ("SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision", "arXiv:2510.02797",
     ["4-layer Transformer d=512 at ~8 Hz over fused SSL features with boundary and function heads; a learned source "
      "embedding makes training on partial, noisy, schema-mismatched labels work",
      "the recipe for the Stage 9 section planner (bar-rate tokens, boundary + role heads, source embedding when mixing "
      "Raveform / Just Dance / Beat Saber statistics)"],
     ["evaluation-metric", "audio-to-light"]),
]
for title, arx, claims, tags in PAPERS:
    hits = [f for f in (VAULT / "papers").glob("*.md") if arx.split(":")[1] in f.read_text()]
    if hits:
        print("paper exists:", hits[0].name); continue
    r = c.call("add_paper", {"title": title, "arxiv_id_or_url": arx, "key_claims": claims, "topic_tags": tags})
    pid = r.get("paper_id") or r.get("id") if isinstance(r, dict) else None
    print("paper:", pid or str(r)[:120])
    if pid:
        c.call("link_paper", {"entity_id": P, "paper_id": pid})

ok(c.call("update_runbook", {"runbook_id": "rb-ls2-09-designed-light", "note": (
    "2026-09-04 late: corpora on disk -- BeatSaver lightshow maps pulled (pull_beatsaver.py: 300 usable of 317 "
    "inspected top-rated maps, 16.3 h of audio, 2.2 GB, /mnt/lightsem-data/beatsaver; most are v2 five-lane "
    "_events maps, some with Chroma colour data, a few v3 event-box maps; Vivify asset-bundle shows are skipped "
    "because their lighting is not in the map data); Raveform (1,423 expert-annotated EDM tracks with beat/downbeat/"
    "section per beat, 4,902 mix records) at /mnt/lightsem-data/raveform; Skip-BART weights at "
    "/mnt/lightsem-data/models/skip-bart; demucs 4.1 installed in the torch venv, htdemucs weights fetched "
    "(~30 s of audio per second on the 5070); xLights sequence downloads need a login (403) and are parked. "
    "ARCHITECTURE PASS (PLAN.md 6.9.9, docs/research/stage9/05_transformer_architectures.md): the decoder runs on "
    "the BEAT GRID (8th/16th notes, 4-8 Hz), never at 40 fps -- 40 fps is the engine's render rate; Beat Saber "
    "event boxes are trained as discrete CE (effect id, group, beat-time, palette/brightness bins, phase spread, "
    "density) on the same type-masked head whose continuous channels train through the histogram renderer on "
    "RPMC-L2 / Just Dance; conditioning = AdaLN/FiLM from the planner's slow knobs + a Skip-BART-style skip of the "
    "time-aligned audio frame; a 4-bin tempo gate on the head (TempoMoE) is the legitimate routing axis, fixture-"
    "type experts are not; the frozen Conformer stays (no Mamba: the sequence is a few thousand beat tokens); no "
    "masked-AR, flow only if the operator vector is multimodal after conditioning (NLL of 2-step MeanFlow must beat "
    "L2 + circular palette loss); symmetry handled by generate-the-look-then-assign (SeqLight, MEGAMI), not an "
    "equivariant CNF over fixture permutations (Hayes 2506.07199 is the diagnosis, not the prescription). "
    "Recommended stack, ~7 M trainable: planner 4 layers d=256 at 1 token/bar, RoPE on bar index, 8 knobs per "
    "section + boundary logit (3-6 h); decoder 3 layers d=256 beat-relative RoPE at 4-8 Hz, ~2.5 M, renderer KL + "
    "event-box CE + look-change BCE, random-rig dropout, RSTC-style clamps at inference; critic 2 layers d=128 "
    "bar-synchronous, InfoNCE with time-shift / other-song / beat-trigger negatives, best-of-8 reranker only "
    "(2-4 h). Falsifiers unchanged from G9.3-G9.5, plus: decoder must beat the ENGINE-ONLY baseline (planner knobs "
    "+ beat-trigger effects, no learned effect id) and the offline-vs-L=0 gap must be reported.")}))

lv = len(list(VAULT.rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov, "optiHealth vault must not change"
c.close()
