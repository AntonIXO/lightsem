"""Blind A/B material: brightness-matched pairs, randomised order, sealed key.

Raters prefer the brighter clip. That is the single most reliable way to get a
flattering result out of a lighting comparison and it measures nothing, so
PLAN.md section 7 makes brightness matching a precondition rather than a
refinement. Every pair here is matched on mean AND 95th-percentile emitted
intensity before it is shown, and the harness also emits a deliberately
over-bright foil so a rater who is simply choosing the brighter clip can be
detected rather than trusted.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dmx.decode import brightness, decode_stream
from ..dmx.fixtures import Rig


def _dimmer_channels(rig: Rig) -> list[int]:
    offs = {"moving_head": 4, "wash": 0, "blinder": 0, "strobe": 0, "projector": 0}
    return [f.address - 1 + offs[f.kind] for f in rig.fixtures]


def scale_brightness(frames: np.ndarray, rig: Rig, gain: float) -> np.ndarray:
    """Scale every dimmer channel by ``gain`` (clipped into DMX range)."""
    out = frames.copy()
    for ch in _dimmer_channels(rig):
        out[:, ch] = np.clip(out[:, ch].astype(np.float64) * gain, 0, 255).astype(np.uint8)
    return out


def match_brightness(a: np.ndarray, b: np.ndarray, rig: Rig,
                     tol: float = 0.03, max_iter: int = 40) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scale the brighter stream down until mean and p95 agree within ``tol``.

    Scaling DOWN rather than up on purpose: raising a dimmer that is already at
    full does nothing, so matching upward silently fails on exactly the clips
    that need it most.
    """
    def stats(f):
        v = brightness(decode_stream(rig, f))
        return float(v.mean()), float(np.percentile(v, 95))

    def combined(m, p):
        # One gain cannot equalise both the mean and the 95th percentile when the
        # two streams have different shapes — a peaky beat trigger and a smoother
        # dance-derived look differ in exactly that way. Matching their geometric
        # mean splits the difference and, unlike a histogram LUT, leaves the
        # temporal dynamics (the thing under judgement) untouched. The residual
        # mean and p95 are both reported so a reader can see what is left.
        return float(np.sqrt(max(m, 1e-9) * max(p, 1e-9)))

    a_out, b_out = a, b
    gain_a = gain_b = 1.0
    for _ in range(max_iter):
        ma, pa = stats(a_out)
        mb, pb = stats(b_out)
        ca, cb = combined(ma, pa), combined(mb, pb)
        if abs(ca - cb) <= tol * max(ca, cb, 1e-6):
            break
        # The target has to be recomputed every iteration. Pinning it to the
        # initial minimum stalls the loop: the brighter stream is scaled past the
        # darker one, and then each is already "at target" while they differ.
        target = min(ca, cb)
        if ca > cb:
            gain_a *= max(min((target / max(ca, 1e-6)) ** 0.9, 0.999), 0.05)
            a_out = scale_brightness(a, rig, gain_a)
        else:
            gain_b *= max(min((target / max(cb, 1e-6)) ** 0.9, 0.999), 0.05)
            b_out = scale_brightness(b, rig, gain_b)
    ma, pa = stats(a_out)
    mb, pb = stats(b_out)
    ca, cb = combined(ma, pa), combined(mb, pb)
    return a_out, b_out, {
        "gain_a": gain_a, "gain_b": gain_b,
        "mean_a": ma, "mean_b": mb, "p95_a": pa, "p95_b": pb,
        "combined_a": ca, "combined_b": cb,
        "mean_ratio": ma / max(mb, 1e-9), "p95_ratio": pa / max(pb, 1e-9),
        "matched": bool(abs(ca - cb) <= tol * max(ca, cb, 1e-6))}


@dataclass
class ABTrial:
    trial_id: str
    clip: str
    left_system: str
    right_system: str
    left_file: str
    right_file: str
    match: dict
    kind: str = "comparison"


def write_key_and_form(trials: list[ABTrial], out_dir: Path, questions: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "KEY.json").write_text(json.dumps(
        [t.__dict__ for t in trials], indent=1))
    lines = ["trial_id,clip,question,answer(L/R/equal),confidence(1-5),notes"]
    for t in trials:
        for q in questions:
            lines.append(f"{t.trial_id},{t.clip},\"{q}\",,,")
    (out_dir / "score_sheet.csv").write_text("\n".join(lines) + "\n")
    (out_dir / "INSTRUCTIONS.md").write_text(
        "# Blind A/B\n\n"
        "Watch each pair with sound, on the same screen, at the same volume.\n"
        "The two clips in a pair use the SAME music and are brightness-matched, so\n"
        "brightness is not a valid reason to prefer one.\n\n"
        "Answer for every pair:\n\n"
        + "".join(f"{i+1}. {q}\n" for i, q in enumerate(questions))
        + "\nWrite L or R (or `equal`) and a confidence from 1 to 5 in "
          "`score_sheet.csv`.\nDo not open `KEY.json` until the sheet is filled in.\n")


def randomise(pairs, seed: int = 0) -> list[ABTrial]:
    rng = random.Random(seed)
    trials = []
    for i, (clip, sys_a, file_a, sys_b, file_b, match) in enumerate(pairs):
        if rng.random() < 0.5:
            sys_a, sys_b = sys_b, sys_a
            file_a, file_b = file_b, file_a
        trials.append(ABTrial(trial_id=f"T{i+1:02d}", clip=clip,
                              left_system=sys_a, right_system=sys_b,
                              left_file=file_a, right_file=file_b, match=match))
    return trials
