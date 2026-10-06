"""Restore the screenshot demo's loudness envelope without changing motion/RGB.

The full-track audio envelope and six-head reference are historical equations.
Applying the modest regional term to V3's fixed anatomical roles, and applying
the envelope to its bulb field, are explicit capability adaptations. They are
not a claim of byte-identical fixtures across different installations.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import json

import numpy as np


HISTORICAL_SOURCE = Path(
    "/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/"
    "pose_stage_01/frozen/source/lightsem/experiments/motion_study/lighting_map.py"
)
HISTORICAL_SHA256 = "55faff3b7f327b88b88db72a7a4e65fbd292520eda8b669b5bed7d703665a99d"
FPS = 40.0


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_verified_loudness(receipt_path, audio_path):
    """Read a full-track LUFS cache after checking its audio and array hashes."""
    path = Path(receipt_path)
    report = json.loads(path.read_text())
    if "payload" in report:
        payload = report["payload"]
        # Older campaign envelopes use sorted compact JSON for the seal.
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != report["sha256"]:
            raise ValueError("Loudness receipt authentication failed")
        report = payload
    else:
        seal = json.loads(Path(str(path)+".sha256.json").read_text())
        if _sha(path) != seal["sha256"]:
            raise ValueError("Loudness receipt authentication failed")
    if _sha(audio_path) != report["audio_sha256"] or _sha(report["path"]) != report["sha256"]:
        raise ValueError("Loudness audio or arrays changed")
    with np.load(report["path"], allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ("time", "momentary", "short_term")}
    return result, {"receipt": str(path), "receipt_sha256": _sha(path), **report}


def _clock(times):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Expected finite one-dimensional lighting timestamps")
    if not np.allclose(times, np.arange(len(times))/FPS, rtol=0, atol=1e-8):
        raise ValueError("Brightness requires the full zero-based 40 Hz clock; crop only afterward")
    return times


def _body(body, frames):
    body = np.asarray(body, dtype=np.float64)
    if body.shape != (frames, 22, 3) or not np.isfinite(body).all():
        raise ValueError("Expected finite 40 Hz world body [T,22,3]")
    return body


def _ema(values, seconds):
    # Operation order matches the authenticated historical source exactly.
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    alpha = -np.expm1(-1/(max(seconds, 1e-6)*FPS))
    for index in range(1, len(x)):
        out[index] = out[index-1]+alpha*(x[index]-out[index-1])
    return out


def historical_envelope(loudness_dict, times):
    """Return activity, level, LUFS traces and source evidence for a whole song.

    Supply the complete song's LUFS cache even for a shorter body prefix: the
    90th-percentile anchor is over that complete cache, never the visible clip.
    The caller binds the cache's full-song duration and waveform hash.
    """
    times = _clock(times)
    values = {key: np.asarray(loudness_dict[key], dtype=np.float64)
              for key in ("time", "momentary", "short_term")}
    clock = values["time"]
    if (clock.ndim != 1 or not len(clock) or np.any(np.diff(clock) <= 0)
            or clock[0] < 0 or clock[0] > .11
            or any(value.shape != clock.shape or not np.isfinite(value).all()
                   for value in values.values())):
        raise ValueError("Invalid full-track LUFS curves")
    if clock[-1] < times[-1]-.11:
        raise ValueError("LUFS cache does not cover the complete lighting interval")
    momentary = np.interp(times, clock, values["momentary"])
    short = np.interp(times, clock, values["short_term"])
    valid = values["short_term"] > -70
    anchor = float(np.quantile(values["short_term"][valid], .90)) if valid.any() else -23.
    x = np.clip((.7*momentary+.3*short-(anchor-12.))/12., 0, 1)
    raw_level = x*x*(3-2*x)
    activity = _ema(raw_level, .30)
    level = .12+(.88-.12)*activity**1.35
    return {"activity": activity, "level": level, "momentary_lufs": momentary,
            "short_term_lufs": short, "raw_activity": raw_level,
            "evidence": {
                "historical_source": str(HISTORICAL_SOURCE),
                "historical_source_sha256": HISTORICAL_SHA256,
                "historical_lines": [207, 215, 229, 284, 285],
                "anchor_lufs": anchor, "anchor_quantile": .90,
                "loudness_frames": len(clock), "loudness_end_s": float(clock[-1]),
                "fps": FPS, "quiet_level": .12, "active_level": .88,
                "loudness_range_lu": 12., "momentary_weight": .7,
                "short_term_weight": .3, "ema_s": .30, "power": 1.35,
                "scope": "Historical full-track loudness control; no clip normalization."}}


def historical_motion_accent(body):
    """The original small accent is body speed, not inferred source onsets."""
    body = _body(body, len(body))
    speed = np.linalg.norm(np.diff(body, axis=0, prepend=body[:1])*FPS, axis=-1)
    return _ema(np.clip((speed[:, [20, 21, 10, 11]].mean(1)-.35)/2., 0, 1), .15)


def historical_six_head_dimmers(body, times, loudness_dict):
    """Exact old six-head brightness equations in left-to-right fixture order.

    Used only to audit restoration against historical data. Does not generate
    light motion or replace any current installation's anatomical assignments.
    """
    envelope = historical_envelope(loudness_dict, times)
    body = _body(body, len(times))
    activity, level = envelope["activity"], envelope["level"]
    accent = historical_motion_accent(body)
    root = body[:, 0]
    result = []
    for rank in range(6):
        side = -1 if rank < 3 else 1
        pair = min(rank, 5-rank)
        if pair == 0:
            regional = body[:, 20 if side < 0 else 21]-root
        elif pair == 1:
            regional = .65*(body[:, 10 if side < 0 else 11]-root)+.35*(body[:, 4 if side < 0 else 5]-root)
        else:
            regional = .65*(body[:, 12]-root)+.35*(body[:, 15]-root)
            regional[:, 0] += side*.12
        regional_accent = .025*accent*activity
        result.append(np.clip(level*(.94+.06*side*np.tanh(regional[:, 0]))+regional_accent, 0, .95))
    return np.stack(result, axis=1)


def apply_brightness(body, times, patch, base_intent, loudness):
    """Return ``(updated_intent_or_dimmers, trace)`` without mutating inputs.

    Call before compile_intent. ``patch`` may be make_outputs' dictionaries or
    emitted patch dictionaries; fixed role_index/blend identities are required.
    Only dimmer is replaced. Body, positions, targets, colors and angular limits
    are neither recomputed nor edited here.
    """
    envelope = historical_envelope(loudness, times)
    body = _body(body, len(times))
    is_intent = isinstance(base_intent, Mapping)
    original = np.asarray(base_intent["dimmer"] if is_intent else base_intent, dtype=np.float64)
    if original.shape != (len(times), len(patch)) or not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
        raise ValueError("Expected finite per-fixture original dimmers in [0,1]")
    if not patch or len({fixture["name"] for fixture in patch}) != len(patch):
        raise ValueError("Expected nonempty, unique fixed fixture identities")
    level, activity = envelope["level"], envelope["activity"]
    accent = historical_motion_accent(body)
    relative = body-body[:, :1]
    dimmer = np.empty_like(original)
    regional_factor = np.ones_like(original)
    attribution = []
    for index, fixture in enumerate(patch):
        kind = fixture["kind"]
        if kind == "bulb":
            # .9 is the analytical maximum of the V3 bulb field. The fixed
            # normalization preserves all spatial ratios and arbitrary-N IDs.
            if np.any(original[:, index] > .9+1e-12):
                raise ValueError("Unexpected V3 bulb range; refuse silent renormalization")
            regional_factor[:, index] = original[:, index]/.9
            dimmer[:, index] = level*regional_factor[:, index]
            attribution.append({"name": fixture["name"], "kind": kind,
                                "brightness_source": "audio envelope times unchanged V3 anatomical bulb field / .9"})
            continue
        if kind not in ("beam", "laser"):
            raise ValueError("Unsupported brightness fixture capability")
        role, secondary = fixture["role_index"], fixture["secondary_role"]
        blend = float(fixture["role_blend"])
        if (isinstance(role, bool) or isinstance(secondary, bool)
                or not isinstance(role, (int, np.integer)) or not isinstance(secondary, (int, np.integer))
                or role not in range(22) or secondary not in range(22)
                or not np.isfinite(blend) or not 0 <= blend <= 1):
            raise ValueError("Invalid anatomical brightness role")
        x_position = float(fixture["position"][0])
        if not np.isfinite(x_position):
            raise ValueError("Nonfinite fixture position")
        side = -1 if x_position < 0 else 1
        regional = (1-blend)*relative[:, role]+blend*relative[:, secondary]
        regional_factor[:, index] = .94+.06*side*np.tanh(regional[:, 0])
        dimmer[:, index] = np.clip(level*regional_factor[:, index]+.025*accent*activity, 0, .95)
        attribution.append({"name": fixture["name"], "kind": kind, "role_index": int(role),
                            "secondary_role": int(secondary), "role_blend": blend, "mount_side": side,
                            "brightness_source": "historical audio law and modest body modulation; V3 anatomical roles"})
    trace = {**envelope, "motion_accent": accent, "regional_factor": regional_factor,
             "evidence": {**envelope["evidence"], "fixture_brightness_attribution": attribution,
                          "adaptation": "Beam/laser regional readout uses fixed V3 roles; bulbs retain relative V3 field. Historical six-head DMX identity is not claimed.",
                          "body_accent_formula": ".025 * audio_activity * EMA_150ms(clip((mean wrist/foot world speed-.35)/2,0,1))"}}
    if is_intent:
        updated = dict(base_intent)
        updated["dimmer"] = dimmer
    else:
        updated = dimmer
    return updated, trace
