"""Human-independent health checks for native dance/light trainings.

The functions in this module intentionally report measurements and conservative
classifications.  They do not decide that a checkpoint is artistically better.
All motion statistics are computed on the supplied native samples, before
interpolation, DMX emission, filtering, or rendering.

The accepted array conventions are deliberately small:

* a body is ``(time, joints, 3)`` (``(time, 3)`` is accepted as one joint),
* streams are ``(time, outputs, channels)`` or ``(time, channels)``.

This module only requires NumPy.  It is therefore safe to run on a CPU after
an inference job and can be used to audit outputs produced by older revisions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-12


def _json_value(value: Any) -> Any:
    """Convert NumPy values to strict-JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _array(value: Any, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be numeric, got {arr.dtype}")
    return arr


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "std": None, "p50": None, "p95": None, "p99": None}
    q = np.percentile(values, [50, 95, 99])
    return {"count": int(values.size), "min": float(values.min()),
            "max": float(values.max()), "mean": float(values.mean()),
            "std": float(values.std()), "p50": float(q[0]),
            "p95": float(q[1]), "p99": float(q[2])}


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    if not value.size:
        return 0.0
    return float(np.sqrt(np.mean(value * value)))


def _finite_transition(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a).all(axis=tuple(range(1, a.ndim))) & np.isfinite(b).all(axis=tuple(range(1, b.ndim)))
    return a[mask], b[mask]


def motion_metrics(body: Any, fps: float = 30.0, active_threshold: float = 1e-5) -> dict[str, Any]:
    """Measure unfiltered body motion, including raw high-frequency residuals.

    ``fps`` is only used to convert frame differences into derivatives.  No
    temporal smoothing is applied.  Missing/nonfinite frames remain visible in
    the returned ``finite`` and ``valid_transition_fraction`` fields.
    """
    arr = _array(body, name="body").astype(np.float64, copy=False)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        arr = arr[:, None, :]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"body must have shape (T,J,3), got {arr.shape}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    finite = bool(np.isfinite(arr).all())
    valid_frames = np.isfinite(arr).all(axis=(1, 2))
    frame_count, joint_count = arr.shape[:2]
    if frame_count < 2:
        raise ValueError("body requires at least two frames")
    # Invalid pairs are omitted from each derivative, but never silently filled.
    pair_valid = valid_frames[1:] & valid_frames[:-1]
    delta = np.diff(arr, axis=0)
    velocity = delta[pair_valid] * float(fps)
    speed = np.linalg.norm(velocity, axis=-1).reshape(-1)
    accel = np.diff(arr, n=2, axis=0) * float(fps * fps)
    accel_valid = valid_frames[2:] & valid_frames[1:-1] & valid_frames[:-2]
    acceleration = np.linalg.norm(accel[accel_valid], axis=-1).reshape(-1)
    jerk = np.diff(arr, n=3, axis=0) * float(fps ** 3)
    jerk_valid = valid_frames[3:] & valid_frames[2:-1] & valid_frames[1:-2] & valid_frames[:-3]
    jerk_magnitude = np.linalg.norm(jerk[jerk_valid], axis=-1).reshape(-1)
    root = arr[:, 0, :]
    root_valid = valid_frames
    root_xyz = root[root_valid]
    root_excursion = float(np.ptp(root_xyz, axis=0).max()) if root_xyz.size else None
    position_range = float(np.ptp(arr[np.isfinite(arr).all(axis=(1, 2))])) if finite else None
    speed_rms = _rms(speed)
    jerk_rms = _rms(jerk_magnitude)
    active_frames = speed.reshape(-1, joint_count) if speed.size and joint_count else np.empty((0, joint_count))
    frame_speed = np.linalg.norm(velocity.reshape(-1, joint_count, 3), axis=-1).mean(axis=1) if velocity.size else np.empty(0)
    constant = bool(speed.size == 0 or np.all(speed <= active_threshold))
    return _json_value({
        "shape": list(arr.shape), "fps": float(fps), "finite": finite,
        "all_finite": finite,
        "finite_frame_fraction": float(valid_frames.mean()),
        "finite_fraction": float(valid_frames.mean()),
        "finite_ratio": float(valid_frames.mean()),
        "valid_transition_fraction": float(pair_valid.mean()),
        "nonfinite_count": int((~np.isfinite(arr)).sum()),
        "position": _finite_stats(arr.reshape(-1)),
        "position_range": position_range,
        "speed_raw": _finite_stats(speed),
        # Report the frame-level profile for gates; flattening all joints would
        # make a sparse articulated gesture appear to have a zero 95th
        # percentile simply because most joints are held.
        "speed_p95": float(np.percentile(frame_speed, 95)) if frame_speed.size else None,
        "p95_speed": float(np.percentile(frame_speed, 95)) if frame_speed.size else None,
        "p95_speed_m_s": float(np.percentile(frame_speed, 95)) if frame_speed.size else None,
        "acceleration_raw": _finite_stats(acceleration),
        "jerk_raw": _finite_stats(jerk_magnitude),
        "frame_speed_raw": _finite_stats(frame_speed),
        "active_frame_fraction": float(np.mean(frame_speed > active_threshold)) if frame_speed.size else None,
        "inactive_frame_fraction": float(np.mean(frame_speed <= active_threshold)) if frame_speed.size else None,
        "constant": constant,
        "constant_body": constant,
        "exact_constant": bool(np.all(arr == arr[:1])),
        "root_xyz_range_m": root_excursion,
        "root_z_stats": _finite_stats(root_xyz[:, 2]) if root_xyz.size else _finite_stats(np.array([])),
        # This is a diagnostic ratio, not a low-pass residual or quality score.
        "jerk_to_speed_rms_ratio": float(jerk_rms / max(speed_rms, EPS)) if speed.size else None,
        "raw_unfiltered": True,
        "support_contact_proxy": None,
    })


def _stream_view(streams: Any) -> tuple[np.ndarray, int, int]:
    arr = _array(streams, name="streams")
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim < 2:
        raise ValueError(f"streams must have time as axis 0, got {arr.shape}")
    # A 3-D DMX tensor is one stream per output. Higher-dimensional tensors are
    # flattened into output records while retaining channels for each record.
    time = arr.shape[0]
    if arr.ndim == 2:
        view = arr[:, :, None]
    else:
        view = arr.reshape(time, int(np.prod(arr.shape[1:-1])), arr.shape[-1])
    return view.astype(np.float64, copy=False), int(view.shape[1]), int(view.shape[2])


def stream_metrics(streams: Any, *, tolerance: float = 1e-9,
                   rgb_channels: Sequence[int] | None = None,
                   brightness_channels: Sequence[int] | None = None) -> dict[str, Any]:
    """Audit emitted streams while keeping color and brightness separate.

    ``rgb_channels`` and ``brightness_channels`` are explicit because a generic
    DMX universe does not identify channel semantics.  If omitted, color and
    brightness changes are reported as unavailable rather than inferred.
    """
    source = _array(streams, name="streams")
    source_integer = bool(np.issubdtype(source.dtype, np.integer))
    view, output_count, channel_count = _stream_view(source)
    finite = bool(np.isfinite(view).all())
    output_hashes = [_digest(view[:, i]) for i in range(output_count)]
    groups: dict[str, list[int]] = {}
    for idx, digest in enumerate(output_hashes):
        groups.setdefault(digest, []).append(idx)
    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    valid = np.isfinite(view).all(axis=2)
    active = np.abs(view) > tolerance
    transitions = np.abs(np.diff(view, axis=0)) > tolerance
    changed = np.any(transitions, axis=2) if transitions.size else np.empty((0, output_count), bool)
    result: dict[str, Any] = {
        "shape": list(view.shape), "finite": finite,
        "nonfinite_count": int((~np.isfinite(view)).sum()),
        "output_count": output_count, "channel_count": channel_count,
        "valid_output_frame_fraction": float(valid.mean()),
        "active_value_fraction": float(active.mean()),
        # ``inactive_fraction`` is the fraction of output streams that never
        # emit a nonzero value.  Keep per-value inactivity separate so an
        # isolated zero sample at t=0 cannot make an active fixture fail.
        "inactive_value_fraction": float(1.0 - active.mean()),
        "inactive_fraction": float(np.mean(np.all(~active, axis=(0, 2)))),
        "inactive_stream_fraction": float(np.mean(np.all(~active, axis=(0, 2)))),
        "changed_output_frame_fraction": float(changed.mean()) if changed.size else None,
        "per_output_active_fraction": [float(x) for x in active.mean(axis=(0, 2))],
        "per_output_changed_fraction": [float(x) for x in changed.mean(axis=0)] if changed.size else [None] * output_count,
        "channel_stats": [_finite_stats(view[:, :, c].reshape(-1)) for c in range(channel_count)],
        "saturation": None,
        "output_sha256": output_hashes,
        "exact_duplicate_groups": duplicate_groups,
        "duplicate_groups": duplicate_groups,
        "exact_duplicate_output_count": int(sum(len(x) for x in duplicate_groups)),
        "rgb": None, "brightness": None,
        "scope": "raw supplied streams; no interpolation, fixture caps, smoothing, or visual inference",
    }
    def semantic(channels: Sequence[int], label: str) -> dict[str, Any]:
        idx = [int(c) for c in channels]
        if not idx or any(c < 0 or c >= channel_count for c in idx):
            raise ValueError(f"{label} channel index outside 0..{channel_count - 1}")
        data = view[:, :, idx]
        delta = np.abs(np.diff(data, axis=0)) if len(data) > 1 else np.empty((0, output_count, len(idx)))
        return {"channels": idx, "value_stats": _finite_stats(data.reshape(-1)),
                "active_fraction": float((np.abs(data) > tolerance).mean()),
                "changed_value_fraction": float((delta > tolerance).mean()) if delta.size else None,
                "changed_frame_fraction": float(np.any(delta > tolerance, axis=2).mean()) if delta.size else None,
                "unique_frame_count": int(np.unique(np.round(data, 12).reshape(data.shape[0], -1), axis=0).shape[0])}
    if rgb_channels is not None:
        result["rgb"] = semantic(rgb_channels, "rgb")
    if brightness_channels is not None:
        result["brightness"] = semantic(brightness_channels, "brightness")
    # Saturation is only meaningful when the serialized stream has a known
    # integer range. Floating point streams may use arbitrary model units, so
    # report it as unavailable rather than inventing a [0,1] or [0,255] scale.
    if source_integer:
        info = np.iinfo(source.dtype)
        result["saturation"] = {
            "dtype": str(source.dtype), "minimum": int(info.min), "maximum": int(info.max),
            "low_clipped_fraction": float(np.mean(view <= info.min)),
            "high_clipped_fraction": float(np.mean(view >= info.max)),
            "saturated_fraction": float(np.mean((view <= info.min) | (view >= info.max))),
        }
    return _json_value(result)


def _pairwise(samples: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for i, j in itertools.combinations(range(len(samples)), 2):
        a, b = np.asarray(samples[i], dtype=np.float64), np.asarray(samples[j], dtype=np.float64)
        if a.shape != b.shape:
            rows.append({"i": i, "j": j, "shape_match": False, "rms": None, "normalized_rms": None})
            continue
        finite = np.isfinite(a) & np.isfinite(b)
        delta = (a - b)[finite]
        ar = a[finite]
        denom = max(_rms(ar), EPS)
        rows.append({"i": i, "j": j, "shape_match": True, "finite_fraction": float(finite.mean()),
                     "rms": _rms(delta), "normalized_rms": float(_rms(delta) / denom),
                     "exact": bool(np.array_equal(a, b))})
    return rows


def collapse_metrics(samples: Sequence[Any], *, near_duplicate_threshold: float = 1e-6) -> dict[str, Any]:
    """Measure sample diversity and report collapse mechanisms separately.

    A zero event count or quiet output is not called representation collapse.
    ``output_inactivity`` belongs in :func:`stream_metrics`; this function only
    compares multiple samples from a model under fixed or changed inputs.
    """
    arrays = [_array(x, name=f"sample[{i}]").astype(np.float64, copy=False) for i, x in enumerate(samples)]
    pairs = _pairwise(arrays)
    comparable = [row for row in pairs if row.get("shape_match") and row.get("finite_fraction") == 1.0]
    exact = bool(comparable) and all(bool(row["exact"]) for row in comparable)
    near = bool(comparable) and all(float(row["normalized_rms"]) <= near_duplicate_threshold for row in comparable)
    matrix = np.stack([x.reshape(-1) for x in arrays], axis=0) if arrays and len({x.size for x in arrays}) == 1 else None
    effective_rank = None
    rank_fraction = None
    if matrix is not None and matrix.shape[0] > 1:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        energy = singular * singular
        effective_rank = int(np.sum(singular > max(singular[0] if singular.size else 0.0, EPS) * 1e-8)) if singular.size else 0
        rank_fraction = float(effective_rank / min(centered.shape)) if min(centered.shape) else 0.0
    status = "unmeasurable" if len(arrays) < 2 else ("exact_duplicate" if exact else ("near_duplicate" if near else "varied"))
    mean_pairwise = float(np.mean([row["rms"] for row in comparable])) if comparable else None
    return _json_value({"sample_count": len(arrays), "shapes": [list(x.shape) for x in arrays],
                        "finite": [bool(np.isfinite(x).all()) for x in arrays],
                        "pairwise": pairs, "status": status, "exact_duplicate": exact,
                        "near_duplicate": near, "effective_rank": effective_rank,
                        "rank_fraction": rank_fraction,
                        "representation_collapse": bool(exact or near),
                        "collapsed": bool(exact or near), "collapse": bool(exact or near),
                        "mean_pairwise_distance": mean_pairwise,
                        "diversity": mean_pairwise,
                        "scope": "sample diversity only; inactivity and audio response are separate diagnostics"})


def input_sensitivity(reference: Any, interventions: Mapping[str, Any], *, tolerance: float = 1e-6) -> dict[str, Any]:
    """Compare a fixed-input output against audio/limb/seed interventions."""
    base = _array(reference, name="reference").astype(np.float64, copy=False)
    rows = []
    for label, value in interventions.items():
        candidate = _array(value, name=f"intervention[{label}]").astype(np.float64, copy=False)
        if candidate.shape != base.shape:
            rows.append({"label": str(label), "status": "shape_mismatch", "reference_shape": list(base.shape),
                         "candidate_shape": list(candidate.shape)})
            continue
        finite = np.isfinite(base) & np.isfinite(candidate)
        delta = candidate - base
        observed = delta[finite]
        denom = max(_rms(base[finite]), EPS) if finite.any() else EPS
        rms = _rms(observed)
        changed = bool(rms > tolerance)
        rows.append({"label": str(label), "status": "response" if changed else "no_response",
                     "changed": changed, "sensitive": changed,
                     "finite_fraction": float(finite.mean()), "rms": rms,
                     "rms_difference": rms, "difference_rms": rms,
                     "normalized_rms": float(rms / denom), "maximum_absolute": float(np.max(np.abs(observed))) if observed.size else 0.0,
                     "fraction_values_changed": float(np.mean(np.abs(delta[finite]) > tolerance)) if finite.any() else None,
                     "exact": bool(np.array_equal(base, candidate))})
    row_map = {str(row["label"]): row for row in rows}
    return _json_value({"reference_shape": list(base.shape), "interventions": row_map,
                        "intervention_rows": rows,
                        "finite": bool(np.isfinite(base).all() and all(np.isfinite(_array(v)).all() for v in interventions.values())),
                        "responsive_intervention_count": sum(r.get("status") == "response" for r in rows),
                        "scope": "output response under declared interventions; response does not establish choreography"})


def provenance_report(receipt: Mapping[str, Any], required_fields: Iterable[str] = (),
                      *, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Check required provenance and hash-linked files without trusting labels."""
    required = [str(x) for x in required_fields]
    missing = [key for key in required if key not in receipt or receipt[key] in (None, "")]
    invalid_hash_fields = [
        key for key in required
        if key.endswith("_sha256") and key in receipt
        and (not isinstance(receipt[key], str) or re.fullmatch(r"[0-9a-fA-F]{64}", receipt[key]) is None)
    ]
    paths: dict[str, Any] = {}
    for key, value in receipt.items():
        if not key.endswith("_path") or not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        if base_dir is not None and not path.is_absolute():
            path = Path(base_dir) / path
        digest_key = key[:-5] + "_sha256"
        row: dict[str, Any] = {"path": str(path), "exists": path.exists(), "hash_field": digest_key}
        if path.exists() and digest_key in receipt:
            row["sha256"] = _digest(np.frombuffer(path.read_bytes(), dtype=np.uint8))
            row["hash_matches"] = bool(row["sha256"] == receipt[digest_key])
        paths[key] = row
    mismatches = [key for key, row in paths.items() if row.get("exists") and row.get("hash_matches") is False]
    missing_paths = [key for key, row in paths.items() if not row.get("exists")]
    complete = not missing and not invalid_hash_fields and not mismatches and not missing_paths
    status = "qualified" if complete else "unqualified"
    return _json_value({"status": status, "verification_status": "verified" if complete else "incomplete",
                        "verdict": status, "complete": complete, "provenance_complete": complete,
                        "required_fields": required, "missing": missing, "missing_fields": missing,
                        "invalid_fields": invalid_hash_fields,
                        "path_checks": paths, "hash_mismatches": mismatches, "missing_paths": missing_paths,
                        "scope": "metadata/path/hash consistency; does not infer training quality"})


def compare_training_runs(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Conservative comparison of two generated audit reports.

    This intentionally uses explicit gate values supplied by the caller.  It
    avoids promoting a model from one metric (for example lower filtered speed
    error) while raw stability, activity, sensitivity, or provenance fails.
    """
    gates: dict[str, str] = {}
    for name in ("finite", "provenance", "collapse", "input_sensitivity", "raw_dynamics",
                 "streams", "duplicate_streams", "output_activity"):
        left, right = baseline.get("gates", {}).get(name), candidate.get("gates", {}).get(name)
        if left is None or right is None:
            gates[name] = "unmeasurable"
        elif right is True and left is not True:
            gates[name] = "candidate_passes"
        elif right is not True and left is True:
            gates[name] = "candidate_fails"
        else:
            gates[name] = "unchanged"
    failed = [name for name, value in gates.items() if value == "candidate_fails"]
    improved = [name for name, value in gates.items() if value == "candidate_passes"]
    complete = not any(value == "unmeasurable" for value in gates.values())
    status = "regression" if failed else ("improvement" if improved else "unchanged_or_unmeasurable")
    qualified_status = (
        "qualified_regression" if complete and failed else
        "qualified_improvement" if complete and improved else
        "unqualified" if not complete else
        "diagnostic_only"
    )
    return _json_value({"status": status, "qualified_status": qualified_status,
                        "complete": complete, "gate_comparison": gates,
                        "candidate_failed_gates": failed, "candidate_improved_gates": improved,
                        "scope": "automated health gates only; no artistic or musical acceptance"})


# Name used by early audit notebooks; keep it as a stable compatibility alias.
compare_bundle = compare_training_runs


def load_npz_array(path: str | Path, keys: Sequence[str]) -> np.ndarray:
    """Load the first matching array from a saved output file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as bundle:
        for key in keys:
            if key in bundle:
                return np.asarray(bundle[key])
        if len(bundle.files) == 1:
            return np.asarray(bundle[bundle.files[0]])
    raise KeyError(f"None of {tuple(keys)} found in {path}")


def dump_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
