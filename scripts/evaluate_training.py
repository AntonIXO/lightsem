#!/usr/bin/env python3
"""Run CPU-only health checks on saved training outputs.

Examples (all inputs are read-only)::

  bash run.sh scripts/evaluate_training.py \
    --body /mnt/.../song/body.npz --stream /mnt/.../song/universes.npz \
    --sample shpongle=/mnt/.../shpongle/body.npz \
    --sample cure_for_me=/mnt/.../cure_for_me/body.npz \
    --intervention limb_left=/mnt/.../limb_left/body.npz \
    --provenance /mnt/.../inference_receipt.json \
    --require-provenance checkpoint_sha256,audio_input_sha256,source_body_sha256 \
    --out /mnt/.../training_health.json

The output is a new report.  Existing reports are never overwritten, which
prevents an audit from silently changing a sealed experiment receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lightsem.eval.training_health import (
    collapse_metrics,
    compare_training_runs,
    dump_json,
    input_sensitivity,
    load_npz_array,
    motion_metrics,
    provenance_report,
    stream_metrics,
)


def _label_path(values: list[str], flag: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{flag} expects LABEL=PATH, got {raw!r}")
        label, path = raw.split("=", 1)
        if not label or not path:
            raise ValueError(f"{flag} expects LABEL=PATH, got {raw!r}")
        if label in result:
            raise ValueError(f"duplicate {flag} label {label!r}")
        result[label] = Path(path)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON report must be an object: {path}")
    return value


def _plot(path: Path, body: np.ndarray, streams: np.ndarray | None, fps: float) -> None:
    # Plotting is optional so a CPU audit never needs matplotlib installed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = np.asarray(body, dtype=float)
    if body.ndim == 2:
        body = body[:, None, :]
    figure, axes = plt.subplots(2 if streams is not None else 1, 1, figsize=(12, 6), squeeze=False)
    ax = axes[0, 0]
    if len(body) > 1:
        speed = np.linalg.norm(np.diff(body, axis=0) * fps, axis=2).mean(axis=1)
        ax.plot(np.arange(len(speed)) / fps, speed, color="#245a9c", linewidth=0.8)
    ax.set_title("Raw native body mean joint speed")
    ax.set_ylabel("m/s (or model units)")
    ax.set_xlabel("song seconds")
    if streams is not None:
        ax = axes[1, 0]
        arr = np.asarray(streams, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        active = np.any(np.abs(arr) > 1e-9, axis=tuple(range(2, arr.ndim))) if arr.ndim > 2 else np.abs(arr) > 1e-9
        ax.plot(np.arange(len(active)) / fps, active.mean(axis=tuple(range(1, active.ndim))), color="#a64040", linewidth=0.8)
        ax.set_title("Raw emitted output activity")
        ax.set_ylabel("active output fraction")
        ax.set_xlabel("song seconds")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing plot: {path}")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _comparison_plot(path: Path, baseline: np.ndarray, candidate: np.ndarray, fps: float) -> None:
    """Plot raw reference/candidate speed and signed difference on one pair scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def speed(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if value.ndim == 2:
            value = value[:, None, :]
        return np.linalg.norm(np.diff(value, axis=0) * fps, axis=2).mean(axis=1)
    ref_raw, cand_raw = speed(baseline), speed(candidate)
    n = max(len(ref_raw), len(cand_raw))
    ref = np.full(n, np.nan, dtype=float)
    cand = np.full(n, np.nan, dtype=float)
    ref[:len(ref_raw)], cand[:len(cand_raw)] = ref_raw, cand_raw
    time = np.arange(n) / fps
    # Shared pair scale includes every finite raw peak from both roles.
    finite = np.isfinite(np.concatenate((ref, cand)))
    ymax = float(np.max(np.concatenate((ref, cand))[finite])) if finite.any() else 1.0
    ymax = max(ymax, 1e-12)
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time, ref, color="#245a9c", linewidth=0.8, label="reference")
    axes[0].plot(time, cand, color="#a64040", linewidth=0.8, label="candidate")
    axes[0].set_ylim(0.0, ymax * 1.05)
    axes[0].set_ylabel("raw mean speed")
    axes[0].set_title("Reference/candidate raw native speed (shared pair scale)")
    axes[0].legend(loc="upper right")
    difference = np.where(np.isfinite(ref) & np.isfinite(cand), cand - ref, np.nan)
    axes[1].plot(time, difference, color="#444444", linewidth=0.8)
    axes[1].axhline(0.0, color="#999999", linewidth=0.5)
    axes[1].set_ylabel("candidate − reference")
    axes[1].set_xlabel("song seconds")
    axes[1].set_title("Signed raw difference; no clipping or smoothing")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing comparison plot: {path}")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _apply_manifest(args: argparse.Namespace) -> argparse.Namespace:
    """Fill CLI options from a small, repository-portable manifest."""
    if not args.manifest:
        if not args.body:
            raise SystemExit("either --body or --manifest is required")
        return args
    manifest_path = Path(args.manifest)
    value = _load_json(manifest_path)
    if value.get("schema") not in (None, "lightsem.training_health.manifest.v1"):
        raise SystemExit(f"unsupported training-health manifest schema: {value.get('schema')!r}")
    base = manifest_path.parent
    def resolve(item: Any) -> Any:
        if item is None:
            return None
        path = Path(item)
        return path if path.is_absolute() else base / path
    def pairs(item: Any) -> list[str]:
        if item is None:
            return []
        if isinstance(item, Mapping):
            return [f"{key}={resolve(path)}" for key, path in item.items()]
        if isinstance(item, list):
            return [str(x) for x in item]
        raise SystemExit("manifest samples/interventions must be an object or list")
    if args.body is None:
        args.body = resolve(value.get("body"))
    if args.stream is None:
        args.stream = resolve(value.get("stream"))
    if not args.sample:
        args.sample = pairs(value.get("samples"))
    if not args.intervention:
        args.intervention = pairs(value.get("interventions"))
    if args.provenance is None:
        args.provenance = resolve(value.get("provenance"))
    if not args.require_provenance and value.get("required_provenance"):
        args.require_provenance = ",".join(value["required_provenance"])
    if value.get("fps") is not None and args.fps == 30.0:
        args.fps = float(value["fps"])
    limits = value.get("limits") or {}
    for name in ("max_speed", "max_acceleration", "max_jerk", "min_body_activity", "min_output_activity"):
        if getattr(args, name, None) is None and name in limits:
            setattr(args, name, float(limits[name]))
    if args.rgb_channels is None and value.get("rgb_channels") is not None:
        args.rgb_channels = list(value["rgb_channels"])
    if args.brightness_channels is None and value.get("brightness_channels") is not None:
        args.brightness_channels = list(value["brightness_channels"])
    if args.body is None:
        raise SystemExit("training-health manifest is missing required field 'body'")
    return args


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    body = load_npz_array(args.body, ("body", "world", "native", "positions"))
    report: dict[str, Any] = {
        "schema": "lightsem.training_health.v1",
        "inputs": {"body": str(Path(args.body).resolve())},
        "metrics": {"motion": motion_metrics(body, fps=args.fps)},
        "gates": {},
    }
    stream = None
    if args.stream:
        stream = load_npz_array(args.stream, ("universes", "dmx", "streams", "commands", "rgb"))
        report["inputs"]["stream"] = str(Path(args.stream).resolve())
        report["metrics"]["streams"] = stream_metrics(
            stream, rgb_channels=args.rgb_channels, brightness_channels=args.brightness_channels)
    samples = _label_path(args.sample, "--sample")
    if samples:
        arrays = [load_npz_array(path, ("body", "world", "native", "positions")) for path in samples.values()]
        report["inputs"]["samples"] = {label: str(path.resolve()) for label, path in samples.items()}
        report["metrics"]["collapse"] = collapse_metrics(arrays)
        report["metrics"]["collapse"]["labels"] = list(samples)
    interventions = _label_path(args.intervention, "--intervention")
    if interventions:
        values = {label: load_npz_array(path, ("body", "world", "native", "positions")) for label, path in interventions.items()}
        report["inputs"]["interventions"] = {label: str(path.resolve()) for label, path in interventions.items()}
        report["metrics"]["input_sensitivity"] = input_sensitivity(body, values)
    if args.provenance:
        receipt = _load_json(Path(args.provenance))
        required = [x for x in args.require_provenance.split(",") if x]
        report["inputs"]["provenance"] = str(Path(args.provenance).resolve())
        report["metrics"]["provenance"] = provenance_report(
            receipt, required, base_dir=Path(args.provenance).parent)

    motion = report["metrics"]["motion"]
    streams = report["metrics"].get("streams")
    collapse = report["metrics"].get("collapse")
    sensitivity = report["metrics"].get("input_sensitivity")
    provenance = report["metrics"].get("provenance")
    max_speed = getattr(args, "max_speed", None)
    max_acceleration = getattr(args, "max_acceleration", None)
    max_jerk = getattr(args, "max_jerk", None)
    min_body_activity = getattr(args, "min_body_activity", None)
    min_output_activity = getattr(args, "min_output_activity", None)
    dynamic_limits = {
        "max_speed_p95": max_speed,
        "max_acceleration_p95": max_acceleration,
        "max_jerk_p95": max_jerk,
        "min_body_activity": min_body_activity,
    }
    dynamic_failures = []
    if max_speed is not None and motion["speed_p95"] is not None and motion["speed_p95"] > max_speed:
        dynamic_failures.append("max_speed_p95")
    if max_acceleration is not None and motion["acceleration_raw"]["p95"] is not None and motion["acceleration_raw"]["p95"] > max_acceleration:
        dynamic_failures.append("max_acceleration_p95")
    if max_jerk is not None and motion["jerk_raw"]["p95"] is not None and motion["jerk_raw"]["p95"] > max_jerk:
        dynamic_failures.append("max_jerk_p95")
    if min_body_activity is not None and (motion["active_frame_fraction"] is None or motion["active_frame_fraction"] < min_body_activity):
        dynamic_failures.append("min_body_activity")
    report["metrics"]["raw_dynamics_limits"] = {"limits": dynamic_limits, "failures": dynamic_failures,
                                                   "applied": bool(any(value is not None for value in dynamic_limits.values()))}
    report["gates"] = {
        "finite": bool(motion["finite"] and (streams is None or streams["finite"])),
        "raw_dynamics": bool(motion["finite"] and motion["valid_transition_fraction"] > 0 and not dynamic_failures),
        "streams": None if streams is None else bool(streams["finite"]),
        "duplicate_streams": None if streams is None else bool(not streams["exact_duplicate_groups"]),
        "output_activity": None if streams is None else bool(
            streams["active_value_fraction"] > (min_output_activity if min_output_activity is not None else 0.0)),
        "collapse": None if collapse is None else bool(not collapse["representation_collapse"]),
        "input_sensitivity": None if sensitivity is None else bool(
            sensitivity["responsive_intervention_count"] == len(sensitivity["interventions"])
            and bool(sensitivity["interventions"])),
        "provenance": None if provenance is None else bool(
            provenance.get("status") in {"verified", "qualified", "complete", "pass"}
            or provenance.get("verification_status") == "verified"
        ),
    }
    failed = [name for name, value in report["gates"].items() if value is False]
    unmeasured = [name for name, value in report["gates"].items() if value is None]
    # Keep the public status vocabulary explicit: a report can be useful while
    # still being diagnostic-only, and passing checks never establishes artistic
    # improvement.  ``health_status`` retains the compact pass/fail summary.
    safety_failures = [name for name in ("finite", "raw_dynamics", "streams", "output_activity", "provenance")
                       if report["gates"].get(name) is False]
    if collapse is not None and collapse["representation_collapse"]:
        status = "collapsed"
    elif safety_failures:
        status = "failed_safety"
    elif unmeasured:
        status = "unqualified"
    else:
        status = "diagnostic_only"
    report["status"] = status
    report["health_status"] = "failed" if failed else ("unmeasurable" if unmeasured else "healthy_checks_passed")
    report["failed_gates"] = failed
    report["unmeasured_gates"] = unmeasured
    report["interpretation"] = (
        "Automated health checks passed; this does not establish musical or artistic improvement."
        if report["health_status"] == "healthy_checks_passed" else
        "A failed component is named by failed_gates; do not promote the checkpoint from this report."
        if failed else
        "At least one required diagnostic was not supplied; this report cannot qualify a checkpoint."
    )
    if args.plot:
        _plot(Path(args.plot), body, stream, args.fps)
        report["plot"] = str(Path(args.plot).resolve())
    if getattr(args, "comparison_plot", None):
        if not getattr(args, "compare_baseline", None):
            raise ValueError("--comparison-plot requires --compare-baseline")
        baseline = _load_json(args.compare_baseline)
        baseline_body_path = baseline.get("inputs", {}).get("body")
        if not baseline_body_path:
            raise ValueError("baseline report has no inputs.body for comparison plot")
        baseline_body = load_npz_array(baseline_body_path, ("body", "world", "native", "positions"))
        _comparison_plot(Path(args.comparison_plot), baseline_body, body, args.fps)
        report["comparison_plot"] = str(Path(args.comparison_plot).resolve())
        report["comparison_plot_scope"] = (
            "raw native mean joint speed; shared pair scale; every raw peak retained; "
            "signed candidate-minus-reference only where both clocks overlap; no smoothing"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, help="JSON manifest describing a complete run bundle")
    parser.add_argument("--body", type=Path, help="saved native body .npz")
    parser.add_argument("--stream", type=Path, help="saved emitted DMX/universe .npz")
    parser.add_argument("--sample", action="append", default=[], help="repeatable LABEL=body.npz for diversity/cross-song checks")
    parser.add_argument("--intervention", action="append", default=[], help="repeatable LABEL=body.npz changed input check")
    parser.add_argument("--provenance", type=Path, help="inference receipt JSON")
    parser.add_argument("--require-provenance", default="", help="comma-separated receipt fields required for verification")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-speed", type=float, help="optional raw frame-speed p95 upper bound")
    parser.add_argument("--max-acceleration", type=float, help="optional raw acceleration p95 upper bound")
    parser.add_argument("--max-jerk", type=float, help="optional raw jerk p95 upper bound")
    parser.add_argument("--min-body-activity", type=float, help="optional minimum active raw-body frame fraction")
    parser.add_argument("--min-output-activity", type=float, help="optional minimum emitted-stream active-value fraction")
    parser.add_argument("--rgb-channels", type=int, nargs="+", help="explicit RGB channel indices")
    parser.add_argument("--brightness-channels", type=int, nargs="+", help="explicit brightness/dimmer channel indices")
    parser.add_argument("--plot", type=Path, help="optional raw speed/activity PNG")
    parser.add_argument("--comparison-plot", type=Path,
                        help="optional reference/candidate raw overlay + signed difference PNG; requires --compare-baseline")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True,
                        help="new JSON report; existing files are never overwritten")
    parser.add_argument("--compare-baseline", type=Path, help="prior report for conservative gate comparison")
    args = _apply_manifest(parser.parse_args())
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing report: {args.out}")
    report = build_report(args)
    if args.compare_baseline:
        report["comparison"] = compare_training_runs(_load_json(args.compare_baseline), report)
    dump_json(args.out, report)
    print(json.dumps({"status": report["status"], "health_status": report["health_status"],
                      "comparison_status": report.get("comparison", {}).get("status"),
                      "comparison_qualified_status": report.get("comparison", {}).get("qualified_status"),
                      "failed_gates": report["failed_gates"],
                      "unmeasured_gates": report["unmeasured_gates"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
