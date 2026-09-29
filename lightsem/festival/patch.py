"""Deterministic venue fanout of LightSem's accepted DMX, with no model changes.

The exported bytes are the renderer's source of truth. Every fixture byte is
copied from an explicitly recorded canonical-rig slot. More fixtures therefore
mean more installed outputs, not more independently learned lighting decisions.
Coordinates are metres: x right, y towards audience, z up; stage is at y=-14.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from lightsem.conductor.rig import canonical_rig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "configs/light_layers/baseline.json"
COUNTS = {"moving_head": 24, "wash": 12, "projector": 4, "laser": 4, "garland": 12}
CHANNELS = {
    "moving_head": ["pan_coarse", "pan_fine", "tilt_coarse", "tilt_fine",
                    "dimmer", "strobe", "red", "green", "blue", "zoom", "focus", "gobo"],
    "wash": ["dimmer", "strobe", "red", "green", "blue", "zoom", "saturation"],
    "projector": ["dimmer", "strobe", "red", "green", "blue", "zoom", "focus",
                  "gobo", "pan", "tilt"],
    "garland": ["dimmer", "red", "green", "blue"],
}
CHANNELS["laser"] = CHANNELS["moving_head"].copy()
CONTROL_KEYS = ("dimmer", "rgb", "pan", "tilt", "zoom", "strobe", "focus", "gobo")


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return int(value)


def _vector(value, name: str) -> list[float]:
    out = np.asarray(value, dtype=float)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return out.tolist()


def allocate_fixtures(fixtures: list[dict], first_universe: int = 1) -> list[dict]:
    """Allocate contiguous personalities, spilling to further Art-Net universes.

    Universe 0 is reserved for the untouched original rig. Art-Net port-address
    numbers are zero based (0..32767), DMX channel addresses are one based.
    """
    universe = _integer(first_universe, "first_universe", 1, 32767)
    address = 1
    result = copy.deepcopy(fixtures)
    for fixture in result:
        n = len(CHANNELS[fixture["kind"]])
        if address + n - 1 > 512:
            universe += 1
            address = 1
        if universe > 32767:
            raise ValueError("patch exceeds Art-Net's 15-bit universe address space")
        fixture.update(universe=universe, address=address, n_channels=n)
        address += n
    return result


def _finish_patch(fixtures: list[dict]) -> dict:
    patch = {
        "schema_version": 1,
        "name": "LightSem psychedelic festival",
        "coordinate_system": "metres; x right, y audience depth (stage -14), z up",
        "source_universe": 0,
        "universe_ids": [0] + sorted({f["universe"] for f in fixtures}),
        "profiles": {k: {"channels": v, "virtual": k == "laser"}
                     for k, v in CHANNELS.items()},
        "control_origin": "Deterministic byte fanout of accepted baseline C; no new model outputs",
        "projector_note": "Generic simulated media-server DMX, not a manufacturer fixture personality",
        "laser_note": "Virtual beam geometry only; this profile is not ILDA or a physical laser controller",
        "fixtures": fixtures,
    }
    validate_patch(patch)
    return patch


def build_patch(counts: dict[str, int] | None = None) -> dict:
    """Build a scalable installation. Unspecified fixture counts use defaults."""
    counts = {**COUNTS, **(counts or {})}
    if set(counts) != set(COUNTS):
        raise ValueError(f"fixture kinds must be drawn from {list(COUNTS)}")
    source = canonical_rig()
    fixtures = []
    for kind, count in counts.items():
        _integer(count, f"{kind} count", 0, 1_000_000)
        source_kind = {"laser": "moving_head", "garland": "wash"}.get(kind, kind)
        sources = source.by_kind(source_kind)
        for i in range(count):
            src = sources[i % len(sources)]
            offsets = [0, 2, 3, 4] if kind == "garland" else list(range(src.n_channels))
            if kind == "moving_head":
                row, column = divmod(i, 6)
                position = [-10 + column * 4, -11 + row * 9, 9.5 if row % 4 in (0, 3) else 10.5]
                target = [-position[0] * .25, position[1] + 7, 1.5]
            elif kind == "wash":
                side = -1 if i % 2 == 0 else 1
                position = [side * 15, -10 + (i // 2) * 5.3, 6]
                target = [side * 4, position[1], 3]
            elif kind == "projector":
                targets = [[0, -15, 6.5], [-9, -13, 5.5], [9, -13, 5.5], [0, -12, 2.2]]
                target = targets[i % 4].copy()
                target[0] += (i // 4) * 22
                position = [target[0], 3, 8]
            elif kind == "laser":
                position = [-12 + (i % 4) * 8, -10 + (i // 4) * 8, 7.5]
                target = [-position[0] * .3, position[1] + 20, 8]
            else:
                angle = 2 * math.pi * (i % 12) / 12
                ring = 1 + i // 12
                position = [16 * ring * math.cos(angle), 5 + 16 * ring * math.sin(angle), 8]
                target = [0, 5, 12]
            fixture = {
                "name": f"{kind}_{i+1:03d}", "kind": kind,
                "position": position, "target": target,
                "source_fixture": src.name, "source_kind": src.kind,
                "source_channels": [src.address + j for j in offsets],
                "pan_range_deg": 110.0 if kind == "laser" else 540.0,
                "tilt_range_deg": 50.0 if kind == "laser" else 270.0,
                "beam_min_deg": {"moving_head": 2.0, "wash": 25.0, "projector": 28.0,
                                 "laser": .08, "garland": 180.0}[kind],
                "beam_max_deg": {"moving_head": 14.0, "wash": 65.0, "projector": 55.0,
                                 "laser": .3, "garland": 180.0}[kind],
                "virtual": kind == "laser",
            }
            fixtures.append(fixture)
    return _finish_patch(allocate_fixtures(fixtures))


def repeat_patch(patch: dict, copies: int, translation_step=(36., 0., 0.)) -> dict:
    """Repeat a fixture arrangement, translate geometry, and allocate all slots."""
    validate_patch(patch)
    _integer(copies, "copies", 1, 1_000_000)
    step = np.asarray(_vector(translation_step, "translation_step"))
    fixtures = []
    for repeat in range(copies):
        for original in patch["fixtures"]:
            fixture = copy.deepcopy(original)
            fixture["name"] = f"{original['name']}_r{repeat+1:03d}"
            for key in ("position", "target"):
                fixture[key] = (np.asarray(fixture[key]) + repeat * step).tolist()
            fixtures.append(fixture)
    return _finish_patch(allocate_fixtures(fixtures))


def validate_patch(patch: dict) -> None:
    """Reject overlaps, malformed personalities and untraceable source mappings."""
    if patch.get("schema_version") != 1 or patch.get("source_universe") != 0:
        raise ValueError("expected patch schema 1 with canonical source universe 0")
    ids = patch.get("universe_ids", [])
    if not isinstance(ids, list) or not ids or ids != sorted(set(ids)) or ids[0] != 0:
        raise ValueError("universe_ids must be sorted, unique and begin with 0")
    for u in ids:
        _integer(u, "universe", 0, 32767)
    originals = {f.name: f for f in canonical_rig().fixtures}
    occupied, names, used = set(), set(), {0}
    for fixture in patch["fixtures"]:
        name, kind = fixture["name"], fixture["kind"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"empty or duplicate fixture name: {name}")
        names.add(name)
        if kind not in CHANNELS:
            raise ValueError(f"unknown fixture kind {kind}")
        if patch.get("profiles", {}).get(kind, {}).get("channels") != CHANNELS[kind]:
            raise ValueError(f"profile channel semantics differ from the decoder: {kind}")
        if fixture.get("virtual") is not (kind == "laser"):
            raise ValueError(f"virtual flag must identify simulated laser profiles: {name}")
        n = _integer(fixture["n_channels"], "n_channels", 1, 512)
        if n != len(CHANNELS[kind]):
            raise ValueError(f"invalid {kind} personality width")
        universe = _integer(fixture["universe"], "universe", 1, 32767)
        address = _integer(fixture["address"], "address", 1, 512)
        if address + n - 1 > 512:
            raise ValueError(f"fixture crosses universe boundary: {name}")
        for channel in range(address, address + n):
            if (universe, channel) in occupied:
                raise ValueError(f"DMX overlap at universe {universe}, channel {channel}")
            occupied.add((universe, channel))
        used.add(universe)
        if universe not in ids:
            raise ValueError(f"fixture universe missing from universe_ids: {name}")
        source_name = fixture["source_fixture"]
        if source_name not in originals:
            raise ValueError(f"unknown canonical source fixture {source_name}")
        src = originals[source_name]
        expected_kind = {"laser": "moving_head", "garland": "wash"}.get(kind, kind)
        if src.kind != expected_kind or fixture.get("source_kind") != src.kind:
            raise ValueError(f"incompatible source personality: {name}")
        channels = fixture["source_channels"]
        expected = [src.address + j for j in ([0, 2, 3, 4] if kind == "garland" else range(n))]
        if channels != expected:
            raise ValueError(f"source channels must preserve the {kind} channel semantics: {name}")
        for ch in channels:
            _integer(ch, "source_channel", 1, 512)
        for key in ("position", "target"):
            _vector(fixture[key], f"{name}.{key}")
        if np.linalg.norm(np.asarray(fixture["target"]) - fixture["position"]) < 1e-6:
            raise ValueError(f"position and target must differ: {name}")
        for key in ("pan_range_deg", "tilt_range_deg", "beam_min_deg", "beam_max_deg"):
            value = fixture[key]
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}.{key}")
        if fixture["beam_min_deg"] > fixture["beam_max_deg"] or fixture["beam_max_deg"] > 180:
            raise ValueError(f"invalid beam range: {name}")
    if used != set(ids):
        raise ValueError("universe_ids includes unused fixture universes")


def validate_source(source_dmx: np.ndarray, times: np.ndarray | None = None) -> None:
    if source_dmx.dtype != np.uint8 or source_dmx.ndim != 2 or source_dmx.shape[1] != 512:
        raise ValueError("source DMX must have shape (T,512), dtype uint8")
    if source_dmx.shape[0] == 0:
        raise ValueError("source timeline is empty")
    if times is not None:
        if times.shape != (len(source_dmx),) or not np.isfinite(times).all():
            raise ValueError("times must have one finite timestamp per source frame")
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be nonnegative and strictly increasing")


def fanout_dmx(source_dmx: np.ndarray, patch: dict | None = None) -> np.ndarray:
    """Return (T,U,512) byte streams, in patch['universe_ids'] order."""
    patch = build_patch() if patch is None else patch
    validate_patch(patch)
    validate_source(source_dmx)
    indices = {u: i for i, u in enumerate(patch["universe_ids"])}
    output = np.zeros((len(source_dmx), len(indices), 512), dtype=np.uint8)
    output[:, indices[0]] = source_dmx
    for fixture in patch["fixtures"]:
        start = fixture["address"] - 1
        output[:, indices[fixture["universe"]], start:start + fixture["n_channels"]] = (
            source_dmx[:, np.asarray(fixture["source_channels"]) - 1])
    return output


def decode_festival(universes: np.ndarray, patch: dict) -> dict[str, np.ndarray]:
    """Decode only exported bytes; floats are normalized except integer gobo."""
    validate_patch(patch)
    if universes.dtype != np.uint8 or universes.ndim != 3 or universes.shape[1:] != (len(patch["universe_ids"]), 512):
        raise ValueError("universes must have shape (T,len(universe_ids),512), dtype uint8")
    shape = (len(universes), len(patch["fixtures"]))
    out = {key: np.full(shape, .5 if key in ("pan", "tilt", "zoom", "focus") else 0., dtype=np.float32)
           for key in CONTROL_KEYS if key not in ("rgb", "gobo")}
    out["rgb"] = np.zeros(shape + (3,), np.float32)
    out["gobo"] = np.zeros(shape, np.uint8)
    indices = {u: i for i, u in enumerate(patch["universe_ids"])}
    for i, fixture in enumerate(patch["fixtures"]):
        start = fixture["address"] - 1
        raw = universes[:, indices[fixture["universe"]], start:start + fixture["n_channels"]]
        values = raw.astype(np.float32) / 255.
        kind = fixture["kind"]
        if kind in ("moving_head", "laser"):
            out["pan"][:, i] = (raw[:, 0].astype(np.uint32) * 256 + raw[:, 1]) / 65535.
            out["tilt"][:, i] = (raw[:, 2].astype(np.uint32) * 256 + raw[:, 3]) / 65535.
            dimmer, strobe, rgb, zoom, focus, gobo = 4, 5, 6, 9, 10, 11
        elif kind == "projector":
            dimmer, strobe, rgb, zoom, focus, gobo = 0, 1, 2, 5, 6, 7
            out["pan"][:, i], out["tilt"][:, i] = values[:, 8], values[:, 9]
        elif kind == "wash":
            dimmer, strobe, rgb, zoom, focus, gobo = 0, 1, 2, 5, None, None
        else:
            dimmer, strobe, rgb, zoom, focus, gobo = 0, None, 1, None, None, None
        out["dimmer"][:, i] = values[:, dimmer]
        out["rgb"][:, i] = values[:, rgb:rgb+3]
        for key, offset in (("strobe", strobe), ("zoom", zoom), ("focus", focus)):
            if offset is not None:
                out[key][:, i] = values[:, offset]
        if gobo is not None:
            out["gobo"][:, i] = raw[:, gobo]
    return out


def sample_timeline(timeline: dict[str, np.ndarray], seconds) -> dict[str, np.ndarray]:
    """Causal zero-order hold, clamped to first/last frame beyond saved coverage.

    Accepts scalar or array timestamps. There is no interpolation, anticipation,
    retiming, or render-fps-dependent smoothing of the accepted controller.
    """
    times = np.asarray(timeline["time"])
    query = np.asarray(seconds, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timeline time must be finite and strictly increasing")
    if not np.isfinite(query).all():
        raise ValueError("sample time must be finite")
    index = np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 1)
    result = {k: timeline[k][index] for k in CONTROL_KEYS if k in timeline}
    result.update(time=times[index], source_index=index)
    if "universes" in timeline:
        result["universes"] = timeline["universes"][index]
    return result


def load_timeline(directory: str | Path) -> dict[str, np.ndarray]:
    directory = Path(directory)
    patch = json.loads((directory / "fixture_patch.json").read_text())
    validate_patch(patch)
    timeline_path = directory / "timeline.npz"
    if patch.get("timeline", {}).get("sha256") != _sha256(timeline_path):
        raise ValueError("exported timeline hash does not match its manifest")
    with np.load(timeline_path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    validate_source(result["source_dmx"], result["time"])
    if not np.array_equal(result["universe_ids"], patch["universe_ids"]):
        raise ValueError("timeline universe IDs disagree with the patch")
    shape = (len(result["time"]), len(patch["fixtures"]))
    for key in CONTROL_KEYS:
        expected_shape = shape + (3,) if key == "rgb" else shape
        if result[key].shape != expected_shape or not np.isfinite(result[key]).all():
            raise ValueError(f"invalid timeline control shape or value: {key}")
    if result["universes"].dtype != np.uint8 or result["universes"].shape != (shape[0], len(patch["universe_ids"]), 512):
        raise ValueError("invalid exported universe byte array")
    if not np.array_equal(result["universes"][:, 0], result["source_dmx"]):
        raise ValueError("export changed source-universe bytes")
    # A patch edit must be re-exported. Otherwise cached float controls could
    # silently disagree with the bytes sent to the live Blender receiver.
    regenerated = fanout_dmx(result["source_dmx"], patch)
    if not np.array_equal(result["universes"], regenerated):
        raise ValueError("fixture mapping and exported DMX bytes disagree; re-export the patch")
    del regenerated
    decoded = decode_festival(result["universes"], patch)
    for key in CONTROL_KEYS:
        if not np.array_equal(result[key], decoded[key]):
            raise ValueError(f"saved {key} controls disagree with exported DMX bytes")
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_timeline(output_dir: str | Path, patch: dict | None = None,
                    baseline_path: str | Path = DEFAULT_BASELINE) -> dict:
    """Authenticate accepted source and export reusable Blender/Art-Net inputs."""
    output_dir, baseline_path = Path(output_dir), Path(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    stream_path = Path(baseline["reference_streams"])
    actual_hash = _sha256(stream_path)
    if actual_hash != baseline["reference_streams_sha256"]:
        raise ValueError("baseline source DMX hash does not match the accepted configuration")
    patch = copy.deepcopy(build_patch() if patch is None else patch)
    validate_patch(patch)
    with np.load(stream_path, allow_pickle=False) as data:
        source, times = data["candidate"], data["time"]
    validate_source(source, times)
    universes = fanout_dmx(source, patch)
    arrays = decode_festival(universes, patch)
    arrays.update(time=times, times=times, source_dmx=source, universes=universes,
                  universe_ids=np.asarray(patch["universe_ids"], dtype=np.uint16))
    plan_path = Path(baseline["review_plan"])
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    audio = json.loads(plan_path.read_text())["inputs"]["audio"]
    audio_path = Path(audio["path"])
    if not audio_path.is_file() or _sha256(audio_path) != audio["sha256"]:
        raise ValueError("source audio does not match the accepted review plan")
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.
    patch["timeline"] = {
        "file": "timeline.npz", "source": str(stream_path), "source_sha256": actual_hash,
        "baseline_config": str(baseline_path.resolve()), "baseline_id": baseline["id"],
        "baseline_sha256": _sha256(baseline_path), "audio": audio,
        "frames": len(times), "fps": 1. / period if period else None,
        "start_s": float(times[0]), "last_sample_s": float(times[-1]),
        "duration_s": float(times[-1] + period), "sample_policy": "causal_zero_order_hold",
        "fixture_order": [fixture["name"] for fixture in patch["fixtures"]],
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Export directories are immutable receipts; partial prior writes are preserved.
    for name in ("timeline.npz", "fixture_patch.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / name}")
    np.savez_compressed(output_dir / "timeline.npz", **arrays)
    patch["timeline"]["sha256"] = _sha256(output_dir / "timeline.npz")
    (output_dir / "fixture_patch.json").write_text(json.dumps(patch, indent=2) + "\n")
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--patch", type=Path, help="Custom fixture_patch.json; explicit positions, sources and addresses")
    for kind in COUNTS:
        parser.add_argument("--" + kind.replace("_", "-") + "-count", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat-step", type=float, nargs=3, default=(36., 0., 0.))
    args = parser.parse_args()
    counts = {kind: getattr(args, kind + "_count") for kind in COUNTS if getattr(args, kind + "_count") is not None}
    if args.patch and counts:
        parser.error("--patch and fixture count overrides are mutually exclusive")
    patch = json.loads(args.patch.read_text()) if args.patch else build_patch(counts)
    if args.repeat != 1:
        patch = repeat_patch(patch, args.repeat, args.repeat_step)
    patch = export_timeline(args.output_dir, patch, args.baseline)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "fixtures": len(patch["fixtures"]),
                      "universe_ids": patch["universe_ids"], "duration_s": patch["timeline"]["duration_s"]}))


if __name__ == "__main__":
    main()
