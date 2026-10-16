"""Causal SSL-to-light adapter for the selected native-C baseline.

The JEPA checkpoint remains a representation model.  This module supplies the
explicit, bounded bridge that turns its causal latent into a small pan/tilt
residual on top of the already-emitted mapper-01 DMX.  Body, P2 RGB and the
restored LUFS dimmer remain fixed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import ControlledJepaConfig, ControlledJepaModel


FRAME_RATE = 40.0
SSL_RATE = 10.0
WINDOW = 192
STRIDE = 96


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _nearest(values: np.ndarray, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(times, targets, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose = np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right)
    return values[choose]


def _motion_proxy(world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world = np.asarray(world, dtype=np.float32)
    root = world[:, 0:1]
    rel = world - root
    joints = rel[:, (13, 14, 16, 17, 18, 19, 20, 21)]
    horizontal = np.maximum(np.linalg.norm(joints[..., (0, 2)], axis=-1), 1e-5)
    pan = np.arctan2(joints[..., 0], -joints[..., 2]) / np.pi
    tilt = np.arctan2(joints[..., 1], horizontal) / (np.pi / 2)
    angle = np.stack((pan, tilt), -1).astype(np.float32)
    unwrapped = np.unwrap(angle.astype(np.float64), axis=0, period=2.0)
    speed_raw = np.linalg.norm(np.diff(unwrapped, axis=0, prepend=unwrapped[:1]), axis=-1, keepdims=True) * SSL_RATE
    speed = np.clip(speed_raw / 4.0, 0, 1).astype(np.float32)
    accel_raw = np.diff(speed_raw, axis=0, prepend=speed_raw[:1]) * SSL_RATE
    accel = np.clip(accel_raw / 20.0, -1, 1).astype(np.float32)
    return rel.reshape(len(rel), -1), angle, speed, accel


def _fixture_state(world10: np.ndarray, controls: dict, patch: list[dict], time40: np.ndarray,
                   color_active40: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    moving = [i for i, fixture in enumerate(patch) if fixture.get("kind") != "bulb"]
    if len(moving) < 8:
        raise ValueError("SSL light adapter requires at least eight moving fixtures")
    selected = moving[:8]
    rel, angle, speed, acceleration = _motion_proxy(world10)
    target_time = np.arange(len(world10), dtype=np.float64) / SSL_RATE
    rgb = _nearest(np.asarray(controls["rgb"][:, selected], dtype=np.float32), time40, target_time)
    brightness = _nearest(np.asarray(controls["dimmer"][:, selected], dtype=np.float32), time40, target_time)[..., None]
    if color_active40 is None:
        event = np.zeros((len(target_time), 1), dtype=np.float32)
    else:
        event = _nearest(np.asarray(color_active40, dtype=np.float32), time40, target_time).reshape(-1, 1)
    fixture = np.concatenate((angle[:, :8], speed[:, :8], acceleration[:, :8], rgb, brightness,
                              np.broadcast_to(event[:, None, :], (len(target_time), 8, 1))), axis=-1)
    return fixture.astype(np.float32), np.asarray(selected, dtype=np.int64)


def _semantic_audio(music, base92_path: Path, body_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(base92_path, allow_pickle=False) as values:
        base = np.asarray(values["x30"], dtype=np.float32)
        base_time = np.asarray(values["time30"], dtype=np.float64)
    body_time = np.arange(len(body_world), dtype=np.float64) / 30.0
    if base.ndim != 2 or base.shape[1] != 92 or base_time.shape != (len(base),):
        raise ValueError("SSL adapter requires the authenticated Base92 [T,92] cache")
    if not np.allclose(base_time, np.arange(len(base_time)) / 30.0, rtol=0, atol=1e-8):
        raise ValueError("Base92 cache does not use the native 30 Hz clock")
    n = min(len(body_world), len(base))
    body_world = np.asarray(body_world[:n], dtype=np.float32)
    times = body_time[:n]
    muq = np.asarray(music.muq, dtype=np.float32)
    muq_time = np.asarray(music.muq_time, dtype=np.float64)
    if muq.ndim != 2 or muq.shape[1] != 1024 or muq_time.shape != (len(muq),):
        raise ValueError("SSL adapter requires the authenticated MuQ [T,1024] cache")
    audio = np.concatenate((_nearest(muq, muq_time, times), base[:n]), axis=-1).astype(np.float32)
    body10 = body_world[::3]
    body10 = body10 - body10[:, :1]
    audio10 = audio[::3]
    return audio, body10, audio10


def _prepare_training_rows(stats: dict[str, np.ndarray]):
    from lightsem.experiments.audio_light_foundation.data import load_rows

    rows = []
    for row in load_rows(sample_rate_hz=10):
        audio = np.asarray(row["audio"], dtype=np.float32)
        body = np.asarray(row["body"], dtype=np.float32)
        parts = [np.asarray(row["motion"], dtype=np.float32),
                 np.asarray(row["speed"], dtype=np.float32),
                 np.asarray(row["acceleration"], dtype=np.float32),
                 np.asarray(row["rgb"], dtype=np.float32),
                 np.asarray(row["brightness"], dtype=np.float32),
                 np.broadcast_to(np.asarray(row["event"], dtype=np.float32)[:, None, :],
                                 (len(row["event"]), 8, 1))]
        n = min(len(audio), len(body), *(len(part) for part in parts))
        fixture = np.concatenate([part[:n] for part in parts], axis=-1)
        rows.append({"id": row["id"], "audio": audio[:n], "body": body[:n],
                     "light": fixture[:n].mean(1), "fixture": fixture[:n]})
    return rows


@dataclass
class SSLAdapter:
    model: ControlledJepaModel
    stats: dict[str, np.ndarray]
    decoder: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    checkpoint_path: Path
    checkpoint_sha256: str
    train_ids: tuple[str, ...]

    @classmethod
    def load(cls, checkpoint: Path, device, train_ids: tuple[str, ...] = ()) -> "SSLAdapter":
        import torch

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SSL light checkpoint missing: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        keys = ("audio_dim", "body_dim", "light_dim", "width", "embedding_dim", "heads", "layers",
                "feedforward_mult", "dropout", "ema_decay", "mask_ratio", "action_semantics")
        config = ControlledJepaConfig(**{key: saved["config"][key] for key in keys})
        if (config.audio_dim, config.body_dim, config.light_dim) != (1116, 66, 9):
            raise ValueError("SSL light checkpoint is not the 1116/66/9 baseline contract")
        model = ControlledJepaModel(config).to(device).eval()
        model.load_state_dict(saved["state_dict"], strict=True)
        stats = {key: np.asarray(value, dtype=np.float32) for key, value in saved["stats"].items()}
        required = {"audio_mean", "audio_std", "body_mean", "body_std", "light_mean", "light_std"}
        if set(stats) != required or any(not np.isfinite(value).all() for value in stats.values()):
            raise ValueError("SSL checkpoint has incomplete or nonfinite normalization stats")
        train_ids = tuple(train_ids) if train_ids else tuple(saved.get("train_ids", ()))
        decoder, target_mean, target_std = cls._fit_decoder(model, stats, device, train_ids)
        return cls(model, stats, decoder, target_mean, target_std, checkpoint, _sha(checkpoint), train_ids)

    @staticmethod
    def _fit_decoder(model, stats, device, train_ids):
        import torch

        rows = [row for row in _prepare_training_rows(stats) if not train_ids or row["id"] in set(train_ids)]
        if len(rows) < 1:
            raise ValueError("SSL decoder has no admitted training rows")
        target_values = np.concatenate([row["fixture"].reshape(-1, 72) for row in rows], axis=0)
        target_mean = target_values.mean(0).astype(np.float32)
        target_std = np.maximum(target_values.std(0), 1e-4).astype(np.float32)
        xs, ys = [], []
        for row in rows:
            audio = (row["audio"] - stats["audio_mean"]) / stats["audio_std"]
            body = (row["body"] - stats["body_mean"]) / stats["body_std"]
            light = (row["light"] - stats["light_mean"]) / stats["light_std"]
            fixture = (row["fixture"] - target_mean.reshape(1, 8, 9)) / target_std.reshape(1, 8, 9)
            for start in range(0, max(1, len(audio) - WINDOW + 1), WINDOW):
                stop = start + WINDOW
                if stop > len(audio):
                    continue
                a = torch.from_numpy(audio[start:stop][None]).float().to(device)
                b = torch.from_numpy(body[start:stop][None]).float().to(device)
                l = torch.from_numpy(light[start:stop][None]).float().to(device)
                ts = torch.arange(start, stop, dtype=torch.float32, device=device)[None] / SSL_RATE
                valid = torch.ones((1, WINDOW), dtype=torch.bool, device=device)
                with torch.no_grad():
                    output = model(a, b, l, ts, valid, torch.zeros_like(valid))
                xs.append(output["causal_context"][0, :-1])
                ys.append(torch.from_numpy(fixture[start + 1:stop].reshape(WINDOW - 1, 72)).float().to(device))
        if not xs:
            raise ValueError("SSL decoder could not construct a full training window")
        x, y = torch.cat(xs), torch.cat(ys)
        design = torch.cat((x, torch.ones((len(x), 1), device=device)), -1)
        regularizer = torch.eye(design.shape[-1], device=device) * 1e-2
        regularizer[-1, -1] = 0
        weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return weights.detach().cpu().numpy().astype(np.float32), target_mean, target_std

    def _predict(self, audio10: np.ndarray, body10: np.ndarray, fixture: np.ndarray, device) -> np.ndarray:
        import torch

        n = min(len(audio10), len(body10), len(fixture))
        audio = (audio10[:n] - self.stats["audio_mean"]) / self.stats["audio_std"]
        body = (body10[:n].reshape(n, -1) - self.stats["body_mean"]) / self.stats["body_std"]
        light = (fixture[:n].mean(1) - self.stats["light_mean"]) / self.stats["light_std"]
        accum = np.zeros((n, 72), dtype=np.float32)
        counts = np.zeros(n, dtype=np.float32)
        weights = torch.from_numpy(self.decoder).to(device)
        for start in sorted(set(list(range(0, max(1, n - WINDOW + 1), STRIDE)) + [max(0, n - WINDOW)])):
            stop = start + WINDOW
            if stop > n:
                continue
            a = torch.from_numpy(audio[start:stop][None]).float().to(device)
            b = torch.from_numpy(body[start:stop][None]).float().to(device)
            l = torch.from_numpy(light[start:stop][None]).float().to(device)
            ts = torch.arange(start, stop, dtype=torch.float32, device=device)[None] / SSL_RATE
            valid = torch.ones((1, WINDOW), dtype=torch.bool, device=device)
            with torch.no_grad():
                output = self.model(a, b, l, ts, valid, torch.zeros_like(valid))
                design = torch.cat((output["causal_context"][0, :-1], torch.ones((WINDOW - 1, 1), device=device)), -1)
                prediction = (design @ weights).cpu().numpy().astype(np.float32)
            accum[start + 1:stop] += prediction
            counts[start + 1:stop] += 1
        predicted = fixture[:n].copy()
        valid = counts > 0
        predicted[valid] = (accum[valid] / counts[valid, None]).reshape(-1, 8, 9)
        return predicted * self.target_std.reshape(1, 8, 9) + self.target_mean.reshape(1, 8, 9)

    @staticmethod
    def _bounded_residual(desired: np.ndarray, max_speed: float, max_accel: float) -> np.ndarray:
        dt = 1.0 / FRAME_RATE
        state = np.zeros(desired.shape[1:], dtype=np.float64)
        velocity = np.zeros_like(state)
        output = np.zeros_like(desired, dtype=np.float64)
        for index, target in enumerate(desired):
            acceleration = np.clip((target - state - dt * velocity) / (dt * dt), -max_accel, max_accel)
            velocity = np.clip(velocity + dt * acceleration, -max_speed, max_speed)
            state = state + dt * velocity
            output[index] = state
        return output.astype(np.float32)

    @staticmethod
    def _kinematic_report(decoded: dict, patch: list[dict]) -> dict[str, float]:
        moving = np.asarray([item.get("kind") != "bulb" for item in patch])
        angles = np.asarray(decoded["angles_deg"][:, moving], dtype=np.float64)
        speed = np.linalg.norm(np.diff(angles, axis=0) * FRAME_RATE, axis=-1)
        accel = np.linalg.norm(np.diff(angles, n=2, axis=0) * FRAME_RATE ** 2, axis=-1)
        return {"speed_p99": float(np.quantile(speed, .99)), "speed_max": float(speed.max()),
                "acceleration_p99": float(np.quantile(accel, .99)), "acceleration_max": float(accel.max())}

    def apply(self, *, body_world: np.ndarray, music, base92_path: Path, controls: dict,
              universes: np.ndarray, patch: list[dict], time40: np.ndarray,
              color_active40: np.ndarray | None, device, max_residual_deg: float = 1.0,
              max_speed: float = 20.0, max_accel: float = 40.0, max_ratio: float = 1.25) -> tuple[np.ndarray, dict]:
        from lightsem.conductor.body_transfer import decode

        body_world = np.asarray(body_world, dtype=np.float32)
        audio30, body10, audio10 = _semantic_audio(music, Path(base92_path), body_world)
        fixture, selected = _fixture_state(body10, controls, patch, np.asarray(time40), color_active40)
        predicted = self._predict(audio10, body10, fixture, device)
        target_time = np.arange(len(body10), dtype=np.float64) / SSL_RATE
        t40 = np.asarray(time40, dtype=np.float64)
        pan10 = np.clip(predicted[:, :, 0] - fixture[:, :, 0], -.25, .25) * (max_residual_deg / .25)
        tilt10 = np.clip(predicted[:, :, 1] - fixture[:, :, 1], -.25, .25) * (max_residual_deg / .25)
        pan = self._bounded_residual(np.stack([np.interp(t40, target_time, pan10[:, i]) for i in range(8)], -1), max_speed, max_accel)
        tilt = self._bounded_residual(np.stack([np.interp(t40, target_time, tilt10[:, i]) for i in range(8)], -1), max_speed, max_accel)
        before = decode(universes, patch)
        candidate = np.asarray(universes, dtype=np.uint8).copy()
        for fixture_index, patch_index in enumerate(selected):
            item = patch[int(patch_index)]
            angle = np.asarray(before["angles_deg"][:, int(patch_index)], dtype=np.float64).copy()
            angle[:, 0] += pan[:, fixture_index]
            angle[:, 1] += tilt[:, fixture_index]
            ranges = np.asarray([item["pan_range_deg"], item["tilt_range_deg"]], dtype=np.float64)
            angle = np.clip(angle, -ranges / 2, ranges / 2)
            ticks = np.rint((angle / ranges + .5) * 65535).clip(0, 65535).astype(np.uint16)
            address = int(item["address"]) - 1
            universe = int(item["universe"])
            candidate[:, universe, address] = (ticks[:, 0] >> 8).astype(np.uint8)
            candidate[:, universe, address + 1] = (ticks[:, 0] & 255).astype(np.uint8)
            candidate[:, universe, address + 2] = (ticks[:, 1] >> 8).astype(np.uint8)
            candidate[:, universe, address + 3] = (ticks[:, 1] & 255).astype(np.uint8)
        after = decode(candidate, patch)
        before_motion = self._kinematic_report(before, patch)
        after_motion = self._kinematic_report(after, patch)
        ratios = {key: after_motion[key] / max(before_motion[key], 1e-6) for key in before_motion}
        if not np.isfinite(candidate).all() or not all(np.isfinite(value).all() for value in after.values()):
            raise ValueError("SSL light adapter produced nonfinite DMX")
        if not np.array_equal(before["rgb"], after["rgb"]) or not np.array_equal(before["dimmer"], after["dimmer"]):
            raise ValueError("SSL light adapter changed fixed P2 RGB or LUFS dimmer")
        if max(ratios.values()) > max_ratio:
            raise ValueError(f"SSL light raw-safety ratio exceeds {max_ratio}: {ratios}")
        body_alt = body10.copy()
        body_alt[:, 1:] = body_alt[:1, 1:]
        alt_fixture, _ = _fixture_state(body_alt, controls, patch, np.asarray(time40), color_active40)
        alt = self._predict(audio10, body_alt, alt_fixture, device)
        intervention = float(np.mean(np.abs(predicted[:, :, :2] - alt[:, :, :2])))
        if not np.isfinite(intervention) or intervention <= 1e-6:
            raise ValueError("SSL latent light adapter is insensitive to the body intervention")
        report = {
            "status": "passed_raw_safety",
            "checkpoint": str(self.checkpoint_path), "checkpoint_sha256": self.checkpoint_sha256,
            "decoder": "ridge on frozen SSL causal_context; train rows are sealed in checkpoint",
            "decoder_sha256": _sha_array(self.decoder),
            "train_ids": list(self.train_ids), "body_generation_changed": False,
            "fixed_controls": {"rgb": True, "dimmer": True},
            "selected_fixture_indices": selected.tolist(),
            "residual_rms_deg": float(np.sqrt(np.mean(np.concatenate((pan, tilt), axis=1) ** 2))),
            "residual_max_abs_deg": float(np.max(np.abs(np.concatenate((pan, tilt), axis=1)))),
            "body_intervention_delta_semantic": intervention,
            "baseline_motion": before_motion, "candidate_motion": after_motion,
            "candidate_baseline_ratios": ratios,
            "changed_dmx_byte_fraction": float(np.mean(candidate != universes)),
            "changed_dmx_frame_fraction": float(np.mean(np.any(candidate != universes, axis=(1, 2)))),
            "max_residual_deg": max_residual_deg, "residual_max_speed_deg_s": max_speed,
            "residual_max_accel_deg_s2": max_accel, "max_candidate_baseline_ratio": max_ratio,
        }
        return candidate, report


__all__ = ["SSLAdapter"]
