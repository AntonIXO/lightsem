"""DMX universe frame stream, cue-list export, and optional Art-Net output."""

from __future__ import annotations

import csv
import json
import socket
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.anticipation import AnticipationCurves
from ..audio.features import AudioFeatures
from ..audio.structure import Segment
from .fixtures import UNIVERSE_SIZE, Rig
from .scenes import SCENES, SceneEngine


@dataclass
class DmxStream:
    """A rendered DMX universe sequence: (n_frames, 512) uint8 at ``fps``."""

    frames: np.ndarray
    fps: float
    times: np.ndarray
    rig_summary: dict
    cues: list[dict]

    @property
    def duration_s(self) -> float:
        return float(self.frames.shape[0] / self.fps)

    def save_npy(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.frames)
        return path

    def save_cue_list(self, json_path: str | Path, csv_path: str | Path | None = None):
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({
            "fps": self.fps,
            "n_frames": int(self.frames.shape[0]),
            "duration_s": self.duration_s,
            "rig": self.rig_summary,
            "cues": self.cues,
        }, indent=2, default=float))
        if csv_path is not None:
            csv_path = Path(csv_path)
            with open(csv_path, "w", newline="") as fh:
                if self.cues:
                    # Cue rows are heterogeneous: structure cues and
                    # anticipation cues carry different keys. Use the UNION of
                    # all keys so no row is silently rejected, and fill the
                    # gaps rather than crashing on the first mismatch.
                    fields: list[str] = []
                    for c in self.cues:
                        for k in c:
                            if k not in fields:
                                fields.append(k)
                    w = csv.DictWriter(fh, fieldnames=fields, restval="")
                    w.writeheader()
                    w.writerows(self.cues)
        return json_path


def build_cue_list(
    segments: list[Segment],
    antic: AnticipationCurves,
    feats: AudioFeatures,
) -> list[dict]:
    """Human-readable cue list: one row per structural cue + anticipation cue."""
    cues: list[dict] = []
    for s in segments:
        scene = SCENES.get(s.label, SCENES["breakdown"])
        cues.append({
            "type": "structure",
            "time_s": round(s.start_s, 3),
            "duration_s": round(s.duration_s, 3),
            "segment": s.label,
            "scene": scene.name,
            "movement": scene.movement,
            "base_dimmer": scene.base_dimmer,
            "zoom": scene.zoom,
            "strobe_rate": scene.strobe_rate,
            "energy": round(s.energy, 4),
            "energy_slope": round(s.energy_slope, 4),
        })
    # Anticipation cue onsets.
    gate = antic.cue_gate.astype(int)
    onsets = antic.frame_times[np.flatnonzero(np.diff(np.concatenate(([0], gate))) == 1)]
    drops = [s.start_s for s in segments if s.label == "drop"]
    for o in onsets:
        nxt = [d - o for d in drops if d >= o]
        cues.append({
            "type": "anticipation",
            "time_s": round(float(o), 3),
            "duration_s": None,
            "segment": None,
            "scene": "anticipatory_lift",
            "movement": None,
            "mode": antic.mode,
            "lookahead_s": antic.lookahead_s,
            "lead_to_next_drop_s": round(min(nxt), 3) if nxt else None,
        })
    cues.sort(key=lambda c: c["time_s"])
    return cues


def render_stream(
    rig: Rig,
    feats: AudioFeatures,
    segments: list[Segment],
    antic: AnticipationCurves,
    fps: float = 40.0,
    duration_s: float | None = None,
) -> DmxStream:
    """Render the full DMX universe sequence at ``fps`` (>= 30 required)."""
    if fps < 30:
        raise ValueError("DMX stream must run at >= 30 fps")
    dur = float(duration_s if duration_s is not None else feats.duration_s)
    n = int(round(dur * fps))
    times = np.arange(n) / fps
    engine = SceneEngine(rig=rig, feats=feats, segments=segments, antic=antic)

    frames = np.zeros((n, UNIVERSE_SIZE), dtype=np.uint8)
    for i, t in enumerate(times):
        engine.apply(float(t))
        frames[i] = rig.universe()

    return DmxStream(
        frames=frames, fps=fps, times=times,
        rig_summary=rig.summary(),
        cues=build_cue_list(segments, antic, feats),
    )


class ArtNetSender:
    """Minimal Art-Net DMX sender (ArtDMX opcode 0x5000).

    Lets the same stream drive real hardware or BlenderDMX. Not exercised
    against physical fixtures here — it is wire-format correct but UNTESTED
    against a real node, and is reported as such.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6454, universe: int = 0):
        self.addr = (host, port)
        self.universe = universe
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._seq = 0

    def send(self, data: np.ndarray) -> int:
        payload = bytes(np.asarray(data, dtype=np.uint8)[:UNIVERSE_SIZE].tobytes())
        if len(payload) < UNIVERSE_SIZE:
            payload += b"\x00" * (UNIVERSE_SIZE - len(payload))
        self._seq = (self._seq % 255) + 1
        pkt = (
            b"Art-Net\x00"
            + struct.pack("<H", 0x5000)
            + struct.pack(">H", 14)
            + bytes([self._seq, 0])
            + struct.pack("<H", self.universe)
            + struct.pack(">H", UNIVERSE_SIZE)
            + payload
        )
        return self.sock.sendto(pkt, self.addr)
