"""ArtDmx byte transport for the festival simulator, using explicit unicast peers.

This is a configured sender/receiver, not a discovery-capable Art-Net console.
Wire layout follows Artistic Licence Art-Net 4, protocol revision 14:
https://art-net.org.uk/downloads/art-net.pdf
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import struct
import time

import numpy as np

from .patch import _integer, load_timeline, sample_timeline

ARTNET_PORT = 6454
ARTNET_ID = b"Art-Net\x00"
OP_DMX = 0x5000


@dataclass(frozen=True)
class ArtDmx:
    universe: int
    sequence: int
    data: bytes
    physical: int = 0


def encode_artdmx(data: bytes | np.ndarray, universe: int = 0,
                  sequence: int = 1, physical: int = 0) -> bytes:
    """Pack standard ArtDmx: LE opcode/address, BE version/length, even data."""
    universe = _integer(universe, "universe", 0, 32767)
    sequence = _integer(sequence, "sequence", 0, 255)
    physical = _integer(physical, "physical", 0, 255)
    if isinstance(data, np.ndarray):
        if data.dtype != np.uint8 or data.ndim != 1:
            raise ValueError("DMX data must be a one-dimensional uint8 array")
        data = data.tobytes()
    if not isinstance(data, (bytes, bytearray)) or not 1 <= len(data) <= 512:
        raise ValueError("DMX data must contain 1..512 bytes")
    data = bytes(data)
    if len(data) % 2:
        data += b"\x00"
    return (ARTNET_ID + struct.pack("<H", OP_DMX) + struct.pack(">H", 14)
            + bytes((sequence, physical)) + struct.pack("<H", universe)
            + struct.pack(">H", len(data)) + data)


def decode_artdmx(packet: bytes) -> ArtDmx:
    """Reject malformed/unsupported datagrams instead of partially applying them."""
    if len(packet) < 18 or packet[:8] != ARTNET_ID:
        raise ValueError("not an Art-Net datagram")
    if struct.unpack_from("<H", packet, 8)[0] != OP_DMX:
        raise ValueError("not an ArtDmx datagram")
    if struct.unpack_from(">H", packet, 10)[0] < 14:
        raise ValueError("unsupported Art-Net protocol version")
    universe = struct.unpack_from("<H", packet, 14)[0]
    length = struct.unpack_from(">H", packet, 16)[0]
    if universe > 32767 or length < 2 or length > 512 or length % 2 or len(packet) != 18 + length:
        raise ValueError("invalid ArtDmx port-address or payload length")
    return ArtDmx(universe=universe, sequence=packet[12], physical=packet[13], data=packet[18:])


class ArtNetState:
    """Latest received bytes in arbitrary manifest universe order.

    A single explicitly configured sender is assumed. Sequence 0 disables
    ordering; after a silence of 2s a restarted sender's sequence is accepted.
    Missing/expired universes are black in a snapshot when timeout_s is given.
    """

    def __init__(self, universe_ids):
        self.universe_ids = list(universe_ids)
        if len(self.universe_ids) != len(set(self.universe_ids)):
            raise ValueError("duplicate universe IDs")
        for universe in self.universe_ids:
            _integer(universe, "universe", 0, 32767)
        self._indices = {universe: i for i, universe in enumerate(self.universe_ids)}
        self._frames = np.zeros((len(self.universe_ids), 512), np.uint8)
        self._sequences = {}
        self._updated = {}

    def ingest(self, packet: bytes, now: float | None = None) -> bool:
        message = decode_artdmx(packet)
        if message.universe not in self._indices:
            return False
        now = time.monotonic() if now is None else float(now)
        if not math.isfinite(now):
            raise ValueError("receive time must be finite")
        previous = self._sequences.get(message.universe, 0)
        age = now - self._updated.get(message.universe, -math.inf)
        if message.sequence and previous and age < 2:
            # Sequence numbering runs 1..255; zero is deliberately excluded.
            distance = (message.sequence - previous) % 255
            if distance == 0 or distance > 127:
                return False
        self._sequences[message.universe] = message.sequence
        self._updated[message.universe] = now
        row = self._frames[self._indices[message.universe]]
        row[:] = 0
        row[:len(message.data)] = np.frombuffer(message.data, np.uint8)
        return True

    def snapshot(self, now: float | None = None, timeout_s: float | None = None) -> np.ndarray:
        result = self._frames.copy()
        if timeout_s is not None:
            if not math.isfinite(timeout_s) or timeout_s < 0:
                raise ValueError("timeout_s must be finite and nonnegative")
            now = time.monotonic() if now is None else float(now)
            for universe, index in self._indices.items():
                if now - self._updated.get(universe, -math.inf) > timeout_s:
                    result[index] = 0
        return result


class ArtNetSender:
    """One fixed unicast destination. No broadcast/discovery is performed."""

    def __init__(self, host: str = "127.0.0.1", port: int = ARTNET_PORT):
        _integer(port, "port", 1, 65535)
        self.destination = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Sequence 0 disables ordering for this first frame and clears the
        # receiver's previous session. Subsequent frames use 1..255 normally,
        # so a quick stop/restart cannot freeze lights behind an old sequence.
        self._sequence = 0

    def send_frame(self, universes: np.ndarray, universe_ids) -> int:
        if universes.dtype != np.uint8 or universes.shape != (len(universe_ids), 512):
            raise ValueError("expected uint8 (U,512) frame with one universe ID per row")
        sent = 0
        for universe, data in zip(universe_ids, universes):
            self._socket.sendto(encode_artdmx(data, int(universe), self._sequence), self.destination)
            sent += 1
        self._sequence = self._sequence % 255 + 1
        return sent

    def close(self) -> None:
        self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def replay_timeline(timeline: dict[str, np.ndarray], host: str = "127.0.0.1",
                    port: int = ARTNET_PORT, start_s: float = 0.,
                    duration_s: float | None = None, speed: float = 1.) -> dict:
    """Replay saved byte frames at their timestamps; late frames are skipped.

    The first frame is held at an arbitrary start offset. A monotonic clock
    prevents cumulative sleep drift. No changes are made to DMX values.
    """
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be finite and nonnegative")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError("duration_s must be finite and positive")
    times = timeline["time"]
    sample_timeline(timeline, start_s)  # validates the temporal contract
    period = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 40
    end_s = float(times[-1] + period)
    if start_s >= end_s:
        raise ValueError("start_s is outside the saved timeline")
    end_s = min(end_s, start_s + duration_s) if duration_s is not None else end_s
    frames, packets, skipped = 0, 0, 0
    ids = timeline["universe_ids"]
    started = time.monotonic()
    previous_index = None
    with ArtNetSender(host, port) as sender:
        while True:
            query = start_s + (time.monotonic() - started) * speed
            if query >= end_s:
                break
            index = int(np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 1))
            if index != previous_index:
                packets += sender.send_frame(timeline["universes"][index], ids)
                frames += 1
                if previous_index is not None:
                    skipped += max(index - previous_index - 1, 0)
                previous_index = index
            next_time = min(float(times[index+1]) if index+1 < len(times) else end_s, end_s)
            wait = (next_time - start_s) / speed - (time.monotonic() - started)
            if wait > 0:
                time.sleep(min(wait, .1))
    return {"host": host, "port": port, "frames_sent": frames, "packets_sent": packets,
            "late_frames_skipped": skipped, "start_s": start_s, "end_s": end_s,
            "universe_ids": [int(u) for u in ids], "elapsed_s": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=ARTNET_PORT)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--speed", type=float, default=1.)
    args = parser.parse_args()
    result = replay_timeline(load_timeline(args.timeline_dir), args.host, args.port,
                             args.start, args.duration, args.speed)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
