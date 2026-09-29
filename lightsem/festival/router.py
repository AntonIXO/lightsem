"""Route live canonical LightSem ArtDmx into a configurable festival patch.

The existing LightSem sender can target 127.0.0.2:6454 (canonical universe 0).
This process copies its bytes into the venue universes and unicasts them to the
Blender listener at 127.0.0.1:6454. No model, audio, GUI or saved timeline is needed.
One canonical controller is assumed; discovery, merging and ArtSync are absent.
"""
from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import math
from pathlib import Path
import select
import socket
import time

from .artnet import ARTNET_PORT, ArtNetState, decode_artdmx, encode_artdmx
from .patch import _integer, fanout_dmx, validate_patch


def _ipv4(host: str) -> str:
    return socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_DGRAM)[0][4][0]


class LiveRouter:
    """Owned nonblocking input/output sockets and an explicit source-byte patch.

    ``poll_once`` supports external event loops; ``run`` supplies a bounded-wait
    loop. Silence causes no sends, allowing the downstream receiver's timeout
    to black out stale fixtures. The untouched source universe is forwarded too.
    """

    def __init__(self, patch: dict, bind: str = "127.0.0.2", port: int = ARTNET_PORT,
                 host: str = "127.0.0.1", output_port: int = ARTNET_PORT,
                 max_packets_per_tick: int = 256):
        validate_patch(patch)
        _integer(port, "port", 0, 65535)
        _integer(output_port, "output_port", 1, 65535)
        _integer(max_packets_per_tick, "max_packets_per_tick", 1, 65536)
        self.patch = copy.deepcopy(patch)
        self.max_packets_per_tick = max_packets_per_tick
        self._state = ArtNetState([0])
        self._source_peer = None
        self._sequence = 0  # First frame resets a previous router session's sequence.
        self._input = self._output = None
        self.closed = False
        self.statistics = {"received": 0, "accepted": 0, "malformed": 0,
                           "ignored_universe": 0, "ignored_sequence": 0,
                           "ignored_echo": 0, "source_restarts": 0,
                           "output_packets": 0}
        destination_ip = _ipv4(host)
        destination_address = ipaddress.ip_address(destination_ip)
        if destination_address.is_multicast or destination_address.is_unspecified or destination_ip == "255.255.255.255":
            raise ValueError("router destination must be a specific unicast IPv4 peer")
        self.destination = (destination_ip, output_port)
        try:
            self._input = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._input.setblocking(False)
            self._input.bind((_ipv4(bind), port))
            self.address = self._input.getsockname()
            # UDP connect on a disposable socket only selects a local interface;
            # it sends no packet. Bind the real sender so its endpoint is known
            # before accepting input, including when output routes back locally.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(self.destination)
                outgoing_ip = probe.getsockname()[0]
            same_input = destination_ip == self.address[0]
            if self.address[0] == "0.0.0.0":
                same_input = destination_ip == outgoing_ip or destination_address.is_loopback
            if self.address[1] == output_port and same_input:
                raise ValueError("router input and destination resolve to the same endpoint (source echo)")
            self._output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._output.bind((outgoing_ip, 0))
            self.sender_address = self._output.getsockname()
        except BaseException:
            self.close()
            raise

    def poll_once(self, timeout: float = 0.) -> int:
        """Drain a bounded number of packets; return accepted source frame count."""
        if self.closed:
            raise RuntimeError("router is closed")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 <= timeout <= .1:
            raise ValueError("timeout must be finite and between 0 and 0.1 seconds")
        if not select.select([self._input], [], [], timeout)[0]:
            return 0
        accepted = 0
        for _ in range(self.max_packets_per_tick):
            try:
                packet, peer = self._input.recvfrom(65535)
            except BlockingIOError:
                break
            self.statistics["received"] += 1
            if peer == self.sender_address:
                self.statistics["ignored_echo"] += 1
                continue
            try:
                message = decode_artdmx(packet)
            except (ValueError, TypeError):
                self.statistics["malformed"] += 1
                continue
            if message.universe != 0:
                self.statistics["ignored_universe"] += 1
                continue
            # The canonical LightSem sender allocates a new ephemeral UDP port
            # when restarted. With the documented single-controller contract,
            # that endpoint change starts a new sequence session. Reordering
            # protection remains active within each sender endpoint.
            if peer != self._source_peer:
                if self._source_peer is not None:
                    self.statistics["source_restarts"] += 1
                self._state = ArtNetState([0])
                self._source_peer = peer
            if not self._state.ingest(packet):
                self.statistics["ignored_sequence"] += 1
                continue
            output = fanout_dmx(self._state.snapshot(), self.patch)[0]
            for universe, row in zip(self.patch["universe_ids"], output):
                datagram = encode_artdmx(row, universe, sequence=self._sequence)
                self._output.sendto(datagram, self.destination)
                self.statistics["output_packets"] += 1
            self._sequence = self._sequence % 255 + 1
            self.statistics["accepted"] += 1
            accepted += 1
        return accepted

    def run(self, duration_s: float | None = None) -> dict:
        """Run until duration expires or Ctrl-C; close sockets on every exit path."""
        if duration_s is not None and (isinstance(duration_s, bool) or not math.isfinite(duration_s) or duration_s <= 0):
            self.close()
            raise ValueError("duration_s must be finite and positive")
        started = time.monotonic()
        try:
            while True:
                remaining = None if duration_s is None else duration_s - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    break
                self.poll_once(.1 if remaining is None else min(.1, remaining))
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        return {"input": list(self.address), "destination": list(self.destination),
                "universe_ids": self.patch["universe_ids"], "fixtures": len(self.patch["fixtures"]),
                "elapsed_s": time.monotonic() - started, **self.statistics}

    def close(self) -> None:
        self.closed = True
        for stream in (self._input, self._output):
            if stream is not None:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=Path, required=True, help="fixture_patch.json (no timeline NPZ required)")
    parser.add_argument("--bind", default="127.0.0.2")
    parser.add_argument("--port", type=int, default=ARTNET_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--output-port", type=int, default=ARTNET_PORT)
    parser.add_argument("--duration", type=float, help="Optional bounded runtime in seconds")
    args = parser.parse_args()
    patch = json.loads(args.patch.read_text())
    # Validate before opening sockets; the context also cleans up invalid runtime arguments.
    with LiveRouter(patch, args.bind, args.port, args.host, args.output_port) as router:
        print(json.dumps({"status": "listening", "input": router.address,
                          "destination": router.destination, "universe_ids": patch["universe_ids"]}), flush=True)
        result = router.run(args.duration)
    print(json.dumps({"status": "stopped", **result}))


if __name__ == "__main__":
    main()
