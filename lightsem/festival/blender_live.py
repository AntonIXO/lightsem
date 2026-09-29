"""Live ArtDmx input for an existing Blender festival scene.

Use from Blender's Python console or a ``--python`` script after building the
scene::

    from lightsem.festival.blender_live import install
    receiver = install(runtime, patch)
    # Later, from Blender's main thread:
    receiver.stop()

The default bind is localhost. A launcher must explicitly pass ``--bind`` (and
forward it here) to receive on another interface, for example ``0.0.0.0``.
This listener emits no network packets. It supports every universe in the patch,
without a fixed fixture/universe limit. Source passthrough universes without
scene fixtures are ignored. One sender is assumed; Art-Net sequence ordering is
honored. Controller merging and ArtSync synchronization are not implemented.

All Blender access occurs in ``bpy.app.timers`` on Blender's main thread. A
per-universe timeout blacks out only its fixture dimmers while retaining the
latest color and position. The live media/strobe clock is monotonic elapsed
time since installation; ArtDmx itself carries no source-song timestamp.
"""
from __future__ import annotations

import copy
import logging
import math
import socket
import time

import numpy as np

from .patch import decode_festival, validate_patch

log = logging.getLogger(__name__)


class ArtNetReceiver:
    """Owned socket, timer and latest universe state; call ``stop`` to release."""

    def __init__(self, runtime, patch, bind, port, timeout_s, poll_interval,
                 max_packets_per_tick):
        import bpy
        from .artnet import ArtNetState
        from .blender_scene import apply_dmx as legacy_apply_dmx

        validate_patch(patch)
        for name, value in (("timeout_s", timeout_s), ("poll_interval", poll_interval)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer in 0..65535")
        if (isinstance(max_packets_per_tick, bool)
                or not isinstance(max_packets_per_tick, int)
                or max_packets_per_tick <= 0):
            raise ValueError("max_packets_per_tick must be a positive integer")

        self.runtime = runtime
        self.patch = copy.deepcopy(patch)
        self.timeout_s = float(timeout_s)
        self.poll_interval = float(poll_interval)
        self.max_packets_per_tick = max_packets_per_tick
        self._bpy = bpy
        self._apply_dmx = runtime.get('apply_dmx', legacy_apply_dmx)
        self._indices = {u: i for i, u in enumerate(self.patch["universe_ids"])}
        self._state = ArtNetState(self.patch["universe_ids"])
        self._mapped = {f["universe"] for f in self.patch["fixtures"]}
        self._fixture_rows = np.array(
            [self._indices[f["universe"]] for f in self.patch["fixtures"]], dtype=np.intp)
        self.universes = np.zeros((len(self._indices), 512), dtype=np.uint8)
        self._last_seen = np.full(len(self._indices), -np.inf)
        self._controls = None
        self._dirty = True
        self._started = time.monotonic()
        self.running = False
        self.last_error = None
        self.statistics = {"received": 0, "accepted": 0, "ignored": 0, "malformed": 0}
        # Retain stable callback identities for unregister and load-handler removal.
        self._timer = self._tick
        self._load_handler = self._before_load
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.setblocking(False)
            self._socket.bind((bind, port))
            self.address = self._socket.getsockname()
            self.running = True
            bpy.app.handlers.load_pre.append(self._load_handler)
            bpy.app.timers.register(self._timer, first_interval=0.0, persistent=False)
        except Exception:
            self.stop()
            raise

    def _receive(self, now):
        from .artnet import decode_artdmx

        # Bounded draining keeps a sustained network flood from freezing Blender.
        for _ in range(self.max_packets_per_tick):
            try:
                packet, _peer = self._socket.recvfrom(65535)
            except BlockingIOError:
                break
            self.statistics["received"] += 1
            try:
                message = decode_artdmx(packet)
            except (ValueError, TypeError):
                self.statistics["malformed"] += 1
                continue
            universe = message.universe
            if universe not in self._mapped:
                self.statistics["ignored"] += 1
                continue
            if not self._state.ingest(packet, now=now):
                self.statistics["ignored"] += 1
                continue
            row = self._indices[universe]
            self._last_seen[row] = now
            self._dirty = True
            self.statistics["accepted"] += 1

    def _tick(self):
        if not self.running:
            return None
        try:
            now = time.monotonic()
            self._receive(now)
            if self._dirty:
                self.universes = self._state.snapshot()
                decoded = decode_festival(self.universes[None], self.patch)
                self._controls = {key: value[0] for key, value in decoded.items()}
                self._dirty = False
            controls = dict(self._controls)
            stale = now - self._last_seen[self._fixture_rows] >= self.timeout_s
            controls["dimmer"] = np.where(stale, 0.0, self._controls["dimmer"])
            # Update even without fresh packets: strobe/media clocks still advance.
            self._apply_dmx(self.runtime, controls, now - self._started)
            return self.poll_interval
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            log.exception("Festival live receiver stopped after an update error")
            self._release(unregister_timer=False)
            return None

    def _before_load(self, *_args):
        # A new .blend invalidates cached runtime object references.
        self.stop()

    def _release(self, unregister_timer=True):
        self.running = False
        self._socket.close()
        if self._load_handler in self._bpy.app.handlers.load_pre:
            self._bpy.app.handlers.load_pre.remove(self._load_handler)
        if unregister_timer and self._bpy.app.timers.is_registered(self._timer):
            self._bpy.app.timers.unregister(self._timer)

    def stop(self):
        """Close the socket and unregister callbacks; safe to call repeatedly.

        Call from Blender's main thread. This does not delete the scene or alter
        its last visible frame. Stale network input is handled by the timer's
        independent universe blackouts while the receiver is running.
        """
        self._release()

    close = stop


def install(runtime, patch, bind="127.0.0.1", port=6454, *, timeout_s=2.0,
            poll_interval=1.0 / 60.0, max_packets_per_tick=4096):
    """Install a main-thread live input timer over an already built scene.

    Return an :class:`ArtNetReceiver`. Bind errors are raised synchronously;
    asynchronous errors stop the receiver and are exposed through ``last_error``.
    Set ``timeout_s`` to configure per-universe loss-of-input blackout time.
    Port 0 is supported for isolated local tests; ``receiver.address`` then gives
    the operating system's allocated port.
    """
    return ArtNetReceiver(runtime, patch, bind, port, timeout_s, poll_interval,
                          max_packets_per_tick)
