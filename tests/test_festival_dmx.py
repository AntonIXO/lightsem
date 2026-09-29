"""Byte ownership, scalable patching, causal sampling and actual UDP transport."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lightsem.festival.artnet import (
    ArtNetSender, ArtNetState, decode_artdmx, encode_artdmx,
)
from lightsem.festival.patch import (
    allocate_fixtures, build_patch, decode_festival, fanout_dmx,
    load_timeline, repeat_patch, sample_timeline, validate_patch,
)


class FestivalPatchTests(unittest.TestCase):
    def setUp(self):
        self.patch = build_patch()
        self.source = np.random.default_rng(49).integers(0, 256, (5, 512), dtype=np.uint8)

    def test_default_capacity_and_exact_original_passthrough(self):
        self.assertEqual(len(self.patch["fixtures"]), 56)
        self.assertEqual(self.patch["universe_ids"], [0, 1])
        self.assertEqual(sum(f["n_channels"] for f in self.patch["fixtures"]), 508)
        before = self.source.copy()
        output = fanout_dmx(self.source, self.patch)
        np.testing.assert_array_equal(output[:, 0], self.source)
        np.testing.assert_array_equal(self.source, before)
        self.assertFalse(output[:, 1, 508:].any())
        # Actual final garland channels must come from wash4 (94,96,97,98),
        # not the neighboring wash strobe/saturation channels.
        np.testing.assert_array_equal(output[:, 1, 504:508], self.source[:, [93, 95, 96, 97]])

    def test_scalable_allocation_and_sparse_custom_universe_ids(self):
        patch = build_patch({"moving_head": 130})
        self.assertGreater(len(patch["universe_ids"]), 3)
        validate_patch(patch)
        output = fanout_dmx(self.source, patch)
        self.assertEqual(output.shape, (5, len(patch["universe_ids"]), 512))
        patch["fixtures"] = allocate_fixtures(patch["fixtures"], first_universe=257)
        patch["universe_ids"] = [0] + sorted({f["universe"] for f in patch["fixtures"]})
        output = fanout_dmx(self.source, patch)
        decoded = decode_festival(output, patch)
        self.assertEqual(decoded["rgb"].shape, (5, 162, 3))
        np.testing.assert_array_equal(output[:, 0], self.source)

    def test_repeated_fixture_positions_targets_and_repatch(self):
        repeat = repeat_patch(self.patch, 3, (40, 5, 1))
        self.assertEqual(len(repeat["fixtures"]), 168)
        first, second = repeat["fixtures"][0], repeat["fixtures"][56]
        for key in ("position", "target"):
            np.testing.assert_allclose(np.asarray(second[key]) - first[key], [40, 5, 1])
        self.assertEqual(first["source_channels"], second["source_channels"])
        self.assertNotEqual(first["name"], second["name"])
        validate_patch(repeat)

    def test_reject_overlap_boundary_source_or_profile_mismatch(self):
        mutations = [
            lambda p: p["fixtures"][1].update(address=1),
            lambda p: p["fixtures"][0].update(address=510),
            lambda p: p["fixtures"][0].update(source_fixture="missing"),
            lambda p: p["fixtures"][0].update(source_channels=list(range(2, 14))),
            lambda p: p["fixtures"][0].update(universe=0),
            lambda p: p["fixtures"][0].update(address=True),
            lambda p: p["profiles"]["moving_head"]["channels"].reverse(),
        ]
        for mutate in mutations:
            patch = copy.deepcopy(self.patch)
            mutate(patch)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_patch(patch)

    def test_decode_bytes_has_16_bit_motion_and_independent_dimmer_rgb(self):
        self.source[:, :12] = [128, 255, 1, 35, 51, 102, 255, 128, 0, 200, 20, 7]
        universes = fanout_dmx(self.source, self.patch)
        result = decode_festival(universes, self.patch)
        for index in (0, 6, 12, 18, 40):  # MH1 clones and first virtual laser
            np.testing.assert_allclose(result["pan"][:, index], 33023 / 65535)
            np.testing.assert_allclose(result["tilt"][:, index], 291 / 65535)
            np.testing.assert_allclose(result["dimmer"][:, index], .2)
            np.testing.assert_allclose(result["rgb"][0, index], [1, 128 / 255, 0])
            np.testing.assert_array_equal(result["gobo"][:, index], 7)
        # Decoding/requantizing all moving head pan/tilt must recover fine bytes.
        for field, off in (("pan", 0), ("tilt", 2)):
            encoded = np.rint(result[field][:, 1] * 65535).astype(np.uint32)
            np.testing.assert_array_equal(encoded >> 8, self.source[:, 12+off])
            np.testing.assert_array_equal(encoded & 255, self.source[:, 13+off])

    def test_source_contract(self):
        for source in (self.source.astype(float), self.source[:, :511], self.source[:0]):
            with self.assertRaises(ValueError):
                fanout_dmx(source, self.patch)
        with self.assertRaises(ValueError):
            build_patch({"garland": -1})
        with self.assertRaises(ValueError):
            allocate_fixtures(self.patch["fixtures"] * 2, first_universe=32767)

    def test_causal_sample_never_interpolates_or_anticipates(self):
        timeline = {"time": np.array([0., .025, .05]),
                    "dimmer": np.array([[.1], [.4], [.9]])}
        result = sample_timeline(timeline, [-1, 0, .024999, .025, .049999, .05, 9])
        np.testing.assert_array_equal(result["source_index"], [0, 0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(result["dimmer"][:, 0], [.1, .1, .1, .4, .4, .9, .9])
        self.assertEqual(sample_timeline(timeline, .03)["source_index"], 1)
        with self.assertRaises(ValueError):
            sample_timeline(timeline, float("nan"))
        with self.assertRaises(ValueError):
            sample_timeline({"time": np.array([0, 0])}, 0)

    def test_fixture_import_does_not_require_audio_analysis_in_blender(self):
        result = subprocess.run([sys.executable, "-c", (
            "import sys; import lightsem.dmx as dmx; "
            "assert dmx.Fixture.__name__ == 'Fixture'; "
            "assert 'SceneEngine' in dmx.__all__; "
            "assert 'librosa' not in sys.modules; "
            "assert 'torch' not in sys.modules"
        )], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ArtNetTests(unittest.TestCase):
    def test_saved_npz_to_udp_to_renderer_controls_are_identical(self):
        patch = build_patch({"moving_head": 65})
        source = np.random.default_rng(76).integers(0, 256, (2, 512), dtype=np.uint8)
        universes = fanout_dmx(source, patch)
        saved = decode_festival(universes, patch)
        saved.update(time=np.array([0., .025]), source_dmx=source, universes=universes,
                     universe_ids=np.array(patch["universe_ids"], dtype=np.uint16))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(root / "timeline.npz", **saved)
            patch["timeline"] = {"sha256": hashlib.sha256((root / "timeline.npz").read_bytes()).hexdigest()}
            (root / "fixture_patch.json").write_text(json.dumps(patch))
            loaded = load_timeline(root)
            frame = sample_timeline(loaded, .03)
            receiver_state = ArtNetState(loaded["universe_ids"])
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                receiver.bind(("127.0.0.1", 0))
                receiver.settimeout(2)
                with ArtNetSender(port=receiver.getsockname()[1]) as sender:
                    count = sender.send_frame(frame["universes"], loaded["universe_ids"])
                    for _ in range(count):
                        self.assertTrue(receiver_state.ingest(receiver.recvfrom(1024)[0]))
            np.testing.assert_array_equal(receiver_state.snapshot(), universes[1])
            received_controls = decode_festival(receiver_state.snapshot()[None], patch)
            for key, array in received_controls.items():
                np.testing.assert_array_equal(array[0], frame[key])
            changed_patch = copy.deepcopy(patch)
            changed_patch["fixtures"][0]["source_fixture"] = "mh2"
            changed_patch["fixtures"][0]["source_channels"] = list(range(13, 25))
            (root / "fixture_patch.json").write_text(json.dumps(changed_patch))
            with self.assertRaisesRegex(ValueError, "mapping"):
                load_timeline(root)
            (root / "fixture_patch.json").write_text(json.dumps(patch))
            saved["dimmer"][0, 0] = .12345
            np.savez_compressed(root / "timeline.npz", **saved)
            patch["timeline"]["sha256"] = hashlib.sha256((root / "timeline.npz").read_bytes()).hexdigest()
            (root / "fixture_patch.json").write_text(json.dumps(patch))
            with self.assertRaisesRegex(ValueError, "dimmer controls"):
                load_timeline(root)
            # Appending an otherwise harmless ZIP tail must still fail provenance.
            with (root / "timeline.npz").open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_timeline(root)

    def test_known_wire_encoding_and_decoding(self):
        # Independent literal confirms each endian field and 15-bit port address.
        expected = b"Art-Net\x00\x00\x50\x00\x0e\xff\x02\x23\x01\x00\x04\x01\x02\x03\x00"
        packet = encode_artdmx(b"\x01\x02\x03", universe=0x123, sequence=255, physical=2)
        self.assertEqual(packet, expected)
        decoded = decode_artdmx(expected)
        self.assertEqual((decoded.universe, decoded.sequence, decoded.physical), (0x123, 255, 2))
        self.assertEqual(decoded.data, bytes([1, 2, 3, 0]))
        # Largest universe and a full 512-slot payload survive unchanged.
        data = bytes(range(256)) * 2
        self.assertEqual(decode_artdmx(encode_artdmx(data, 32767, 0)).data, data)

    def test_malformed_wire_never_applies_partial_values(self):
        valid = encode_artdmx(b"\x01\x02")
        bad_packets = [b"", valid[:-1], valid + b"x", b"not-net!" + valid[8:],
                       valid[:8] + b"\x00\x20" + valid[10:],
                       valid[:10] + b"\x00\x0d" + valid[12:],
                       valid[:14] + b"\x00\x80" + valid[16:],
                       valid[:16] + b"\x00\x03" + valid[18:] + b"x"]
        for packet in bad_packets:
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                decode_artdmx(packet)
        for kwargs in ({"universe": 32768}, {"sequence": 256}, {"universe": True}):
            with self.assertRaises(ValueError):
                encode_artdmx(b"\x01\x02", **kwargs)

    def test_receiver_sequence_wrap_restart_and_timeout(self):
        state = ArtNetState([0, 257])
        self.assertTrue(state.ingest(encode_artdmx(b"\x12\x34", 257, 255), now=0))
        self.assertTrue(state.ingest(encode_artdmx(b"\x23\x45", 257, 1), now=.01))
        self.assertFalse(state.ingest(encode_artdmx(b"\xff\xff", 257, 255), now=.02))
        self.assertFalse(state.ingest(encode_artdmx(b"\xff\xff", 258, 1), now=.02))
        self.assertEqual(state.snapshot()[1, 0], 0x23)
        self.assertFalse(state.snapshot(now=3, timeout_s=2).any())
        self.assertTrue(state.ingest(encode_artdmx(b"\x56\x78", 257, 200), now=3))
        self.assertTrue(state.ingest(encode_artdmx(b"\x9a\xbc", 257, 0), now=3.01))
        self.assertEqual(state.snapshot()[1, 0], 0x9a)

    def test_real_local_udp_carries_all_universe_bytes(self):
        # Actual sockets prove sender integration, beyond codec self-consistency.
        frames = np.random.default_rng(3).integers(0, 256, (3, 512), dtype=np.uint8)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(2)
            with ArtNetSender(port=receiver.getsockname()[1]) as sender:
                self.assertEqual(sender.send_frame(frames, [0, 1, 257]), 3)
                received = [receiver.recvfrom(1024)[0] for _ in range(3)]
        for frame, universe, packet in zip(frames, [0, 1, 257], received):
            self.assertEqual(packet[:8], b"Art-Net\x00")
            self.assertEqual(struct.unpack_from("<H", packet, 14)[0], universe)
            self.assertEqual(struct.unpack_from(">H", packet, 16)[0], 512)
            self.assertEqual(packet[18:], frame.tobytes())

    def test_restarted_sender_resets_old_sequence_without_a_two_second_freeze(self):
        state = ArtNetState([1])
        state.ingest(encode_artdmx(bytes([9, 9]), 1, sequence=80), now=10.)
        frame = np.full((1, 512), 23, dtype=np.uint8)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(2)
            for restart in range(2):
                with ArtNetSender(port=receiver.getsockname()[1]) as sender:
                    for offset in range(2):
                        frame[:] = 23 + restart * 2 + offset
                        sender.send_frame(frame, [1])
                        packet = receiver.recvfrom(1024)[0]
                        self.assertEqual(decode_artdmx(packet).sequence, offset)
                        self.assertTrue(state.ingest(packet, now=10.01 + .025 * (restart * 2 + offset)))
                        np.testing.assert_array_equal(state.snapshot(), frame)


if __name__ == "__main__":
    unittest.main()
