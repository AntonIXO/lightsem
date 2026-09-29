"""Live canonical LightSem input → patch fanout → real UDP output checks."""
import socket
import unittest

import numpy as np

from lightsem.festival.artnet import ArtNetState, encode_artdmx
from lightsem.festival.patch import build_patch, decode_festival, fanout_dmx
from lightsem.festival.router import LiveRouter


class LiveRouterTests(unittest.TestCase):
    def setUp(self):
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.settimeout(.1)
        self.source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.patch = build_patch({"moving_head": 65})

    def tearDown(self):
        self.receiver.close()
        self.source.close()

    def router(self):
        return LiveRouter(self.patch, port=0, output_port=self.receiver.getsockname()[1])

    def capture(self):
        state = ArtNetState(self.patch["universe_ids"])
        for _ in self.patch["universe_ids"]:
            packet, _ = self.receiver.recvfrom(1024)
            self.assertTrue(state.ingest(packet))
        return state.snapshot()

    def test_real_canonical_udp_drives_three_output_universes(self):
        source = np.random.default_rng(82).integers(0, 256, 512, dtype=np.uint8)
        self.assertEqual(len(self.patch["universe_ids"]), 3)
        with self.router() as router:
            self.source.sendto(encode_artdmx(source, universe=0, sequence=4), router.address)
            self.assertEqual(router.poll_once(.1), 1)
            received = self.capture()
            expected = fanout_dmx(source[None], self.patch)[0]
            np.testing.assert_array_equal(received, expected)
            np.testing.assert_array_equal(received[0], source)
            decoded = decode_festival(received[None], self.patch)
            expected_controls = decode_festival(expected[None], self.patch)
            for key in decoded:
                np.testing.assert_array_equal(decoded[key], expected_controls[key])
            self.assertEqual(router.statistics["output_packets"], 3)
            self.assertEqual(router.poll_once(.01), 0)
            with self.assertRaises(socket.timeout):
                self.receiver.recvfrom(1024)

    def test_ignores_bad_wrong_universe_and_old_sequence_input(self):
        with self.router() as router:
            address = router.address
            self.source.sendto(encode_artdmx(b"\x01\x02", 0, 100), address)
            self.assertEqual(router.poll_once(.1), 1)
            first = self.capture()
            np.testing.assert_array_equal(first[0, :2], [1, 2])
            self.assertFalse(first[0, 2:].any())
            self.source.sendto(b"not Art-Net", address)
            self.source.sendto(encode_artdmx(b"\xff\xff", 1, 101), address)
            self.source.sendto(encode_artdmx(b"\xff\xff", 0, 99), address)
            self.assertEqual(router.poll_once(.1), 0)
            self.assertEqual(router.statistics["malformed"], 1)
            self.assertEqual(router.statistics["ignored_universe"], 1)
            self.assertEqual(router.statistics["ignored_sequence"], 1)
            with self.assertRaises(socket.timeout):
                self.receiver.recvfrom(1024)

    def test_existing_lightsem_sender_connects_without_protocol_changes(self):
        from lightsem.dmx.universe import ArtNetSender as ExistingLightSemSender

        source = np.arange(512, dtype=np.uint16).astype(np.uint8)
        with self.router() as router:
            sender = ExistingLightSemSender(host=router.address[0], port=router.address[1])
            try:
                self.assertEqual(sender.send(source), 530)
                self.assertEqual(router.poll_once(.1), 1)
                np.testing.assert_array_equal(self.capture(), fanout_dmx(source[None], self.patch)[0])
            finally:
                sender.sock.close()

    def test_restarted_legacy_sender_starts_a_new_input_sequence_session(self):
        from lightsem.dmx.universe import ArtNetSender as ExistingLightSemSender

        source = np.full(512, 17, dtype=np.uint8)
        with self.router() as router:
            old = ExistingLightSemSender(host=router.address[0], port=router.address[1])
            restarted = ExistingLightSemSender(host=router.address[0], port=router.address[1])
            try:
                old._seq = 79  # Last frame of a prior source run had sequence 80.
                old.send(source)
                self.assertEqual(router.poll_once(.1), 1)
                np.testing.assert_array_equal(self.capture(), fanout_dmx(source[None], self.patch)[0])
                source[:] = 193
                restarted.send(source)  # Its first normal frame has sequence 1.
                self.assertNotEqual(old.sock.getsockname(), restarted.sock.getsockname())
                old.sock.close()
                self.assertEqual(router.poll_once(.1), 1)
                np.testing.assert_array_equal(self.capture(), fanout_dmx(source[None], self.patch)[0])
                self.assertEqual(router.statistics["source_restarts"], 1)
                # An old packet from the current endpoint remains rejected.
                restarted.sock.sendto(encode_artdmx(source, 0, sequence=255), router.address)
                self.assertEqual(router.poll_once(.1), 0)
                self.assertEqual(router.statistics["ignored_sequence"], 1)
            finally:
                old.sock.close()
                restarted.sock.close()

    def test_rejects_self_destination_and_filters_source_echo(self):
        # Reserve then release a local port. The router must reject the loop
        # before emitting anything, and release the input socket on failure.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        for bind in ("127.0.0.1", "0.0.0.0"):
            with self.assertRaisesRegex(ValueError, "same endpoint"):
                LiveRouter(self.patch, bind=bind, port=port, host="localhost", output_port=port)
        with self.router() as router:
            # A misdirected source-passthrough packet retains our own sender
            # endpoint and must never feed the router again.
            router._output.sendto(encode_artdmx(b"\xff\xff", 0), router.address)
            self.assertEqual(router.poll_once(.1), 0)
            self.assertEqual(router.statistics["ignored_echo"], 1)

    def test_bounded_quiet_run_closes_sockets_and_sends_nothing(self):
        router = self.router()
        input_address, output_address = router.address, router.sender_address
        result = router.run(.03)
        self.assertTrue(router.closed)
        self.assertEqual(result["output_packets"], 0)
        self.assertGreaterEqual(result["elapsed_s"], .025)
        self.assertLess(result["elapsed_s"], .3)
        router.close()  # Idempotent cleanup.
        for address in (input_address, output_address):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as replacement:
                replacement.bind(address)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            router.poll_once()
        invalid = self.router()
        with self.assertRaises(ValueError):
            invalid.run(float("nan"))
        self.assertTrue(invalid.closed)


if __name__ == "__main__":
    unittest.main()
