import unittest
import numpy as np

from lightsem.dmx.encode import pack_universe
from lightsem.dmx.decode import decode_stream
from lightsem.dmx.fixtures import Fixture, Rig


class NativeDmxValidationTests(unittest.TestCase):
    def setUp(self):
        self.rig = Rig([Fixture('head', 'moving_head', 1)])
        self.params = {k: np.full((2, 1), .5) for k in
                       ('dimmer', 'strobe', 'hue', 'sat', 'val', 'pan', 'tilt', 'zoom')}
        self.params['gobo'] = np.zeros((2, 1))

    def test_pack_rejects_out_of_range_and_nonfinite_values(self):
        for key, value in (('dimmer', 1.1), ('pan', -0.1), ('sat', np.nan), ('gobo', 256.)):
            params = {k: v.copy() for k, v in self.params.items()}
            params[key][0, 0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pack_universe(self.rig, params)

    def test_pack_rejects_bad_patch_and_decode_requires_full_universe(self):
        bad = Rig([Fixture('bad', 'moving_head', 505)])
        with self.assertRaises(ValueError):
            pack_universe(bad, self.params)
        with self.assertRaises(ValueError):
            decode_stream(self.rig, np.zeros((2, 511), dtype=np.uint8))
        with self.assertRaises(ValueError):
            decode_stream(self.rig, np.zeros((2, 512), dtype=np.float32))

    def test_round_trip_preserves_valid_stream(self):
        frames = pack_universe(self.rig, self.params)
        decoded = decode_stream(self.rig, frames)
        self.assertEqual(frames.shape, (2, 512))
        self.assertTrue(np.isfinite(decoded['pan']).all())


if __name__ == '__main__':
    unittest.main()
