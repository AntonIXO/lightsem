import numpy as np

from lightsem.dance.color_field import color_field_metrics, continuous_color_field, oklab_to_srgb, srgb_to_oklab
from lightsem.conductor.body_transfer import TransferConfig, compile_intent, decode
from lightsem.conductor.body_transfer_v2 import make_outputs


def test_oklab_round_trip_and_arbitrary_n_flow():
    rng = np.random.default_rng(4)
    rgb = rng.random((12, 3))
    assert np.max(np.abs(oklab_to_srgb(srgb_to_oklab(rgb)) - rgb)) < 1e-8
    world = np.zeros((48, 22, 3), dtype=float)
    world[:, 7, 0] = np.linspace(-.5, .8, len(world))
    world[:, 11, 2] = np.sin(np.linspace(0, 2, len(world)))
    left = np.repeat([[20, 80, 220]], len(world), axis=0).astype(np.uint8)
    right = np.repeat([[240, 40, 100]], len(world), axis=0).astype(np.uint8)
    outputs = make_outputs(beams=11, lasers=3, bulbs=7)
    field, meta = continuous_color_field(world, left, right, outputs)
    assert field.shape == (len(world), len(outputs), 3)
    assert field.dtype == np.uint8 and np.isfinite(field).all()
    assert meta['arbitrary_n'] and meta['brightness_separate']
    metrics = color_field_metrics(field)
    assert metrics['unique_fixture_streams'] > 1
    assert metrics['spatial_gradient_active_fraction'] > .1


def test_fixture_rgb_is_encoded_per_output_and_dimmer_stays_separate():
    outputs = make_outputs(beams=2, lasers=0, bulbs=1)
    n = 5
    targets = np.zeros((n, len(outputs), 3), dtype=float); targets[..., 1] = 1.
    levels = np.full((n, len(outputs)), .4)
    side = np.full((n, 3), 17, dtype=np.uint8)
    fixture = np.zeros((n, len(outputs), 3), dtype=np.uint8)
    fixture[:, 0] = [255, 0, 0]; fixture[:, 1] = [0, 255, 0]; fixture[:, 2] = [0, 0, 255]
    dmx, patch, _ = compile_intent({'targets': targets, 'dimmer': levels}, outputs, side, side,
                                   TransferConfig(apply_motion_limits=False), fixture_rgb=fixture)
    decoded = decode(dmx, patch)
    assert np.array_equal(np.rint(decoded['rgb'] * 255).astype(np.uint8), fixture)
    assert np.allclose(decoded['dimmer'], levels)

