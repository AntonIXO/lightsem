"""Analytic geometry and decoded-control preservation checks; never create EGL."""
from contextlib import nullcontext
import shutil
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest

from lightsem.sim import lit_body_renderer as lit
from lightsem.sim.body_stage_renderer import BodyStageRenderer, stage_shader
from lightsem.sim.shaders import VERTEX_SHADER


@pytest.mark.parametrize('origin,direction,expected', [
    ([3, 0, 1], [-1, 0, 0], 2.),
    ([1, -2, 1], [0, 1, 0], 2.),
    ([1.01, -2, 1], [0, 1, 0], np.inf),
    ([0, 0, 5], [0, 0, -1], 2.),
    ([2, 0, 5], [0, 0, -1], np.inf),
    ([0, 0, 1], [1, 0, 0], 1.),
    ([0, 0, .5], [0, 0, 1], 2.5),  # Interior sphere seam is not a surface.
    ([3, 0, 2.5], [-1, 0, 0], 3-np.sqrt(3)/2),
    ([3, 0, 2], [-1, 0, 0], 2.),
    ([3, 0, 4], [-1, 0, 0], np.inf),
])
def test_capsule_known_surface_intersections(origin, direction, expected):
    value = lit.ray_capsule_distance(origin, direction, [0, 0, 0], [0, 0, 2], 1.)
    assert value == pytest.approx(expected)


def test_degenerate_bone_and_camera_inside_sphere():
    assert lit.ray_capsule_distance([0, 0, 3], [0, 0, -1], [0, 0, 0], [0, 0, 0], 1) == 2
    assert lit.ray_capsule_distance([0, 0, 0], [0, 0, -1], [0, 0, 0], [0, 0, 0], 1) == 1


def test_geometry_is_invariant_to_endpoint_order_and_fixed_rigid_transform():
    origin, direction, a, b = (np.array(x, dtype=float) for x in
        ([3, 0, 1], [-1, 0, 0], [0, 0, 0], [0, 0, 2]))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([5., 3., -7.])
    for first, second in ((a, b), (b, a)):
        distance = lit.ray_capsule_distance(rotation@origin+translation, rotation@direction,
            rotation@first+translation, rotation@second+translation, 1.)
        assert distance == pytest.approx(2.)


def test_fixed_metric_transform_preserves_source_translation_and_does_not_mutate_input():
    pose = np.arange(66, dtype=float).reshape(22, 3) / 100
    before = pose.copy()
    out = lit.transform_body(pose, scale=2., offset=[1, 2, 3])
    np.testing.assert_allclose(out, 2*pose+[1, 2, 3], rtol=1e-6)
    translated = lit.transform_body(pose+[2, 0, -1], scale=2., offset=[1, 2, 3])
    np.testing.assert_allclose(translated-out, np.tile([4, 0, -2], (22, 1)), atol=1e-6)
    np.testing.assert_array_equal(pose, before)


def test_bad_geometry_fails_before_context_creation(monkeypatch):
    monkeypatch.setattr(lit.moderngl, 'create_context', lambda **kwargs: pytest.fail('EGL must not be touched'))
    with pytest.raises(ValueError, match='scale'):
        lit.LitBodyStageRenderer([], None, body_scale=0)
    with pytest.raises(ValueError, match='fill'):
        lit.LitBodyStageRenderer([], None, neutral_fill=np.nan)
    for pose in (np.zeros((21, 3)), np.full((22, 3), np.nan)):
        with pytest.raises(ValueError, match='22 finite'):
            lit.transform_body(pose)


def test_shader_preserves_light_optics_and_avoids_body_sdf_inside_raymarch():
    original, shader = stage_shader(), lit.lit_body_shader()
    # The actual cone, falloff, angular edges, laser profile, RGB and dimmer
    # equations must remain identical; geometry cannot re-aim or recolor them.
    assert original.split('vec3 lightAt(', 1)[1].split('void main()', 1)[0] == \
           shader.split('vec3 lightAt(', 1)[1].split('void main()', 1)[0]
    assert shader.count('float bodyHit = intersectBody(') == 1
    march = shader.split('for (int s = 0; s < 96; s++)', 1)[1].split('float bodyHit', 1)[0]
    assert 'hitCapsule' not in march and 'sdScene' in march
    assert 'proj > tHit + (mat == 3 ? 0.0 : 0.08)' in shader
    assert shader.index('if (u_body_diagnostic == 2) {') < shader.index('// ---------- volumetric pass')


def test_shader_compiles_offline_without_egl(tmp_path):
    validator = shutil.which('glslangValidator')
    if validator is None:
        pytest.skip('Offline GLSL validator is unavailable')
    for suffix, source in [('frag', lit.lit_body_shader()), ('vert', VERTEX_SHADER)]:
        path = tmp_path / ('lit_body.' + suffix)
        path.write_text(source)
        subprocess.run([validator, '-S', suffix, str(path)], check=True, capture_output=True, text=True)


def cpu_renderer():
    renderer = lit.LitBodyStageRenderer.__new__(lit.LitBodyStageRenderer)
    renderer.body_scale, renderer.body_offset, renderer.neutral_fill = 1., (0., 0., 0.), .12
    renderer.ctx = nullcontext()
    renderer.fbo = Mock()
    renderer._array, renderer._require_uniform = Mock(), Mock()
    return renderer


def test_pose_changes_never_change_uploaded_fixture_controls(monkeypatch):
    renderer = cpu_renderer()
    directions, rgb, dimmer = np.eye(3), np.ones((3, 3)), np.array([0., .4, .8])
    snapshots = [x.copy() for x in (directions, rgb, dimmer)]
    received = []

    def fake_base(self, dirs, colors, levels):
        received.append((dirs, colors, levels))
        return np.zeros((3, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(BodyStageRenderer, 'render_controls', fake_base)
    pose = np.zeros((22, 3))
    renderer.render_controls(directions, rgb, dimmer, pose)
    pose[20] = [1., 1., 0.]
    renderer.render_controls(directions, rgb, dimmer, pose)
    assert len(received) == 2
    for args in received:
        assert all(a is b for a, b in zip(args, (directions, rgb, dimmer)))
    for before, after in zip(snapshots, (directions, rgb, dimmer)):
        np.testing.assert_array_equal(before, after)
    assert {call.args[0] for call in renderer._array.call_args_list} == {'u_body_joints'}
    assert renderer.fbo.use.call_count == 2


def test_diagnostic_controls_use_zero_dimmer_without_altering_actual_inputs():
    renderer = cpu_renderer()
    calls = []

    def fake(dirs, colors, levels, pose, *, diagnostic='scene'):
        calls.append((levels.copy(), diagnostic))
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        value = {'body_mask': 255, 'direct_body_light': 8}.get(diagnostic, 20 if levels.any() else 3)
        image[1:3, 2:4] = value
        return image

    renderer.render_controls = fake
    dimmer = np.array([.2, .6])
    result = renderer.render_diagnostics(np.zeros((2, 3)), np.ones((2, 3)), dimmer, np.zeros((22, 3)))
    np.testing.assert_array_equal(dimmer, [.2, .6])
    np.testing.assert_array_equal(calls[1][0], [0., 0.])
    assert [x[1] for x in calls] == ['scene', 'scene', 'body_mask', 'direct_body_light']
    assert result['metrics']['body_visible_pixels'] == 4
    assert result['metrics']['body_bbox_xyxy'] == [2, 1, 3, 2]
    assert result['metrics']['body_lights_off_mean_255'] == 3
    assert result['metrics']['body_direct_light_mean_255'] == 8
    assert result['metrics']['body_direct_lit_pixels_above_1'] == 4
