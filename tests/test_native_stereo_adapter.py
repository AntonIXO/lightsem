"""Native-C compatibility tests for the optional StereoAudioV1 branch."""
import torch

from lightsem.experiments.dynamic_dance.learning import DynamicFlow, DynamicFlowConfig
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel


def _models():
    config = MusicConfig(width=16, frames=48, dilations=(1, 2), steps=32)
    norms = {"mean": [0.] * 139, "std": [1.] * 139, "delta_scale": [1.] * 139}
    teacher = DynamicFlow(config, norms, DynamicFlowConfig(10, arm="D7"))
    model = NativeDynamicsModel(config, norms, Config(arm="joint", extra_dim=10, stereo_dim=4,
                                                       planner_width=16, planner_heads=4, frames=48))
    model.warmstart(teacher.state_dict())
    return teacher, model


def test_zero_stereo_adapters_preserve_incumbent_field_exactly():
    teacher, model = _models()
    fine = torch.randn(2, 48, 102)
    coarse = torch.randn(2, 8, 10)
    coarse_stereo = torch.randn(2, 8, 4)
    stereo = torch.randn(2, 48, 4)
    valid = torch.ones(2, 8, dtype=torch.bool)
    indices = torch.arange(48)[None].expand(2, -1)
    conditioning, _ = model.condition(fine, coarse, valid, indices, stereo, coarse_stereo)
    state = torch.randn(2, 48, 139)
    time = torch.tensor([.2, .7])
    reference = torch.cat((fine, torch.zeros(2, 48, 34)), -1)
    torch.testing.assert_close(model.flow.vector_field(conditioning, state, time),
                               teacher.vector_field(reference, state, time), rtol=0, atol=0)


def test_stereo_context_changes_field_only_after_nonzero_adapter_weights():
    teacher, model = _models()
    with torch.no_grad():
        model.flow.stereo_projection.weight.fill_(.01)
        model.flow.stereo_adapters[0].weight.fill_(.01)
    fine = torch.randn(1, 48, 102)
    coarse = torch.randn(1, 8, 10)
    valid = torch.ones(1, 8, dtype=torch.bool)
    indices = torch.arange(48)[None]
    zero = torch.zeros(1, 48, 4)
    nonzero = torch.ones(1, 48, 4)
    coarse_zero = torch.zeros(1, 8, 4)
    a, _ = model.condition(fine, coarse, valid, indices, zero, coarse_zero)
    b, _ = model.condition(fine, coarse, valid, indices, nonzero, coarse_zero)
    _, a_bias = model.flow.prepare_conditioning(a)
    _, b_bias = model.flow.prepare_conditioning(b)
    assert float((a_bias - b_bias).abs().max().detach()) > 0.


def test_dual_cross_attention_preserves_two_stream_width_at_zero_output_init():
    config = MusicConfig(width=16, frames=48, dilations=(1, 2), steps=32)
    norms = {"mean": [0.] * 139, "std": [1.] * 139, "delta_scale": [1.] * 139}
    model = NativeDynamicsModel(
        config, norms,
        Config(arm="joint", extra_dim=10, stereo_dim=128,
               stereo_fusion="dual_cross_attention", planner_width=16,
               planner_heads=4, frames=48),
    )
    value = torch.randn(2, 17, 128)
    fused = model.flow.flow_stereo(value)
    assert fused.shape == value.shape
    # The two MHA output projections are zero-initialized, so the new branch
    # starts as an exact concat control before end-to-end updates.
    torch.testing.assert_close(fused, value, rtol=0, atol=0)
