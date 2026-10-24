import numpy as np
import torch

from lightsem.experiments.foundation_audio_light import (
    AudioEncoder,
    AudioEncoderConfig,
    MuQContract,
    configure_muq_scope,
    inject_lora,
    trainable_parameter_report,
    validate_base92,
    validate_muq_features,
)
from lightsem.experiments.foundation_audio_light.layer_selection import (
    FeatureFamily,
    compare_feature_families,
    geometry,
)
from lightsem.experiments.foundation_audio_light.retention import compare_features
from lightsem.experiments.foundation_audio_light.peft import (
    AdapterConfig,
    CausalTransformerAdapter,
    HierarchicalTemporalAdapter,
    LocalConvAdapter,
)


def test_contract_validates_pinned_dimensions_and_finiteness():
    contract = MuQContract()
    assert validate_muq_features(np.zeros((4, 1024)), contract=contract).shape == (4, 1024)
    assert validate_base92(np.zeros((4, 92)), contract=contract).shape == (4, 92)
    for bad in (np.zeros((4, 1023)), np.full((4, 1024), np.nan)):
        try:
            validate_muq_features(bad, contract=contract)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MuQ feature array accepted")


def test_zero_initialized_adapters_preserve_frozen_feature_baseline():
    x = torch.randn(2, 32, 16, generator=torch.Generator().manual_seed(3))
    valid = torch.ones(2, 32, dtype=torch.bool)
    for adapter in (
        LocalConvAdapter(16, 4),
        CausalTransformerAdapter(16, 4, 4, 8),
        HierarchicalTemporalAdapter(AdapterConfig(kind="hierarchical", width=16, bottleneck=4, heads=4)),
    ):
        adapter.eval()
        with torch.no_grad():
            y = adapter(x, valid)
        assert torch.allclose(x, y, atol=1e-7, rtol=1e-7)


def test_causal_adapter_does_not_use_future_frames():
    adapter = CausalTransformerAdapter(16, 4, 4, 8).eval()
    x = torch.randn(1, 24, 16, generator=torch.Generator().manual_seed(4))
    changed = x.clone()
    changed[:, 12:] += 20.0
    with torch.no_grad():
        a = adapter(x)
        b = adapter(changed)
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-6, rtol=1e-6)


def test_audio_encoder_adapter_is_optional_and_finite():
    x = torch.randn(2, 20, 12, generator=torch.Generator().manual_seed(5))
    valid = torch.ones(2, 20, dtype=torch.bool)
    base = AudioEncoder(AudioEncoderConfig(12, 16, adapter_kind="none"))
    adapted = AudioEncoder(AudioEncoderConfig(12, 16, adapter_kind="hierarchical"))
    for model in (base, adapted):
        y = model(x, valid=valid)
        assert y.shape == (2, 20, 16)
        assert torch.isfinite(y).all()


def test_lora_replacement_freezes_base_and_is_zero_initialized():
    module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    original = module(torch.ones(2, 8))
    replaced = inject_lora(module, rank=2, alpha=4, name_fragments=("0",))
    assert replaced == ["0"]
    report = trainable_parameter_report(module)
    assert report["trainable"] < report["total"]
    after = module(torch.ones(2, 8))
    assert torch.allclose(original, after, atol=1e-7, rtol=1e-7)


def test_last_block_scope_is_explicit():
    module = torch.nn.Module()
    module.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
    report = configure_muq_scope(module, "last_blocks", last_blocks=2)
    assert report["trainable_parameters"] == 2 * (4 * 4 + 4)
    assert all(not p.requires_grad for p in module.layers[0].parameters())
    assert all(p.requires_grad for p in module.layers[-1].parameters())


def test_layer_probe_reports_geometry_and_grouped_gain():
    t = np.arange(80, dtype=float) / 10.0
    target = np.sin(t)
    informative = target[:, None]
    noise = np.random.default_rng(0).normal(size=(80, 5))
    report = compare_feature_families(
        [FeatureFamily("Toy", np.column_stack([informative, noise]), t)],
        target_time=t,
        target=target,
        groups=np.repeat(np.arange(4), 20),
        pca_components=3,
        folds=4,
    )
    assert report["families"]["Toy"]["geometry"]["effective_rank"] > 1
    assert report["families"]["Toy"]["probe"]["grouped_folds"] is True
    assert report["selection"]["ranked_by_gain"] == ["Toy"]
    assert geometry(np.zeros((4, 3)))["constant_channel_fraction"] == 1.0


def test_retention_reports_feature_drift_without_calling_it_quality():
    reference = np.arange(40, dtype=float).reshape(10, 4)
    candidate = reference + 0.01
    report = compare_features(reference, candidate)
    assert report["finite"] is True
    assert report["frames"] == 10
    assert report["mse"] > 0
    assert report["scope"].startswith("representation retention")

