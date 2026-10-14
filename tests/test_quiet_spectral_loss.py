import torch

from scripts.motion.train_native_stereo_e2e import quiet_spectral_excess


def test_quiet_spectral_loss_penalizes_high_frequency_velocity_and_backpropagates():
    time = torch.arange(60, dtype=torch.float32)
    signal = torch.sin(2 * torch.pi * 10 * time / 30)[:, None].repeat(1, 22)
    signal[:, 0] = 0.
    generated = torch.stack((signal, signal * .2, signal * .1), -1)[None].repeat(2, 1, 1, 1).requires_grad_()
    baseline = torch.zeros_like(generated)
    quiet = torch.ones(2, 60, dtype=torch.bool)
    valid = torch.ones_like(quiet)
    loss = quiet_spectral_excess(generated, baseline, quiet, valid, target_ratio=.7, cutoff_hz=6.)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()
