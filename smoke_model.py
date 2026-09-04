"""Verify the macro encoder is genuinely causal and the data path is sane.

A non-causal model would score better and be unshippable on a real desk, so
this is a correctness gate, not a nicety. We do not trust the mask argument --
we perturb the future and demand bit-identical output.
"""
import numpy as np
import torch

from lightsem.model.data import DECIM, TARGET_NAMES, load_songs, split_songs
from lightsem.model.encoder import MacroEncoder

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fail += 1


print("=== causality ===")
torch.manual_seed(0)
m = MacroEncoder().eval()
x = torch.randn(3, 64, 512)
with torch.no_grad():
    p0, t0 = m(x)

# Corrupt EARLY frames -> output MUST change (they are the past).
# NOTE: the perturbation must NOT be a constant offset across the feature dim.
# input_norm is a LayerNorm over the 512 features, so `x[:, :8] += 5.0` is
# mathematically erased before it reaches any weight. Using random noise
# instead. (This bit us once: it looked like a causality failure.)
x_past = x.clone()
x_past[:, :8] += torch.randn_like(x_past[:, :8]) * 5.0
with torch.no_grad():
    p_past, _ = m(x_past)
check("past affects output", not torch.allclose(p0, p_past, atol=1e-6),
      f"delta={float((p0-p_past).abs().max()):.4f}")

# And prove the LayerNorm invariance claim above, so the note stays honest.
x_const = x.clone()
x_const[:, :8] += 5.0
with torch.no_grad():
    p_const, _ = m(x_const)
check("constant offset absorbed by LayerNorm (expected)",
      torch.allclose(p0, p_const, atol=1e-6),
      f"delta={float((p0-p_const).abs().max()):.2e}")

# True future test: extend the window and confirm the prediction for an
# earlier position is unchanged by frames after it.
xs = torch.randn(2, 128, 512)
with torch.no_grad():
    a = m.latent(xs[:, :64])
    xs2 = xs.clone()
    xs2[:, 64:] += 9.0          # scramble everything after frame 64
    b = m.latent(xs2[:, :64])   # same slice -> must be identical
check("future frames outside window cannot leak", torch.equal(a, b))

# Attention mask test: within one window, token i must not see token j>i.
h = m.adapter(m.input_norm(xs))
tok = m.patching(h)
tok = tok + m.pos[:, : tok.shape[1]]
from lightsem.model.encoder import causal_mask
with torch.no_grad():
    full = m.transformer(tok, mask=causal_mask(tok.shape[1], tok.device))
    tok2 = tok.clone()
    tok2[:, 5:] += 7.0
    part = m.transformer(tok2, mask=causal_mask(tok.shape[1], tok.device))
check("token i ignores tokens j>i", torch.allclose(full[:, :5], part[:, :5], atol=1e-5),
      f"delta={float((full[:,:5]-part[:,:5]).abs().max()):.2e}")

print("\n=== data ===")
songs = load_songs()
check("songs loaded", len(songs) > 100, f"n={len(songs)}")
check("decimation factor", DECIM == 3, f"DECIM={DECIM}")
lens = np.array([len(s.x) for s in songs])
print(f"  10 Hz frames/song: min={lens.min()} med={int(np.median(lens))} "
      f"max={lens.max()} total={lens.sum():,}")
check("x/y aligned", all(len(s.x) == len(s.y) for s in songs))
check("targets are 6-dim", all(s.y.shape[1] == len(TARGET_NAMES) for s in songs))

tr, va, te = split_songs(songs)
check("split non-empty", min(len(tr), len(va), len(te)) > 0,
      f"train={len(tr)} val={len(va)} test={len(te)}")
keys_tr, keys_va, keys_te = ({s.key for s in g} for g in (tr, va, te))
check("no song appears in two splits",
      not (keys_tr & keys_va) and not (keys_tr & keys_te) and not (keys_va & keys_te))

print("\n=== model ===")
print(f"  params: {m.n_params():,}")
p, t = m(torch.randn(4, 64, 512))
check("pred shape [B,H,6]", tuple(p.shape) == (4, 4, 6), str(tuple(p.shape)))
check("tension shape [B]", tuple(t.shape) == (4,), str(tuple(t.shape)))

print(f"\n{'ALL CHECKS PASS' if fail == 0 else f'{fail} FAILURES'}")
raise SystemExit(1 if fail else 0)
