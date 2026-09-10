"""Fast defaults for new runs; explicit legacy decoding for frozen protocols."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrainingRuntime:
    precision: str = 'bf16'
    optimizer: str = 'adamw'
    compile: str = 'default'
    prefetch: int = 2

    @classmethod
    def legacy(cls, precision='fp32'):
        return cls(precision=precision, optimizer='adamw', compile='off', prefetch=0)

    @classmethod
    def from_training(cls, config):
        # Missing fields in an old serialized protocol mean the OLD defaults.
        values=cls.legacy(config.get('precision','fp32')).as_dict()
        values.update(config.get('runtime',{}))
        runtime=cls(**values)
        if config.get('precision',runtime.precision)!=runtime.precision:
            raise ValueError('Conflicting training precision and runtime identity')
        return runtime

    def __post_init__(self):
        if self.precision not in ('fp32', 'tf32', 'bf16'):
            raise ValueError('Unknown precision')
        if self.optimizer not in ('adamw', 'adamw_fused'):
            raise ValueError('Unknown optimizer')
        if self.compile not in ('off', 'default') or self.prefetch not in (0, 2):
            raise ValueError('Unsupported compile/prefetch option')

    def as_dict(self):
        return asdict(self)

    def apply(self):
        """Call AFTER seeding: seed_all intentionally restores legacy settings."""
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; CPU fallback is prohibited')
        if self.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Native BF16 is required')
        torch.backends.cuda.matmul.allow_tf32 = self.precision == 'tf32'
        torch.backends.cudnn.allow_tf32 = self.precision == 'tf32'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.compile != 'off':
            from torch._functorch import config as compiler_config
            from torch._dynamo import config as dynamo_config
            compiler_config.backward_pass_autocast = 'off'
            dynamo_config.fail_on_recompile_limit_hit = True

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext()

    def make_optimizer(self, model, lr):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01,
                                 **({'fused': True} if self.optimizer == 'adamw_fused' else {}))

    def loss_function(self, model):
        fn = model.losses_from_draws
        if self.compile == 'default':
            # Each model in a sequential campaign gets its own guard budget.
            torch.compiler.reset()
        return torch.compile(fn, mode='default', dynamic=False) if self.compile == 'default' else fn


def backward_checked(losses, model):
    """One host decision per update, without weakening fail-before-step checks."""
    losses['total'].backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    finite = torch.stack([v.detach().isfinite() for v in losses.values()] + [grad.isfinite()]).all()
    if not bool(finite):
        raise FloatingPointError('Nonfinite loss or gradient; optimizer not updated')
    return grad


def runtime_loss(model, fn, runtime, x, y, b, pos, generator, beat_dropout=.25):
    # Explicit flow RNG stays outside Dynamo; targets and draws stay FP32.
    x, z, t = model.loss_inputs(x, y, generator, beat_dropout)
    with runtime.autocast():
        return fn(x, y, b, pos, z, t)


class BatchStream:
    """Two pinned buffers, one CPU producer, independent consumed RNG cursor.

    The producer owns a cloned NumPy generator. The public generator advances
    only when a batch is consumed, so discarded prefetched batches replay on
    resume. CPU buffers are not reused until their H2D event has completed.
    """
    def __init__(self, data, size, frames, rng, depth=0):
        if depth not in (0, 2):
            raise ValueError('depth must be 0 or 2')
        self.data, self.size, self.frames, self.rng = data, size, frames, rng
        self.depth = depth
        self.closed = False
        if not depth:
            return
        self.producer_rng = np.random.default_rng()
        self.producer_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lightsem-crops')
        self.copy_stream = torch.cuda.Stream()
        self.slots = []
        for _ in range(depth):
            self.slots.append({
                'index': torch.empty((size, frames), dtype=torch.int64, pin_memory=True),
                'aux': (torch.empty((size, frames, data.aux_dim), dtype=torch.float16, pin_memory=True)
                        if data.host_aux is not None else None), 'event': None})
        self.pending = [(slot, self.pool.submit(self._fill, slot)) for slot in self.slots]

    def _fill(self, slot):
        if slot['event'] is not None:
            slot['event'].synchronize()
        ix = self.data.draw_indices(self.size, self.frames, self.producer_rng)
        np.copyto(slot['index'].numpy(), ix)
        if slot['aux'] is not None:
            np.take(self.data.host_aux, ix, axis=0, out=slot['aux'].numpy())
        return deepcopy(self.producer_rng.bit_generator.state)

    def next(self):
        if self.closed:
            raise RuntimeError('Closed batch stream')
        if not self.depth:
            return self.data.batch(self.size, self.frames, self.rng)
        slot, future = self.pending.pop(0)
        consumed_state = future.result()
        with torch.cuda.stream(self.copy_stream):
            index = slot['index'].to('cuda', non_blocking=True)
            aux = slot['aux'].to('cuda', non_blocking=True) if slot['aux'] is not None else None
            done = torch.cuda.Event()
            done.record()
        slot['event'] = done
        current = torch.cuda.current_stream()
        current.wait_event(done)
        index.record_stream(current)
        x = self.data.x[index]
        if aux is not None:
            aux.record_stream(current)
            x = torch.cat((x, (aux.float() - self.data.aux_mean) / self.data.aux_std), dim=2)
        result = x, self.data.y[index], self.data.b[index]
        self.rng.bit_generator.state = consumed_state
        self.pending.append((slot, self.pool.submit(self._fill, slot)))
        return result

    def close(self):
        if not self.closed and self.depth:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.copy_stream.synchronize()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
