"""Controlled self-supervised body-to-light representation loop.

The package is intentionally an offline representation probe.  It consumes
already-audited native body trajectories, frozen audio features, and semantic
light state; it does not generate a replacement body, raw DMX, or fallback
motion.  A policy may use this representation only after the input-sensitivity
and raw-body gates pass.
"""

from .loop import ControlledSSLLoop, audit_input_sensitivity
from .model import ControlledJepaConfig, ControlledJepaModel, make_mask
from .losses import controlled_ssl_loss
from .adapter import SSLAdapter

__all__ = [
    "ControlledJepaConfig",
    "ControlledJepaModel",
    "ControlledSSLLoop",
    "audit_input_sensitivity",
    "controlled_ssl_loss",
    "make_mask",
    "SSLAdapter",
]
