# Controlled body-to-light SSL loop — protocol v0

This protocol defines a CPU-testable representation loop before any native
training or DMX generation. It is deliberately narrower than an RL policy.

## Inputs

Each row contains independently authenticated frozen audio features, an
already-audited native 30 Hz body, and semantic light state. Semantic light
state may contain normalized aim/motion, angular speed/acceleration, P2/RGB
state, brightness and cue probability. Raw DMX universes are not a training
input to this loop.

The caller supplies physical timestamps. The model is not allowed to invent an
independent temporal clock. Shpongle and AURORA — Cure For Me retain separate
audio, body, light and provenance streams.

## Objective

The student receives audio at every frame, while body and semantic light are
masked in reproducible temporal spans. An EMA target encoder sees the clean
sequence. The predictor matches target embeddings only at masked valid frames.
An additional causal head predicts the next target embedding conditioned on the
previous semantic light action. Variance and covariance terms are logged to
prevent a low-loss constant representation from being called success.

The target encoder is representation supervision, not a causal light-to-body
claim. That claim requires randomized safe light interventions and live dancer
responses later.

## Locked checks before native use

- finite values and external timestamp monotonicity;
- masked-context no-leak test;
- target encoder has no gradients;
- finite loss and gradient norm;
- embedding standard deviation and effective rank;
- body intervention changes the body-conditioned representation;
- semantic-light intervention changes the action-conditioned future prediction;
- audio/body/light shuffle controls are recorded separately;
- native raw-body speed, acceleration, jerk, root/support and duplicate-stream
  gates remain mandatory before mapping, interpolation, DMX or rendering.

No checkpoint from this protocol is a product baseline. A future policy may use
the encoder only after paired held-out validation and source-dependence checks.
