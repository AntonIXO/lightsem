#!/usr/bin/env bash
# H-D, corrected: the hazard features are computed from the target's own past
# events, and a symmetric window reads their FUTURE (a 10x lift with a
# needle-sharp shift curve, 2026-09-04). Offline arms are audio-only, and the
# causal counterparts are audio-only too so the pair is like for like; the
# rhythm baseline (causal hazard ridge) stays the reference.
set -e
cd /root/lightsem
bash run_torch.sh scripts/training/train_operator_head.py --horizon-s 4 --run-tag _h4
bash run_torch.sh scripts/training/train_operator_head.py --offline --horizon-s 4 --run-tag _off_h4
bash run_torch.sh scripts/training/train_operator_head.py --offline --run-tag _off
echo HD2_DONE
