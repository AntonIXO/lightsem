#!/usr/bin/env bash
# H-D: operator head with anticipation. Same 699-show split, same features.
#   (i)  causal, horizon 4 s   -- the L=0 counterpart at the same horizon
#   (ii) offline, horizon 4 s  -- the claim
#   (iii) offline, horizon 0   -- against the causal +0.3145 at horizon 0
set -e
cd /root/lightsem
bash run_torch.sh train_operator_head.py --with-hazard --horizon-s 4 --run-tag _h4
bash run_torch.sh train_operator_head.py --with-hazard --offline --horizon-s 4 --run-tag _off_h4
bash run_torch.sh train_operator_head.py --with-hazard --offline --run-tag _off
echo HD_DONE
