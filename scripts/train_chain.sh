#!/usr/bin/env bash
# Train several configs back to back. Run from anywhere:
#
#   systemd-inhibit --what=sleep:idle bash scripts/train_chain.sh qv_r4 qv_r16 attn_r8 all7_r8
#
# Runs are independent, so a failed run (e.g. OOM) is recorded in runs/chain.log and the chain
# moves on to the next config. Each run's stdout goes to runs/<name>.log.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1

[ $# -gt 0 ] || { echo "usage: $0 <run_name>..." >&2; exit 2; }
for name in "$@"; do
    [ -f "configs/train/$name.yaml" ] || { echo "no configs/train/$name.yaml" >&2; exit 2; }
done

for name in "$@"; do
    echo "$(date -Is) start $name" | tee -a runs/chain.log
    python -m lora_sql.train --config "configs/train/$name.yaml" 2>&1 | tee "runs/$name.log"
    echo "$(date -Is) $name exit ${PIPESTATUS[0]}" | tee -a runs/chain.log
done
