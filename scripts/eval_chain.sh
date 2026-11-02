#!/usr/bin/env bash
# Score the base model and fine-tuned runs on Spider dev, back to back. Run from anywhere:
#
#   systemd-inhibit --what=sleep:idle bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
#
# "base" evaluates Coder-7B with no adapter; any other name loads runs/<name>/final via load_lora.
# Results go to results/eval/<name>.json (+ .jsonl, written as it goes); stdout to
# runs/eval_<name>.log. A failed eval is recorded in runs/eval_chain.log and the chain moves on.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
CONFIG=configs/eval/coder7b_spider_dev.yaml

[ $# -gt 0 ] || { echo "usage: $0 base|<run_name>..." >&2; exit 2; }
for name in "$@"; do
    [ "$name" = base ] || [ -f "runs/$name/final/lora_config.json" ] \
        || { echo "no finished run at runs/$name/final" >&2; exit 2; }
done

for name in "$@"; do
    ckpt_args=()
    [ "$name" = base ] || ckpt_args=(--lora-checkpoint "runs/$name/final")
    echo "$(date -Is) start $name" | tee -a runs/eval_chain.log
    python -m lora_sql.eval.runner --config "$CONFIG" "${ckpt_args[@]}" --out "results/eval/$name.json" \
        2>&1 | tee "runs/eval_$name.log"
    echo "$(date -Is) $name exit ${PIPESTATUS[0]}" | tee -a runs/eval_chain.log
done
