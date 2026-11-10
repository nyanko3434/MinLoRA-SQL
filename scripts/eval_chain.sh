#!/usr/bin/env bash
# Score the base model and fine-tuned runs back to back. Run from anywhere:
#
#   systemd-inhibit --what=sleep:idle bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
#   systemd-inhibit --what=sleep:idle bash scripts/eval_chain.sh \
#       --config configs/eval/coder7b_bird_dev.yaml --out-dir results/eval/bird base qv_r8 qv_r16 all7_r8
#
# Defaults: Spider dev (configs/eval/coder7b_spider_dev.yaml) into results/eval/.
# "base" evaluates Coder-7B with no adapter; any other name loads runs/<name>/final via load_lora.
# Results go to <out-dir>/<name>.json (+ .jsonl, written as it goes); stdout to
# runs/eval_<tag><name>.log. A failed eval is recorded in runs/eval_chain.log and the chain moves on.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
CONFIG=configs/eval/coder7b_spider_dev.yaml
OUT_DIR=results/eval

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) break ;;
    esac
done
[ -f "$CONFIG" ] || { echo "no config at $CONFIG" >&2; exit 2; }
[ $# -gt 0 ] || { echo "usage: $0 [--config FILE] [--out-dir DIR] base|<run_name>..." >&2; exit 2; }
for name in "$@"; do
    [ "$name" = base ] || [ -f "runs/$name/final/lora_config.json" ] \
        || { echo "no finished run at runs/$name/final" >&2; exit 2; }
done
# Log names stay as before for Spider; other out-dirs get a prefix so logs don't collide.
TAG=""
[ "$OUT_DIR" = results/eval ] || TAG="$(basename "$OUT_DIR")_"

for name in "$@"; do
    ckpt_args=()
    [ "$name" = base ] || ckpt_args=(--lora-checkpoint "runs/$name/final")
    echo "$(date -Is) start $TAG$name ($CONFIG)" | tee -a runs/eval_chain.log
    python -m lora_sql.eval.runner --config "$CONFIG" "${ckpt_args[@]}" --out "$OUT_DIR/$name.json" \
        2>&1 | tee "runs/eval_$TAG$name.log"
    echo "$(date -Is) $TAG$name exit ${PIPESTATUS[0]}" | tee -a runs/eval_chain.log
done
