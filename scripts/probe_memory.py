"""Session 5: peak memory + throughput probe for one training config. Run from repo root.

Sets the final data.max_seq_len and projects epoch time. For each candidate total sequence length
it runs real training steps (completion_loss -> backward -> AdamW step) on synthetic ids, and records
peak memory and sec/step. OOM is recorded, not fatal. It then interpolates sec/step over the real
training-set length distribution. Results go to results/probe_<run_name>.json, rewritten after
every length so a crash keeps what finished.

    HF_HUB_OFFLINE=1 python scripts/probe_memory.py --config configs/train/qv_r8.yaml \
        --seq-lens 512 1024 1536 2048 2560
"""
import argparse
import bisect
import gc
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lora_sql.config import load_config
from lora_sql.data import build_train_example, load_examples
from lora_sql.lora import inject_lora
from lora_sql.qlora import assert_training_dtypes, completion_loss, load_model, prepare_for_training

GB = 1024**3

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 1536, 2048, 2560])
parser.add_argument("--completion-len", type=int, default=202)
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()
assert args.steps >= 2, "need >= 2 steps: step 1 is warm-up and excluded from sec/step"
seq_lens = sorted(args.seq_lens)

cfg = load_config(args.config)
torch.manual_seed(cfg.seed)
out_path = Path("results") / f"probe_{cfg.run_name}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
report = {"config": args.config, "run_name": cfg.run_name, "completion_len": args.completion_len,
          "steps_per_length": args.steps}


def write_report():
    out_path.write_text(json.dumps(report, indent=2))


def percentiles(values):
    s = sorted(values)
    pick = lambda q: s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]
    return {"p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99), "max": s[-1]}


# --- real train length distribution (CPU first: survives a GPU failure) ------------------
print("=== train length distribution ===")
tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
lengths, completions, db_ids = [], [], []
for path in cfg.data.train_paths:  # config.py guarantees only train_spider / train_others
    for ex in load_examples(str(path)):
        ids, n_completion = build_train_example(tokenizer, ex)
        lengths.append(len(ids))
        completions.append(n_completion)
        db_ids.append(ex.db_id)
report["train_examples"] = len(lengths)
report["total_len"] = percentiles(lengths)
report["completion_len_dist"] = percentiles(completions)
report["dropped_by_max_seq_len"] = {
    str(c): {
        "n_dropped": sum(1 for l in lengths if l > c),
        "db_ids": dict(Counter(d for l, d in zip(lengths, db_ids) if l > c).most_common()),
    }
    for c in seq_lens
}
print(f"examples {len(lengths)}  total_len {report['total_len']}  completion {report['completion_len_dist']}")
for c in seq_lens:
    print(f"  max_seq_len {c:>5}: drops {report['dropped_by_max_seq_len'][str(c)]['n_dropped']}")
write_report()

# --- model ----------------------------------------------------------------------------
print("\n=== model ===")
model = load_model(cfg.model)
prepare_for_training(model, cfg.model.gradient_checkpointing)
inject_lora(model, target_modules=cfg.lora.target_modules, rank=cfg.lora.rank,
            alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)
assert_training_dtypes(model, getattr(torch, cfg.model.dtype))

embed, head = model.get_input_embeddings(), model.get_output_embeddings()
lora_params = [p for p in model.parameters() if p.requires_grad]
report["model"] = {
    "embed_tokens": f"{type(embed).__name__} {embed.weight.dtype}",
    "lm_head": f"{type(head).__name__} {head.weight.dtype}",
    "memory_footprint_gb": model.get_memory_footprint() / GB,
    "allocated_after_load_gb": torch.cuda.memory_allocated() / GB,
    "lora_params": sum(p.numel() for p in lora_params),
}
for k, v in report["model"].items():
    print(f"{k:<26} {v}")
write_report()

optimizer = torch.optim.AdamW(lora_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

# --- per-length steps -------------------------------------------------------------------
print("\n=== steps ===")
report["lengths"] = []
gen = torch.Generator().manual_seed(cfg.seed)
for seq_len in seq_lens:
    n = args.completion_len
    assert seq_len > n + 1, f"seq_len {seq_len} must exceed completion_len + 1"
    # Below tokenizer.vocab_size: ordinary tokens only, no special ids.
    ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), generator=gen).cuda()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_times, oom = [], False
    try:
        for _ in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_sum, n_tokens = completion_loss(model, ids, n)
            (loss_sum / n_tokens).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:  # cleanup outside the except block, so its traceback no longer pins tensors
        loss_sum = n_tokens = None
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    row = {
        "seq_len": seq_len,
        "status": "OOM" if oom else "ok",
        "max_allocated_gb": torch.cuda.max_memory_allocated() / GB,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / GB,
        "step_times": step_times,
        "sec_per_step": None if oom else statistics.median(step_times[1:]),
    }
    report["lengths"].append(row)
    write_report()
    timing = "OOM" if oom else f"{row['sec_per_step']:.2f} s/step"
    print(f"seq_len {seq_len:>5}: alloc {row['max_allocated_gb']:.2f} GB  "
          f"reserved {row['max_reserved_gb']:.2f} GB  {timing}")

# --- epoch projection -------------------------------------------------------------------
ok = [(r["seq_len"], r["sec_per_step"]) for r in report["lengths"] if r["status"] == "ok"]
xs, ys = [x for x, _ in ok], [y for _, y in ok]


def sec_for(length):
    # Linear between measured points; clamped at the shortest measured length, so short
    # examples are charged as if they were that long (conservative).
    if length <= xs[0]:
        return ys[0]
    i = bisect.bisect_left(xs, length)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (length - x0) / (x1 - x0)


# The probe takes an AdamW step every micro-step; real training steps every grad_accum_steps,
# so this slightly overstates optimizer cost (negligible for a few M LoRA params).
report["projection"] = {}
print("\n=== projected epoch time (only lengths that ran) ===")
for c in xs:
    kept = [l for l in lengths if l <= c]
    hours = sum(sec_for(l) for l in kept) * cfg.train.epochs / 3600
    report["projection"][str(c)] = {
        "kept_examples": len(kept),
        "optimizer_steps": math.ceil(len(kept) / cfg.train.grad_accum_steps) * cfg.train.epochs,
        "epoch_hours": hours,
        "within_time_budget": hours <= cfg.train.time_budget_hours,
    }
    print(f"max_seq_len {c:>5}: {len(kept)} examples, {hours:.2f} h "
          f"(budget {cfg.train.time_budget_hours} h)")
if not ok:
    print("no length ran; nothing to project")
write_report()
print(f"\nwrote {out_path}")
