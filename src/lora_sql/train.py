"""Config-driven QLoRA training loop. Run from repo root:

    python -m lora_sql.train --config configs/train/qv_r8.yaml [--limit-examples N] [--overwrite]

Micro-batch 1, token-weighted gradient accumulation, completion-only logits, fp32 adapters on a
base left in its load dtype, no torch.autocast. There is no resume: an OOM preflight on the
longest example fails the run in minute 1, and adapters are saved every save_every steps.
--limit-examples is for smoke tests only; it truncates after the shuffle and the length filter.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import random
import shutil
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from lora_sql.checkpoint import save_lora
from lora_sql.config import RunConfig, load_config
from lora_sql.data import build_train_example, load_examples
from lora_sql.lora import inject_lora, lora_param_stats
from lora_sql.qlora import assert_training_dtypes, completion_loss, load_model, prepare_for_training

GB = 1024**3


def config_sha256(cfg: RunConfig) -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()).hexdigest()


def warmup_steps(warmup_ratio: float, total: int) -> int:
    return max(1, round(warmup_ratio * total))


def lr_lambda(step: int, warm: int, total: int) -> float:
    """Linear warmup over `warm` steps, then cosine to 0 at `total`. The clamp and max(1, ...)
    only matter at step == total (LambdaLR evaluates it after the last optimizer step), where
    total == warm would otherwise divide by zero."""
    if step < warm:
        return (step + 1) / warm
    d = max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(step - warm, d) / d))


def compute_total_steps(n_kept: int, grad_accum_steps: int, epochs: int, max_steps: int | None) -> int:
    """FLOOR: the trailing partial window is dropped, so every step sees grad_accum_steps examples."""
    if max_steps is not None:
        return max_steps
    return (n_kept // grad_accum_steps) * epochs


def accumulate_window(model, examples: list[tuple[list[int], int]]) -> tuple[float, int]:
    """Backprop one optimizer step's worth of examples. Each summed loss is divided by the
    window's TOTAL completion tokens, so the accumulated gradient equals that of one big batch
    (not a mean of per-example means). Returns (summed loss, completion tokens)."""
    total = sum(n for _, n in examples)
    loss_sum, n_tokens = torch.zeros((), device=model.device), 0
    for ids, n in examples:
        example_loss, k = completion_loss(model, torch.tensor([ids], device=model.device), n)
        (example_loss / total).backward()
        loss_sum += example_loss.detach()
        n_tokens += k
    return loss_sum.item(), n_tokens


def _order_sha256(examples: list[tuple[list[int], int]]) -> str:
    """Covers order AND content: a tokenizer or prompt change moves this hash too."""
    h = hashlib.sha256()
    for ids, n in examples:  # length-prefixed, so row boundaries are unambiguous
        h.update(np.asarray([len(ids), n, *ids], dtype=np.int32).tobytes())
    return h.hexdigest()


def _to_plain(cfg: RunConfig) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(cfg)))  # tuples -> lists, for yaml.safe_dump


def main(cfg: RunConfig, limit_examples: int | None = None, overwrite: bool = False):
    out_dir = Path(cfg.train.out_dir)
    if (out_dir / "final").exists() and not overwrite:
        raise FileExistsError(f"{out_dir / 'final'} exists (a finished run); pass --overwrite to replace it")
    if overwrite:  # only our own checkpoint dirs, never the whole out_dir
        for old in [out_dir / "final", *out_dir.glob("step_*")]:
            if old.is_dir():
                shutil.rmtree(old)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(_to_plain(cfg), sort_keys=False))
    cfg_sha = config_sha256(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # --- data ---------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    rows = []
    for path in cfg.data.train_paths:  # config.py guarantees only train_spider / train_others
        for ex in load_examples(str(path)):
            ids, n = build_train_example(tokenizer, ex)
            rows.append((ids, n, ex.db_id))
    too_long = Counter(db for ids, _, db in rows if len(ids) > cfg.data.max_seq_len)
    kept = [(ids, n) for ids, n, _ in rows if len(ids) <= cfg.data.max_seq_len]

    order = list(range(len(kept)))
    random.Random(cfg.seed).shuffle(order)
    if limit_examples is not None:
        order = order[:limit_examples]
    examples = [kept[i] for i in order]
    order_sha = _order_sha256(examples)

    ga = cfg.train.grad_accum_steps
    n_windows = len(examples) // ga
    if n_windows == 0:
        raise ValueError(f"{len(examples)} examples < grad_accum_steps={ga}: not one full step")
    total = compute_total_steps(len(examples), ga, cfg.train.epochs, cfg.train.max_steps)

    report = {
        "n_loaded": len(rows),
        "max_seq_len": cfg.data.max_seq_len,
        "n_dropped": sum(too_long.values()),
        "dropped_by_db_id": dict(too_long.most_common()),
        "n_kept": len(kept),
        "limit_examples": limit_examples,
        "n_used": len(examples),
        "n_discarded_tail_per_epoch": len(examples) % ga,
        "total_steps": total,
        "data_order_sha256": order_sha,
    }
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"data: loaded {report['n_loaded']}, dropped {report['n_dropped']} > {cfg.data.max_seq_len} "
          f"{report['dropped_by_db_id']}, kept {len(kept)}, used {len(examples)}", flush=True)
    print(f"steps: {total} (grad_accum {ga}, epochs {cfg.train.epochs}, max_steps {cfg.train.max_steps}); "
          f"trailing partial window discards {len(examples) % ga} per epoch; order sha256 {order_sha}",
          flush=True)

    assert cfg.train.micro_batch_size == 1, "completion_loss assumes micro-batch 1 (no padding)"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        f"pad ({tokenizer.pad_token_id}) == eos ({tokenizer.eos_token_id}): pad-masking would clobber eos"
    )

    # --- model --------------------------------------------------------------------------
    model = load_model(cfg.model)
    prepare_for_training(model, cfg.model.gradient_checkpointing)
    inject_lora(model, target_modules=cfg.lora.target_modules, rank=cfg.lora.rank,
                alpha=cfg.lora.alpha, dropout=cfg.lora.dropout, adapter_dtype=torch.float32)
    assert_training_dtypes(model, getattr(torch, cfg.model.dtype))
    print(f"lora: {lora_param_stats(model)}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    warm = warmup_steps(cfg.train.warmup_ratio, total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(lr_lambda, warm=warm, total=total))

    # --- OOM preflight: forward + backward on the longest example, no step ---------------
    longest_ids, longest_n = max(examples, key=lambda r: len(r[0]))
    loss_sum, n_tokens = completion_loss(model, torch.tensor([longest_ids], device=model.device), longest_n)
    (loss_sum / n_tokens).backward()
    del loss_sum
    optimizer.zero_grad(set_to_none=True)
    print(f"preflight: len {len(longest_ids)} ok, peak reserved "
          f"{torch.cuda.max_memory_reserved() / GB:.2f} GB", flush=True)

    # --- train --------------------------------------------------------------------------
    meta = {"config_sha256": cfg_sha, "data_order_sha256": order_sha}
    budget_s = cfg.train.time_budget_hours * 3600
    t0 = time.monotonic()
    with open(out_dir / "log.jsonl", "w", encoding="utf-8") as log_f:  # no resume: always a fresh log
        def write_log(record):
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

        for step in range(1, total + 1):
            w = (step - 1) % n_windows  # epochs > 1 (or max_steps past one epoch) reuse the order
            loss_sum, n_tokens = accumulate_window(model, examples[w * ga:(w + 1) * ga])
            lr = optimizer.param_groups[0]["lr"]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.max_grad_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            elapsed = time.monotonic() - t0
            stop = step < total and elapsed > budget_s
            if step == 1 or step % cfg.train.log_every == 0 or step == total or stop:
                record = {
                    "step": step,
                    "loss": loss_sum / n_tokens,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens": n_tokens,
                    "elapsed_s": elapsed,
                    "peak_alloc_gb": torch.cuda.max_memory_allocated() / GB,
                    "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB,
                }
                write_log(record)
                print(f"step {step}/{total}  loss {record['loss']:.4f}  lr {lr:.2e}  "
                      f"gnorm {grad_norm:.3f}  {elapsed / 60:.1f} min  "
                      f"peak {record['peak_reserved_gb']:.2f} GB", flush=True)
            if step % cfg.train.save_every == 0:
                save_lora(model, out_dir / f"step_{step}", {"step": step, **meta})
            if stop:
                save_lora(model, out_dir / "final", {"step": step, **meta})
                write_log({"event": "stopped_by_budget", "step": step})
                print(f"time budget {cfg.train.time_budget_hours} h hit at step {step}; saved final/",
                      flush=True)
                return model

        save_lora(model, out_dir / "final", {"step": total, **meta})
    print(f"done: {out_dir / 'final'}", flush=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-examples", type=int, default=None, help="smoke tests only")
    parser.add_argument("--overwrite", action="store_true", help="replace a finished run's checkpoints")
    args = parser.parse_args()
    main(load_config(args.config), limit_examples=args.limit_examples, overwrite=args.overwrite)
