"""Generate SQL from a model, execute it, and score against gold — end to end.

Usage:
    python -m lora_sql.eval.runner --config configs/eval/coder7b_spider_dev.yaml \
        [--lora-checkpoint runs/qv_r8/final] [--out results/eval/qv_r8.json] [--limit N]

Without --lora-checkpoint the plain base model is evaluated (no adapters injected). With it, the
checkpoint dir is loaded through checkpoint.load_lora, which injects from the sidecar and asserts
every key mapped on. --limit is for smoke tests only.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from lora_sql.checkpoint import SIDECAR_FILE, git_commit, load_lora
from lora_sql.config import ModelConfig
from lora_sql.data import Example, encode_prompt, load_bird_examples, load_examples
from lora_sql.eval.execute import DEFAULT_TIMEOUT_S, ExecutionError, execute_and_compare, order_matters, run_query
from lora_sql.eval.metrics import EvalResult, compute_metrics, eval_difficulties
from lora_sql.qlora import load_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# The only execution targets per eval format: Spider's database/ (never test_database/) and BIRD's
# dev_databases/. BIRD is zero-shot only and has no public test set.
DB_DIR_BY_FORMAT = {"spider": "database", "bird": "dev_databases"}


def eval_format(cfg: dict) -> str:
    fmt = cfg["data"].get("format", "spider")
    if fmt not in DB_DIR_BY_FORMAT:
        raise ValueError(f"data.format must be one of {sorted(DB_DIR_BY_FORMAT)}, got {fmt!r}")
    return fmt


def check_held_out(cfg: dict) -> None:
    """The held-out rule, enforced in code: Spider test files and test_database/ are never an
    eval target. Spider runs only against database/, BIRD only against dev_databases/."""
    fmt = eval_format(cfg)
    eval_path = Path(cfg["data"]["eval_path"])
    if eval_path.name.startswith("test"):
        raise ValueError(f"data.eval_path {eval_path} is held out; evaluate on dev only")
    db_dir = Path(cfg["data"]["db_dir"])
    if db_dir.name != DB_DIR_BY_FORMAT[fmt]:
        raise ValueError(f"data.db_dir for format {fmt!r} must be a {DB_DIR_BY_FORMAT[fmt]}/ directory, got {db_dir} "
                         "(test_database/ is held out)")


def load_eval_examples(cfg: dict) -> list[Example]:
    d = cfg["data"]
    if eval_format(cfg) == "bird":
        return load_bird_examples(d["eval_path"], d["tables_path"], d.get("use_evidence", True))
    return load_examples(d["eval_path"])


def _db_path(db_dir: Path, db_id: str) -> Path:
    return db_dir / db_id / f"{db_id}.sqlite"


def gold_preflight(examples: list[Example], db_dir: Path, timeout: float = DEFAULT_TIMEOUT_S) -> list[list[tuple]]:
    """Execute every gold query before any model is loaded, so a broken gold query or a wrong
    db_dir fails in seconds rather than partway through generation. Returns the gold rows."""
    gold_rows, failures = [], []
    for ex in examples:
        try:
            gold_rows.append(run_query(_db_path(db_dir, ex.db_id), ex.query, timeout))
        except ExecutionError as e:
            gold_rows.append(None)
            failures.append(f"[{ex.db_id}] {ex.question!r}: {e}")
    if failures:
        raise RuntimeError(f"{len(failures)} gold queries failed to execute, e.g.:\n" + "\n".join(failures[:5]))
    return gold_rows


def load_model_for_eval(cfg: dict, lora_checkpoint: str | None = None):
    m = cfg["model"]
    if m["quant"] == "nf4" and m["dtype"] != "bfloat16":
        raise ValueError(f"model.quant=nf4 computes in bfloat16; model.dtype must be bfloat16, got {m['dtype']!r}")
    tokenizer = AutoTokenizer.from_pretrained(m["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Same loader as training: nf4 4-bit with bf16 compute, or plain from_pretrained in m["dtype"].
    model = load_model(ModelConfig(name_or_path=m["name_or_path"], quant=m["quant"],
                                   double_quant=m["double_quant"], dtype=m["dtype"],
                                   gradient_checkpointing=False))
    if lora_checkpoint:
        load_lora(model, lora_checkpoint)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_sql(model, tokenizer, example: Example, max_new_tokens: int, device: str) -> str:
    input_ids = torch.tensor([encode_prompt(tokenizer, example)], device=device)
    attention_mask = torch.ones_like(input_ids)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_eval(cfg: dict, lora_checkpoint: str | None = None, out: str | None = None,
             limit: int | None = None) -> dict:
    check_held_out(cfg)
    examples = load_eval_examples(cfg)
    # Same file, same order as the loader, so the two lists join by index.
    difficulties = eval_difficulties(cfg["data"]["eval_path"])
    assert len(difficulties) == len(examples), "difficulty list does not line up with the examples"
    if limit is not None:
        examples, difficulties = examples[:limit], difficulties[:limit]
    db_dir = Path(cfg["data"]["db_dir"])

    t0 = time.monotonic()
    # Predictions always get DEFAULT_TIMEOUT_S. A longer gold deadline only lets a slow gold query
    # finish; a prediction that runs as slowly still times out and scores wrong.
    gold_rows = gold_preflight(examples, db_dir, cfg["data"].get("gold_timeout_s", DEFAULT_TIMEOUT_S))
    print(f"gold preflight: {len(examples)} gold queries executed ok", flush=True)

    model, tokenizer = load_model_for_eval(cfg, lora_checkpoint)

    output_path = Path(out or cfg["eval"]["results_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results, n_exec_errors = [], 0
    # One line per example as it goes: a crash late in a long eval keeps what finished.
    with open(output_path.with_suffix(".jsonl"), "w", encoding="utf-8") as partial:
        for i, (example, gold, difficulty) in enumerate(zip(examples, gold_rows, difficulties), start=1):
            predicted_sql = generate_sql(model, tokenizer, example, cfg["eval"]["max_new_tokens"], model.device)
            correct, exec_error = execute_and_compare(
                _db_path(db_dir, example.db_id), predicted_sql, gold, order_matters(example.query)
            )
            n_exec_errors += exec_error is not None
            result = EvalResult(
                db_id=example.db_id,
                question=example.question,
                predicted_sql=predicted_sql,
                gold_sql=example.query,
                correct=correct,
                difficulty=difficulty,
            )
            results.append(result)
            partial.write(json.dumps({**vars(result), "exec_error": exec_error}) + "\n")
            partial.flush()
            if i % 50 == 0 or i == len(examples):
                acc = sum(r.correct for r in results) / i
                elapsed = time.monotonic() - t0
                print(f"{i}/{len(examples)}  EX {acc:.3f}  exec errors {n_exec_errors}  "
                      f"{elapsed / 60:.1f} min  eta {elapsed / i * (len(examples) - i) / 60:.1f} min", flush=True)

    summary = {**compute_metrics(results).to_dict(), "n_pred_exec_errors": n_exec_errors}
    sidecar = None
    if lora_checkpoint:
        sidecar = json.loads((Path(lora_checkpoint) / SIDECAR_FILE).read_text())
    meta = {
        "model": cfg["model"],
        "lora_checkpoint": lora_checkpoint,
        "checkpoint_sidecar": sidecar,
        "eval_path": cfg["data"]["eval_path"],
        "data": cfg["data"],  # format, db_dir, use_evidence, gold_timeout_s, ...
        "limit": limit,
        "max_new_tokens": cfg["eval"]["max_new_tokens"],
        "eval_git_commit": git_commit(),
        "elapsed_s": time.monotonic() - t0,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "meta": meta, "results": [vars(r) for r in results]}, f, indent=2)
    print(f"wrote {output_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an eval config YAML")
    parser.add_argument("--lora-checkpoint", default=None,
                        help="LoRA checkpoint dir (adapter.safetensors + lora_config.json); omit for the base model")
    parser.add_argument("--out", default=None, help="Results JSON path (overrides eval.results_path)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples (smoke tests)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = run_eval(cfg, args.lora_checkpoint, out=args.out, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
