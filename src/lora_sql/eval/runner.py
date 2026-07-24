"""Generate SQL from a model, execute it, and score against gold — end to end.

Usage:
    python -m lora_sql.eval.runner --config configs/week1_parity.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_sql.data import Example, format_prompt, load_examples
from lora_sql.eval.execute import execute_and_compare
from lora_sql.eval.metrics import EvalResult, compute_metrics
from lora_sql.lora import inject_lora


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_for_eval(cfg: dict, lora_checkpoint: str | None = None):
    dtype = getattr(torch, cfg["model"]["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name_or_path"], torch_dtype=dtype)
    model = inject_lora(
        model,
        target_modules=cfg["lora"]["target_modules"],
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        dropout=0.0,
    )
    if lora_checkpoint:
        state = torch.load(lora_checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_sql(model, tokenizer, example: Example, max_new_tokens: int, device: str) -> str:
    prompt = format_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_eval(cfg: dict, lora_checkpoint: str | None = None) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_for_eval(cfg, lora_checkpoint)
    model.to(device)

    examples = load_examples(cfg["data"]["eval_path"])
    db_dir = Path(cfg["data"]["db_dir"])

    results = []
    for example in examples:
        predicted_sql = generate_sql(model, tokenizer, example, cfg["eval"]["max_new_tokens"], device)
        db_path = db_dir / example.db_id / f"{example.db_id}.sqlite"
        correct = execute_and_compare(db_path, predicted_sql, example.query)
        results.append(
            EvalResult(
                db_id=example.db_id,
                question=example.question,
                predicted_sql=predicted_sql,
                gold_sql=example.query,
                correct=correct,
            )
        )

    summary = compute_metrics(results)

    output_path = Path(cfg["eval"]["results_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary.to_dict(),
                "results": [vars(r) for r in results],
            },
            f,
            indent=2,
        )

    return summary.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a run config YAML")
    parser.add_argument("--lora-checkpoint", default=None, help="Optional LoRA weights to load before eval")
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = run_eval(cfg, args.lora_checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
