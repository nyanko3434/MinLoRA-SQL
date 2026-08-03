# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch LoRA implementation validated numerically against HuggingFace `peft`, applied to
text-to-SQL fine-tuning on Spider with an execution-based eval harness. The package is `lora_sql`,
laid out as a `src/` package and installed editable into `.venv`.

**Status: Week 1 nearly complete.** The core LoRA implementation and its verification are done and
committed — `src/lora_sql/lora.py` (`LoRALinear`, `inject_lora`, `lora_param_stats`),
`tests/test_lora_injection.py` (10 passing), and `tests/test_lora_parity.py` (8 passing:
forward + backward numerical parity vs `peft`, fp32, atol=1e-6). `pyproject.toml` is configured
(setuptools backend + pytest config).

**Still empty placeholders**, committed on purpose so module paths and imports stay stable:
`src/lora_sql/data.py`, `src/lora_sql/train.py`, `configs/week1_parity.yaml`. Downstream code
already imports names these must provide — treat them as the contract to implement, not a bug:
- `lora_sql.data` must provide `Example`, `format_prompt(example)`, `load_examples(path)`.
  `Example` must carry at least `db_id`, `question`, `query` (gold SQL) — `eval/runner.py`
  destructures those fields.
- `configs/week1_parity.yaml` must provide keys read by `eval/runner.py`: `model.name_or_path`,
  `model.dtype`, `lora.target_modules`, `lora.rank`, `lora.alpha`, `data.eval_path`, `data.db_dir`,
  `eval.max_new_tokens`, `eval.results_path`. **`data.db_dir` must point at `database/`, never
  `test_database/`** (held-out rule).

**Already implemented:** `src/lora_sql/lora.py`, `src/lora_sql/eval/` (runner.py, execute.py,
metrics.py), both test files, `pyproject.toml`, and the two `scripts/` files.

`inject_lora` is already built and its signature is fixed by `eval/runner.py` (which calls it by
keyword and assigns the result directly): `inject_lora(model, target_modules, rank, alpha, dropout)
-> model` (returns the model alone, not a tuple). It freezes the whole base model before wrapping.
Do not change this signature without updating `runner.py`.

## Environment & commands

Python 3.11, venv at `.venv/` (already created; dependencies pinned in `requirements.txt`).

```bash
source .venv/bin/activate
pip install -r requirements.txt   # includes `-e .` editable install of lora_sql

python -c "import lora_sql; print('ok')"   # packaging gate — run after any pyproject/packaging change

# Sanity-check the training environment (CUDA, bitsandbytes, model load, LoRA target modules)
python scripts/check_env.py

# Inspect a Spider dev example end-to-end (question, gold SQL, execution result, schema)
python scripts/check_spider_database.py

# Run the eval loop (generate -> execute -> score) once data.py/config are implemented
python -m lora_sql.eval.runner --config configs/week1_parity.yaml
python -m lora_sql.eval.runner --config configs/week1_parity.yaml --lora-checkpoint path/to/weights.pt

# Tests: pytest is in requirements.txt and configured in pyproject.toml
# (testpaths = ["tests"]). 18 passing as of Session 3 (10 injection + 8 parity).
pytest -v
pytest -v tests/test_lora_parity.py   # just the parity suite
```

There is no lint/format tooling configured in this repo yet.

## Data

`data/spider_data/` (untracked, gitignored — populated locally from `data/spider.zip`) is the Spider
text-to-SQL dataset: `train_spider.json` / `train_others.json` (train), `dev.json` (dev, 1034
examples / 20 dbs), `tables.json` (schemas for all dbs), and
`database/<db_id>/<db_id>.sqlite` (per-database SQLite files used both as few-shot schema
context and as the execution target for scoring). `results/` and `runs/` are also gitignored,
kept in git only via `.gitkeep`, and are the expected output locations for eval results and
training runs respectively.

## Eval architecture (`src/lora_sql/eval/`)

The eval pipeline is generate → execute → score, split across three modules with a clean
separation of concerns — keep new eval code in the matching module rather than adding a fourth:

- `runner.py` — orchestration only. Loads a YAML config, loads the base model + tokenizer, calls
  `inject_lora` (optionally loading a LoRA checkpoint via `load_state_dict(..., strict=False)`),
  generates SQL greedily (`do_sample=False`) per example, and writes a JSON report
  (`{"summary": ..., "results": [...]}`) to `cfg["eval"]["results_path"]`.
- `execute.py` — SQLite execution and comparison, no model/tokenizer code. `run_query` executes
  against the per-`db_id` SQLite file and converts any `sqlite3.Error` into `ExecutionError`, so a
  broken generation is a scoring signal rather than a crash. `results_match` does an
  order-insensitive comparison of result sets with float rounding, matching Spider's execution
  accuracy convention. `execute_and_compare` treats a predicted query that fails to execute as
  simply "wrong," not an error.
- `metrics.py` — pure aggregation over `EvalResult` dataclasses into a `MetricsSummary`, with an
  overall accuracy plus a breakdown by `difficulty` (defaults to `"unknown"` if not set upstream).

When implementing `data.py`, keep the same shape: `Example` should carry at least `db_id`,
`question`, `query` (gold SQL) since `runner.py` already destructures those fields. `inject_lora`
already returns a model compatible with the standard HF `generate()` call.

## Model roles
- `Qwen/Qwen2.5-0.5B-Instruct` — Week 1 numerical-parity fixture ONLY. Never fine-tuned.
  Chosen because its module names are identical to the 7B, so target strings transfer verbatim.
- `Qwen/Qwen2.5-Coder-7B-Instruct` — the actual QLoRA fine-tune target, Week 2 onward.

## Hard rules
- `data/spider_data/test.json` and `test_database/` are HELD OUT. Never read them, never
  evaluate against them, never tune on them. `database/` is the only execution target.
- Never use `base_layer.weight.shape` to get layer dimensions. Under bitsandbytes `Linear4bit`
  the weight is packed and its shape is wrong. Read `.in_features` / `.out_features`.
- transformers is pinned to 5.14.1 (v5). Use `dtype=`, never `torch_dtype=`.
  `apply_chat_template` returns a BatchEncoding: `.to(device)` + `model.generate(**inputs)`.
- Parity tests run in fp32. bf16 carries ~3 significant decimal digits and cannot reach
  atol=1e-6. Training runs in bf16.
- peft `scaling` is a dict `{"default": alpha/r}` — index `["default"]`, never assume it's a
  scalar. (Session 3.)
- Parity / weight-transplant tests must randomize peft's `lora_B` before copying, or the
  comparison is vacuous (`base == base`) and `lora_A.grad` is zero. (Session 3.)
- LoRA checkpoints: save only keys containing `lora_`, and after `load_state_dict(..., strict=False)`
  assert `unexpected_keys == []` and that the expected lora keys were consumed. `strict=False`
  silently loading nothing is a real failure mode here.
- Ablations are new files in `configs/`, never edited copies of `train.py`.
- GPU is an RTX 4060 Laptop, 8.2 GB VRAM. Assume memory is tight.