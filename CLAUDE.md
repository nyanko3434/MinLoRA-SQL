# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch LoRA implementation validated numerically against HuggingFace `peft`, applied to
text-to-SQL fine-tuning on Spider with an execution-based eval harness. The package is `lora_sql`,
laid out as a `src/` package and installed editable into `.venv`.

**Status: Week 1 COMPLETE.** The core LoRA implementation and its verification are done and
committed — `src/lora_sql/lora.py` (`LoRALinear`, `inject_lora`, `lora_param_stats`),
`tests/test_lora_injection.py` (10 passing), and `tests/test_lora_parity.py` (8 passing:
forward + backward numerical parity vs `peft`, fp32, atol=1e-6). Session 4 added `src/lora_sql/data.py`
(`tests/test_data.py`, 4 passing) and a toy training loop whose loss curve is bit-identical to peft's
(max per-step loss delta = 0.000000). `pyproject.toml` is configured (setuptools backend + pytest config).
**`pytest -v` -> 22 passing** (10 injection + 8 parity + 4 data).

**Still empty placeholders**, committed on purpose so module paths and imports stay stable:
`src/lora_sql/train.py`, `configs/week1_parity.yaml`. Downstream code already imports names these
must provide — treat them as the contract to implement, not a bug:
- `configs/week1_parity.yaml` must provide keys read by `eval/runner.py`: `model.name_or_path`,
  `model.dtype`, `lora.target_modules`, `lora.rank`, `lora.alpha`, `data.eval_path`, `data.db_dir`,
  `eval.max_new_tokens`, `eval.results_path`. **`data.db_dir` must point at `database/`, never
  `test_database/`** (held-out rule). This is the only remaining blocker to running the eval harness
  end-to-end.
- `src/lora_sql/train.py` — the Week 2 custom training loop (Session 5). Reuses `data.py`'s
  `format_prompt` and ID-level completion+eos masking; config-driven so ablations are N YAML files,
  not N edited scripts.

**Already implemented:** `src/lora_sql/lora.py`, `src/lora_sql/data.py`, `src/lora_sql/eval/`
(runner.py, execute.py, metrics.py), all three test files, `pyproject.toml`, and the `scripts/`
files (`check_env.py`, `check_spider_database.py`, `toy_train.py`, `plot_toy.py`).

`inject_lora` is already built and its signature is fixed by `eval/runner.py` (which calls it by
keyword and assigns the result directly): `inject_lora(model, target_modules, rank, alpha, dropout)
-> model` (returns the model alone, not a tuple). It freezes the whole base model before wrapping.
Do not change this signature without updating `runner.py`.

## data.py contract (implemented Session 4 — do not change signatures without updating runner.py)

`eval/runner.py` imports `Example`, `format_prompt`, `load_examples` from `lora_sql.data` and
destructures `.db_id` / `.question` / `.query`. As built:
- `Example` is a dataclass with `db_id`, `question`, `query`, and `schema` (the rendered DDL).
- `format_prompt(example) -> str` — receives NO tokenizer; its output is fed straight to
  `tokenizer(prompt)`. So NO `apply_chat_template` / NO ChatML at this layer. Renders schema
  (CREATE TABLE DDL: columns+types, PRIMARY KEY, FOREIGN KEY, skipping the `-1 *` wildcard) then
  the question, ending at `SQL: `. This is the shared train+eval contract — Week 2's `train.py`
  reuses it verbatim. Training target is `format_prompt(example) + example.query + eos`, built at
  the ID level (not string concat).
- `load_examples(path, tables_path=None) -> list[Example]` — single positional arg is what
  `runner.py` calls; `tables_path` defaults to `Path(path).parent / "tables.json"`. Schema is
  cached per-`db_id` inside each call.
- `render_schema(table_entry) -> str` — public, takes a whole `tables.json` entry; golden-string
  tested byte-for-byte against the real `perpetrator` entry.
- ChatML is a Week 2 decision, reachable only in `train.py` (which owns its own tokenization),
  NOT through the `runner.py`/`format_prompt` contract.

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

# Toy LoRA training figure (ours vs peft) — run from repo root
python scripts/toy_train.py 2>&1 | tee runs/toy_train.log
python scripts/plot_toy.py   # writes results/toy_loss_curves.png, prints max per-step loss delta

# Run the eval loop (generate -> execute -> score) once configs/week1_parity.yaml is implemented
python -m lora_sql.eval.runner --config configs/week1_parity.yaml
python -m lora_sql.eval.runner --config configs/week1_parity.yaml --lora-checkpoint path/to/weights.pt

# Tests: pytest is in requirements.txt and configured in pyproject.toml
# (testpaths = ["tests"]). 22 passing as of Session 4 (10 injection + 8 parity + 4 data).
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
training runs respectively. Note: the committed toy-figure artifacts
(`results/toy_loss_curves.png`, `results/toy_loss_*.json`) are exceptions kept in git.

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

## Eval-harness hardening (latent — fix before any real Week 3 eval run)

These do not bite until a checkpoint is loaded / generation / execution actually happens, but must
be fixed before eval numbers mean anything:
- `runner.py` checkpoint load is unguarded — capture the `load_state_dict(..., strict=False)` return,
  assert `unexpected_keys == []`, and assert the checkpoint's `lora_` keys actually mapped on (a
  prefix mismatch silently loads nothing and evaluates the untrained base while reporting success).
- `execute.py` opens DBs read-write — open read-only:
  `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)`.
- `results_match` crashes on NULLs (`sorted()` → `None < number` → TypeError) — replace with a
  `Counter`-based multiset compare.
- `run_query` leaks the connection on a failed query (use try/finally); gold execution is unguarded.
- `load_model_for_eval` has no 4-bit path — Week 3 eval of the fine-tuned Coder-7B needs a `bnb`
  quantization branch to fit the 8 GB 4060.
- Wiring `eval_hardness()` is a two-place change — compute difficulty AND thread it into the
  `EvalResult(...)` construction in `runner.py` (currently omitted, so `difficulty` is `"unknown"`).

## Model roles
- `Qwen/Qwen2.5-0.5B-Instruct` — Week 1 numerical-parity fixture ONLY. Never fine-tuned.
  Chosen because its module names are identical to the 7B, so target strings transfer verbatim.
- `Qwen/Qwen2.5-Coder-7B-Instruct` — the actual QLoRA fine-tune target, Week 2 onward. Download
  verified (Session 4): Qwen2ForCausalLM, 28 layers, hidden 3584, 28 q / 4 kv heads (GQA), 4
  safetensor shards ~15.2 GB, eos `<|im_end|>`, pad `<|endoftext|>`.

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
  comparison is vacuous (`base == base`) and `lora_A.grad` is zero. (Session 3.) NOTE: this is a
  TEST-ONLY technique — real training (and the toy loop) must leave `lora_B` at zero-init.
- LoRA checkpoints: save only keys containing `lora_`, and after `load_state_dict(..., strict=False)`
  assert `unexpected_keys == []` and that the expected lora keys were consumed. `strict=False`
  silently loading nothing is a real failure mode here. This is the highest-value trap in the project.
- Qwen2.5 pad (`<|endoftext|>`) and eos (`<|im_end|>`) are DISTINCT token IDs — add
  `assert pad_id != eos_id` once batching (bs>1) starts in Week 2, so pad-masking never clobbers eos.
- Ablations are new files in `configs/`, never edited copies of `train.py`.
- GPU is an RTX 4060 Laptop, 8.2 GB VRAM. Assume memory is tight.