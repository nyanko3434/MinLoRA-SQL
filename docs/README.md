> **CURRENT STATE** (overwrite at the end of every session)
> Repo: https://github.com/nyanko3434/MinLoRA-SQL.git
> Last commit: see `git log` (dev) | pytest: 85 passing
> Status: Session 6 COMPLETE. All five configs trained (536 steps each) and scored on Spider dev
> with the hardened harness. Base EX 74.95%; best qv_r16 77.47% / all7_r8 77.56% (+2.5 pts,
> nominal p≈0.035, not significant after Holm correction). See `results/eval/summary.md`.
> Next task: decide what Week 4 covers (difficulty breakdown, BIRD-dev, seed repeats, write-up).
> Not in git: data/ (Spider), Coder-7B weights (~15 GB), .venv/, runs/

# MinLoRA-SQL

**A from-scratch LoRA implementation, numerically verified against HuggingFace `peft`, applied to
text-to-SQL fine-tuning with an execution-based evaluation harness.**
 
## Context (this file is the Project knowledge base — any new chat starts here)
 
I'm a Computer Engineering student (B.E., graduating 2027) building a resume project.
Background relevant here: completed a 3-month AI/ML internship at Fusemachines building
a production RAG chatbot (Qdrant, AWS EC2, Knowledge Ingestion API), so I already have
hands-on RAG/vector-DB experience — this new project is deliberately going deeper into
**fine-tuning** instead, so it doesn't overlap with that.
 
**Timeframe:** 2-4 weeks, at **1-2 focused hours per day** (~40 hours total). This budget
is the binding constraint on the whole plan — see "Scope cuts" below.
 
**Goal / resume angle:** "Deep/research-y," not just "I called an API and fine-tuned a model."
The differentiator is doing two things most student projects skip:
 
1. Implementing LoRA myself (not just using HF's `peft`) and **numerically verifying** it
   against `peft` — forward and backward agreement to 1e-6, not just "loss curves look similar."
   **✅ DONE as of Session 3 — this claim is now proven, not aspirational.** Session 4 adds the
   corroborating training-curve figure (200 steps, ours vs peft, max per-step loss δ = 0.0).
2. Building a rigorous, objective evaluation harness (execution-based accuracy), instead of
   eyeballing outputs or using an LLM-judge.
 
The name undersells it slightly: the **numerical parity claim is the actual differentiator**,
not the fact that the implementation is small. Any framing (resume bullet, README intro,
writeup abstract) should lead with parity, not with "minimal."
 
**Domain: SQL generation (text-to-SQL).** Chosen over code review/bug-fixing because
SQL eval is free and objective — run the generated query against a real DB and diff the
result set — whereas rigorous bug-fix eval (running original test suites per repo, SWE-bench
style) would eat most of the budget on infra instead of the fine-tuning/LoRA work
that's actually the point of the project.
 
## Hardware / environment (verified working)
 
| Item | Value |
|---|---|
| GPU | RTX 4060 Laptop, 8.2 GB VRAM |
| Compute capability | 8.9 (Ada Lovelace) — native bf16 |
| OS | Arch Linux (native, no WSL) |
| Python | 3.11, venv at `.venv/` |
| torch | 2.5.1+cu121 |
| transformers | **5.14.1 — pinned, decision closed (see below)** |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 — verified working, `Linear4bit` forward passes in bf16 |
| pytest | 9.1.1 |
 
Repo path: `~/Documents/Projects/SQL_Fine-Tune`.
 
Cloud burst to RunPod/Vast.ai (a few $/hr for an A100) remains available for heavier runs
if the 4060 proves too slow for Week 2.
 
## How I'm using Claude Pro for this
 
- **This Project (browser/app)** — planning, decisions, conceptual discussion, spec writing,
  reviewing results, writing up the final report. The knowledge base holds this README, so any
  new chat inside the Project already has full context — no need to re-paste it.
- **Claude Code** — actual implementation, run against the local repo.
- **Working pattern that has worked well so far:** decisions and a **detailed spec** get written
  in the Project chat, then pasted into Claude Code **in plan mode (Shift+Tab)**. Review the plan
  before approving — the plan is where design gaps surface cheaply. Split large specs into halves
  (module first, tests second) and commit between them, so a bad generation is one
  `git checkout -- <path>` away from gone. I don't want full code pasted into chat; specs +
  checklists are the right granularity.
- **`CLAUDE.md` lives in the repo root** and is Claude Code's equivalent of this README. It is
  loaded at the start of every Claude Code session. Durable constraints go there; see
  "CLAUDE.md hard rules" below for what it currently contains.
- **Rhythm:** Pro usage is shared across chat and Claude Code, so batch Claude Code into focused
  sessions rather than leaving it open idle.
- Whenever a chat produces a decision or result worth keeping, it gets folded back into
  this README and re-uploaded to the Project's knowledge base, since chats inside a
  Project don't share history with each other automatically.
- **Session rule (from the 1-2 hr/day constraint):** every session should *end* by launching
  something long-running and walking away. The GPU (or the network) should be working while I'm not.
 
---
 
# PROGRESS
 
## Session 1 — COMPLETE (environment + data)
 
**Environment:** venv created, all packages installed, `requirements.txt` frozen, repo
scaffolded and committed. `scripts/check_env.py` passes end-to-end.
 
**bitsandbytes verified early on purpose** — it's the single most likely thing to break
Week 2, and it breaks at install level rather than usage level. Confirmed working:
`Linear4bit(64, 64, compute_dtype=torch.bfloat16)` forward pass returns bf16.
 
**Spider downloaded and verified.** Google Drive file ID `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`
(linked from yale-lily.github.io/spider), ~197 MB zip, extracts to `data/spider_data/`:
 
- `train_spider.json` (7,000 examples), `train_others.json`, `dev.json` (**1,034** — matches
  the published count), `tables.json`, `dev_gold.sql`, `train_gold.sql`
- `database/` — **166** `.sqlite` files (train + dev pool)
- `test_database/` — **206** `.sqlite` files, plus `test.json` / `test_tables.json` / `test_gold.sql`
- `__MACOSX/` and `.DS_Store` junk — stripped
 
Verified the databases are actually *populated*, not empty schema shells: ran `dev.json[0]`
gold query (`SELECT count(*) FROM singer` on `concert_singer`) → returned `[(6,)]`.
 
**Qwen2.5-0.5B-Instruct loads and generates.** Architecture confirmed:
 
| Property | Value |
|---|---|
| hidden size | 896 |
| layers | 24 |
| attention heads | 14 query / 2 KV (GQA), head_dim 64 |
| `q_proj`, `o_proj` | 896 → 896 |
| `k_proj`, `v_proj` | 896 → 128 (GQA asymmetry) |
| `gate_proj`, `up_proj` | 896 → 4864 |
| `down_proj` | 4864 → 896 |
 
**Also built in Session 1 but not recorded at the time:** the entire `src/lora_sql/eval/`
package (`runner.py`, `execute.py`, `metrics.py`) is *already implemented*. This was discovered
when Claude Code's `/init` summarised the repo. See "Week 3 status" below — this materially
loosens the remaining budget.
 
## Session 2 — COMPLETE (`LoRALinear` + injection + tests)
 
`src/lora_sql/lora.py` and `tests/test_lora_injection.py` are implemented and committed.
**`pytest -v` → 10 passed.**
 
**Packaging blockers found and fixed first** (both would have failed the test imports):
 
- `pyproject.toml` was empty — no build backend, so `pip install -e .` could not have worked.
  Now has `[build-system]` (setuptools), `[project]`, `[tool.setuptools.packages.find] where = ["src"]`,
  and `[tool.pytest.ini_options] testpaths = ["tests"]`.
- `pytest` was missing from `requirements.txt`.
- Gate to re-run after any packaging change: `python -c "import lora_sql; print('ok')"`.
 
**Also fixed:** `eval/runner.py` was calling `from_pretrained(..., torch_dtype=dtype)`, the v4
keyword. Changed to `dtype=`.
 
### The `inject_lora` contract, as built
 
```python
inject_lora(model, target_modules=("q_proj","v_proj"), rank=8, alpha=16, dropout=0.0) -> model
```
 
This contract is **fixed by `eval/runner.py`**, which already calls it by keyword and assigns
the result directly (`model = inject_lora(...)`). Parameter names are exact; the return is the
**model alone, not a tuple**. Wrap counts are exposed separately via `lora_param_stats(model)`,
which returns `{"total", "trainable", "lora", "n_lora_layers"}`.
 
**`inject_lora` freezes the entire base model** as its first action, before wrapping — matching
peft's `get_peft_model` behaviour. The caller does not need to freeze anything. Ordering matters:
`lora_A` / `lora_B` are created with `requires_grad=True` during wrapping, so freezing after
wrapping would freeze the adapters too and train nothing. (This gap was caught by the freeze test,
not by inspection — the test paid for itself immediately.)
 
### Design decisions inside `lora.py`
 
- **Shape convention matches peft exactly.** `lora_A` is `(r, in_features)`, `lora_B` is
  `(out_features, r)` — identical to peft's `nn.Linear(in_features, r).weight`. Session 3's weight
  transplant is therefore a bare `.copy_()` with no transpose, and `kaiming_uniform_(a=sqrt(5))`
  computes the same fan_in peft does, so even the init distributions match. **Do not "simplify"
  these shapes.** *(Confirmed correct in Session 3 — see below.)*
- **Dimensions read from `.in_features` / `.out_features`, never `.weight.shape`** — under
  bitsandbytes 4-bit the weight is packed and its shape lies.
- **Adapters held in `adapter_dtype` (default fp32), independent of the base dtype** *(changed
  Session 5; originally resolved from the base's `compute_dtype`)* — matches peft's
  `autocast_adapter_dtype=True`. Forward mirrors peft's bnb `Linear4bit` cast order: `x` cast to the
  adapter dtype *before* dropout, `scaling` applied in fp32 *before* casting down to the base output
  dtype. On the Week 1 fp32 fixture every cast is a no-op, so parity is unaffected.
- **Forward keeps explicit parentheses:** `(h @ lora_A.T) @ lora_B.T`. The other grouping
  materialises an `out_features × in_features` matrix — exactly what LoRA exists to avoid.
- **Exact name matching in injection**, not substring. `"up_proj" in "gate_up_proj"` is `True`,
  which would silently wrap the wrong layer on any architecture with fused projections. Qwen2.5
  has no fused names, but exact matching converts a future silent wrong-wrap into a loud zero-wrap.
- **Two-pass injection:** collect `(parent, child_name, child)` triples first, then `setattr`.
  Mutating the module tree while iterating it is fragile.
- **Zero-wrap raises `ValueError`** listing both the requested targets and a sample of the
  `nn.Linear` child names actually present — this failure is always a typo, so the error
  diagnoses itself.
 
### The test suite (`tests/test_lora_injection.py`, 10 cases)
 
Module-scoped fixture loads Qwen2.5-0.5B-Instruct in **fp32 on CUDA** and stays pristine;
each test that injects works on a `copy.deepcopy`, since injection is in-place. CUDA absence
is an `assert`, not a `pytest.skip` — on a single-GPU dev machine a missing GPU is a broken
environment, and silent skips are the failure mode this suite exists to prevent.
 
1. **Wrap count** — exact, parametrized over three target sets. Catches over-wrapping as well as under-wrapping.
2. **Param count** — exact, same parametrization. Matches the hand-computed table below.
3. **Bit-identical at init** — base logits captured from the pristine fixture *before* any
   injection, then `torch.equal` (not `allclose`). With `B = 0` the update is exactly `0.0`,
   so this must hold bitwise.
4. **Freeze check** — `trainable == lora > 0`, plus a name-level assertion that every
   `requires_grad` parameter contains `lora_`, reporting *which* one leaked on failure.
5. **Device/dtype placement** — per-`LoRALinear`, and asserts at least one was checked.
6. **Scaling** — `alpha/r == 2.0` for r=8/alpha=16. No GPU. Catches an alpha/r inversion that
   would otherwise surface as a mystifying Session 3 failure.
 
### Precomputed parameter counts for Qwen2.5-0.5B (r=8) — verified by tests 1 & 2
 
Formula: `r × (in_features + out_features)` per wrapped layer, × 24 layers.
 
| Target set | Wrapped layers | Per layer | Total |
|---|---|---|---|
| `q_proj`, `v_proj` | 48 | 22,528 | **540,672** |
| attention (`q`,`k`,`v`,`o`) | 96 | 45,056 | **1,081,344** |
| all seven | 168 | 183,296 | **4,399,104** |
 
Scales linearly for r=4 and r=16.
 
**Notable:** the three MLP projections account for 138,240 of the 183,296 per-layer total —
**75% of LoRA parameters live in the MLP**, because 4,864 is a much fatter dimension than 896.
This motivates the target-set ablation.
 
## Session 3 — COMPLETE (the `peft` parity test — the actual resume claim)
 
`tests/test_lora_parity.py` implemented and committed. **`pytest -v` → 18 passed** (10 injection
+ 8 parity). This is the correctness proof the whole project is pitched on.
 
**What the test does.** Builds a matched peft model (`LoraConfig` with the same r, alpha,
target modules, `lora_dropout=0.0`, `use_rslora=False`), randomizes peft's `lora_B`, transplants
**both** adapter matrices into our `LoRALinear` with a bare `.copy_()` (no transpose), then asserts:
 
- **Forward:** `torch.allclose(atol=1e-6)` on logits, in fp32, on a shared batch.
- **Backward:** one shared seeded target, `.mean()` MSE loss on both models, then
  `torch.allclose(atol=1e-6)` on **both** `lora_A.grad` and `lora_B.grad`, per paired layer.
 
Parametrized over **r ∈ {4, 8, 16}** on q/v, plus an **all-seven** spot check (confirms the GQA
k/v `out_features` asymmetry and the fat MLP dims all transplant correctly). Empirically the
agreement is *bit-identical* in fp32 on a toy verification (maxdiff `0.00e+00`), so `atol=1e-6`
passes with room to spare — the parity is proven, not merely "close."
 
### Two facts pinned this session (verified against real peft output, not from memory)
 
- **peft's `scaling` is a dict**, `{"default": alpha/r}`, not a scalar. Index `["default"]` if it
  ever needs reading. We recompute `alpha/r` ourselves so we don't depend on it — but don't assume
  it's scalar.
- **Both implementations init `lora_B = 0`.** A naive parity comparison at init is therefore
  *vacuously* true (`base == base`) and proves nothing. peft's `lora_B` **must be randomized
  before transplant**. This also matters for backward: with `B = 0`, `dL/dA = 0`, so the
  `lora_A.grad` assertion would be vacuous too.
- **Shape convention confirmed correct.** peft's `lora_A["default"].weight` is `(r, in_features)`
  and `lora_B["default"].weight` is `(out_features, r)` — identical to `LoRALinear`. Transplant is
  a bare `.copy_()`, no transpose, exactly as designed in Session 2.
 
### peft attribute map (recorded for future reference)
 
- peft LoRA layers are `peft.tuners.lora.Linear`, found via `named_modules()`.
- Named `base_model.model.<original_qualified_name>`, so
  `name.removeprefix("base_model.model.")` pairs them to our `LoRALinear` (which keeps the wrapped
  layer's original qualified name after `setattr`).
- Adapter weights: `layer.lora_A["default"].weight` / `layer.lora_B["default"].weight`;
  gradients at `.grad` on those tensors.
 
### Housekeeping follow-up — DONE
 
The three shared fixtures were duplicated across `test_lora_injection.py` and
`test_lora_parity.py`; both are now extracted into `tests/conftest.py`, so the 0.5B loads once
for the whole suite. `base_model_and_tokenizer` and `batch` are **session-scoped**; `fresh_model`
was converted to a **factory fixture** (returns a callable — tests call `fresh_model()` for a
fresh deepcopy per test), which is the pytest-idiomatic form and removes the non-idiomatic
`from conftest import fresh_model`. Session scope is safe because every injection runs on a
`fresh_model()` deepcopy and `test_bit_identical_at_init` only reads the base under `no_grad`.
Pure refactor, still **18 tests green**. Committed separately from the parity test.
 
## Session 4 — COMPLETE (`data.py` + toy training figure)
 
Two committed halves. **`pytest -v` → 22 passed** (18 prior + 4 new data tests).
 
**Coder-7B download verified first (closes OPEN ITEM #1).** Config smoke test with
`HF_HUB_OFFLINE=1` confirms `Qwen2ForCausalLM`, **28 layers, hidden 3584, 28 query / 4 KV heads (GQA).** Vocab: config.vocab_size = 152064 (embedding/logit width); len(tokenizer) = 151665; tokenizer.vocab_size = 151643 (base vocab only — this was the number previously recorded as “vocab”). tie_word_embeddings = False — embed_tokens and lm_head are separate ≈1.09 GB bf16 tensors each., eos `<|im_end|>`, pad `<|endoftext|>`. 4 safetensor shards,
~15.2 GB total (4.88 + 4.93 + 4.33 + 1.09 GB). Ready for Week 2 QLoRA. The GQA asymmetry
(28 q vs 4 kv) is the same shape `LoRALinear` already handles on the 0.5B fixture.
 
**Half 1 — `src/lora_sql/data.py` + `tests/test_data.py` (committed).** Implements the
`runner.py` contract: `Example` (now carrying a `schema` field alongside `db_id`/`question`/
`query`), `render_schema`, `format_prompt`, `load_examples`.
 
- **Schema-in-prompt design chosen** (over question-only): `format_prompt` is the shared
  train+eval contract reused in Week 2, so it renders the DB schema as CREATE TABLE DDL —
  columns + types (`number→INTEGER, text→TEXT, time→TIME, boolean→BOOLEAN, else→TEXT`),
  `PRIMARY KEY (...)`, and `FOREIGN KEY (...) REFERENCES ...`, skipping the `-1 *` wildcard
  column. Identifiers with spaces/non-word chars are double-quoted (`"Home Town"`).
- `render_schema(table_entry)` takes the whole `tables.json` entry (public, not underscored)
  so the golden test calls it directly on real data — no fixture file. Schema is cached
  per-`db_id` inside each `load_examples` call (many examples share a db).
- `format_prompt` receives **no tokenizer** and its output goes straight to `tokenizer(prompt)`
  — so **no `apply_chat_template`, no ChatML**. Plain instruction text; ChatML deferred to
  Week 2 (which owns its own training path and can reach the template).
- **4 tests:** golden-string equality of `render_schema("perpetrator")` byte-for-byte;
  `format_prompt` contains schema+question, excludes the query, ends with `SQL: `;
  `load_examples` maps fields correctly and the per-db cache is *shared* (`is` identity, not
  just equal strings); default sibling-`tables.json` path resolution (what `runner.py`'s
  single-positional-arg call depends on).
 
**Half 2 — `scripts/toy_train.py` + `scripts/plot_toy.py` (committed, then run).** Toy loop:
ours (`inject_lora`) vs peft (`get_peft_model`), Qwen2.5-0.5B fixture, fp32, bs=1, 200 steps,
AdamW(lr=2e-4), one pass over the first 200 `train_spider.json` rows. Both runs start from an
**identical `lora_A` snapshot** taken from a peft model (built, snapshotted, deleted before
training), with `lora_B=0` on both (NOT randomized — that was the parity-test-only trick;
real training needs `lora_B=0`). Base stays on CPU; each run `deepcopy`s it and moves only the
copy to CUDA, sequential with `empty_cache` between — one ~2 GB model resident at a time.
Loss via HF's built-in masked shifted CE so the loss math is identical across impls. matplotlib
added to `requirements.txt`.
 
**Result — the supporting figure is earned.** Loss descends 2.07 → ~0.2 over 200 steps, and
every printed step is identical across ours and peft (`2.0694`, `1.3771`, `0.2335`, …).
**`max per-step |loss_ours − loss_peft| = 0.000000`** — bit-identical training trajectories.
This is Session 3's 1e-6 forward/backward parity shown as a training curve, on the same fixture.
Artifacts committed: `results/toy_loss_curves.png`, `results/toy_loss_ours.json`,
`results/toy_loss_peft.json`.
 
**Writeup caption (drafted):** *"200-step LoRA training on Qwen2.5-0.5B, from-scratch vs peft,
identical init/data/seed — max per-step loss divergence 0.0. The parity of the 1e-6
forward/backward proof, shown as a training trajectory."* Lead the caption with the `0.000000`.
 
**Masking trap — downgraded.** The "pad==eos masking" trap flagged for the toy loop does not
bite on Qwen2.5: pad `<|endoftext|>` and eos `<|im_end|>` are **distinct token IDs**, so
pad-masking never clobbers a learned eos, and the toy loop is bs=1 (no padding) regardless.
Downgraded from live bug to: **add `assert pad_id != eos_id` when batching starts in Week 2.**
 
## Session 5 — pre-work (verified)
 
- **Token IDs:** pad `<|endoftext|>` = 151643, eos `<|im_end|>` = 151645,
  `generation_config.eos_token_id = [151645, 151643]`. pad ≠ eos confirmed.
- **`build_example` lives in `scripts/toy_train.py:36`, not `data.py`** — moves to `data.py` in
  Session 5 Half 1.
- **peft 0.19.1:** `get_peft_model(..., autocast_adapter_dtype=True)` by default;
  `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None)`.
- **Memory arithmetic for the 8.2 GB 4060:** untied embed+head ≈ 2.18 GB bf16 (4.36 GB if upcast
  to fp32 — must not happen); 4-bit body ≈ 3.3 GB; full-sequence fp32 logits ≈ 1.2 GB at seq 1024
  with grad. **Gradient checkpointing, completion-only logits, and no fp32 upcast of embed/head
  are required.**
- **Open question carried into the Session 5 spec:** adapters held in fp32 on the 4-bit path
  (pure-bf16 adapter weights lose small AdamW updates to rounding). Hard rule "training in bf16"
  to be reworded as "base compute in bf16."

## Session 5 — Half 1 progress

- **1a — fp32 adapters (DONE).** `LoRALinear` / `inject_lora` take `adapter_dtype=torch.float32`;
  forward mirrors peft's bnb `Linear4bit` cast order (cast `x` before dropout, scale in fp32, then
  downcast). Resolves the open question above. fp32 parity unchanged.
- **1b — `tests/test_lora_4bit.py` (DONE, 10 tests).** First time the 4-bit path runs in the suite.
  - Unit, single `bnb.nn.Linear4bit(256, 128, nf4, bf16)`: `weight.shape == (16384, 1)` (packed) yet
    adapters are `(8, 256)` / `(128, 8)`; adapters fp32, output bf16; `torch.equal` with base at init;
    `lora_A`/`lora_B` grads fp32 and non-zero, no base param gets a grad.
  - **QLoRA parity vs peft (0.5B, nf4, bf16 compute): bit-identical.** Logits and every per-layer
    `lora_A`/`lora_B` grad pass `torch.equal`; **max abs diff 0.000e+00** for q/v r=8, q/v r=8 α=12,
    and all-seven r=8. Two separate 4-bit loads are first checked bit-identical (precondition), and
    the adapted logits are checked to differ from base (non-vacuity).
  - **Why the α=12 case:** at scaling = α/r = 2.0 the multiply is exact in bf16, so scaling before vs
    after the downcast can't be told apart — the pre-1a order still passes r8-qv. At scaling 1.5 the
    old order is off from peft by 0.32 in the logits; the new order is 0.0.
- **1c — ChatML prompt contract (DONE, +4 tests in `test_data.py`).** `SYSTEM_PROMPT`,
  `encode_prompt` (only source of prompt ids, train + eval), `build_train_example` (ID-level
  prompt + SQL + `<|im_end|>`, returns `n_completion`); `runner.generate_sql` switched over (closes
  OPEN ITEM #4).
  - SQL tokenize→decode round-trips **byte-exact for all 8,659 training queries** (incl. Spider's
    irregular double spaces); Coder-7B `clean_up_tokenization_spaces=False`.
  - ID-level concat matched joined-string tokenization on 2,000/2,000 training examples checked —
    we build at the ID level anyway. Longest prompt in that sample: 707 tokens.
  - Smoke (0.5B, untrained, 2 dev examples): prompt ≈ 256 tokens, output is bare SQL — no prose or
    code fences.
- **1d — config, QLoRA loader, completion loss, memory probe (DONE, +18 tests).**
  - `config.py`: frozen dataclasses; unknown / missing keys and wrong types raise with the dotted
    path; floats coerced (YAML 1.1 reads `2e-4` as a string); `train_files` restricted to
    `train_spider.json` / `train_others.json` by exact name; `micro_batch_size` must be 1.
    `configs/train/`: qv_r8 (reference), qv_r4, qv_r16, attn_r8, all7_r8 — alpha = 2r.
  - `qlora.py`: `load_model` (nf4 + double quant, bf16 compute / `none`), `prepare_for_training`
    (`use_cache=False`, non-reentrant checkpointing, no kbit prep, no upcast, no input-grad hook —
    test confirms LoRA grads flow on 4-bit without it), `assert_training_dtypes(model, base_dtype)`,
    `completion_loss` via `logits_to_keep=n+1`.
  - **`logits_to_keep` is not bit-identical to full logits, by mechanism:** `lm_head` on n+1 rows vs
    the full sequence picks a different cuBLAS fp32 GEMM kernel (different reduction order over
    hidden). Kept logits differ by ≤2.5e-5 at |logit|≈32; vs HF's labels loss: loss ≤2.7e-6, LoRA
    grads ≤5.5e-6. So the test is two-tier: our CE indexing on *full* logits matches HF at 1e-6 with
    bit-identical grads (alignment proven), and the end-to-end `logits_to_keep` path is checked at
    1e-5. An off-by-one misalignment moves the loss 0.785 → 11.8 — six orders above the tolerance.
  - **Train length distribution (Coder-7B tokenizer, both train files, 8,659 examples):** total p50
    374 / p90 707 / p99 1310 / max 2228; completion p50 28 / p99 121 / max 202. Dropped by
    `max_seq_len`: 512 → 2365; 1024 → 184 (baseball_1 82, cre_Drama_Workshop_Groups 82, sakila_1 13,
    cre_Theme_park 6, formula_1 1); 1536 and 2048 → 82 (all baseball_1, 2049–2228 tokens); 2560 → 0.
  - **Pre-flight (qv_r8, 7B nf4):** embed / lm_head stay bf16; footprint 5.08 GB; seq 512 peaks at
    5.67 GB allocated, 1.03 s/step; peak host RAM during load ≈ 8.7 GB of 13.
  - **Memory probe (7B nf4, 3 AdamW steps per length on synthetic ids with a 202-token completion;
    sec/step = median of steps 2–3; `results/probe_{qv_r8,all7_r8}.json`).** Peak alloc / reserved GB,
    sec/step:

    | seq_len | qv_r8 (2.52M LoRA params) | all7_r8 (20.2M LoRA params) |
    | --- | --- | --- |
    | 512  | 5.67 / 5.75, 1.06 | 5.87 / 6.20, 1.41 |
    | 1024 | 5.78 / 6.00, 1.87 | 6.11 / 6.45, 2.64 |
    | 1536 | 6.00 / 6.23, 2.76 | 6.45 / 6.94, 3.95 |
    | 2048 | 6.22 / 6.57, 3.75 | 6.79 / 7.25, 5.41 |
    | 2560 | 6.46 / 6.81, 4.72 | **OOM** (7.14 / 7.52 before failing) |

    Projected epoch at 2048 (8,577 kept, 537 optimizer steps): qv_r8 2.72 h, all7_r8 3.66 h, vs the
    6 h budget. Slight overestimates: the probe steps AdamW every micro-step and charges examples
    under 512 tokens as 512.
  - **`max_seq_len = 2048` for all five configs** (one value, so the ablations train on identical
    data). It is the largest probed length where all7_r8 fits (~0.75 GB reserved headroom). It drops
    82 examples (0.95%), all from `baseball_1`; Spider's train and dev databases are disjoint, so no
    dev database loses coverage. Keeping everything (~2240) would leave ~0.6 GB on all7_r8 for an
    8,600-step run — too tight.

## Session 5 — Half 2 progress

- **2a — `checkpoint.py` (DONE, +6 tests).**
  - `save_lora` writes `adapter.safetensors` (only `lora_` keys, fp32) and a `lora_config.json`
    sidecar. The sidecar has base_model, target_modules, rank, alpha, dropout, adapter_dtype, step,
    git_commit (`+dirty` if the worktree is dirty), config_sha256, n_keys, n_params and any extra
    meta.
  - The structural fields are read from the `LoRALinear` modules, not from the caller, and saving
    fails if the layers disagree.
  - `load_lora` takes an un-injected base. It checks base_model, calls `inject_lora` from the
    sidecar (so a wrong rank or target set fails at load), then runs a shape / dtype pre-check.
    After `load_state_dict(strict=False)` it asserts: no unexpected keys; every saved key consumed;
    model `lora_` params == saved keys; tensors `torch.equal`; at least one `lora_B` nonzero.
  - **Tests:**
    - The roundtrip gives bit-identical logits.
    - The file holds only `lora_` keys, and its size is exactly 8 + header + n_params×4.
    - Four bad checkpoints all raise: a `base_model.model.` key prefix, a tampered rank, an
      all-zero `lora_B` and a wrong base model.
- **2b — `train.py` (DONE, +6 tests).** `python -m lora_sql.train --config ... [--limit-examples N]
  [--overwrite]`.
  - **Run directory:**
    - Refuses to start if `final/` exists; `--overwrite` removes only `final/` + `step_*/`.
    - Writes `config.yaml` and `data_report.json`.
  - **Data:**
    - Seeded shuffle. `data_order_sha256` hashes the ordered token ids, so it covers both the
      order and the content.
    - FLOOR step count: 8,577 kept // 16 = **536 steps**, discarding 1 example. The probe's
      "537" above is its ceil.
  - **Before the first step:**
    - asserts pad ≠ eos
    - `assert_training_dtypes` against `cfg.model.dtype` (already in the "unchanged from load"
      form, so `qlora.py` was not edited)
    - OOM preflight: forward + backward on the longest example, with no optimizer step
  - **Each step:**
    - Token-weighted `accumulate_window`, then clip → AdamW → custom warmup + cosine
      `LambdaLR` → zero_grad.
    - `log.jsonl` rows record step, loss, lr, grad_norm, tokens, elapsed and peak memory.
    - Adapters save to `step_<n>/` and, at the end, to `final/`.
    - The time-budget guard saves `final/` and logs `stopped_by_budget`.
  - No autocast.
  - **Tests (0.5B fp32):**
    - Accumulation equals one big batch within 1e-6.
    - Schedule boundaries and step count are checked.
    - Smoke run: logs steps 1–3, `step_2/` + `final/`, real-data drop `{"baseball_1": 82}`, and
      `config.yaml` reloads equal.
    - Reloading `final/` gives logits `torch.equal` to the trained model (eval mode, dropout 0.05
      on).
    - Budget guard and no-clobber both behave as specified.
  - The five configs were already at `max_seq_len 2048 / max_steps null / epochs 1`; no YAML
    changed.
- **Known gap:** `runner.py` still loads checkpoints with `torch.load` + a bare
  `load_state_dict(strict=False)`, which can't read the new format. It must switch to
  `checkpoint.load_lora` before Week 3 (OPEN ITEM 5).

## Session 6 — COMPLETE (training runs + eval)

- **Run #1 `qv_r8` (DONE).**
  - 536/536 optimizer steps in **2.29 h** (the probe projected 2.72 h), peak **6.23 GB**
    reserved, no budget stop.
  - `data_report`: 8,659 loaded, baseball_1: 82 dropped, 8,577 used, 1 tail example discarded.
  - Logged loss: 0.593 at step 1, levelling off around **0.08** from roughly step 250 (mean of
    the last 10 logged points 0.087). Each logged point is one window's loss, so it is noisy.
  - Figure: `results/train_loss.png`. Evidence (log, data report, sidecar) is in
    `results/train/qv_r8/`; adapter weights stay in gitignored `runs/`.
- **Reload check on the real 7B (nf4).** `load_lora(model, "runs/qv_r8/final")` passes all five
  checks. On the first two dev questions the base writes `COUNT(*)` and qv_r8 writes Spider's
  `count(*)`, matching gold exactly, so the adapter visibly loaded. Generation peaked at 5.49 GB.
- **Ablation chain (DONE):** `scripts/train_chain.sh qv_r4 qv_r16 attn_r8 all7_r8` ran
  2026-11-02 00:28–10:24, all four `exit 0`, each `final/` at step 536.
  - Train hours: qv_r4 2.27, qv_r16 2.25, attn_r8 2.31, all7_r8 3.06.
  - Peak reserved: 6.21 / 6.26 / 6.27 / 6.74 GB.
  - Mean of the last 10 logged losses: qv_r8 0.087, qv_r4 0.090, qv_r16 0.084, attn_r8 0.080,
    all7_r8 0.065.
  - All runs share one data order, so their loss spikes line up step for step
    (`results/train_loss.png`).
- **GPU rule while the chain runs:** all7_r8 peaks at 7.25 GB of 8.2, so only CPU tests run
  meanwhile, always with `CUDA_VISIBLE_DEVICES=""`. No pytest that loads a model.
- **Eval-harness hardening (DONE CPU-side; closes OPEN ITEMS 3 and 5 except difficulty).**
  - `execute.py`:
    - Read-only URI connections: generated writes raise and the DB is untouched, and a missing
      file is never created.
    - A wall-clock query deadline via `set_progress_handler` (default 30 s). sqlite's `timeout` is
      only a lock wait, so a generated cross join could otherwise hang the eval.
    - `try/finally` close.
    - NULL-safe, duplicate-sensitive `Counter` compare. Row order is compared only when the gold
      has ORDER BY (the test-suite convention).
    - `execute_and_compare` returns `(correct, exec_error)` against precomputed gold rows.
    - Known difference from the official test-suite eval: columns are compared positionally (no
      permutation search), so ours is slightly stricter.
  - `runner.py`:
    - `check_held_out` rejects `test*.json` and anything but `database/`.
    - `gold_preflight` executes all gold queries before the model loads.
    - Models load via `qlora.load_model` (nf4 path); checkpoints load via `checkpoint.load_lora`;
      the base path injects nothing. The last bare `load_state_dict(strict=False)` in the repo is
      gone.
    - Writes a per-example `.jsonl` as it goes; the final JSON has `summary` (+
      `n_pred_exec_errors`), `meta` (checkpoint sidecar, eval git commit) and `results`.
    - CLI: `--out`, `--limit`.
  - **Real-data check (CPU):** all **1,034 dev gold queries execute in 1.6 s**. 237 have ORDER BY.
    **9 gold results contain NULLs**, so the old `sorted()` compare could have crashed a real
    eval.
  - **Configs:** `configs/week1_parity.yaml` is filled in as the 0.5B fp32 smoke eval (results
    to `runs/`). New `configs/eval/coder7b_spider_dev.yaml` is the real one (nf4, dev,
    `max_new_tokens` 256). Neither has a `lora` section: the shape comes from the sidecar.
  - **Scripts:** new `scripts/eval_chain.sh` evaluates base + runs back to back. It refuses to
    start if a named run has no `final/`.
  - **Tests:** `tests/test_eval_execute.py` (16, CPU) and `tests/test_eval_runner.py` (3, GPU).
    Full `pytest -v` after the chain: **85 passed**.
  - `checkpoint._git_commit` is now public `git_commit` (also used for eval provenance).
- **Spider dev results (DONE, C).** `scripts/eval_chain.sh` ran all six, about 20–28 min each.
  Every eval covers all 1,034 questions, at eval commit `c1b3174`. Each eval's `meta` holds the
  checkpoint sidecar, so every number traces back to a training config. Full table:
  `results/eval/summary.md`, generated by `scripts/summarize_eval.py`.

  | run | LoRA params | dev EX % | Δ vs base | McNemar p (Holm) |
  |---|---|---|---|---|
  | base | 0 | 74.95 | | |
  | qv_r8 | 2.52M | 75.92 | +0.97 | 0.453 (0.900) |
  | qv_r4 | 1.26M | 75.92 | +0.97 | 0.450 (0.900) |
  | qv_r16 | 5.05M | 77.47 | +2.51 | 0.037 (0.164) |
  | attn_r8 | 5.05M | 76.79 | +1.84 | 0.113 (0.338) |
  | all7_r8 | 20.19M | 77.56 | +2.61 | 0.033 (0.164) |

  - **The gains are small.** No run is significantly better than base after Holm correction.
    Each run gains 74–88 questions and loses 55–67, so fine-tuning mostly reshuffles which
    questions are right.
  - **Rank 16 beats rank 8 on q/v** (+24/−8, p=0.007 unadjusted). all7_r8 has much the lowest
    training loss but ties qv_r16 on EX (+40/−39). One seed per config, so seed variance is unmeasured.
  - **The base is probably contaminated.** Untuned Coder-7B already writes Spider-gold quirks
    (`Country ,  COUNT(*)` with the double space), which caps the headroom.
  - **Column-order robustness check.** Scoring with any column permutation allowed (as the
    test-suite eval does) raises every run 2.3–3.2 pts, base the most. The strict Δ is
    therefore the larger estimate. qv_r16 and all7_r8 swap places at the top; the rest of the
    order holds.
  - **What changes:**
    - Gains come from Spider conventions: INTERSECT, `=` instead of `LIKE`, literals as
      written, fewer needless joins.
    - Losses come from invented snake_case column names (`pet_type` for `PetType`) and dropped
      DISTINCTs.
  - A 20-example probe before the chain confirmed that base emits bare SQL with no fences or
    prose, so base isn't penalised for formatting.

---
 
# DECISIONS MADE
 
**transformers: staying on 5.14.1 — CLOSED (Session 2).** Earlier leaning was to pin to 4.x;
reversed. Reasons: peft ≥ 0.18.0 is required for v5 compatibility and 0.19.1 is already installed,
so downgrading transformers would produce an *untested* combination rather than a safer one;
4.x is now the deprecated line; torch 2.5.1 satisfies v5's requirements; TRL/Unsloth/Axolotl all
maintain v5 compatibility. The translation cost is bounded and enumerated: `dtype=` not
`torch_dtype=`, `apply_chat_template` returns a `BatchEncoding` (`.to(device)` +
`generate(**inputs)`). Pinned explicitly in `requirements.txt` so `pip install -r` can't drift.
 
**Week 2 training loop: custom, not TRL's `SFTTrainer` — decided Session 2.** Roughly 60 lines
(AdamW, gradient accumulation, cosine schedule, bf16 autocast). Removes almost all remaining
version-coupling surface, and "I wrote the training loop" reads better on a project whose pitch
is *from scratch*.
 
**Precision split: parity in fp32, training in bf16 — decided Session 2.** bf16 carries ~3
significant decimal digits, so two mathematically identical implementations that accumulate in a
different order disagree around 1e-2. `atol=1e-6` is physically unreachable in bf16 regardless of
correctness. The 0.5B in fp32 is ~2 GB, comfortable on the 4060. **State this explicitly in the
writeup** — it's a point in favour, not a caveat. *(Vindicated in Session 3: fp32 parity is
bit-identical.)*
 
**Session 5 design — decided (pre-implementation).**
 
- **Prompt:** ChatML via `data.encode_prompt` with an explicit `SYSTEM_PROMPT`. It is the single
  source for train and eval, and `runner.generate_sql` switches to it (closes OPEN ITEM #4).
- **Precision:** 4-bit nf4 base with bf16 compute, fp32 LoRA adapters (matches peft's
  `autocast_adapter_dtype=True` default), no autocast. This supersedes "bf16 autocast" in the
  training-loop decision.
- **`prepare_model_for_kbit_training` is NOT used.** Verified source: it casts every
  non-`Params4bit` bf16 param to fp32, and on the untied 7B embed + head that's 4.36 GB. We keep
  only gradient checkpointing with `use_reentrant=False`, which per peft's source needs no
  input-grad hook.
- **Loss:** completion-only logits via `logits_to_keep=n_completion+1` (verified present in
  v5.14.1 `Qwen2ForCausalLM.forward`); fp32 CE, summed. Micro-bs 1, grad accum 16,
  token-weighted accumulation.
- **Data:** `train_spider` + `train_others` (8,659). Measured prompt tokens: `train_spider`
  p50/p95/p99/max = 255/855/2041/2067; `train_others` max 648; dev max 673. SQL max 201. Longest
  training sequence ≈ 2,310 tokens. `max_seq_len` is set by the memory probe, and any dropped
  `db_id`s get reported in the writeup.
- **Hyperparameters:** lr 2e-4, warmup 3% then cosine to 0, wd 0, clip 1.0, dropout 0.05,
  1 epoch, ≤ 6 h per run.
- **Ablations:** alpha = 2r (scaling fixed at 2.0), so the rank ablation doesn't also change
  effective LR. Configs: `qv_r8` (reference), `qv_r4`, `qv_r16`, `attn_r8`, `all7_r8`.
- **Checkpoints:** `adapter.safetensors` (only `lora_` keys, under our module names) plus a
  `lora_config.json` sidecar. `load_lora` rebuilds via `inject_lora` and asserts no unexpected
  keys, every saved key consumed, every model `lora_` key covered, tensors equal, and at least one
  `lora_B` nonzero. A peft-format export is deferred to Week 4.
- **Cut:** resume-from-checkpoint. Mitigated by periodic adapter saves and `systemd-inhibit`.
- **New claim to earn:** parity on the QLoRA path (4-bit 0.5B, ours vs peft). Record the measured
  max diff.
 
**Week 1 validation model: Qwen2.5-0.5B-Instruct** (locked). Chosen over Gemma-2B because
its module names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
are *identical* to Qwen2.5-Coder-7B, so `inject_lora` target strings transfer verbatim with
zero edits. It also uses GQA, so `k_proj`/`v_proj` have much smaller `out_features` than
`q_proj` — a real shape asymmetry the `LoRALinear` must handle correctly, and one the 7B
shares. **This model is a parity fixture only. It is never fine-tuned.**
 
**Base model (Week 2 QLoRA target): Qwen2.5-Coder-7B-Instruct.** Code-specialized
pretraining gives a much stronger out-of-the-box SQL baseline than a general model like
Llama-3.1-8B-Instruct. Dense — no MoE routing complexity like the newer Qwen3-Coder line, whose
smallest variant jumps to a 30B MoE, too heavy for local QLoRA on 8GB. Apache 2.0, well supported
by standard LoRA/QLoRA tooling.
 
**Dataset: Spider (train + primary eval), BIRD-dev as a held-out generalization test.**
Spider's difficulty tiers map directly to the Week 3 breakdown, and its ~200 databases are a
manageable footprint for local execution-based eval. BIRD is harder and more realistic (noisy
real-world values, external-knowledge reasoning) — running the Spider-fine-tuned model zero-shot
against BIRD-dev tests generalization to an unseen, harder benchmark, a stronger claim than
in-distribution accuracy alone. WikiSQL skipped (mostly single-table, minimal joins). A custom
schema stays in scope as a qualitative bonus demo for the writeup, not as a quantitative
benchmark — self-authored eval sets are easy for a reviewer to be skeptical of.
 
**Spider splits — strict discipline:**
 
- `train_spider.json` → training only
- `dev.json` → the *only* in-distribution eval set, used throughout (keeps numbers
  comparable to published work, which is all dev-column)
- `test.json` / `test_database/` → **do not touch.** Publicly released Feb 2024 and ships in the
  download, but using it for tuning would destroy its value. Reserved at most for a single final
  sanity check after all decisions are frozen.
- BIRD-dev → zero-shot only, never trained on
 
**HF `xlangai/spider` is NOT usable for this project.** ~960 KB of text pairs only
(`db_id`, `query`, `question`, tokens). No SQLite files → no execution-based eval. No
`tables.json` → no schema to put in the prompt. The official Google Drive release is
required. (Reconstructing empty DBs from schema is a trap: every query returns an empty
set, so every query "matches.")
 
## Scope cuts (forced by the ~40 hour budget)
 
- ❌ **DoRA** — nice-to-have, requires a whole second implementation.
- ❌ **RAG baseline** — already on my resume from Fusemachines; adds nothing new, costs real infra time.
- ✅ **Kept: rank ablation** (r = 4/8/16) — same script three times, mostly unattended GPU hours.
- ✅ **Kept: BIRD-dev zero-shot** — cheap once the harness exists, and the strongest single claim.
- ✅ **Added: target-set ablation** — q/v vs. full attention vs. all-seven. "Does targeting the MLP
  justify 4× the adapter size" is a more interesting question than rank alone.
 
## Corrections to earlier assumptions
 
**1. Spider does NOT ship difficulty tiers as a data field.** No `difficulty` key in `dev.json`.
The tiers are **computed** by `eval_hardness()` in `evaluation.py` from the
[taoyds/spider](https://github.com/taoyds/spider) repo, which parses each gold SQL and counts
components (joins, nesting, aggregates, `GROUP BY`). Consequence: `process_sql.py` from that repo
is a dependency of the Week 3 difficulty breakdown. `metrics.py` currently defaults `difficulty`
to `"unknown"` — this is the gap to fill.
 
**2. `bnb.nn.Linear4bit` IS a subclass of `nn.Linear`** (verified: `isinstance` → `True`). An
earlier assumption that it wasn't turned out to be wrong. Good news — `inject_lora`'s
`isinstance(child, nn.Linear)` filter will correctly find quantized projections in Week 2.
The *real* `Linear4bit` gotchas are different:
 
- `weight.shape` reports **packed** dimensions (a 64×64 layer shows `(2048, 1)` — 4,096
  weights at 4 bits = 2,048 bytes). Read `in_features` / `out_features` attributes instead.
- `lora_A` / `lora_B` must not be created in the packed storage dtype (`uint8`). *(Superseded
  Session 5:)* they are created in **fp32** via `adapter_dtype`, not the bf16 compute dtype —
  peft does the same (`autocast_adapter_dtype=True`), and bf16 adapters lose small AdamW updates.
 
**3. Official Spider metric is Test Suite Accuracy**, not plain execution accuracy — separate
codebase at [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval). Worth
citing in the writeup to show awareness of the standard even while implementing single-DB
execution accuracy myself.
 
**4. Claude Code's `/init` output is not trustworthy as written.** The generated `CLAUDE.md`
contained two real errors: it described the project as fine-tuning the 0.5B (wrong model role),
and it listed `test_database/` as a legitimate execution target for scoring — directly
contradicting the held-out rule. Both were corrected by hand. `/init` is a scaffolding tool;
everything it asserts about *intent* needs review.
 
**5. Claude's predictions about library behaviour need verification against actual output.**
Two errors in Session 1 (the `Linear4bit` subclass question above, and a broken `gdown --fuzzy`
command with the wrong URL type). Worth staying alert to.
 
**6. peft internals confirmed empirically in Session 3.** Before writing the parity spec, peft's
LoRA layer was introspected against real output rather than trusted from memory (per correction #5).
Result: `lora_A["default"].weight` is `(r, in_features)`, `lora_B["default"].weight` is
`(out_features, r)` — transplant is a bare `.copy_()`, no transpose. `scaling` is a **dict**
`{"default": alpha/r}`, not a scalar. Both sides init `lora_B = 0`, so peft's `lora_B` must be
randomized before any comparison or the test is vacuous. All three matched the Session 2 design
assumptions — but they were checked, not assumed.
 
---
 
# CLAUDE.md hard rules (repo root, loaded by every Claude Code session)
 
Recorded here so they survive even if the file is regenerated:
 
- `data/spider_data/test.json` and `test_database/` are **HELD OUT**. Never read, never evaluate
  against, never tune on. `database/` is the only execution target.
- Never use `base_layer.weight.shape` for layer dimensions — packed under `Linear4bit`.
  Use `.in_features` / `.out_features`.
- transformers is pinned to 5.14.1 (v5). Use `dtype=`, never `torch_dtype=`.
- Parity tests run in fp32. Training: 4-bit base with bf16 compute, LoRA adapters fp32, no
  `torch.autocast`, explicit casts in `LoRALinear` that mirror peft's order. Never call
  `peft.prepare_model_for_kbit_training` and never upcast base params — it fp32-casts the untied
  embed + lm_head (4.36 GB).
- Prompt ids come only from `data.encode_prompt` (ChatML + `SYSTEM_PROMPT`), in both train and eval.
- Training files are only `train_spider.json` / `train_others.json`. `config.py` enforces this;
  dev is eval-only.
- LoRA checkpoints go through `checkpoint.save_lora` / `load_lora` only. Never call a bare
  `load_state_dict(strict=False)`.
- Training is micro-bs 1 with token-weighted gradient accumulation and completion-only logits
  (`logits_to_keep`).
- **peft `scaling` is a dict** `{"default": alpha/r}` — index `["default"]`, never assume it's a
  scalar. (Session 3.)
- **Parity / weight-transplant tests must randomize peft's `lora_B` before copying**, or the
  comparison is vacuous (`base == base`) and `lora_A.grad` is zero. (Session 3.)
- **LoRA checkpoints:** save only keys containing `lora_`, and after
  `load_state_dict(..., strict=False)` assert `unexpected_keys == []` and that the expected LoRA
  keys were actually consumed. `strict=False` silently loading *nothing* is a real failure mode
  here — injection renames base params (`q_proj.weight` → `q_proj.base_layer.weight`), so a
  prefix mismatch between the save side and the load side would evaluate the base model while
  reporting success. This is the highest-value trap in the project.
- Ablations are new files in `configs/`, never edited copies of `train.py`.
- Model roles: 0.5B = parity fixture only; Coder-7B = the actual fine-tune target.
- GPU is an RTX 4060 Laptop, 8.2 GB VRAM. Assume memory is tight.
- **Parity tests must include a non-power-of-two scaling case** (e.g. r=8, alpha=12 → 1.5).
  Power-of-two scaling is exact in every float format and commutes with dtype casts, which hides
  cast-order bugs. Every training config uses alpha=2r (scaling 2.0), so training configs alone
  can never catch this. (Session 5: the pre-1a order was off by 0.32 at scaling 1.5, 0.0 at 2.0.)
- `LoRALinear` cast order mirrors peft 0.19.1 bnb `Linear4bit.forward` exactly: `x` → adapter dtype
  (fp32) BEFORE dropout; `lora_B(lora_A(dropout(x))) * scaling` in fp32; output → base output dtype,
  then add. Do not reorder; 4-bit parity is bit-identical only in this order.
- All `configs/train/*.yaml` share identical data settings (`max_seq_len` 2048, train files, seed);
  ablations differ only in `run_name`, `lora.*`, `train.out_dir`. A test enforces this.
 
---
 
# PLAN
 
**Week 1 — LoRA from scratch** *(Sessions 1-4 — ALL DONE)*
 
- ✅ Custom `LoRALinear` in raw PyTorch: frozen base weight, trainable low-rank A (r×d) and
  B (d×r), scaled by alpha/r.
- ✅ Wrap attention/MLP projections of Qwen2.5-0.5B-Instruct; injection tests green.
- ✅ **Numerical parity vs. `peft`**: weight-transplant → forward `allclose` → backward `allclose`,
  fp32, `atol=1e-6`, parametrized over r∈{4,8,16} + all-seven. **18 tests green.**
- ✅ Loss-curve comparison (Session 4): ours vs peft, 200 steps, fp32, identical init/data.
  **Max per-step loss δ = 0.000000** — bit-identical trajectories. Figure at
  `results/toy_loss_curves.png`. **Week 1 is complete.**
 
**Week 2 — QLoRA on the real task**
 
- Base model: Qwen2.5-Coder-7B-Instruct, QLoRA (4-bit via `bitsandbytes`,
  `bnb_4bit_compute_dtype=torch.bfloat16`) — fits the 8GB 4060 for iteration; cloud burst
  for a bigger run if wanted.
- Custom training loop (see "Session 5 design" above): 4-bit nf4 base with bf16 compute, fp32
  adapters, no autocast, micro-bs 1 with token-weighted grad accum 16, completion-only logits.
- Dataset: `train_spider` + `train_others` (8,659). BIRD-dev reserved purely for Week 3.
- `src/lora_sql/data.py` — ✅ **built in Session 4** (`Example`+`schema`, `render_schema`,
  `format_prompt`, `load_examples`). Session 5 adds `encode_prompt` (ChatML + `SYSTEM_PROMPT`,
  the single source of prompt ids for train and eval) and moves `build_example` in from
  `scripts/toy_train.py`.
- ✅ `checkpoint.py` + `train.py` built and tested (Session 5 Half 2).
- ✅ Run #1 `qv_r8` on the 7B, then the `qv_r4`, `qv_r16`, `attn_r8`, `all7_r8` chain (Session 6).
 
**Week 3 — Evaluation harness — built in Session 1, hardened in Session 6** (see "Session 6"
above for what changed; the Session 1 description below is kept for history)
 
- `runner.py` — orchestration. Loads YAML config, loads base model + tokenizer, calls
  `inject_lora`, optionally loads a LoRA checkpoint, generates greedily (`do_sample=False`),
  writes `{"summary": ..., "results": [...]}` JSON.
- `execute.py` — SQLite execution and comparison. `run_query` converts any `sqlite3.Error` into
  `ExecutionError` so a broken generation is a scoring signal rather than a crash. `results_match`
  does order-insensitive comparison with float rounding, matching Spider's convention.
  `execute_and_compare` treats an unexecutable prediction as simply wrong.
- `metrics.py` — aggregation over `EvalResult` into `MetricsSummary`; overall accuracy plus
  breakdown by `difficulty`.
 
**Remaining Week 3 work:**
 
- ✅ Eval configs (`configs/week1_parity.yaml` smoke, `configs/eval/coder7b_spider_dev.yaml` real);
  the held-out rule is enforced in code by `runner.check_held_out`.
- ✅ Score Spider dev: base + five fine-tuned runs (Session 6; `results/eval/summary.md`).
- ⬜ (deferred) Wire `eval_hardness()` from taoyds/spider so `difficulty` stops defaulting to `"unknown"`.
- BIRD-dev zero-shot run.
- Comparison table: base model vs. mine vs. peft-fine-tuned sanity check.
 
*(Closed Session 5, 1c: `generate_sql` uses `data.encode_prompt`, the same ChatML ids training
uses.)*
 
**Week 4 — Polish**
 
- Write up like a mini paper: motivation, method, results, ablations.
- Ablations: rank (r = 4/8/16) **and** target set (q/v vs. attention vs. all-seven).
- Clean GitHub repo, README, loss curve + eval score plots.
 
---
 
# REPO LAYOUT
 
```
SQL_Fine-Tune/
├── CLAUDE.md               # Claude Code's context file — hard rules, model roles
├── pyproject.toml          # ✅ setuptools backend + pytest config
├── requirements.txt        # frozen; transformers pinned to 5.14.1
├── .gitignore              # data/, runs/, .venv/
├── configs/
│   ├── week1_parity.yaml   # ✅ S6 — 0.5B fp32 smoke eval config
│   ├── eval/               # ✅ S6 — coder7b_spider_dev.yaml (real eval, nf4)
│   └── train/              # ✅ qv_r8 (reference), qv_r4, qv_r16, attn_r8, all7_r8
├── src/lora_sql/
│   ├── lora.py             # ✅ LoRALinear, inject_lora, lora_param_stats
│   ├── data.py             # ✅ Example+schema, render_schema, format_prompt, load_examples,
│   │                       #    encode_prompt, build_train_example
│   ├── config.py           # ✅ S5 — strict frozen-dataclass train config
│   ├── qlora.py            # ✅ S5 — load_model, prepare_for_training, completion_loss
│   ├── checkpoint.py       # ✅ S5 — save_lora / load_lora (safetensors + sidecar)
│   ├── train.py            # ✅ S5 — config-driven QLoRA training loop
│   └── eval/
│       ├── execute.py      # ✅ implemented
│       ├── metrics.py      # ✅ implemented
│       └── runner.py       # ✅ implemented
├── scripts/
│   ├── check_env.py                 # ✅ passing
│   ├── check_spider_database.py     # ✅ passing
│   ├── toy_train.py                 # ✅ S4 — ours vs peft toy loop (run; δ=0.0)
│   ├── plot_toy.py                  # ✅ S4 — loss-curve plot + max-δ report
│   ├── probe_memory.py              # ✅ S5 — peak memory / sec-per-step probe per config
│   ├── train_chain.sh               # ✅ S6 — train several configs back to back
│   ├── plot_train.py                # ✅ S6 — loss + LR plot, copies run evidence to results/train/
│   ├── eval_chain.sh                # ✅ S6 — score base + runs on Spider dev back to back
│   └── summarize_eval.py            # ✅ S6 — results/eval/summary.md (table, McNemar, bootstrap CI)
├── tests/
│   ├── conftest.py                  # ✅ shared fixtures: session-scoped base, fresh_model factory
│   ├── test_lora_injection.py       # ✅ 10 passed
│   ├── test_lora_parity.py          # ✅ 8 passed (Session 3) — peft forward+backward parity
│   ├── test_data.py                 # ✅ 8 passed — schema/prompt/load_examples/ChatML
│   ├── test_lora_4bit.py            # ✅ 10 passed (S5) — 4-bit unit + QLoRA parity vs peft
│   ├── test_config.py               # ✅ 14 passed (S5)
│   ├── test_qlora.py                # ✅ 4 passed (S5) — completion loss vs HF, nf4 path
│   ├── test_checkpoint.py           # ✅ 6 passed (S5) — roundtrip + load-time traps
│   ├── test_train.py                # ✅ 6 passed (S5) — accumulation, schedule, smoke run
│   ├── test_eval_execute.py         # ✅ 16 passed (S6, CPU) — read-only, deadline, NULLs, guard
│   └── test_eval_runner.py          # ✅ 3 passed (S6, GPU) — load_lora path, base path, smoke run_eval
├── data/                   # gitignored — spider_data/ lives here
├── runs/                   # gitignored — checkpoints, logs (+ toy_train.log)
└── results/                # COMMITTED — eval JSONs, plots (toy_loss_*, probe_*, train_loss.png,
                            #   train/<run>/{log.jsonl, data_report.json, lora_config.json},
                            #   eval/<name>.json + .jsonl for base and the five runs, eval/summary.md)
```
 
Two choices that pay off later:
 
**`configs/` as YAML, `train.py` takes a config path.** The ablations become N config files
and N commands, not N edited copies of a script. This is the difference between the ablation
being an afternoon and being "I ran out of time."
 
**`runs/` ignored, `results/` committed.** Checkpoints are gigabytes of noise; eval JSONs and
plots are the project. A reviewer cloning the repo should see the numbers without downloading
weights.
 
---
 
# COMMANDS
 
```bash
source .venv/bin/activate
 
python -c "import lora_sql; print('ok')"   # packaging gate
python scripts/check_env.py
python scripts/check_spider_database.py
pytest -v                                   # 85 passing as of Session 6
pytest -v tests/test_lora_parity.py         # just the parity suite
 
# Session 4 toy training figure (run from repo root):
python scripts/toy_train.py 2>&1 | tee runs/toy_train.log   # ours vs peft, ~mins on the 4060
python scripts/plot_toy.py                                   # writes results/toy_loss_curves.png, prints max δ

# Training (Session 5) -> runs/<run_name>/ (log.jsonl, data_report.json, step_<n>/, final/)
HF_HUB_OFFLINE=1 python -m lora_sql.train --config configs/train/qv_r8.yaml --limit-examples 32  # 7B sanity, then rm -rf runs/qv_r8
systemd-inhibit --what=sleep:idle python -m lora_sql.train --config configs/train/qv_r8.yaml 2>&1 | tee runs/qv_r8.log
systemd-inhibit --what=sleep:idle bash scripts/train_chain.sh qv_r4 qv_r16 attn_r8 all7_r8   # -> runs/chain.log
python scripts/plot_train.py --runs qv_r8 qv_r4 qv_r16 attn_r8 all7_r8   # results/train_loss.png + results/train/<run>/
 
# Eval (Session 6): smoke on the 0.5B, then the real thing on Coder-7B nf4
python -m lora_sql.eval.runner --config configs/week1_parity.yaml --limit 5
python -m lora_sql.eval.runner --config configs/eval/coder7b_spider_dev.yaml --lora-checkpoint runs/qv_r8/final --out results/eval/qv_r8.json
systemd-inhibit --what=sleep:idle bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
python scripts/summarize_eval.py            # -> results/eval/summary.md (CPU; re-executes predictions)
CUDA_VISIBLE_DEVICES="" pytest -v tests/test_eval_execute.py tests/test_config.py tests/test_data.py   # CPU-only subset
```
 
---
 
# OPEN ITEMS
 
1. ✅ **Qwen2.5-Coder-7B-Instruct download (~15 GB) — VERIFIED (Session 4).** Config smoke
   test (`HF_HUB_OFFLINE=1`) confirmed `Qwen2ForCausalLM`, 28 layers, hidden 3584, 28 q / 4 kv
   heads (GQA), 4 safetensor shards ≈15.2 GB, eos `<|im_end|>`, pad `<|endoftext|>`. Ready for
   Week 2 QLoRA.
2. ✅ **`data.py` — DONE (Session 4).** `Example`+`schema`, `render_schema`, `format_prompt`,
   `load_examples`; 4 tests green. Eval runner is now executable end-to-end once the config
   YAML exists (item 3).
3. ✅ **`configs/week1_parity.yaml` — DONE (Session 6).** Filled in as the 0.5B smoke eval;
   the real eval config is `configs/eval/coder7b_spider_dev.yaml`. `runner.check_held_out`
   enforces `database/`.
4. ✅ **ChatML in `runner.py` — CLOSED (Session 5, 1c).** `runner.generate_sql` builds inputs via
   `data.encode_prompt` and passes an explicit attention mask.
5. ✅ **Eval-harness hardening — DONE (Session 6), except the deferred difficulty wiring.** The
   original review is kept below for history; see "Session 6" for the fixes.
   Reviewed the `eval/` package in a Session 3 tail; three real issues, all latent because no
   checkpoint/generation/execution happens before Week 3:
   - **`runner.py` checkpoint load is unguarded** — `model.load_state_dict(state, strict=False)`
     throws away the return. This is the exact CLAUDE.md "highest-value trap": a prefix mismatch
     makes `strict=False` load *nothing*, leaving `lora_B` at zero-init (adapter is a no-op) while
     eval reports success on the untrained base. **Updated fix (Session 5 Half 2):** `runner.py`
     also can't read the training checkpoint format: it uses `torch.load` on a `.pt`, while
     training writes `adapter.safetensors` + `lora_config.json` dirs. Replace its `inject_lora` +
     `load_state_dict` with `checkpoint.load_lora(base, ckpt_dir)`, which already asserts
     everything above. `--lora-checkpoint` then takes e.g. `runs/qv_r8/final`.
   - **`execute.py` opens DBs read-write** — generated SQL runs against the real
     `database/*.sqlite` with write access, so a generated `DROP`/`DELETE` could persist and
     silently corrupt the eval set. Open read-only:
     `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)`. A generated write
     then correctly raises → `ExecutionError` → scored wrong.
   - **`results_match` crashes on NULLs** — `sorted()` compares tuples element-wise, and
     `None < number` raises `TypeError`, which nothing catches (the `try/except ExecutionError`
     only wraps the *predicted* query). Spider result sets contain NULLs routinely. Replace with a
     multiset compare: `Counter(map(_normalize_row, predicted)) == Counter(map(_normalize_row, gold))`
     — order-insensitive, duplicate-sensitive, no element comparison.
   - Minor: `run_query` leaks the connection on a failed query (raises before `conn.close()` —
     use `try/finally`); gold execution is unguarded (a failing gold query kills the run).
   - Forward-looking: `load_model_for_eval` has **no 4-bit path**, so Week 3 eval of the
     fine-tuned Coder-7B won't fit in bf16 on the 8 GB 4060 — needs a `bnb` quantization branch.
   - Reminder: wiring `eval_hardness()` is a **two-place** change — compute difficulty *and*
     thread it into the `EvalResult(...)` construction in `runner.py` (currently omitted, so
     `difficulty` always defaults to `"unknown"`).
 
*(Done since last README: `tests/conftest.py` fixture extraction + `fresh_model` factory — see
Session 3 housekeeping note above.)*
 
# WHAT I WANT HELP WITH NEXT

Session 6 COMPLETE: five configs trained, base + five scored on Spider dev
(`results/eval/summary.md`). The fine-tuning gains are small (+1 to +2.6 pts) and not
significant after correction, against a base that is probably Spider-contaminated. Candidate
next steps (my decision):

1. Spider difficulty breakdown (vendor `process_sql.py` + `eval_hardness`, thread `difficulty`
   into `EvalResult`). It shows where the gains and losses sit.
2. BIRD-dev zero-shot, base vs best run: a less contaminated generalization test.
3. Seed repeats of qv_r16 (or qv_r8) to measure the seed-to-seed variance the ablation gaps are
   compared against.
4. Optional harness switch to column-permutation-invariant compare (test-suite convention).
   `summarize_eval.py` already reports it as a robustness column.
5. peft sanity comparison, then the Week 4 write-up.
