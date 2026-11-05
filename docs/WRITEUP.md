# MinLoRA-SQL: a peft-exact LoRA/QLoRA implementation, and a Spider fine-tune that does not transfer to BIRD

## Abstract

I implemented LoRA and its 4-bit QLoRA variant from scratch and tested them against Hugging Face
`peft` 0.19.1. In fp32 the forward logits and the adapter gradients match `peft` with a maximum
absolute difference of 0.0; on the 4-bit path they are bit-identical (asserted with
`torch.equal`), including at a non-power-of-two scaling that exposes cast-order bugs; and a
200-step training run matches `peft`'s loss at every step. With this implementation I fine-tuned
Qwen2.5-Coder-7B-Instruct for text-to-SQL on one 8 GB laptop GPU (five adapter configurations,
2.3–3.1 h each, peak memory ≤ 6.74 GB) and scored every model by executing its SQL. On Spider dev
the best adapters gain about 2.5 points over the untuned model (74.95% → 77.56%), which is not
significant after correcting for multiple comparisons. On BIRD dev, which the adapters never saw,
the same adapters lose 4.2 and 4.8 points (45.96% → 41.79% and 41.13%, Holm-adjusted p ≤ 1e-4).
Counting the predictions shows why: the adapters learned Spider's conventions, and on BIRD those
conventions are wrong.

## 1. Introduction

Most LoRA projects use `peft` and report a loss curve. This one asks two narrower questions and
tries to answer each with evidence that does not depend on judgment:

1. Can a from-scratch LoRA be shown to compute exactly what `peft` computes? The bar is identical
   numbers on both the fp32 and the 4-bit path, which two similar-looking loss curves cannot show.
2. Does fine-tuning make the model better at writing SQL? I measure this by running each generated
   query against the real database and comparing result rows, with paired significance tests, on
   the training benchmark and on a second benchmark the model was not trained on.

The contributions are:

- A LoRA/QLoRA implementation (`src/lora_sql/lora.py`, `qlora.py`) with a parity suite that
  compares it with `peft` exactly, and a description of the two test design choices without which
  such a suite passes vacuously (Section 4).
- A training loop that fine-tunes a 7.6B-parameter model in 4-bit on 8 GB, and a hardened
  execution-accuracy harness (read-only, time-limited, NULL-safe, held-out guard, provenance on
  every result) for Spider and BIRD (Sections 5–6).
- A rank and target-module ablation on Spider, with paired tests (Section 7.1–7.2).
- A measured negative transfer result: Spider fine-tuning lowers BIRD accuracy by 4–5 points, with
  counted evidence of the mechanism (Sections 7.3–7.4).

## 2. Background

**LoRA** (Hu et al., 2021) freezes a pretrained weight matrix `W` (shape `out × in`) and learns a
low-rank update: the layer computes `W x + (alpha / r) · B A x`, where `A` is `r × in`, `B` is
`out × r`, and `r` is much smaller than `in` and `out`. `B` starts at zero, so training starts from
the unmodified model. Only `A` and `B` are trained.

**QLoRA** (Dettmers et al., 2023) stores the frozen base in 4-bit NormalFloat (NF4), computes in
bf16, and keeps the LoRA matrices in higher precision. This is what makes a 7B model trainable on
an 8 GB GPU.

**Text-to-SQL** maps a question and a database schema to a SQL query. **Spider** (Yu et al., 2018)
is the standard cross-database benchmark: its dev set has 1,034 questions over 20 databases that do
not appear in training. **BIRD** (Li et al., 2023) is newer and harder: 1,534 dev questions over 11
larger, messier databases, each question paired with an expert hint ("evidence"). Both are scored
by **execution accuracy (EX)**: a prediction is correct if it returns the same rows as the gold
query. Spider's official metric executes on several database variants to cut false positives
(Zhong et al., 2020); this project executes on the released databases only (see Limitations).

## 3. Models and hardware

- The fine-tuning target is Qwen2.5-Coder-7B-Instruct (Hui et al., 2024), 28 layers, hidden size
  3,584, grouped-query attention with 28 query and 4 key/value heads.
- The parity fixture is Qwen2.5-0.5B-Instruct. It has the same module names as the 7B
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), so target-module
  strings carry over unchanged, and it is small enough to run in fp32. It is never fine-tuned.
- The hardware is one RTX 4060 Laptop GPU with 8.2 GB of memory. Software: torch 2.5.1, transformers
  5.14.1, peft 0.19.1 (reference only), bitsandbytes 0.49.2.

## 4. Implementation and parity verification

### 4.1 The layer

`LoRALinear` wraps an existing linear layer (a plain `nn.Linear` or a bitsandbytes `Linear4bit`),
freezes it, and adds `A` and `B` in fp32. `inject_lora` replaces the named target modules
throughout the model and freezes everything else. Two details matter on the 4-bit path:

- Layer dimensions come from `in_features` / `out_features`, never from the weight's shape.
  Under `Linear4bit` the stored weight is packed and its shape is wrong.
- The order of dtype casts mirrors `peft`'s `Linear4bit.forward` exactly. The input is cast to
  the adapter dtype (fp32) before dropout; `B(A(dropout(x))) · scaling` is computed in fp32; the
  result is cast to the base output dtype (bf16) and then added. The base parameters are never
  upcast and `torch.autocast` is not used.

### 4.2 How the parity tests avoid passing vacuously

Two design choices turned out to be necessary.

**Randomise `B` before comparing.** Both implementations initialise `B` to zero, so a freshly
built adapter is a no-op and both models output exactly the base model's logits. A parity test
that compares them at initialisation compares the base model with itself and passes for almost any
bug; with `B = 0` the gradient of `A` is also zero. The tests randomise `peft`'s `B`, copy both
matrices into the from-scratch layers, and only then compare.

**Test a scaling that is not a power of two.** Every training configuration uses `alpha = 2r`, a
scaling of 2.0. Multiplying by 2.0 is exact in every floating-point format, so it commutes with
dtype casts and hides errors in cast order. With `alpha = 12, r = 8` (scaling 1.5) an earlier
version of the layer, which applied the scaling after the downcast, differed from `peft` by 0.32
in the logits, while still passing every scaling-2.0 test. The 4-bit suite keeps a scaling-1.5
case permanently for this reason.

**Run parity in fp32.** bf16 carries about three significant decimal digits, so two correct
implementations that sum in a different order can disagree at the 1e-2 level, and a tolerance of
1e-6 is unreachable. The fp32 suite therefore runs on the 0.5B model in fp32 (about 2 GB); the
4-bit suite runs the 0.5B in NF4 with bf16 compute, as in training.

### 4.3 Results

| Check | Setting | Result |
|---|---|---|
| Forward and backward | 0.5B, fp32; r = 4, 8, 16 on q/v, and r = 8 on all seven modules | max abs difference 0.0 in logits and in every `lora_A` / `lora_B` gradient (asserted at atol 1e-6) |
| 4-bit (QLoRA path) | 0.5B in NF4, bf16 compute, fp32 adapters; r = 8 on q/v at scaling 2.0 and 1.5, and on all seven modules | bit-identical logits and gradients (`torch.equal`) |
| Training trajectory | 0.5B, r = 8 on q/v, 200 optimizer steps, lr 2e-4, same init, data and seed | max per-step loss difference 0.000000 |

Sources: `tests/test_lora_parity.py`, `tests/test_lora_4bit.py`, `results/toy_loss_*.json`.

![Figure 1](../results/toy_loss_curves.png)

*Figure 1. 200-step LoRA training on Qwen2.5-0.5B, from-scratch vs `peft`, identical
initialisation, data and seed: the maximum per-step loss difference is 0.000000. The loss falls
from 2.07 to 0.14 and the two curves coincide at every step.*

### 4.4 Checkpoints

The most dangerous failure in this kind of pipeline is silent: `load_state_dict(strict=False)`
with mismatched key names loads nothing, raises nothing, and the "fine-tuned" evaluation is really
the base model. Adapters are therefore saved as `lora_` keys only, with a JSON sidecar that records
the targets, rank, alpha and dtype read from the modules themselves, plus the config hash and git
commit. Loading rebuilds the adapter from the sidecar and asserts that there are no unexpected
keys, that every saved tensor was consumed and is `torch.equal` to the file, and that at least one
`B` is nonzero.

## 5. Training setup

**Data.** Spider `train_spider.json` and `train_others.json`, 8,659 examples. Spider dev is used
only for evaluation and the Spider test split is never read. A memory probe on the largest
configuration (`results/probe_all7_r8.json`) fits sequences of 2,048 tokens but not 2,560; at 2,048
exactly the 82 examples of one database (`baseball_1`) are dropped, leaving 8,577
(`results/train/*/data_report.json`).

**Prompt.** The schema is rendered as `CREATE TABLE` statements with column types, primary keys
and foreign keys, followed by the question. One function builds the ChatML token ids (with a fixed
system prompt) for both training and evaluation, so evaluation uses exactly the training format.

**Objective.** Loss on the SQL completion only (the vocabulary projection is computed only at
completion positions), micro-batch 1 with 16-step gradient accumulation, weighted by token count
so each update equals that of a true batch of 16.

**Hyperparameters** (shared by all runs): AdamW, learning rate 2e-4, 3% linear warmup then cosine
decay to 0, weight decay 0, gradient clipping 1.0, LoRA dropout 0.05, `alpha = 2r`, one epoch =
536 optimizer steps, seed 0. NF4 with double quantisation, bf16 compute, gradient checkpointing.
All five configurations use identical data settings and data order (a test enforces this) and
differ only in LoRA rank and targets.

| Run | Targets | Rank | LoRA params | Train time | Peak reserved memory | Loss (mean of last 10 logged) |
|---|---|---|---|---|---|---|
| `qv_r4` | q, v | 4 | 1.26M | 2.27 h | 6.21 GB | 0.090 |
| `qv_r8` | q, v | 8 | 2.52M | 2.29 h | 6.23 GB | 0.087 |
| `qv_r16` | q, v | 16 | 5.05M | 2.25 h | 6.26 GB | 0.084 |
| `attn_r8` | q, k, v, o | 8 | 5.05M | 2.31 h | 6.27 GB | 0.080 |
| `all7_r8` | all seven | 8 | 20.19M | 3.06 h | 6.74 GB | 0.065 |

Source: `results/eval/summary.md` (from `results/train/<run>/`).

![Figure 2](../results/train_loss.png)

*Figure 2. Training loss (rolling mean over logged windows) and learning rate for the five runs.*

## 6. Evaluation harness

**Generation.** Greedy decoding with the training prompt; up to 256 new tokens on Spider and 640
on BIRD (its longest gold query is 541 tokens).

**Execution.** Each prediction runs on a read-only SQLite connection (a generated `DROP TABLE`
fails instead of damaging the database) with a 30-second wall-clock limit enforced through
SQLite's progress handler. Every gold query is executed before the model loads, so a broken path
or gold query fails in seconds.

**Comparison.** Result rows are compared as multisets (duplicate-sensitive and NULL-safe; an early
version sorted rows and would have crashed on the 9 Spider dev gold results that contain NULL),
with floats rounded to six places. Row order matters only when the gold query has `ORDER BY`.
Columns are compared by position, which is stricter than Spider's test-suite evaluation. As a
robustness check, `scripts/summarize_eval.py` re-scores every prediction allowing any column
order.

**Guards and provenance.** The runner refuses Spider's test split and test databases, and allows
each benchmark to execute only against its own database directory. Every result file records the
adapter's sidecar (targets, rank, training step, config hash) and the evaluation commit, and the
summary scripts assert that each evaluated checkpoint is the one whose training log is committed.

**Statistics.** All models answer the same questions, so comparisons are paired: an exact
two-sided McNemar test on the questions where two models disagree, Holm correction across the
comparisons with the base model, and a 95% paired bootstrap confidence interval (5,000 resamples)
on the accuracy difference.

**BIRD.** Four adaptations were needed. BIRD lists composite primary keys as nested lists and uses
type names (`integer`, `real`, `date`, `datetime`) that Spider does not; the schema renderer handles
both without changing Spider's output (golden-string tests). Each question's evidence is appended
to the question text as `Hint: ...`, the standard BIRD setting, so the prompt template is
unchanged. Two gold queries take 39 s and 188 s, so gold queries get up to 300 s while predictions
keep 30 s. And BIRD's official rule, `set(pred) == set(gold)`, which ignores row order and
duplicates, is reported alongside the strict score.

## 7. Results

### 7.1 Spider dev

| Model | Dev EX (%) | Δ vs base | 95% CI | Won / lost vs base | McNemar p (Holm) |
|---|---|---|---|---|---|
| Base | 74.95 | | | | |
| `qv_r4` | 75.92 | +0.97 | [−1.26, +3.19] | +76 / −66 | 0.450 (0.900) |
| `qv_r8` | 75.92 | +0.97 | [−1.35, +3.19] | +77 / −67 | 0.453 (0.900) |
| `qv_r16` | 77.47 | +2.51 | [+0.19, +4.74] | +85 / −59 | 0.037 (0.164) |
| `attn_r8` | 76.79 | +1.84 | [−0.39, +3.97] | +74 / −55 | 0.113 (0.338) |
| `all7_r8` | 77.56 | +2.61 | [+0.29, +4.93] | +88 / −61 | 0.033 (0.164) |

Source: `results/eval/summary.md`.

Every adapter scores above the base, by 1 to 2.6 points. Two are nominally significant
(p ≈ 0.035) and none survives Holm correction. Each adapter fixes 74–88 questions the base got
wrong and breaks 55–67 it got right, so the net gain is a small difference between two large
numbers. Allowing any column order raises every model by 2.3–3.2 points and the base the most
(74.95 → 78.14), so the strict comparison, if anything, favours the adapters.

By Spider's official difficulty tiers (computed with Spider's own `eval_hardness`, which
reproduces the official dev split of 248 / 446 / 174 / 166), the gain sits mostly on medium
questions: every adapter nets +10 to +20 questions there and between −5 and +9 on the other three
tiers combined. Easy questions are near the ceiling (base 90.3%).

### 7.2 Ablations

- *Rank on q/v.* r = 4 and r = 8 are tied (785 correct each; 22 won, 22 lost against each other).
  r = 16 beats r = 8 on 24 questions against 8 (unadjusted p = 0.007).
- *Target modules.* At r = 8, adding k/o (`attn_r8`) and the MLP (`all7_r8`) scores above q/v
  alone (+35/−26 and +50/−33), but neither difference is significant (p = 0.31 and 0.08).
  `all7_r8` reaches by far the lowest training loss (0.065 against 0.080–0.090) with four times the
  parameters of `qv_r16`, and the two tie on dev (40 won, 39 lost; p = 1.0).

Lower training loss did not mean higher accuracy on Spider, and on BIRD (Section 7.3) both the
lowest-loss adapter and the smaller one fall below the base.

### 7.3 BIRD dev, zero-shot

The two best Spider adapters were scored on BIRD dev against the base model.

| Model | BIRD EX strict (%) | Δ vs base | 95% CI | Won / lost vs base | McNemar p (Holm) | BIRD EX official (%) |
|---|---|---|---|---|---|---|
| Base | 45.96 | | | | | 50.39 |
| `qv_r16` | 41.13 | −4.82 | [−6.98, −2.74] | +99 / −173 | 8.5e-6 (1.7e-5) | 45.37 |
| `all7_r8` | 41.79 | −4.17 | [−6.19, −2.02] | +101 / −165 | 1.0e-4 (1.0e-4) | 46.22 |

Source: `results/eval/bird/summary.md`.

Both adapters are significantly *worse* than the model they started from, under either scoring
rule. The drop is broad: it appears on all three BIRD difficulty levels (largest on challenging:
−9.0 and −6.2 points) and on 9 and 8 of the 11 databases; both adapters improve only on
`student_club` and `thrombosis_prediction`. The two adapters tie with each other again (+91/−81,
p = 0.49).

![Figure 3](../results/spider_vs_bird.png)

*Figure 3. Change in execution accuracy against the untuned base, with 95% paired bootstrap
intervals. On Spider (left) every adapter is above the base and the intervals of the two best
barely clear zero. On BIRD (right) both adapters evaluated there are well below the base, with
intervals entirely below zero.*

### 7.4 What changed

Most of the questions the adapters lose on BIRD (71% for `qv_r16`, 68% for `all7_r8`) are not
crashes: the SQL runs and returns the wrong rows. Execution errors rise only from 189 to 208 and
211. To see what changed, `scripts/summarize_bird.py` counts features of every prediction:

| Measure | BIRD gold | Base | `qv_r16` | `all7_r8` |
|---|---|---|---|---|
| Spider gold spacing (`a  =  b`, `x ,  y`), % | 0.0 | 0.6 | 74.1 | 77.2 |
| String literal in double quotes, % | 0.0 | 0.2 | 51.1 | 54.4 |
| Uses `YEAR(...)` when the hint itself writes `YEAR(` (n = 78), % | 0.0 | 5.1 | 28.2 | 28.2 |
| Uses SQLite's `STRFTIME(...)` on those same 78, % | 87.2 | 85.9 | 50.0 | 44.9 |
| Uses `CAST(...)`, % | 9.9 | 7.5 | 7.6 | 7.0 |
| Execution errors: `no such column` | | 171 | 165 | 173 |
| Execution errors: `no such function` | | 5 | 25 | 21 |

Source: `results/eval/bird/summary.md`.

**The Spider style is imprinted.** Spider's gold queries are written with doubled spaces around
operators and double-quoted string values; three quarters of the adapters' BIRD predictions now
look like that, against almost none of the base model's. The formatting is harmless by itself, but
it shows how completely 536 steps on Spider reshaped the output.

**Literals are copied as written.** Spider's gold queries tend to copy values exactly as the
question words them, and in Section 7.1 this was one of the habits the adapters gained. BIRD's
databases store values in their own formats. Examples from the result files:

- Question 1177 asks about a lab test "on 1995-9-4". The base writes `Date = '1995-09-04'`, the
  stored format. `qv_r16` writes `Date = "1995-9-4"` and matches nothing.
- Question 1523 asks about "card owner of No.667467". The base writes `CardID = 667467`;
  `qv_r16` writes `CardID = "No.667467"`.
- Question 815 counts "good female superheroes". The base follows the hint's ids; `qv_r16` writes
  `gender = "female" AND alignment = "good"`, but the stored values are capitalised.

**Hints are copied instead of translated.** BIRD's evidence is often written in pseudo-SQL, such as
`YEAR(CreationDate) = 2011`. SQLite has no `YEAR` function. On the 78 questions whose hint does
this, the base model translates it to `STRFTIME('%Y', ...)` 86% of the time, as the gold queries do,
and copies `YEAR(` 5% of the time. The adapters translate only 45–50% of the time and copy 28% of
the time, and those queries fail with `no such function: YEAR` (Question 662 is lost this way by
both adapters).

**What did not change.** `CAST` usage and `no such column` errors are flat, so the drop is not a
general loss of SQL competence or a sudden inability to read the schema. A hand-read random sample
of 15 lost questions per adapter adds two further patterns that are harder to count: hint
instructions that are partly ignored (a requested `CASE`, a symmetric `OR` condition, a
`SUM ... GROUP BY`), and output columns pared down or swapped in Spider's style.

## 8. Discussion

**The two results are one result.** On Spider the adapters gain by adopting the benchmark's
conventions: `INTERSECT` for "both X and Y", exact equality instead of `LIKE` patterns, literals
copied as written, fewer unnecessary joins (from reading a sample of won and lost Spider
questions). On BIRD the same conventions are wrong: literal formats differ, hints must be
translated into SQLite, and schemas are larger. Fine-tuning on 8,577 Spider examples mostly
taught the model how Spider's answer key is written.

**The base model probably knew Spider already.** Untuned, Qwen2.5-Coder reproduces quirks of
Spider's gold queries, such as `Country ,  COUNT(*)` with the doubled space. Spider has been public
since 2018 and is very likely in the pretraining data. That leaves little real headroom on Spider
and explains why the in-domain gain is small: much of the task was already learned, and what
remained to learn was style.

**Training loss is not the target.** The configuration with the lowest training loss
(`all7_r8`, 0.065) ties a configuration with a quarter of its parameters on Spider, and on BIRD the
two lose about the same amount (−4.2 and −4.8 points; head to head p = 0.49). More capacity fitted Spider more closely; it did not generalise better.

**What would test this further** (not done here):

- *Scaling the adapter down at inference* (for example half its `alpha / r`) on both benchmarks
  would show whether the trade-off is continuous: whether BIRD recovers faster than Spider falls.
- *Mixing training data* from more than one benchmark, or regularising toward the base model,
  would test whether the convention imprint can be avoided rather than only measured.
- *Seed repeats* would bound the training noise on the Spider ablation gaps. They are unlikely to
  change the BIRD conclusion: a 4–5 point drop with p < 0.001 is far larger than the 1–2 point gaps
  between Spider configurations.

## 9. Limitations

- One seed per configuration: differences between Spider configurations include unmeasured
  seed variance.
- One model: all results are for Qwen2.5-Coder-7B-Instruct with one prompt format.
- Execution on the released databases only. Spider's test-suite metric runs each query on several
  database variants; these Spider numbers are therefore not directly comparable with published
  test-suite results. The BIRD official-rule column is computed with BIRD's own comparison.
- Strict comparison. The harness compares columns by position. Allowing any column order
  raises Spider scores by 2.3–3.2 points without changing the conclusions.
- BIRD may not be clean. BIRD (2023) predates Qwen2.5-Coder (2024), so it is less likely than
  Spider to be in the pretraining data, not certainly absent.
- One training database is excluded. The 82 `baseball_1` examples exceed the 2,048-token limit
  the largest configuration fits in 8 GB.

## 10. Reproducibility

Everything needed to regenerate the numbers above is in the repository. Per-question results are
committed (`results/eval/*.jsonl`, `results/eval/bird/*.jsonl`); the summary tables and Figure 3
are generated from them on CPU.

```bash
pip install -r requirements.txt
pytest -v                                      # 106 tests; parity and runner tests need a GPU
python -m lora_sql.train --config configs/train/qv_r8.yaml
bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
bash scripts/eval_chain.sh --config configs/eval/coder7b_bird_dev.yaml --out-dir results/eval/bird \
    base qv_r16 all7_r8
python scripts/summarize_eval.py               # -> results/eval/summary.md
python scripts/summarize_bird.py               # -> results/eval/bird/summary.md
python scripts/plot_results.py                 # -> results/spider_vs_bird.png
```

The Spider evaluations ran at commit `c1b3174` and the BIRD evaluations at `5d959f0`; each result
file records its commit and the adapter sidecar. On the RTX 4060 Laptop GPU, a training run takes
2.3–3.1 h, a Spider dev evaluation 20–28 min, and a BIRD dev evaluation about an hour.

## References

- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021.
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* 2023.
- Yu et al. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic
  Parsing and Text-to-SQL Task.* 2018.
- Zhong et al. *Semantic Evaluation for Text-to-SQL with Distilled Test Suites.* 2020.
- Li et al. *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database
  Grounded Text-to-SQLs* (BIRD). 2023.
- Hui et al. *Qwen2.5-Coder Technical Report.* 2024.
- Hugging Face [`peft`](https://github.com/huggingface/peft) and
  [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes).

## Appendix: hyperparameters

| Setting | Value |
|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct, NF4 with double quantisation, bf16 compute |
| Adapter dtype | fp32 |
| LoRA alpha | 2 × rank (scaling 2.0) |
| LoRA dropout | 0.05 |
| Optimiser | AdamW, lr 2e-4, weight decay 0 |
| Schedule | 3% linear warmup, cosine decay to 0 |
| Gradient clipping | 1.0 |
| Batch | micro-batch 1 × 16 accumulation steps, token-weighted |
| Epochs / steps | 1 / 536 |
| Max sequence length | 2,048 tokens |
| Gradient checkpointing | on |
| Seed | 0 (shared data order across runs) |
| Training data | Spider `train_spider` + `train_others`: 8,659 loaded, 8,577 used |
| Decoding | greedy; 256 new tokens (Spider), 640 (BIRD) |
| Query time limit | 30 s for predictions; 300 s for BIRD gold queries |

Source: `configs/train/*.yaml`, `configs/eval/*.yaml`, `results/train/*/data_report.json`.
