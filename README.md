# MinLoRA-SQL

A from-scratch implementation of LoRA and QLoRA, verified bit-for-bit against Hugging Face `peft`,
used to fine-tune Qwen2.5-Coder-7B-Instruct for text-to-SQL on a single 8 GB laptop GPU and
evaluated by execution accuracy on Spider, plus zero-shot on BIRD.

The full write-up, with method, results and discussion, is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

## Summary

- **Numerical parity with `peft`.** The LoRA layer, injection and training path are written from
  scratch and match `peft` 0.19.1 exactly. Forward outputs and gradients are bit-identical in fp32,
  bit-identical on the 4-bit QLoRA path (including a non-power-of-two scaling case that exposes
  cast-order bugs), and a 200-step training run matches step for step.
- **QLoRA on consumer hardware.** A custom training loop fine-tunes a 7.6B-parameter model on an
  RTX 4060 Laptop GPU (8 GB). Five adapter configurations, varying rank and target modules, train
  in 2.3 to 3.1 hours each with peak memory at or below 6.74 GB.
- **Execution-based evaluation.** Generated SQL is run against the Spider databases and its result
  rows are compared with the gold query's. The harness is read-only, time-limited and NULL-safe. It
  refuses held-out data and records the exact checkpoint and commit behind every number.
- **Result: an in-domain gain that does not transfer.**
  - On Spider dev, fine-tuning raises execution accuracy from 75.92% (base) to 79.88% (best
    configuration). The gains of two of the five configurations stay significant after correcting
    for multiple comparisons. The base model has probably already seen Spider.
  - On BIRD dev, which none of the adapters were trained on, the three adapters tested score
    **3.3 to 4.5 points below the base model** (46.68% → 42.18–43.35%, each p ≤ 0.002 after
    correction). The adapters learned Spider's conventions, and those conventions hurt on a
    different benchmark.
- **A decoding default that moved the numbers.** Qwen's `generation_config.json` sets
  `repetition_penalty: 1.1`, and `generate()` applies it even with `do_sample=False`. The first
  round of evaluations inherited it silently. Turning it off changed 40–66% of the predictions and
  raised the fine-tuned models' Spider accuracy by 0.7 to 3.2 points. Results under both settings
  are committed, and the [comparison](#decoding-the-repetition-penalty-ablation) is below.

## Results

### Spider dev execution accuracy

All six models answer the same 1,034 dev questions with greedy decoding (repetition penalty off).

| Model | Target modules | Rank | LoRA params | Dev EX (%) | Δ vs base | Won / lost vs base | McNemar p (Holm) |
|---|---|---|---|---|---|---|---|
| Base (no adapter) | none | | 0 | 75.92 | | | |
| `qv_r4` | q, v | 4 | 1.26M | 78.14 | +2.22 | +74 / −51 | 0.049 (0.125) |
| `qv_r8` | q, v | 8 | 2.52M | 79.11 | +3.19 | +79 / −46 | 0.004 (0.016) |
| `qv_r16` | q, v | 16 | 5.05M | 79.88 | +3.97 | +84 / −43 | 3.5e-04 (0.002) |
| `attn_r8` | q, k, v, o | 8 | 5.05M | 78.24 | +2.32 | +76 / −52 | 0.042 (0.125) |
| `all7_r8` | q, k, v, o, gate, up, down | 8 | 20.19M | 78.24 | +2.32 | +82 / −58 | 0.052 (0.125) |

"Won / lost" counts the questions a fine-tuned model answers correctly that the base model does
not, and the reverse. p-values are exact two-sided McNemar tests on those discordant pairs, with
Holm-adjusted values in parentheses for the five comparisons against the base. Bootstrap confidence
intervals (by question and by database), pairwise comparisons between configurations, a
column-order robustness check and style diagnostics are in
[`results/eval/summary.md`](results/eval/summary.md), which is generated from the committed
per-question results by `scripts/summarize_eval.py`.

### Accuracy by difficulty

Spider assigns each question one of four official difficulty tiers by counting the SQL components
of its gold query. Here the tiers come from the official `eval_hardness` function, vendored from
[taoyds/spider](https://github.com/taoyds/spider). Cells show dev EX (%) with the change against
the base model in parentheses.

| Model | Easy (248) | Medium (446) | Hard (174) | Extra (166) | All (1,034) |
|---|---|---|---|---|---|
| Base (no adapter) | 91.53 | 76.23 | 71.84 | 56.02 | 75.92 |
| `qv_r4` | 91.13 (−0.40) | 80.94 (+4.71) | 72.99 (+1.15) | 56.63 (+0.60) | 78.14 (+2.22) |
| `qv_r8` | 90.32 (−1.21) | 81.84 (+5.61) | 75.29 (+3.45) | 59.04 (+3.01) | 79.11 (+3.19) |
| `qv_r16` | 91.94 (+0.40) | 82.29 (+6.05) | 76.44 (+4.60) | 59.04 (+3.01) | 79.88 (+3.97) |
| `attn_r8` | 91.13 (−0.40) | 79.82 (+3.59) | 75.86 (+4.02) | 57.23 (+1.20) | 78.24 (+2.32) |
| `all7_r8` | 91.53 (+0.00) | 80.49 (+4.26) | 75.29 (+3.45) | 55.42 (−0.60) | 78.24 (+2.32) |

In the smallest tier one question is worth 0.6 points, so the per-tier differences are descriptive
and are not tested for significance. Per-tier won / lost counts are in
[`results/eval/summary.md`](results/eval/summary.md).

### Findings

- **Every configuration improves on the base, by 2.2 to 4.0 points.** The gains of `qv_r16`
  (+3.97) and `qv_r8` (+3.19) stay significant after Holm correction. The other three are at the
  edge of nominal significance (p between 0.04 and 0.06) and do not survive it.
- **The questions are not independent.** They come from 20 databases. Resampling whole databases
  instead of single questions widens every interval, and only `qv_r16`'s then stays above zero
  ([+0.82, +7.56]).
- **Fine-tuning reshuffles answers more than it adds them.** Each model fixes 74 to 84 questions
  the base got wrong and breaks 43 to 58 it got right.
- **The gain sits mostly in medium questions.** Counting questions won minus lost against the
  base, every configuration gains 16 to 27 on medium and 2 to 14 on the other three tiers
  combined. Extra-hard questions change the most (22–26% flip in one direction or the other) but
  show almost no net gain. Easy questions are near the ceiling already (91.5% for the base).
- **Rank.** On the query and value projections, accuracy rises with rank: 78.14, 79.11 and 79.88
  for ranks 4, 8 and 16. Rank 16 beats rank 4 on 30 questions against 12 (unadjusted p = 0.008).
  The two single steps (4 to 8, 8 to 16) are not significant on their own.
- **Target modules.** At rank 8, adapting more projections does not help. `attn_r8` and `all7_r8`
  both score 78.24, below `qv_r8` (not significantly). `all7_r8` reaches the lowest training loss
  by a wide margin (0.065 against roughly 0.085) and still scores below `qv_r16`, which has a
  quarter of the parameters (20 questions won, 37 lost, unadjusted p = 0.033).
- **What changes.** The fine-tuned models write Spider's conventions:
  - Spider's gold spacing appears in 77% of their predictions, against 3% of the base's.
  - `INTERSECT` for "both X and Y" doubles, from 2.1% of predictions to about 4%, the gold's rate.
  - `no such column` errors halve, from 37 for the base to 16–25.
  - Columns come out in the gold's order. Of the 74 to 84 questions a model wins, 15 to 19 are ones
    where the base returned the right rows with its columns in another order.
  - Most losses (81–89%) are queries that run and return the wrong rows, typically an extra or
    wrong join.
- **Likely contamination.** Without any fine-tuning, the base model reproduces formatting quirks
  of Spider's gold queries (for example `Country ,  COUNT(*)` with a doubled space) in 31 of its
  1,034 predictions. That suggests Spider was in its pretraining data, which limits the headroom
  fine-tuning can show.
- **Empty results.** 53 of the 1,034 gold queries return no rows, so any query whose filter
  matches nothing scores as correct there. Leaving those questions out lowers the gains by 0.2
  to 0.5 points (+1.83 to +3.77 on the other 981).

### Out of domain: BIRD dev (zero-shot)

Three adapters were scored on BIRD dev (1,534 questions, 11 databases, release `dev_20240627`):
`qv_r16` and `qv_r8`, the two best on Spider, and `all7_r8`, the largest. BIRD is a newer and harder benchmark, and it was never trained on. Each question's
expert hint ("evidence") is added to the prompt, as in the standard BIRD setting. "Strict" is this
project's harness, the same rule as the Spider table. "Official" is BIRD's own rule
(`set(pred) == set(gold)`, which ignores row order and duplicates).

| Model | BIRD EX strict (%) | Δ vs base | 95% CI | Won / lost vs base | McNemar p (Holm) | BIRD EX official (%) |
|---|---|---|---|---|---|---|
| Base (no adapter) | 46.68 | | | | | 51.37 |
| `qv_r8` | 43.35 | −3.32 | [−5.48, −1.24] | +109 / −160 | 0.002 (0.002) | 48.70 |
| `qv_r16` | 42.89 | −3.78 | [−5.93, −1.76] | +105 / −163 | 4.8e-04 (9.5e-04) | 48.24 |
| `all7_r8` | 42.18 | −4.50 | [−6.52, −2.48] | +91 / −160 | 1.6e-05 (4.7e-05) | 47.33 |

- **Fine-tuning on Spider hurts on BIRD, significantly.**
  - All three adapters lose about 160 questions the base answers correctly and gain 91 to 109.
  - The drop is largest on the challenging questions (−8.3 to −9.0 points). `qv_r8` is level
    with the base on moderate questions; every other level drops for every adapter.
  - By database, 6 of the 11 drop for `qv_r8` and for `qv_r16`, and 9 for `all7_r8`.
  - Most lost answers are still valid SQL that returns the wrong rows (72–75%), not errors.
  - Resampling whole databases, the interval for `all7_r8` stays below zero ([−7.20, −1.92]). The
    ones for `qv_r16` ([−7.92, +0.18]) and `qv_r8` ([−7.18, +0.57]) reach it.
  - The three adapters do not differ significantly from each other (pairwise p 0.22 to 0.54).
- **The Spider style carries over.**
  - Spider's double-spaced gold formatting appears in 78–81% of the adapters' BIRD predictions,
    against under 1% for the base. Double-quoted string literals rise from 0% to 35–39%.
  - Literal values are copied as the question writes them (`"1995-9-4"` for a stored
    `1995-09-04`, `"president"` for `'President'`).
  - The hints' pseudo-SQL is copied verbatim: when a hint writes `YEAR(...)`, the adapters use
    that function, which SQLite lacks, in 24–28% of cases against 5% for the base.
- **What this means.** The Spider gain is mostly learned benchmark conventions, not a general
  improvement in writing SQL. Full tables (by difficulty, by database, style diagnostics) are in
  [`results/eval/bird/summary.md`](results/eval/bird/summary.md), generated by
  `scripts/summarize_bird.py`.

![Change in execution accuracy against the base model on Spider dev and BIRD dev](results/spider_vs_bird.png)

### Decoding: the repetition-penalty ablation

The first round of evaluations (Sessions 6 and 8 of the project log) called `generate()` with
`do_sample=False` and nothing else. Qwen's `generation_config.json` sets `repetition_penalty: 1.1`,
and `transformers` applies it under greedy search too, over the prompt as well as the output. So
every token of the schema was penalised when the model tried to write it. The evaluation config now
requires `eval.repetition_penalty`, the runner passes it explicitly and records it with each
result, and a test checks the output against a hand-written argmax loop.

The earlier results are kept unchanged in `results/eval/rp1.1/` and `results/eval/bird/rp1.1/`.
Same checkpoints, same questions, execution accuracy (%):

| Model | Spider, penalty 1.1 | Spider, off | Δ | BIRD, penalty 1.1 | BIRD, off | Δ |
|---|---|---|---|---|---|---|
| Base (no adapter) | 74.95 | 75.92 | +0.97 | 45.96 | 46.68 | +0.72 |
| `qv_r4` | 75.92 | 78.14 | +2.22 | | | |
| `qv_r8` | 75.92 | 79.11 | +3.19 | not run | 43.35 | |
| `qv_r16` | 77.47 | 79.88 | +2.42 | 41.13 | 42.89 | +1.76 |
| `attn_r8` | 76.79 | 78.24 | +1.45 | | | |
| `all7_r8` | 77.56 | 78.24 | +0.68 | 41.79 | 42.18 | +0.39 |

- **It changed a lot of text.** 40–42% of the Spider predictions and 47–66% of the BIRD
  predictions differ between the two settings.
- **It cost the fine-tuned models more than the base.** On Spider the base moves by 1.0 point and
  the QV adapters by 2.2 to 3.2 (each p ≤ 0.003). With the penalty on, the fine-tuned models made
  32 to 41 `no such column` errors; with it off, 16 to 25. The base stays at 37.
- **It changed one conclusion and left the other.** With the penalty, no Spider gain survived Holm
  correction and `all7_r8` looked like the best configuration. Without it, two gains survive and
  the rank-16 QV adapter leads. The BIRD drop is there under both settings (−4.2 to −4.8 points
  with the penalty, −3.8 to −4.5 without, for the two adapters scored both ways).

### Training

| Run | Steps | Train time | Peak reserved memory | Final loss (mean of last 10 logged) |
|---|---|---|---|---|
| `qv_r4` | 536 | 2.27 h | 6.21 GB | 0.090 |
| `qv_r8` | 536 | 2.29 h | 6.23 GB | 0.087 |
| `qv_r16` | 536 | 2.25 h | 6.26 GB | 0.084 |
| `attn_r8` | 536 | 2.31 h | 6.27 GB | 0.080 |
| `all7_r8` | 536 | 3.06 h | 6.74 GB | 0.065 |

![Training loss and learning rate for the five runs](results/train_loss.png)

All runs see the same examples in the same order, so their loss spikes occur at the same steps.
Per-run logs, data reports and adapter metadata are in [`results/train/`](results/train/).

## Verification against `peft`

The central engineering claim is that the from-scratch implementation computes exactly what `peft`
computes. The test suite checks this directly rather than by comparing loss curves by eye.

| Check | Setting | Result |
|---|---|---|
| Forward and backward parity | Qwen2.5-0.5B-Instruct, fp32, ranks 4, 8 and 16, and all seven target modules | max abs difference 0.0 in logits and in every `lora_A` and `lora_B` gradient (tolerance 1e-6) |
| QLoRA-path parity | 0.5B base in 4-bit NF4 with bf16 compute and fp32 adapters, including scaling 1.5 | max abs difference 0.0 in logits and gradients |
| Training trajectory | 200 optimizer steps, identical initialisation, data and seed | max per-step loss difference 0.000000 |

Two details make these tests meaningful:

- **Randomised `lora_B`.** LoRA initialises `B` to zero, so a freshly initialised adapter is a
  no-op in both implementations, and a naive parity test compares the base model with itself.
  The tests randomise `peft`'s `B`, transplant the weights, and only then compare.
- **Non-power-of-two scaling.** Every training configuration uses alpha = 2r, a scaling of 2.0.
  Multiplication by 2.0 is exact in every floating-point format, so it hides errors in the order
  of dtype casts. A scaling of 1.5 exposed such an error during development (a logit difference of
  0.32), which was fixed by mirroring `peft`'s cast order exactly.

![Toy training run: from-scratch LoRA against peft](results/toy_loss_curves.png)

## Method

**Models.** Qwen2.5-Coder-7B-Instruct is the fine-tuning target. Qwen2.5-0.5B-Instruct shares its
module names and serves only as an fp32 test fixture for the parity suite; it is never fine-tuned.

**Data.** Training uses Spider `train_spider.json` and `train_others.json` (8,659 examples).
Evaluation uses `dev.json` (1,034 examples over 20 databases not seen in training). The Spider
test split is never read. Examples longer than 2,048 tokens are dropped, which removes exactly the
82 examples of one database (`baseball_1`) and leaves 8,577.

**Prompt.** Each database schema is rendered as `CREATE TABLE` statements with column types,
primary keys and foreign keys, followed by the question. A single function builds the ChatML token
ids for both training and evaluation, so the model is evaluated on exactly the format it was
trained on.

**Training.**
- Base: 4-bit NF4 with double quantisation, bf16 compute, gradient checkpointing.
- Adapters: kept in fp32 so that small AdamW updates are not rounded away. Base parameters are
  never upcast, and `torch.autocast` is not used.
- Batching: micro-batch 1 with 16-step gradient accumulation, weighted by token count so that the
  update equals that of a true batch of 16.
- Loss: computed on the SQL completion only, with the vocabulary projection evaluated only for
  completion positions.
- Optimiser: AdamW, learning rate 2e-4, 3% linear warmup and cosine decay, gradient clipping at
  1.0, LoRA dropout 0.05.
- Schedule: one epoch, 536 optimizer steps. All five configurations share the data settings and
  data order (a test enforces this) and differ only in their LoRA settings.

**Checkpoints.** Only adapter weights are saved (safetensors), with a JSON sidecar describing the
adapter and its provenance. Loading rebuilds the adapter from the sidecar and then asserts that
every saved tensor was consumed and matches exactly, and that the adapter is not a no-op. A
`load_state_dict(strict=False)` that silently loads nothing would otherwise evaluate the base model
under a fine-tuned label.

**Evaluation.**
- Decoding: greedy, with the repetition penalty set to 1.0 explicitly, and up to 256 new tokens
  (640 on BIRD).
- Execution: each prediction runs on a read-only connection to its Spider database with a 30-second
  limit. Every gold query is executed before the model loads.
- Comparison: results are compared as multisets of rows, which is NULL-safe and duplicate-sensitive,
  with floats rounded to six places. Row order is compared only when the gold query contains
  `ORDER BY`.
- Columns are compared by position. That is stricter than the official test-suite evaluation, which
  allows any column order. Rescoring with column permutations allowed raises every model by 2.3 to
  3.8 points and the base model the most. Every fine-tuned model still scores above the base
  (by 0.9 to 2.9 points), so the strict gain is the larger of the two estimates.

## Repository layout

```
src/lora_sql/
  lora.py          LoRALinear, inject_lora, parameter statistics
  qlora.py         4-bit model loading, training preparation, dtype assertions, completion loss
  data.py          Spider loading, schema rendering, the single prompt-encoding function
  config.py        strict typed training configuration
  checkpoint.py    save_lora / load_lora with load-time verification
  train.py         the training loop
  eval/            runner.py (generation), execute.py (SQLite execution and comparison),
                   metrics.py (aggregation, Spider difficulty tiers)
configs/
  train/           qv_r4, qv_r8, qv_r16, attn_r8, all7_r8
  eval/            coder7b_spider_dev.yaml, coder7b_bird_dev.yaml
  week1_parity.yaml  smoke-test evaluation on the 0.5B model
scripts/           environment checks, toy parity run, memory probe, training and evaluation
                   chains, plotting (plot_train.py, plot_results.py), results summaries
tests/             109 tests: injection, fp32 and 4-bit parity, data (Spider and BIRD),
                   configuration, loss, checkpointing, training loop, evaluation, statistics
results/           committed evidence: figures, memory probes, training logs, per-question
                   evaluation results and generated summaries (eval/, eval/bird/), and the
                   earlier repetition-penalty-1.1 results (eval/rp1.1/, eval/bird/rp1.1/)
docs/              WRITEUP.md (the mini-paper) and an explanatory guide to the project
```

## Reproducing

**Requirements.** Linux, an NVIDIA GPU with at least 8 GB of memory and bf16 support, CUDA 12.1
drivers, Python 3.11, and about 20 GB of disk for the model weights and data.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pinned versions, installs lora_sql in editable mode
python scripts/check_env.py              # CUDA, bitsandbytes, model loading, target modules
```

Download Spider from the [official site](https://yale-lily.github.io/spider) and extract it so that
`data/spider_data/` contains `train_spider.json`, `train_others.json`, `dev.json`, `tables.json` and
`database/`. The Qwen models are downloaded from the Hugging Face Hub on first use.

```bash
# Tests (the parity and runner tests need the GPU and the 0.5B model; tests whose GPU or data
# is missing are skipped)
pytest -v

# Train one configuration, or several back to back -> runs/<name>/
python -m lora_sql.train --config configs/train/qv_r8.yaml
bash scripts/train_chain.sh qv_r4 qv_r16 attn_r8 all7_r8

# Evaluate the base model and each adapter on Spider dev -> results/eval/<name>.json
bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8

# Figures and the results table
python scripts/plot_train.py --runs qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
python scripts/summarize_eval.py

# BIRD dev, zero-shot -> results/eval/bird/<name>.json, then the BIRD table (CPU)
bash scripts/eval_chain.sh --config configs/eval/coder7b_bird_dev.yaml --out-dir results/eval/bird \
    base qv_r8 qv_r16 all7_r8
python scripts/summarize_bird.py
python scripts/plot_results.py           # Spider vs BIRD figure -> results/spider_vs_bird.png
```

For BIRD, download [`dev.zip`](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) from the
[BIRD site](https://bird-bench.github.io/). Unzip it, and the `dev_databases.zip` inside it, into
`data/bird/`, so that `data/bird/dev_20240627/` contains `dev.json`, `dev_tables.json` and
`dev_databases/`.

On an RTX 4060 Laptop GPU, each training run takes 2.3 to 3.1 hours, each full Spider dev
evaluation takes 20 to 28 minutes, and each BIRD dev evaluation takes about an hour.

## Limitations

- **Single seed.** Each configuration was trained once, so differences between configurations
  include unmeasured seed-to-seed variance. That matters most for the comparisons between
  configurations, whose gaps are 2 points or less.
- **Questions cluster in databases.** Spider dev has 20 databases and BIRD dev 11. The p-values
  treat questions as independent. Intervals that resample whole databases are wider, and are
  reported next to the question-level ones in both summaries.
- **Probable contamination.** Spider is public and widely used. The base model's behaviour suggests
  it was in the pretraining data, which inflates the baseline and compresses fine-tuning gains.
- **Execution accuracy on a single database instance.** The official test-suite metric executes
  queries on several database variants to reduce false positives; this harness uses the released
  databases only. Results are therefore not directly comparable with published test-suite numbers.
  53 of the 1,034 Spider gold queries return an empty result, the easiest case to match by accident.
- **BIRD may not be clean either.** BIRD (2023) predates Qwen2.5-Coder (2024), so the base model
  may have seen it as well. It is less likely to be contaminated than Spider, not certainly clean.
- **One database excluded from training.** The 82 `baseball_1` examples exceed the 2,048-token
  limit that the largest configuration fits in memory.

## License

The code in this repository is released under the [MIT License](LICENSE), except for the Spider
difficulty function (`eval_hardness` and its helpers) in `src/lora_sql/eval/metrics.py`. That code
is vendored from [taoyds/spider](https://github.com/taoyds/spider) under the Apache License 2.0;
its license is in [`licenses/spider-LICENSE`](licenses/spider-LICENSE).

The Spider and BIRD datasets (both CC BY-SA 4.0) and the Qwen2.5 model weights (Apache 2.0) are
not redistributed here and are subject to their own licenses.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023.
- Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic
  Parsing and Text-to-SQL Task*, 2018.
- Zhong et al., *Semantic Evaluation for Text-to-SQL with Distilled Test Suites*, 2020.
- Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database
  Grounded Text-to-SQLs* (BIRD), 2023.
- Hui et al., *Qwen2.5-Coder Technical Report*, 2024.
- Hugging Face [`peft`](https://github.com/huggingface/peft) and
  [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes).
