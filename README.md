# MinLoRA-SQL

[![CI](https://github.com/nyanko3434/MinLoRA-SQL/actions/workflows/ci.yml/badge.svg)](https://github.com/nyanko3434/MinLoRA-SQL/actions/workflows/ci.yml)

A from-scratch implementation of LoRA and QLoRA, verified bit-for-bit against Hugging Face `peft`,
used to fine-tune Qwen2.5-Coder-7B-Instruct for text-to-SQL on a single 8 GB laptop GPU and
evaluated by execution accuracy on Spider, plus zero-shot on BIRD.

The full write-up, with method, results and discussion, is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

## Summary

- **Numerical parity with `peft`.** The LoRA layer, injection and training path are written from
  scratch and match `peft` 0.19.1 exactly. Forward outputs and gradients are bit-identical in fp32
  and on the 4-bit QLoRA path, and a 200-step training run matches step for step.
- **QLoRA on consumer hardware.** A custom training loop fine-tunes a 7.6B-parameter model on an
  RTX 4060 Laptop GPU (8 GB). Five adapter configurations train in 2.3 to 3.1 hours each with
  peak memory at or below 6.74 GB.
- **Execution-based evaluation.** Generated SQL is run against the real databases and its result
  rows are compared with the gold query's. The harness is read-only, time-limited and NULL-safe,
  refuses held-out data, and records the checkpoint and commit behind every number.
- **Result: an in-domain gain that does not transfer.**
  - On Spider dev every adapter beats the base model (75.92%), by 2.2 to 4.0 points. Three seeds
    of the best configuration gain +2.71 to +3.97 (mean +3.16); the reported run, 79.88%, is the
    highest of the three. The base model has probably already seen Spider.
  - On BIRD dev, which no adapter was trained on, the three adapters tested score **3.3 to 4.5
    points below the base model** (46.68% → 42.18–43.35%, each p ≤ 0.002 after correction), and
    the loss comes back on every seed. The adapters learned Spider's conventions, and those
    conventions hurt on a different benchmark.
  - The five runs do not rank the configurations: every other configuration lies within half a
    point of the range the three seeds span ([seed repeats](#seed-repeats)).
  - Run at half its trained strength, with no retraining, the best adapter keeps 68% of its
    Spider gain and shows 26% of its BIRD loss. This is a
    [diagnostic](#adapter-scaling-probe), not a tuned model.
- **A decoding default that moved the numbers.** Qwen's `generation_config.json` sets
  `repetition_penalty: 1.1`, and `generate()` applies it even with `do_sample=False`. The first
  round of evaluations inherited it silently. Turning it off changed 40–66% of the predictions.
  Results under both settings are [committed](#decoding-the-repetition-penalty-ablation).

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
not, and the reverse. p-values are exact two-sided McNemar tests, with Holm-adjusted values in
parentheses for the five comparisons against the base. Each configuration here is one training
run (seed 0).

- **Every configuration improves on the base.** The gains of `qv_r16` and `qv_r8` stay
  significant after Holm correction. The other three (p between 0.04 and 0.06) do not.
- **The questions are not independent.** They come from 20 databases. Resampling whole databases
  widens every interval, and only `qv_r16`'s then stays above zero ([+0.82, +7.56]). That holds
  for the reported seed only.
- **Fine-tuning reshuffles answers more than it adds them.** Each model fixes 74 to 84 questions
  the base got wrong and breaks 43 to 58 it got right.
- **The gain sits mostly in medium questions.** By Spider's official difficulty tiers, every
  configuration nets 16 to 27 questions on medium and 2 to 14 on the other three tiers combined.
  Easy questions are near the ceiling already (91.5% for the base).
- **What changes.** The fine-tuned models write Spider's conventions. Spider's gold spacing is in
  77% of their predictions against 3% of the base's, `no such column` errors halve (37 for the
  base, 16–25 for the adapters), and columns come out in the gold's order: 15 to 19 of the 74 to
  84 questions a model wins are ones where the base returned the right rows in another column
  order. Most losses (81–89%) are queries that run and return the wrong rows.
- **Likely contamination.** Without any fine-tuning, the base model reproduces formatting quirks
  of Spider's gold queries (such as `Country ,  COUNT(*)` with a doubled space) in 31 of its
  1,034 predictions, which suggests Spider was in its pretraining data.
- **How much is the scoring rule.** This harness compares columns by position. Allowing any
  column order raises every model by 2.3 to 3.8 points, and the adapters then lead the base by
  0.9 to 2.9. 53 of the 1,034 gold queries return no rows; leaving them out, the gains are +1.83
  to +3.77.

Confidence intervals, pairwise comparisons, the per-tier table and style diagnostics are in
[`results/eval/summary.md`](results/eval/summary.md), generated from the committed per-question
results by `scripts/summarize_eval.py`.

### Out of domain: BIRD dev (zero-shot)

Three adapters were scored on BIRD dev (1,534 questions, 11 databases, release `dev_20240627`),
a newer and harder benchmark that was never trained on. Each question's expert hint is added to
the prompt, as in the standard BIRD setting. "Strict" is this project's harness; "official" is
BIRD's own rule (`set(pred) == set(gold)`).

| Model | BIRD EX strict (%) | Δ vs base | 95% CI | Won / lost vs base | McNemar p (Holm) | BIRD EX official (%) |
|---|---|---|---|---|---|---|
| Base (no adapter) | 46.68 | | | | | 51.37 |
| `qv_r8` | 43.35 | −3.32 | [−5.48, −1.24] | +109 / −160 | 0.002 (0.002) | 48.70 |
| `qv_r16` | 42.89 | −3.78 | [−5.93, −1.76] | +105 / −163 | 4.8e-04 (9.5e-04) | 48.24 |
| `all7_r8` | 42.18 | −4.50 | [−6.52, −2.48] | +91 / −160 | 1.6e-05 (4.7e-05) | 47.33 |

- **Fine-tuning on Spider hurts on BIRD, significantly.** All three adapters lose about 160
  questions the base answers correctly and gain 91 to 109. The drop is largest on the challenging
  questions (−8.3 to −9.0 points). Most lost answers (72–75%) are valid SQL that returns the
  wrong rows. The three adapters do not differ significantly from each other (pairwise p 0.22 to
  0.54).
- **By database the evidence is thinner.** Resampling whole databases, the interval for
  `all7_r8` stays below zero ([−7.20, −1.92]); the ones for `qv_r16` ([−7.92, +0.18]) and `qv_r8`
  ([−7.18, +0.57]) reach it.
- **The Spider style carries over.** Spider's double-spaced formatting appears in 78–81% of the
  adapters' BIRD predictions, against under 1% for the base. Double-quoted string literals rise
  from 0% to 35–39%. When a hint writes `YEAR(...)`, a function SQLite lacks, the adapters copy
  it in 24–28% of cases against 5% for the base.
- **What this means.** A good part of the Spider gain is learned benchmark conventions, not a
  general improvement in writing SQL.

![Change in execution accuracy against the base model on Spider dev and BIRD dev](results/spider_vs_bird.png)

Tables by difficulty and by database are in
[`results/eval/bird/summary.md`](results/eval/bird/summary.md), generated by
`scripts/summarize_bird.py`.

### Seed repeats

Every number above comes from one training run per configuration (seed 0). `qv_r16` was trained
twice more, with seeds 1 and 2, and each run was scored on both benchmarks. The seed sets the LoRA
initialisation, the dropout masks and the data order together. Seed 0 stays the reported run; no
seed is picked as a result.

| `qv_r16` seed | Spider EX (%) | Δ vs base | 95% CI | BIRD EX (%) | Δ vs base | 95% CI |
|---|---|---|---|---|---|---|
| Base (no adapter) | 75.92 | | | 46.68 | | |
| 0 (the reported run) | 79.88 | +3.97 | [+1.93, +6.09] | 42.89 | −3.78 | [−5.93, −1.76] |
| 1 | 78.63 | +2.71 | [+0.68, +4.84] | 42.70 | −3.98 | [−6.06, −2.02] |
| 2 | 78.72 | +2.80 | [+0.77, +4.93] | 42.31 | −4.37 | [−6.45, −2.35] |
| Mean of the three | 79.08 | +3.16 | [+1.29, +5.16] | 42.63 | −4.04 | [−5.91, −2.19] |

![Change in execution accuracy against the base model for three seeds of qv_r16 and the other configurations](results/seed_spread.png)

- **Both effects replicate in direction.** All three seeds are above the base on Spider and below
  it on BIRD, and every question-level interval excludes zero. The Spider-style formatting on BIRD
  comes back on every seed too.
- **The reported Spider number is the high one.** Seed 0 scores 1.16 to 1.26 points above the
  other two (unadjusted p 0.119 and 0.098). The seeds span 1.26 points on Spider and 0.59 on
  BIRD.
- **The other configurations sit in or next to the seed range.** On Spider, `qv_r8` (79.11) is
  inside it, and `qv_r4`, `attn_r8` and `all7_r8` (78.14–78.24) are 0.39 to 0.48 points below the
  lowest seed, less than the seeds differ among themselves. So "accuracy rises with rank" and
  "`qv_r16` is the best configuration" describe five particular runs, not the configurations.
- **Two seeds disagree on as many questions as two configurations do:** 4.8–5.5% of the Spider
  questions and 8.1–8.9% of the BIRD questions.
- **Limits.** Three seeds give a rough spread, not a variance estimate, and only `qv_r16` was
  repeated.

Full tables are in [`results/eval/seeds/summary.md`](results/eval/seeds/summary.md), generated by
`scripts/summarize_seeds.py`.

### Adapter scaling probe

If the Spider gain and the BIRD loss come from the same learned conventions, turning the adapter
down should shrink both. This probe scores `qv_r16` with its LoRA scaling `alpha / r` multiplied
by 0.5 and by 0.25 at inference (`--lora-scale`), with no retraining.

| Adapter scale | Spider EX (%) | Δ vs base | 95% CI | BIRD EX (%) | Δ vs base | 95% CI |
|---|---|---|---|---|---|---|
| 0 (base) | 75.92 | | | 46.68 | | |
| 0.25 | 76.89 | +0.97 | [−0.48, +2.51] | 46.54 | −0.13 | [−1.56, +1.24] |
| 0.5 | 78.63 | +2.71 | [+0.97, +4.55] | 45.70 | −0.98 | [−2.74, +0.78] |
| 1 (as trained) | 79.88 | +3.97 | [+1.93, +6.09] | 42.89 | −3.78 | [−5.93, −1.76] |

![Change in execution accuracy against the base model by adapter scale, Spider dev and BIRD dev](results/scaling_curve.png)

- **Both effects shrink with the scale, and not at the same rate.** At half scale the adapter
  keeps 68% of its Spider gain and shows 26% of its BIRD loss. At a quarter, neither benchmark
  can be told apart from the base.
- **Against the adapter as trained,** halving the scale recovers 2.80 points on BIRD (p 6.9e-04)
  and gives up 1.26 on Spider (p 0.098, unadjusted). That Spider change is the size of the gap
  between two seeds.
- **The formatting arrives before the errors.** On BIRD at half scale, Spider's double-spaced
  formatting is already in 78.0% of predictions (79.2% as trained), while copying the hints'
  `YEAR(...)` is at 11.5% (28.2% as trained, 5.1% for the base). These counts show what moves
  with the accuracy, not what causes it.
- **This is a diagnostic, not a tuned model.** Both benchmarks are dev sets, so choosing a scale
  from this table would be tuning on dev. It covers one adapter and one seed, at two scales
  fixed in advance.

Full tables are in [`results/eval/scaling/summary.md`](results/eval/scaling/summary.md),
generated by `scripts/summarize_scaling.py`.

### Decoding: the repetition-penalty ablation

The first round of evaluations called `generate()` with `do_sample=False` and nothing else.
Qwen's `generation_config.json` sets `repetition_penalty: 1.1`, and `transformers` applies it
under greedy search too, over the prompt as well as the output, so every schema token was
penalised when the model tried to write it. The evaluation config now requires
`eval.repetition_penalty`, the runner passes it explicitly and records it with each result, and a
test checks the output against a hand-written argmax loop. The earlier results are kept unchanged
in `results/eval/rp1.1/` and `results/eval/bird/rp1.1/`. Same checkpoints, same questions,
execution accuracy (%):

| Model | Spider, penalty 1.1 | Spider, off | Δ | BIRD, penalty 1.1 | BIRD, off | Δ |
|---|---|---|---|---|---|---|
| Base (no adapter) | 74.95 | 75.92 | +0.97 | 45.96 | 46.68 | +0.72 |
| `qv_r4` | 75.92 | 78.14 | +2.22 | | | |
| `qv_r8` | 75.92 | 79.11 | +3.19 | not run | 43.35 | |
| `qv_r16` | 77.47 | 79.88 | +2.42 | 41.13 | 42.89 | +1.76 |
| `attn_r8` | 76.79 | 78.24 | +1.45 | | | |
| `all7_r8` | 77.56 | 78.24 | +0.68 | 41.79 | 42.18 | +0.39 |

The penalty changed 40–42% of the Spider predictions and 47–66% of the BIRD predictions, and it
cost the fine-tuned models more than the base. With it, no Spider gain survived Holm correction;
without it, two do. The BIRD drop is there under both settings (−4.2 to −4.8 points with the
penalty, −3.8 to −4.5 without, for the two adapters scored both ways).

### Training

| Run | Steps | Train time | Peak reserved memory | Final loss (mean of last 10 logged) |
|---|---|---|---|---|
| `qv_r4` | 536 | 2.27 h | 6.21 GB | 0.090 |
| `qv_r8` | 536 | 2.29 h | 6.23 GB | 0.087 |
| `qv_r16` | 536 | 2.25 h | 6.26 GB | 0.084 |
| `attn_r8` | 536 | 2.31 h | 6.27 GB | 0.080 |
| `all7_r8` | 536 | 3.06 h | 6.74 GB | 0.065 |

![Training loss and learning rate for the five runs](results/train_loss.png)

All five runs see the same examples in the same order. The two seed repeats of `qv_r16` see them
in a different order ([`results/train_loss_seeds.png`](results/train_loss_seeds.png)). Per-run
logs, data reports and adapter metadata are in [`results/train/`](results/train/).

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

- **Models.** Qwen2.5-Coder-7B-Instruct is the fine-tuning target. Qwen2.5-0.5B-Instruct shares
  its module names and serves only as an fp32 test fixture for the parity suite.
- **Data.** Training uses Spider `train_spider.json` and `train_others.json` (8,659 examples;
  the 82 examples of `baseball_1` exceed 2,048 tokens and are dropped, leaving 8,577). Evaluation
  uses `dev.json` (1,034 examples over 20 databases not seen in training). The Spider test split
  is never read.
- **Prompt.** Each schema is rendered as `CREATE TABLE` statements, followed by the question. One
  function builds the ChatML token ids for both training and evaluation.
- **Training.** 4-bit NF4 base with bf16 compute and gradient checkpointing; fp32 adapters, so
  small AdamW updates are not rounded away. Micro-batch 1 with 16-step gradient accumulation,
  weighted by token count. Loss on the SQL completion only. AdamW, learning rate 2e-4, 3% warmup
  and cosine decay, LoRA dropout 0.05, one epoch (536 optimizer steps). A test enforces that the
  five configurations differ only in their LoRA settings, and the seed repeats only in the seed.
- **Checkpoints.** Only adapter weights are saved, with a JSON sidecar. Loading asserts that every
  saved tensor was consumed and matches, and that the adapter is not a no-op. A
  `load_state_dict(strict=False)` that silently loads nothing would otherwise evaluate the base
  model under a fine-tuned label.
- **Evaluation.** Greedy decoding with the repetition penalty set to 1.0 explicitly. Each
  prediction runs on a read-only connection with a 30-second limit, and every gold query is
  executed before the model loads. Results are compared as multisets of rows (NULL-safe, floats
  rounded to six places), in order only when the gold query has `ORDER BY`, with columns compared
  by position.

## Repository layout

```
src/lora_sql/
  lora.py          LoRALinear, inject_lora, set_lora_scale, parameter statistics
  qlora.py         4-bit model loading, training preparation, dtype assertions, completion loss
  data.py          Spider loading, schema rendering, the single prompt-encoding function
  config.py        strict typed training configuration
  checkpoint.py    save_lora / load_lora with load-time verification
  train.py         the training loop
  eval/            runner.py (generation), execute.py (SQLite execution and comparison),
                   metrics.py (aggregation, Spider difficulty tiers)
configs/
  train/           qv_r4, qv_r8, qv_r16, attn_r8, all7_r8, and the seed repeats qv_r16_s1,
                   qv_r16_s2
  eval/            coder7b_spider_dev.yaml, coder7b_bird_dev.yaml
  week1_parity.yaml  smoke-test evaluation on the 0.5B model
scripts/           environment checks, toy parity run, memory probe, training and evaluation
                   chains, plotting, results summaries
tests/             121 tests: injection, fp32 and 4-bit parity, data (Spider and BIRD),
                   configuration, loss, checkpointing, training loop, evaluation, statistics
results/           committed evidence: figures, memory probes, training logs, per-question
                   evaluation results and generated summaries (eval/, eval/bird/), the
                   earlier repetition-penalty-1.1 results (rp1.1/), the scaling probe
                   (scaling/) and the seed repeats (seeds/)
docs/              WRITEUP.md (the mini-paper) and an explanatory guide to the project
.github/workflows/ ci.yml: lint and the CPU tests on every push
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
`database/`. For BIRD, download [`dev.zip`](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip)
from the [BIRD site](https://bird-bench.github.io/) and unzip it, and the `dev_databases.zip`
inside it, so that `data/bird/dev_20240627/` contains `dev.json`, `dev_tables.json` and
`dev_databases/`. The Qwen models are downloaded from the Hugging Face Hub on first use.

```bash
# Lint and tests (tests whose GPU, model or data is missing are skipped)
ruff check .
pytest -v

# Train one configuration, or several back to back -> runs/<name>/
python -m lora_sql.train --config configs/train/qv_r8.yaml
bash scripts/train_chain.sh qv_r4 qv_r16 attn_r8 all7_r8

# Evaluate the base model and each adapter on Spider dev -> results/eval/<name>.json
bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
python scripts/plot_train.py --runs qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
python scripts/summarize_eval.py

# BIRD dev, zero-shot -> results/eval/bird/<name>.json, then the BIRD table (CPU)
bash scripts/eval_chain.sh --config configs/eval/coder7b_bird_dev.yaml --out-dir results/eval/bird \
    base qv_r8 qv_r16 all7_r8
python scripts/summarize_bird.py
python scripts/plot_results.py           # Spider vs BIRD figure -> results/spider_vs_bird.png

# Adapter scaling probe: qv_r16 at 0.5x and 0.25x of its trained scaling, on both benchmarks
for S in 0.5 0.25; do
  bash scripts/eval_chain.sh --lora-scale $S --out-dir results/eval/scaling qv_r16
  bash scripts/eval_chain.sh --lora-scale $S --config configs/eval/coder7b_bird_dev.yaml \
      --out-dir results/eval/bird/scaling qv_r16
done
python scripts/summarize_scaling.py      # -> results/eval/scaling/summary.md
python scripts/plot_scaling.py           # -> results/scaling_curve.png

# Seed repeats: qv_r16 with seeds 1 and 2, each scored on both benchmarks
for S in qv_r16_s1 qv_r16_s2; do
  bash scripts/train_chain.sh $S
  bash scripts/eval_chain.sh --out-dir results/eval/seeds $S
  bash scripts/eval_chain.sh --config configs/eval/coder7b_bird_dev.yaml \
      --out-dir results/eval/bird/seeds $S
done
python scripts/plot_train.py --runs qv_r16 qv_r16_s1 qv_r16_s2 --out results/train_loss_seeds.png
python scripts/summarize_seeds.py        # -> results/eval/seeds/summary.md
python scripts/plot_seeds.py             # -> results/seed_spread.png
```

On an RTX 4060 Laptop GPU, each training run takes 2.3 to 3.1 hours, each full Spider dev
evaluation takes 20 to 28 minutes, and each BIRD dev evaluation takes about an hour.

Without a GPU or the datasets, `pip install -r requirements-ci.txt` (CPU-only wheels) is enough
to run the linter, the CPU tests, and the summary scripts that read only the committed results.
The CI job does exactly that.

## Limitations

- **Mostly one seed.** Only `qv_r16` was trained more than once (three seeds). The other four
  configurations have one seed each and lie within half a point of the range those seeds span,
  so the ranking of configurations is not established.
- **Questions cluster in databases.** Spider dev has 20 databases and BIRD dev 11. The p-values
  treat questions as independent. Intervals that resample whole databases are wider, and are
  reported next to the question-level ones in the summaries.
- **Probable contamination.** Spider is public and widely used, and the base model's behaviour
  suggests it was in the pretraining data. BIRD (2023) predates Qwen2.5-Coder (2024), so it is
  less likely to be contaminated than Spider, not certainly clean.
- **Execution accuracy on a single database instance.** The official test-suite metric executes
  queries on several database variants to reduce false positives; this harness uses the released
  databases only, so results are not directly comparable with published test-suite numbers.
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
