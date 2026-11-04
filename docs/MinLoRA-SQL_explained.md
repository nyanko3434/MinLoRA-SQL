# MinLoRA-SQL — The Whole Project, From Zero

*A self-contained guide: what the project is, the background you need, what was built, how it was
run, what the results are, and what is left. Started during run #1 (Session 5); updated at the end of
Session 6, when all five configs had been trained and scored on Spider dev.*

---

## Contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [Background from zero](#2-background-from-zero)
3. [The task: text-to-SQL and the Spider dataset](#3-the-task-text-to-sql-and-the-spider-dataset)
4. [The two claims that make this a "research-y" project](#4-the-two-claims-that-make-this-a-research-y-project)
5. [Key design decisions and why](#5-key-design-decisions-and-why)
6. [What has been built, session by session](#6-what-has-been-built-session-by-session)
7. [Code map: what every file does](#7-code-map-what-every-file-does)
8. [The training command, dissected](#8-the-training-command-dissected)
9. [What `train.py` does, step by step](#9-what-trainpy-does-step-by-step)
10. [Reading a training log: is the run healthy?](#10-reading-a-training-log-is-the-run-healthy)
11. [The evaluation harness](#11-the-evaluation-harness)
12. [Results](#12-results)
13. [What happens next](#13-what-happens-next)
14. [Glossary](#14-glossary)

---

## 1. The project in one paragraph

You are teaching an AI model that already knows how to code (**Qwen2.5-Coder-7B-Instruct**) to get better
at one job: turning an English question about a database into a correct **SQL query**. Instead of
retraining the whole model (impossible on an 8 GB laptop GPU), you use **LoRA**, a technique that freezes
the model and trains a tiny add-on. The twist that makes this a strong resume project: **you wrote LoRA
yourself** and **proved mathematically and empirically that it is bit-for-bit identical** to the industry
standard library (HuggingFace `peft`). Then you measured improvement the objective way: **run every
generated query against a real database and check whether it returns the right rows.** Result: five
fine-tunes trained on the laptop, the best scoring 77.6% on Spider dev vs the untuned model's 75.0%.
That is a real but small gain, and not statistically conclusive. The likely reason is that the base
model already knows Spider well (§12).

---

## 2. Background from zero

### 2.1 What a language model is

A large language model (LLM) is a function that reads a sequence of **tokens** (word pieces) and outputs
a probability for every possible next token. Generating text is just repeating "predict the next token,
append it, predict again."

Inside, the model is a stack of layers (28 in Coder-7B). Each layer is mostly **matrix multiplications**.
The numbers in those matrices are the model's **parameters** (or **weights**). "7B" means about 7.6
billion of them. Everything the model "knows" is encoded in those numbers.

### 2.2 How a model learns: loss, gradients, optimizer

Training is a loop:

1. Show the model an input and the correct answer.
2. Measure how wrong it was. This number is the **loss**. We use **cross-entropy**: for each token of the
   answer, take −log(probability the model gave to the correct token), then average.
3. Compute the **gradient**: for each trainable parameter, which direction to nudge it to lower the loss.
4. The **optimizer** (we use **AdamW**) nudges the parameters. The size of the nudge is controlled by the
   **learning rate** (lr).

**Intuition for loss numbers:** `exp(−loss)` is roughly the average probability the model assigns to the
correct next token.

| loss | model's average confidence in the right token |
|---|---|
| 0.59 | ~55% |
| 0.20 | ~82% |
| 0.10 | ~90% |
| 0.07 | ~93% |

### 2.3 Fine-tuning, and why full fine-tuning is impossible here

The model was already trained (pretrained) on huge amounts of text and code. **Fine-tuning** means
continuing training on your specific task so it gets better at that task.

Full fine-tuning updates all 7.6 billion parameters. That needs memory for the weights, their gradients,
and two AdamW statistics per parameter: well over 100 GB. Your GPU has 8.2 GB. So we need something
smarter.

### 2.4 LoRA (Low-Rank Adaptation)

Take one weight matrix `W` inside the model (say, the `q_proj` matrix of an attention layer, 3584 × 3584
≈ 12.8 million numbers). LoRA says:

- **Freeze `W`.** Never change it.
- Add two small matrices: `A` (shape `r × in`) and `B` (shape `out × r`), where the **rank** `r` is tiny
  (we use 4, 8 or 16).
- The layer now computes: `output = W·x + (alpha / r) · B·(A·x)`

Only `A` and `B` are trained. For `q_proj` with r = 8 that is 8 × (3584 + 3584) = 57,344 numbers instead
of 12.8 million.

Key details:

- **`B` starts at zero**, so at step 0 the add-on contributes exactly nothing. The model starts out
  identical to the original and drifts only as training teaches it.
- **`alpha / r` is the scaling factor.** We always use alpha = 2r, so scaling is always 2.0. That way,
  changing the rank doesn't secretly also change how strong the updates are.
- **Which layers get adapters** is a choice (the "target modules"). Each transformer layer has seven big
  projections: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) and `gate_proj`, `up_proj`, `down_proj`
  (the MLP). Run #1 adapts only `q_proj` and `v_proj`.

In run #1: **2,523,136 trainable numbers out of ~7.6 billion, about 0.03%.**

### 2.5 Number formats: fp32, bf16, 4-bit

Computers store decimals with limited precision:

| format | bits per number | precision | used here for |
|---|---|---|---|
| **fp32** | 32 | ~7 significant digits | the LoRA adapters; the parity tests |
| **bf16** | 16 | ~3 significant digits | all base-model computation |
| **nf4** (4-bit) | 4 | very coarse, specially designed for weights | storing the frozen 7B weights |

Why this matters:

- **Memory.** 7.6B parameters in bf16 = ~15 GB (does not fit). In 4-bit = ~3.3 GB for the body, plus
  ~2.2 GB for the two big vocabulary matrices kept in bf16. Total measured: **~5.1 GB.** Fits.
- **Exactness.** Two correct implementations that add numbers in a different order agree only to the
  precision of the format. In bf16 that is ~1e-2. In fp32 it is ~1e-7. That's why the correctness proof
  (parity tests) runs in fp32, where "identical" can be checked tightly.
- **Adapters stay fp32 even during 4-bit training.** AdamW makes tiny updates (~1e-4 relative). In bf16,
  those updates get rounded away to nothing. So the adapters live in fp32, exactly like `peft` does.

### 2.6 QLoRA

**QLoRA = LoRA on top of a 4-bit quantized base model.** The frozen base is stored in 4-bit (nf4) and
temporarily expanded to bf16 for each multiplication. The LoRA adapters are fp32 and are the only things
trained. This is the only reason a 7B model can be fine-tuned on an 8 GB laptop GPU.

### 2.7 Chat templates (ChatML)

"Instruct" models were trained on conversations wrapped in special tokens. Qwen's format (ChatML) looks
like this:

```
<|im_start|>system
You are a SQLite expert ... output only the SQL query.<|im_end|>
<|im_start|>user
CREATE TABLE singer ( ... );  ... Question: How many singers do we have?  SQL: <|im_end|>
<|im_start|>assistant
SELECT count(*) FROM singer<|im_end|>
```

The model learns to write the part after `assistant` and to stop with `<|im_end|>` (the end-of-sequence
token, id 151645). Prompts are built in **one** function (`data.encode_prompt`) used by both training and
evaluation. So the model is tested on exactly the format it was trained on.

---

## 3. The task: text-to-SQL and the Spider dataset

### 3.1 Why SQL

SQL generation has a **free, objective grader**: run the model's query on the real database and compare
the returned rows to those of the correct query. No human judges, no AI judges. The original alternative
(automatic code review / bug fixing) would have needed per-repository test infrastructure that would eat
the whole time budget.

### 3.2 Spider

Spider is the standard academic text-to-SQL benchmark. Each example has:

- a **database** (a real SQLite file with data in it),
- a **question** in English,
- the **gold SQL** query.

The prompt contains the database **schema** (its tables and columns, written as `CREATE TABLE`
statements) plus the question. The model must produce SQL.

| split | examples | role |
|---|---|---|
| `train_spider.json` | 7,000 | training |
| `train_others.json` | 1,659 | training |
| `dev.json` | 1,034 | **the only evaluation set** (comparable to published numbers) |
| `test.json` + `test_database/` | — | **held out, never touched** (reserved for at most one final check) |

The training and dev sets use **different databases**. Good results on dev therefore mean the model
generalizes to schemas it has never seen. (Caveat, found in Session 6: Spider is public and old,
so the base model has very probably seen it during pretraining. See §12.)

**BIRD-dev** is a harder, more realistic benchmark. It will be used zero-shot (never trained on) to test
generalization.

### 3.3 Execution accuracy

Example: question *"How many singers do we have?"*

- Gold: `SELECT count(*) FROM singer` → `[(6,)]`
- Model: `SELECT COUNT(singer_id) FROM singer` → `[(6,)]` → **correct**, even though the text differs.

Accuracy = fraction of dev questions where the result sets match. (The official Spider metric, "test-suite
accuracy," runs each query on several extra database variants, which this project doesn't have. The
writeup will cite it.)

Two details of *how* rows are compared matter:

- **Row order** counts only when the gold query has `ORDER BY`. Otherwise the rows are compared as a
  bag, so duplicates matter but order doesn't.
- **Column order:** this harness compares columns by position, so `SELECT count(*), grade` is wrong
  when the gold is `SELECT grade, count(*)`. The official eval allows any column order. §12 measures
  how much this matters (2–3 points for everyone, and it does not change the conclusions).

---

## 4. The two claims that make this a "research-y" project

Most student fine-tuning projects call a library and show a loss curve. This one does two things they skip.

### Claim 1: Parity. "My LoRA is identical to peft." **EARNED.**

Proven three ways, each stronger than "the curves look similar":

| evidence | setting | result |
|---|---|---|
| Forward + backward parity | Qwen2.5-0.5B, fp32, r ∈ {4, 8, 16} + all-seven | max difference **0.0** (bit-identical) |
| Toy training run, 200 steps | ours vs peft, identical init/data | max per-step loss difference **0.000000** |
| **QLoRA-path parity** | 0.5B in 4-bit nf4, bf16 compute, incl. scaling 1.5 | max difference **0.0** on logits and gradients |

How the parity test works: build the same model twice, once with your LoRA and once with peft. Copy
peft's adapter weights into yours. Run the same input through both, then compare outputs and gradients.
A naive version of this test is **vacuous**. Both libraries start with `B = 0`, so both models equal the
base model and "agree" trivially. The test therefore **randomizes `B` first**.

The **scaling-1.5 case** is the clever bit. Multiplying by 2.0 is exact in every float format, so a bug in
*when* you convert between fp32 and bf16 is invisible at scaling 2.0. That's exactly what all your
training configs use. At scaling 1.5 the old (buggy) order was off by 0.32; the fixed order is exactly 0.

### Claim 2: Rigorous evaluation. "Measured by execution accuracy." **EARNED (Session 6).**

The harness was hardened (§11), and the base model plus all five fine-tunes were scored on all 1,034
dev questions. Every number traces back to the exact checkpoint and code commit that produced it, and
the differences were tested for statistical significance rather than just eyeballed (§12).

---

## 5. Key design decisions and why

| decision | why |
|---|---|
| **Two models:** Qwen2.5-0.5B = test fixture only; Coder-7B = real target | The 0.5B fits in fp32 (~2 GB) for exact parity tests and has the same layer names as the 7B. It is **never** fine-tuned. |
| **Qwen2.5-Coder-7B-Instruct** as the base | Code-specialized means a strong SQL baseline. Dense (no mixture-of-experts), fits 4-bit on 8 GB, Apache 2.0. |
| **Custom training loop**, not a library trainer | Fits the "from scratch" story and removes version-coupling. |
| **fp32 adapters, bf16 base compute, no autocast** | Matches peft exactly, so QLoRA parity is bit-identical. Avoids losing small updates to bf16 rounding. |
| **Never call `prepare_model_for_kbit_training`** | It upcasts the two giant vocabulary matrices to fp32 (+2.2 GB). That would not fit. |
| **Completion-only loss** (`logits_to_keep`) | Only the SQL tokens are graded, and the final vocabulary projection is computed only for them. Saves ~1 GB per long example. |
| **Micro-batch 1, gradient accumulation 16** | One example at a time fits memory. Gradients are summed over 16 examples before each update, so it behaves like a batch of 16. |
| **Token-weighted accumulation** | Each SQL token counts equally, which is mathematically identical to one batch of 16. |
| **`max_seq_len` 2048 for all five configs** | The largest length the biggest config fits. It drops only `baseball_1` (82 examples, 0.95%), a database with a ~2,000-token schema. All ablations train on identical data. |
| **alpha = 2r** | Scaling stays 2.0, so the rank ablation compares rank alone, not rank-plus-learning-rate. |
| **Checkpoints = only LoRA weights + sidecar JSON**, loaded with strict checks | Prevents the silent failure where a load quietly loads *nothing* and you evaluate the untrained model while thinking it's fine-tuned. |
| **transformers 5.14.1 pinned** | peft 0.19.1 needs it; downgrading would create an untested combination. |
| **Held-out discipline** | `test.json` / `test_database/` are never read. `dev.json` is never trained on (enforced in `config.py`). |
| **Cut:** DoRA, RAG baseline, resume-from-checkpoint | The ~40-hour budget. |

---

## 6. What has been built, session by session

| session | what | result |
|---|---|---|
| **1** | Python env, packages, Spider download (verified DBs contain real data), 0.5B loads | Environment works; bitsandbytes verified early |
| **2** | `LoRALinear` + `inject_lora` (wrap the chosen layers, freeze everything else) | 10 tests: exact wrap counts, exact parameter counts, bit-identical at init, only LoRA trainable |
| **3** | Parity test vs peft (fp32, forward + backward) | 18 tests, **bit-identical** |
| **4** | `data.py` (schema-as-DDL prompts); Coder-7B download verified; 200-step toy training ours vs peft | Loss curves **identical at every step** (figure in `results/`) |
| **5, Half 1** | fp32 adapters with peft's cast order; 4-bit tests + **QLoRA parity** (0.0); ChatML prompt contract; config system; 4-bit model loader; completion-only loss; memory probe | Probe: `qv_r8` peaks ~6.6 GB, `all7_r8` OOMs at 2560 → `max_seq_len` = 2048 |
| **5, Half 2** | `checkpoint.py` (safe save/load) and `train.py` (the training loop) | **66 tests pass.** 32-example sanity run on the 7B OK |
| **6** | Run #1 `qv_r8` (2.29 h); reload check on the real 7B; `train_chain.sh` runs the other four overnight (~10 h); loss plot | All five runs finish 536/536 steps, peak memory ≤ 6.74 GB |
| **6** | Eval harness hardened (§11); `eval_chain.sh` scores base + five runs (~2.5 h); `summarize_eval.py` writes the results table with significance tests | **85 tests pass.** Base 74.95% → best 77.56% (§12) |
| **7** | v1 merged to `main` (PR #1). Spider's official difficulty function vendored (Apache-2.0); every dev question tagged easy / medium / hard / extra; accuracy broken down by tier, with no re-run (§11, §12.4) | **95 tests pass.** Split matches the official 248 / 446 / 174 / 166; the gain sits mostly in medium |

---

## 7. Code map: what every file does

```
src/lora_sql/
├── lora.py         Your LoRA. LoRALinear (frozen base + A, B adapters), inject_lora (wraps the
│                   chosen layers, freezes the rest), lora_param_stats (counts).
├── data.py         Loads Spider examples, renders the schema as CREATE TABLE text,
│                   encode_prompt (ChatML; the ONLY source of prompt tokens),
│                   build_train_example (prompt + SQL + end token, and how many tokens are "the answer").
├── config.py       Reads a training YAML into strict, typed settings. Rejects typos, missing keys,
│                   and any attempt to train on dev/test.
├── qlora.py        load_model (4-bit or plain), prepare_for_training (gradient checkpointing),
│                   assert_training_dtypes (nothing upcast, adapters fp32), completion_loss.
├── checkpoint.py   save_lora / load_lora: save only adapter weights + a JSON describing them;
│                   loading rebuilds the adapters and verifies every weight actually arrived.
├── train.py        The training loop (§9).
└── eval/           runner.py (load model + adapter, generate SQL for every dev question, write
                    results), execute.py (run SQL read-only with a time limit, compare result rows),
                    metrics.py (accuracy, by difficulty too; holds Spider's official difficulty
                    function, copied from the Spider repo under Apache-2.0). See §11.

configs/train/      qv_r8 (reference), qv_r4, qv_r16, attn_r8, all7_r8 — the five runs.
configs/eval/       coder7b_spider_dev.yaml — the real eval (7B in 4-bit, dev, database/).
configs/week1_parity.yaml   a quick smoke eval on the 0.5B (results go to runs/, never reported).
scripts/            check_env, toy_train + plot_toy (Session 4 figure), probe_memory (Session 5),
                    train_chain.sh (train several configs back to back), plot_train.py (loss plot),
                    eval_chain.sh (score several runs back to back), summarize_eval.py (results table).
tests/              95 tests: injection, fp32 parity, 4-bit parity, data, config, loss, checkpoint,
                    train, eval execution (CPU), difficulty tiers (CPU), eval runner (GPU).
licenses/           spider-LICENSE: the Apache-2.0 license of the copied difficulty code.
results/            COMMITTED: plots and JSON: toy curves, probe numbers, train_loss.png,
                    train/<run>/ (log, data report, adapter description), eval/<name>.json + .jsonl
                    (every prediction), eval/summary.md (the results table).
runs/               NOT committed: adapter weights and raw logs from training.
data/               NOT committed: Spider.
```

---

## 8. The training command, dissected

```bash
HF_HUB_OFFLINE=1 systemd-inhibit --what=sleep:idle python -m lora_sql.train \
    --config configs/train/qv_r8.yaml 2>&1 | tee runs/qv_r8.log
```

| piece | meaning |
|---|---|
| `HF_HUB_OFFLINE=1` | Use the already-downloaded model files; never contact the internet. |
| `systemd-inhibit --what=sleep:idle` | Asks Linux not to suspend or idle-sleep while the command runs. **Caveat:** on default systemd settings, *closing the lid* ignores this lock and can still suspend the machine. Keep the lid open (or use `--what=sleep:idle:handle-lid-switch` next time). |
| `python -m lora_sql.train` | Runs `src/lora_sql/train.py`. |
| `--config configs/train/qv_r8.yaml` | Which experiment: LoRA on `q_proj` + `v_proj`, rank 8, alpha 16. |
| `2>&1` | Merge error output into normal output, so both are captured. |
| `\| tee runs/qv_r8.log` | Show output on screen **and** save it to a file. |

Closing the terminal kills the run. It is fine to use the laptop for light work meanwhile. Avoid anything
GPU-heavy: there is ~2 GB of VRAM headroom (only ~1.5 GB during `all7_r8`). That is also why, while
training runs, only CPU tests are run (`CUDA_VISIBLE_DEVICES=""`): loading even the 0.5B test model
would push training out of memory.

The other four runs used the same command inside a loop:

```bash
systemd-inhibit --what=sleep:idle bash scripts/train_chain.sh qv_r4 qv_r16 attn_r8 all7_r8
```

`train_chain.sh` trains each config in turn, records `<name> exit <code>` in `runs/chain.log`, and
moves on if one fails, since the runs are independent.

---

## 9. What `train.py` does, step by step

1. **Safety check:** refuses to start if `runs/qv_r8/final/` already exists (never overwrite a finished
   run). Saves a copy of the config.
2. **Data:** loads 8,659 training examples and turns each into tokens (ChatML prompt + SQL + end token).
   Drops examples longer than 2,048 tokens (exactly `baseball_1`'s 82). Shuffles with a fixed seed and
   prints a **sha256 fingerprint** of the order. Same fingerprint means same data, same order: reproducible.
3. **Step count:** 8,577 examples ÷ 16 per update = **536 optimizer steps** (one leftover example
   discarded).
4. **Model:** loads Coder-7B in 4-bit, turns on **gradient checkpointing** (don't keep every intermediate
   result in memory; recompute during the backward pass, trading time for memory), injects your LoRA into
   all 56 `q_proj`/`v_proj` layers, and asserts nothing is in the wrong precision.
5. **Optimizer + schedule:** AdamW on the 2.5M adapter numbers only. Learning rate: **warmup** from ~0 to
   2e-4 over the first 16 steps (gentle start), then **cosine decay** smoothly down to 0 by step 536.
6. **Preflight:** runs the single longest kept example (1,347 tokens) forward and backward once, with no
   update. If memory were going to run out, it fails in minute 1, not hour 2.
7. **Each optimizer step:**
   - Take the next 16 examples. For each one, run the model, compute the loss **only on the SQL tokens**,
     backpropagate. Gradients add up across the 16, weighted by token count.
   - **Clip** the gradient norm to 1.0 (a safety cap against sudden huge updates).
   - AdamW updates the adapters; the learning-rate schedule advances; gradients reset.
8. **Logging:** every 10 steps, a line on screen and a JSON row in `runs/qv_r8/log.jsonl`.
9. **Checkpoints:** adapters saved at steps 100, 200, 300, 400, 500 (`step_<n>/`) and at the end
   (`final/`). Each is ~10 MB (only the 2.5M fp32 adapter numbers), never the 15 GB base model.
10. **Time guard:** if it ever passes 6 hours, it saves `final/` and stops cleanly.

---

## 10. Reading a training log: is the run healthy?

*(Written live during run #1. All five runs ended up looking like this: see `results/train_loss.png`.)*

### Line by line

```
data: loaded 8659, dropped 82 > 2048 {'baseball_1': 82}, kept 8577, used 8577
```
Exactly the expected drop.

```
steps: 536 (grad_accum 16, ...); trailing partial window discards 1 per epoch; order sha256 c5c7...
```
536 = floor(8577 / 16). Keep the sha256: the ablation runs should print the **same** fingerprint,
proving they trained on identical data.

```
lora: {'total': 4355495424, 'trainable': 2523136, 'lora': 2523136, 'n_lora_layers': 56}
```
- 56 = 28 layers × 2 (q and v).
- Trainable = 28 × 8 × [(3584 + 3584) + (3584 + 512)] = **2,523,136**. Matches the hand calculation, and
  trainable == lora, so nothing else is training.
- "total" shows 4.36B, not 7.6B, because 4-bit weights are stored packed two-per-byte and counted as
  bytes. ~6.5B body weights ÷ 2 + ~1.09B vocabulary weights ≈ 4.36B. Consistent.

```
preflight: len 1347 ok, peak reserved 6.16 GB
```
The longest kept example is 1,347 tokens, so the 2,048 limit is never approached. This also means the
biggest config (`all7_r8`) will be comfortably below its OOM point on real data.

```
step 150/536  loss 0.1128  lr 1.69e-04  gnorm 0.398  38.7 min  peak 6.23 GB
```
- **loss:** cross-entropy on that step's 16 SQL answers.
- **lr:** current learning rate (on the cosine schedule).
- **gnorm:** size of the gradient before clipping.
- **min:** elapsed time.
- **peak:** maximum GPU memory reserved so far.

### Health checklist (as of step 160)

| check | expected | seen | |
|---|---|---|---|
| Data drop | only `baseball_1: 82` | `{'baseball_1': 82}` | ✅ |
| Steps | 536 | 536 | ✅ |
| Trainable params | 2,523,136, only LoRA | 2,523,136 = lora | ✅ |
| Warmup | 1.25e-5 at step 1 → 2e-4 by step 16 | 1.25e-5 → 2.00e-4 at step 20 | ✅ |
| Cosine decay | slowly falling after warmup | 2.00e-4 → 1.65e-4 at step 160 | ✅ |
| Memory | ≈ probe (~6.2 GB) | flat at 6.23 GB | ✅ |
| Speed | probe projected ≤ 2.72 h | ~15.5 s/step → **~2.3 h total** | ✅ |
| Loss | finite, falling | 0.59 → mostly 0.07–0.15 | ✅ |
| Grad norm | below clip (1.0), no spikes | 0.14–0.46 | ✅ |

**Why the loss starts low (0.59):** Coder-7B-Instruct can already write SQL. It is being refined, not
taught from scratch.

**Why the loss jumps around (0.07 ↔ 0.21):** each logged value is one window of only 16 examples. Some
windows contain harder questions. Look at the trend, not single points. `results/train_loss.png` shows
the raw points faintly and a 5-point rolling mean in bold. All five runs saw the same examples in the
same order, so their spikes happen at the same steps.

**Why this training loss is an honest signal:** with 1 epoch, every example is seen exactly once. Each
step's loss is measured on examples the model has never trained on before.

**But low training loss ≠ high accuracy.** A query can be one token wrong and return different rows. Only
the dev execution accuracy tells you whether the model actually improved. §12 shows this concretely:
`all7_r8` reached by far the lowest training loss (0.065 vs ~0.085) but is no more accurate than
`qv_r16`.

### Warning signs to watch for

| sign | likely meaning |
|---|---|
| `loss nan` / `inf` | numerical blow-up; stop the run |
| loss trending **up** for 50+ steps | learning rate too high or a data problem |
| `gnorm` suddenly > 5–10 | instability; usually followed by a loss spike |
| minutes per 10 steps creeping up a lot | laptop thermal throttling. The 6 h guard still saves a checkpoint. |
| `CUDA out of memory` | something else grabbed VRAM; check `nvidia-smi` |
| `stopped_by_budget` in the log | hit 6 hours; `final/` was saved, but the schedule didn't finish |

---

## 11. The evaluation harness

Evaluation is three steps: **generate** SQL for every dev question, **execute** it on the real
database, and **score** it against the gold query's rows. Before Session 6 this code existed but
could not be trusted. Here is what was fixed, and why each fix matters.

| fix | the problem it prevents |
|---|---|
| Checkpoints load through `load_lora` | The old code used an unchecked load that could silently load *nothing*, and then the "fine-tuned" score would really be the base model's. |
| 4-bit loading, same loader as training | The 7B in bf16 (~15 GB) doesn't fit on the GPU. |
| No adapter at all for the base model | The baseline is the true untouched model, not a model with an empty adapter. |
| Databases opened **read-only** | A generated `DROP TABLE` or `DELETE` simply fails and is scored wrong. It can't damage the databases every later eval uses. |
| A **30-second time limit** per query | A generated query with an accidental huge join could otherwise hang the eval forever. (sqlite's own `timeout` only covers waiting for a lock.) |
| **NULL-safe** comparison | The old comparison sorted the rows, and Python can't sort `None` next to text. 9 dev gold answers contain NULLs, so a real eval would have crashed partway through. |
| **Gold preflight** | All 1,034 gold queries are run first (1.6 s). A broken database path fails in seconds, not after 20 minutes of generation. |
| **Held-out guard** | The runner refuses to run on `test.json` or `test_database/`. The rule is enforced in code, not left to memory. |
| Results written **as it goes** (`.jsonl`) | A crash at question 900 keeps the first 899. |
| **Provenance** in every result file | Each result JSON records the adapter's description (rank, targets, training step, config hash) and the code commit, so each number traces back to exactly what produced it. |

Decoding is **greedy** (always pick the most likely token), so every eval is deterministic and
repeatable. Before the full runs, a 20-question probe checked that the base model outputs bare SQL,
with no markdown fences or explanations, so the baseline isn't scored down for formatting.

**Difficulty tiers (Session 7).** Spider labels every question **easy**, **medium**, **hard** or
**extra** (extra-hard), so you can see *where* a model succeeds, not just how often. The label comes
from the gold query alone, by counting its parts:
- "component 1" clauses: WHERE, GROUP BY, ORDER BY, LIMIT, each extra joined table, OR, LIKE;
- "component 2": nested sub-queries and INTERSECT / UNION / EXCEPT;
- "others": several aggregates, several selected columns, several WHERE conditions, several GROUP BY
  columns.

`SELECT count(*) FROM singer` is easy. A JOIN with WHERE, GROUP BY, ORDER BY and LIMIT is extra.

The function that does this counting is Spider's own `eval_hardness`, copied unchanged into
`metrics.py`, with credit and its Apache-2.0 license in `licenses/spider-LICENSE`. It is copied
rather than rewritten because its quirks define the official labels: a "cleaner" rewrite would
label some questions differently and no longer match published Spider numbers.

Spider's `dev.json` already stores each gold query in parsed form (the `sql` field), so the
function runs on that directly, without Spider's SQL parser. A test checks the result against
the official dev counts: 248 easy, 446 medium, 174 hard, 166 extra. New evals record each
question's tier as they run. The six evals from Session 6 were run before this existed, so
`summarize_eval.py` adds the tiers afterwards by question position, after checking that every row
lines up with `dev.json`. Nothing had to be re-run on the GPU.

The commands:

```bash
# one model
python -m lora_sql.eval.runner --config configs/eval/coder7b_spider_dev.yaml \
    --lora-checkpoint runs/qv_r8/final --out results/eval/qv_r8.json
# base + all five, back to back (~20-28 min each)
systemd-inhibit --what=sleep:idle bash scripts/eval_chain.sh base qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
# the results table -> results/eval/summary.md
python scripts/summarize_eval.py
```

---

## 12. Results

### 12.1 Training (all five runs)

| run | adapts | rank | LoRA params | train time | peak memory | final loss (avg of last 10 logs) |
|---|---|---|---|---|---|---|
| `qv_r8` | q, v | 8 | 2.52M | 2.29 h | 6.23 GB | 0.087 |
| `qv_r4` | q, v | 4 | 1.26M | 2.27 h | 6.21 GB | 0.090 |
| `qv_r16` | q, v | 16 | 5.05M | 2.25 h | 6.26 GB | 0.084 |
| `attn_r8` | q, k, v, o | 8 | 5.05M | 2.31 h | 6.27 GB | 0.080 |
| `all7_r8` | all seven | 8 | 20.19M | 3.06 h | 6.74 GB | 0.065 |

Every run completed 536/536 steps with no memory problems.

### 12.2 Spider dev execution accuracy (1,034 questions)

| model | accuracy | change vs base | right that base missed / wrong that base got | p-value (corrected) |
|---|---|---|---|---|
| base (no fine-tuning) | **74.95%** | | | |
| `qv_r8` | 75.92% | +0.97 | +77 / −67 | 0.900 |
| `qv_r4` | 75.92% | +0.97 | +76 / −66 | 0.900 |
| `qv_r16` | 77.47% | +2.51 | +85 / −59 | 0.164 |
| `attn_r8` | 76.79% | +1.84 | +74 / −55 | 0.338 |
| `all7_r8` | **77.56%** | +2.61 | +88 / −61 | 0.164 |

The full table, with confidence intervals and more comparisons, is in `results/eval/summary.md`.

### 12.3 How to read it

**Is a 2.5-point gain real?** All six models answered the same 1,034 questions, so they can be
compared question by question. Only the questions where two models *disagree* carry information. The
**McNemar test** asks whether "fine-tune right, base wrong" outnumbers "base right, fine-tune wrong"
by more than chance would. For `qv_r16` and `all7_r8` the raw p-value is about 0.035 (below the usual
0.05 line). But five fine-tunes were each compared with the base, and testing five things raises the
odds that one looks good by luck. After the **Holm correction** for that, no p-value is below 0.05.
So the honest summary is: *every fine-tune scored higher, the best by about 2.5 points, but the gain is
not statistically conclusive.*

**Fine-tuning reshuffles more than it adds.** Every run fixes 74–88 questions the base got wrong and
breaks 55–67 it got right. The net gain is the small difference between two big numbers.

**Why so little headroom? The base has probably seen Spider.** Untuned, the base already writes the
dataset's odd formatting, like `Country ,  COUNT(*)` with the double space, which is a fingerprint of
Spider's answer key. Spider has been public since 2018, so it is very likely in Qwen2.5-Coder's
pretraining data. A 75% baseline leaves little for fine-tuning to add, and the remaining gains come
from matching the dataset's conventions.

**What fine-tuning actually changed** (from reading questions that flipped):

- *Gains* come from adopting Spider's conventions: `INTERSECT` for "both X and Y", `= 'Brazil'` instead
  of `LIKE '%Brazil%'`, copying values as the question writes them ("Virginia", not "VA"), and fewer
  unnecessary joins.
- *Losses* come mostly from inventing column names in Spider's lower_snake_case style that don't exist
  in the schema (`pet_type` when the real column is `PetType`), and occasionally from dropping a needed
  `DISTINCT`.

**The ablations:**

- *Rank on q/v:* r4 = r8 exactly (785 correct each), and r16 beats r8 on 24 questions versus 8
  (p = 0.007 before correction). Bigger rank helped at 16, but not between 4 and 8.
- *Target set:* at rank 8, adding k/o (`attn_r8`) and the MLP (`all7_r8`) scored higher than q/v
  alone (+35/−26 and +50/−33 flipped questions), but neither difference is significant (p = 0.31 and
  0.08). And `all7_r8`, with 4× the parameters of `qv_r16` and much lower training loss, ties it
  (40 vs 39 flipped questions).
- *Caveat:* each config was trained **once** (one random seed). Without repeats, part of any gap
  between two fine-tunes could be noise from training itself.

**The column-order check.** Rescoring with any column order allowed (as the official eval does) raises
every model by 2.3–3.2 points, the base most of all (74.95 → 78.14). So the strict harness slightly
*overstates* the fine-tuning gains, not understates them. `qv_r16` and `all7_r8` swap first and second
place; the rest of the order is unchanged.

### 12.4 Accuracy by difficulty

Each cell is accuracy on that tier, with the change against the base in parentheses.

| model | easy (248) | medium (446) | hard (174) | extra (166) | all (1,034) |
|---|---|---|---|---|---|
| base | 90.32 | 75.34 | 70.11 | 56.02 | 74.95 |
| `qv_r8` | 89.92 (−0.40) | 78.25 (+2.91) | 68.97 (−1.15) | 56.02 (+0.00) | 75.92 (+0.97) |
| `qv_r4` | 89.52 (−0.81) | 78.70 (+3.36) | 69.54 (−0.57) | 54.82 (−1.20) | 75.92 (+0.97) |
| `qv_r16` | 90.32 (+0.00) | 79.82 (+4.48) | 72.41 (+2.30) | 57.23 (+1.20) | 77.47 (+2.51) |
| `attn_r8` | 90.73 (+0.40) | 77.58 (+2.24) | 72.41 (+2.30) | 58.43 (+2.41) | 76.79 (+1.84) |
| `all7_r8` | 91.13 (+0.81) | 79.37 (+4.04) | 72.99 (+2.87) | 57.23 (+1.20) | 77.56 (+2.61) |

**How to read it:**

- **The tiers behave as labelled.** The base goes from 90% on easy down to 56% on extra.
- **The gain sits mostly in medium.** Take questions won minus lost against the base:
  - On medium, every fine-tune nets +10 to +20.
  - On easy, hard and extra combined, the net is between −5 and +9.
  - `qv_r4` and `qv_r8` gain *only* on medium; they slightly lose on easy and hard.
  - The bigger adapters (`qv_r16`, `attn_r8`, `all7_r8`) also pick up a few hard questions.
- **Extra-hard questions churn but don't improve.** 23–27% of them flip one way or the other in
  every run, the most of any tier, yet the net change is only −2 to +4 questions.
  - One plausible reading, *not yet checked against the flipped questions*: fine-tuning mostly
    teaches Spider's conventions (§12.3), and those fixes pay off most on mid-sized queries.
  - On long queries, a fix in one clause can be undone by a mistake in another.
- **Easy is near the ceiling.** At 90%, there is little left to gain there.
- **Sample sizes are small.** On extra, one question moves accuracy by 0.6 points, so per-tier
  differences of a point or two are descriptive, not tested.

The per-tier won / lost counts are in `results/eval/summary.md`.

### 12.5 Where the claims stand

| claim | status |
|---|---|
| From-scratch LoRA, bit-identical to peft (fp32, forward + backward) | ✅ earned |
| Identical training trajectory vs peft (toy, 200 steps) | ✅ earned |
| Bit-identical to peft on the QLoRA path (4-bit, incl. non-power-of-two scaling) | ✅ earned |
| Fine-tuned 7B on an 8 GB laptop GPU via your own QLoRA loop | ✅ earned: five runs, peak ≤ 6.74 GB |
| A rigorous, reproducible execution-accuracy harness | ✅ earned |
| Improvement in execution accuracy over the base model on Spider dev | ⚠️ measured: +1 to +2.6 points, not statistically conclusive, base likely contaminated |
| Rank and target-set ablation findings | ⚠️ measured, single seed |
| Where the gains sit (Spider difficulty tiers) | ✅ measured: mostly medium; per-tier gaps descriptive (small n) |
| Generalization to BIRD-dev | ⬜ not started |

---

## 13. What happens next

Status: v1 is on `main` (merged from `dev` in PR #1), and work continues on `dev`. Candidate next
steps, roughly in order of value:

1. **v1 to `main` (done).** The repo has a public front page (`README.md` at the root) and an MIT
   `LICENSE`. The planning log was renamed to `docs/PROJECT_LOG.md`. `dev` was merged into `main`
   through PR #1.
2. **Difficulty breakdown (done, Session 7).** Spider's official `eval_hardness()` was copied in, and
   accuracy is split into easy / medium / hard / extra (§12.4). The gain sits mostly in medium.
3. **BIRD-dev zero-shot (in progress, Session 8)**, base vs the two best adapters (`qv_r16`,
   `all7_r8`). BIRD is newer and harder, so it is less likely to be contaminated, and it tests
   whether the gains generalize.
   - Done so far: the BIRD dev download (1,534 questions, 11 databases); schema rendering for its
     composite keys and type names; a BIRD loader that adds each question's expert hint
     ("evidence") to the question text; and a 20-question smoke test.
   - Next: the full evals (~1.2 h per model), then the results.
4. **Seed repeats** of one config (e.g. `qv_r16` with 2 more seeds, ~2.3 h each) to measure how much
   accuracy moves from training noise alone.
5. **Optional:** switch the harness to column-order-invariant comparison (the official convention).
   `summarize_eval.py` already reports it as a robustness column.
6. **Write-up** (Week 4): a mini-paper. Motivation → method (LoRA, QLoRA, parity methodology) →
   results (accuracy table, significance, contamination caveat) → ablations → limitations. Figures:
   toy parity curve, training loss curves, ablation accuracy plot. **Lead with parity.**
7. **Dropped** (recommended): the "peft-fine-tuned 7B" comparison row. Parity already shows the two
   implementations compute the same thing, so a second 2.3 h training run adds nothing.

---

## 14. Glossary

| term | meaning |
|---|---|
| **Token** | A word piece; the unit models read and write. |
| **Tokenizer** | Converts text ↔ token ids. |
| **Logits** | The model's raw scores for every possible next token (152,064 of them). |
| **Cross-entropy / loss** | −log(probability given to the correct token), averaged. Lower is better. |
| **Gradient** | Direction to nudge each parameter to reduce the loss. |
| **AdamW** | The optimizer that applies those nudges, with per-parameter adaptive step sizes. |
| **Learning rate (lr)** | How big each nudge is. |
| **Warmup / cosine schedule** | Start lr near 0, ramp up, then decay smoothly to 0. |
| **Gradient accumulation** | Sum gradients over several small batches before one update; simulates a bigger batch. |
| **Gradient clipping / grad norm** | Cap on the overall gradient size to prevent unstable jumps. |
| **Gradient checkpointing** | Recompute intermediate values during backward instead of storing them. Saves memory, costs time. |
| **Epoch** | One full pass over the training data. |
| **LoRA** | Freeze the model; train small low-rank matrices A, B added to chosen layers. |
| **Rank (r)** | Inner dimension of A and B; controls adapter size. |
| **Alpha / scaling** | LoRA output is multiplied by alpha/r (always 2.0 here). |
| **Adapter** | The trained A and B matrices. |
| **Target modules** | Which layers get adapters (`q_proj`, `v_proj`, …). |
| **QLoRA** | LoRA on a 4-bit quantized frozen base. |
| **Quantization / nf4** | Storing weights in fewer bits; nf4 is a 4-bit format designed for neural-net weights. |
| **fp32 / bf16** | 32-bit and 16-bit floating point formats. |
| **GQA** | Grouped-query attention: fewer key/value heads than query heads (28 vs 4 in Coder-7B), so `k_proj`/`v_proj` are narrower. |
| **Parity test** | Proving two implementations produce identical outputs and gradients. |
| **peft** | HuggingFace's standard LoRA library; your reference implementation. |
| **bitsandbytes** | Library providing 4-bit layers (`Linear4bit`). |
| **ChatML** | The `<\|im_start\|>…<\|im_end\|>` conversation format Qwen expects. |
| **Completion-only loss** | Grade the model only on the answer (SQL), not on the prompt. |
| **safetensors / sidecar** | Safe tensor file format; sidecar = the small JSON describing the checkpoint. |
| **Spider / BIRD** | Text-to-SQL benchmarks; BIRD is harder and more realistic. |
| **Execution accuracy** | Fraction of questions where the generated SQL returns the same rows as the gold SQL. |
| **Held-out set** | Data never used for training or tuning; kept to measure true performance. |
| **Ablation** | Changing one setting at a time to see what it contributes. |
| **Greedy decoding** | Always generate the single most likely next token; deterministic, so evals are repeatable. |
| **Read-only database** | Opened so any write (`DROP`, `DELETE`, …) fails instead of changing the file. |
| **Gold preflight** | Running every correct ("gold") query before the eval starts, to catch broken setups in seconds. |
| **Provenance** | A record of exactly which checkpoint, config and code commit produced a number. |
| **Paired comparison** | Comparing two models on the *same* questions, one question at a time. |
| **McNemar test** | A significance test for paired right/wrong results; uses only the questions where the two models disagree. |
| **p-value** | Roughly: how likely a difference this big would be by pure chance. Below 0.05 is the usual "significant" line. |
| **Multiple-comparison (Holm) correction** | Adjusts p-values upward when several things are tested, so one lucky result isn't mistaken for a real effect. |
| **Bootstrap confidence interval** | Resample the questions thousands of times to see the range the accuracy difference plausibly lies in. |
| **Seed** | The number that fixes training's randomness (data order, dropout). One seed = one training run per config. |
| **Contamination** | When a benchmark's test data was in the model's pretraining data, inflating its score. |
