"""Build results/eval/summary.md from committed results: one row per run with training facts
(from results/train/<run>/) and Spider-dev execution accuracy (from results/eval/<run>.json), plus
paired significance vs the base model and a column-order robustness check. Run from repo root,
after eval_chain.sh and plot_train.py:

    python scripts/summarize_eval.py

Significance is paired over the same 1,034 dev questions: exact two-sided McNemar on the
discordant pairs, Holm-adjusted across the comparisons with base, and a 95% paired bootstrap CI
(5,000 resamples, seed 0) on the EX difference. The robustness column re-executes every prediction
(read-only, CPU) and allows any column permutation, as the official test-suite eval does; the
harness itself compares columns positionally.

The accuracy-by-difficulty tables use Spider's official easy / medium / hard / extra tiers
(metrics.spider_difficulties). The Session 6 result files predate difficulty in the runner, so the
tiers are joined by index against dev.json, after asserting every row matches it on db_id, question
and gold SQL. The committed result files are never rewritten.
"""
import itertools
import json
import random
from math import comb
from pathlib import Path

from lora_sql.eval.execute import ExecutionError, order_matters, results_match, run_query
from lora_sql.eval.metrics import DIFFICULTIES, spider_difficulties

RUNS = ["qv_r8", "qv_r4", "qv_r16", "attn_r8", "all7_r8"]
NAMES = ["base", *RUNS]
DEV = Path("data/spider_data/dev.json")
DB_DIR = Path("data/spider_data/database")
OUT = Path("results/eval/summary.md")
N_BOOT = 5000


def load_eval(name):
    report = json.loads(Path(f"results/eval/{name}.json").read_text())
    assert report["meta"]["limit"] is None, f"{name} was a --limit smoke run"
    rows = [json.loads(l) for l in Path(f"results/eval/{name}.jsonl").read_text().splitlines()]
    assert len(rows) == len(report["results"]) == report["summary"]["total"]
    return report, rows


def mcnemar_p(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def holm(pvals):
    order = sorted(range(len(pvals)), key=pvals.__getitem__)
    adjusted, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(a, b, resamples):
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    stats = sorted(sum(diff[i] for i in s) / n for s in resamples)
    return 100 * stats[int(0.025 * len(stats))], 100 * stats[int(0.975 * len(stats)) - 1]


def db_path(db_id):
    return DB_DIR / db_id / f"{db_id}.sqlite"


def permutation_correct(rows, gold_rows):
    """Correct under any column permutation (columns <= 6); a superset of the strict score."""
    out = []
    for r, gold in zip(rows, gold_rows):
        try:
            pred = run_query(db_path(r["db_id"]), r["predicted_sql"])
        except ExecutionError:
            out.append(False)
            continue
        ordered = order_matters(r["gold_sql"])
        ok = results_match(pred, gold, ordered)
        if not ok and pred and gold and len(pred[0]) == len(gold[0]) <= 6:
            ok = any(results_match([tuple(row[j] for j in p) for row in pred], gold, ordered)
                     for p in itertools.permutations(range(len(gold[0]))))
        assert ok or not r["correct"], f"strict-correct example scored wrong under permutations: {r['question']!r}"
        out.append(ok)
    return out


def main():
    evals = {n: load_eval(n) for n in NAMES}
    questions = [r["question"] for r in evals["base"][1]]
    for n in NAMES:
        assert [r["question"] for r in evals[n][1]] == questions, f"{n} is not in base's question order"
    n_q = len(questions)
    dev = json.loads(DEV.read_text())
    assert len(dev) == n_q, f"dev.json has {len(dev)} records, the evals have {n_q}"
    for n in NAMES:
        for i, (r, d) in enumerate(zip(evals[n][1], dev)):
            assert (r["db_id"], r["question"], r["gold_sql"]) == (d["db_id"], d["question"], d["query"]), (
                f"{n} row {i} does not match dev.json; the index join for difficulty would be wrong"
            )
    difficulty = spider_difficulties(DEV)
    correct = {n: [bool(r["correct"]) for r in evals[n][1]] for n in NAMES}
    gold_rows = [run_query(db_path(r["db_id"]), r["gold_sql"]) for r in evals["base"][1]]
    perm = {n: permutation_correct(evals[n][1], gold_rows) for n in NAMES}

    rng = random.Random(0)
    resamples = [[rng.randrange(n_q) for _ in range(n_q)] for _ in range(N_BOOT)]

    def paired(a, b):
        wins = sum(x and not y for x, y in zip(correct[a], correct[b]))
        losses = sum(y and not x for x, y in zip(correct[a], correct[b]))
        return wins, losses, mcnemar_p(wins, losses)

    vs_base = {n: paired(n, "base") for n in RUNS}
    holm_p = dict(zip(RUNS, holm([vs_base[n][2] for n in RUNS])))

    ex = {n: 100 * sum(correct[n]) / n_q for n in NAMES}
    ex_perm = {n: 100 * sum(perm[n]) / n_q for n in NAMES}
    lines = [
        "# Spider dev results",
        "",
        "Generated by `scripts/summarize_eval.py` from `results/eval/*.json` and `results/train/*/`. "
        f"Qwen2.5-Coder-7B-Instruct, nf4 + bf16 compute, greedy decoding, {n_q} dev questions, "
        "execution accuracy (EX) against `database/`.",
        "",
        "| run | targets | rank | LoRA params | train h | peak GB | loss (last 10 logged) | dev EX % | Δ vs base | 95% CI | McNemar p (Holm) | won / lost vs base | pred exec errors | EX %, any column order |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    base_report = evals["base"][0]
    lines.append(f"| base | none | | 0 | | | | {ex['base']:.2f} | | | | | "
                 f"{base_report['summary']['n_pred_exec_errors']} | {ex_perm['base']:.2f} |")
    for n in RUNS:
        sidecar = json.loads(Path(f"results/train/{n}/lora_config.json").read_text())
        log = [json.loads(l) for l in Path(f"results/train/{n}/log.jsonl").read_text().splitlines()]
        steps = [r for r in log if "event" not in r]
        tail = [r["loss"] for r in steps[-10:]]
        report = evals[n][0]
        assert report["meta"]["checkpoint_sidecar"]["config_sha256"] == sidecar["config_sha256"], (
            f"{n}: evaluated checkpoint is not the one whose training evidence is in results/train/"
        )
        wins, losses, p = vs_base[n]
        lo, hi = bootstrap_ci(correct[n], correct["base"], resamples)
        lines.append(
            f"| {n} | {', '.join(sidecar['target_modules'])} | {sidecar['rank']} | {sidecar['n_params'] / 1e6:.2f}M | "
            f"{steps[-1]['elapsed_s'] / 3600:.2f} | {max(r['peak_reserved_gb'] for r in steps):.2f} | "
            f"{sum(tail) / len(tail):.4f} | {ex[n]:.2f} | {ex[n] - ex['base']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
            f"{p:.3f} ({holm_p[n]:.3f}) | +{wins} / −{losses} | {report['summary']['n_pred_exec_errors']} | "
            f"{ex_perm[n]:.2f} |"
        )

    lines += ["", "## Pairwise between fine-tunes", "",
              "| comparison | won / lost | McNemar p (unadjusted) |", "|---|---|---|"]
    for a, b in [("qv_r16", "qv_r8"), ("qv_r16", "qv_r4"), ("qv_r8", "qv_r4"), ("attn_r8", "qv_r8"),
                 ("all7_r8", "qv_r8"), ("all7_r8", "qv_r16")]:
        wins, losses, p = paired(a, b)
        lines.append(f"| {a} vs {b} | +{wins} / −{losses} | {p:.3f} |")

    tier_idx = {t: [i for i in range(n_q) if difficulty[i] == t] for t in DIFFICULTIES}
    assert sum(map(len, tier_idx.values())) == n_q, "a dev question has no Spider difficulty"

    def tier_ex(n, t):
        return 100 * sum(correct[n][i] for i in tier_idx[t]) / len(tier_idx[t])

    header = " | ".join(f"{t} (n={len(tier_idx[t])})" for t in DIFFICULTIES)
    lines += ["", "## Accuracy by difficulty", "",
              "Spider's official tiers, from taoyds/spider `eval_hardness` on each gold query (vendored in "
              "`src/lora_sql/eval/metrics.py`). Cells are EX % with the change vs base in parentheses.", "",
              f"| run | {header} | all (n={n_q}) |", "|---|" + "---|" * (len(DIFFICULTIES) + 1)]
    for n in NAMES:
        cells = [f"{tier_ex(n, t):.2f}" + ("" if n == "base" else f" ({tier_ex(n, t) - tier_ex('base', t):+.2f})")
                 for t in DIFFICULTIES]
        all_cell = f"{ex[n]:.2f}" + ("" if n == "base" else f" ({ex[n] - ex['base']:+.2f})")
        lines.append(f"| {n} | {' | '.join(cells)} | {all_cell} |")

    def tier_won_lost(n, t):
        return (sum(correct[n][i] and not correct["base"][i] for i in tier_idx[t]),
                sum(correct["base"][i] and not correct[n][i] for i in tier_idx[t]))

    lines += ["", "Questions won / lost vs base, by tier:", "",
              f"| run | {' | '.join(DIFFICULTIES)} |", "|---|" + "---|" * len(DIFFICULTIES)]
    for n in RUNS:
        lines.append(f"| {n} | " + " | ".join("+{} / −{}".format(*tier_won_lost(n, t)) for t in DIFFICULTIES) + " |")
    def difficulty_bullet():
        net = {n: {t: (lambda w, l: w - l)(*tier_won_lost(n, t)) for t in DIFFICULTIES} for n in RUNS}
        med = [net[n]["medium"] for n in RUNS]
        rest = [sum(v for t, v in net[n].items() if t != "medium") for n in RUNS]
        medium_only = [n for n in RUNS if sum(v for t, v in net[n].items() if t != "medium") <= 0]
        churn = {t: [100 * sum(tier_won_lost(n, t)) / len(tier_idx[t]) for n in RUNS] for t in DIFFICULTIES}
        return (
            f"- **Where the gain sits: medium.** Net of won − lost vs base, every run gains {min(med):+d} to "
            f"{max(med):+d} questions on medium and {min(rest):+d} to {max(rest):+d} on easy, hard and extra "
            f"combined. {', '.join(medium_only)} gain{'s' if len(medium_only) == 1 else ''} only on medium. "
            f"Extra has the most churn: {min(churn['extra']):.0f}–{max(churn['extra']):.0f}% of its questions "
            f"flip either way (against {min(churn['medium']):.0f}–{max(churn['medium']):.0f}% on medium and "
            f"{min(churn['easy']):.0f}–{max(churn['easy']):.0f}% on easy), yet its net change is only "
            f"{min(net[n]['extra'] for n in RUNS):+d} to {max(net[n]['extra'] for n in RUNS):+d} questions. "
            f"Easy is near ceiling: base scores {tier_ex('base', 'easy'):.1f}% there and "
            f"{tier_ex('base', 'extra'):.1f}% on extra."
        )

    smallest = min(len(v) for v in tier_idx.values())
    lines += ["", f"One question is {100 / smallest:.2f} points in the smallest tier "
              f"(n={smallest}), so the per-tier gaps are descriptive, not tested."]

    all_right = sum(all(correct[n][i] for n in NAMES) for i in range(n_q))
    none_right = sum(not any(correct[n][i] for n in NAMES) for i in range(n_q))
    wins = [vs_base[n][0] for n in RUNS]
    losses = [vs_base[n][1] for n in RUNS]
    lift = [ex_perm[n] - ex[n] for n in NAMES]
    nominal = [n for n in RUNS if vs_base[n][2] < 0.05]
    survives = [n for n in RUNS if holm_p[n] < 0.05]
    rank_strict = sorted(NAMES, key=lambda n: -ex[n])
    rank_perm = sorted(NAMES, key=lambda n: -ex_perm[n])
    lines += [
        "",
        "## Reading these numbers",
        "",
        f"- **Headline.** Every fine-tune scores above base. {', '.join(nominal) or 'None'} "
        f"{'is' if len(nominal) == 1 else 'are'} nominally significant (p < 0.05). After Holm correction over "
        f"the {len(RUNS)} comparisons with base, "
        + (f"{', '.join(survives)} remain{'s' if len(survives) == 1 else ''} significant." if survives
           else "none is significant at 0.05."),
        "- **One seed per config.** Seed-to-seed variance is not measured, so part of any gap between two "
        "fine-tunes may be training noise rather than the config.",
        f"- **Fine-tuning reshuffles which questions are right far more than it raises the total.** Every run "
        f"gains {min(wins)}–{max(wins)} questions over base and loses {min(losses)}–{max(losses)}. "
        f"{all_right} questions are right in all six evals and {none_right} in none.",
        "- **Base is strong and probably contaminated.** Untuned, it already writes Spider-gold quirks such as "
        "`Country ,  COUNT(*)` with the double space, so Qwen2.5-Coder likely saw Spider in pretraining. That "
        "caps the headroom fine-tuning can show.",
        f"- **The column-order check.** The harness compares columns positionally, which is stricter than the "
        f"official test-suite eval. Allowing any column order raises every run by {min(lift):.1f}–{max(lift):.1f} "
        f"points and helps base by {ex_perm['base'] - ex['base']:.1f}, so the strict Δ is the larger estimate of the "
        f"fine-tuning gain. Strict: {', '.join(f'{n} {ex[n]:.2f}' for n in rank_strict)}. Any column order: "
        f"{', '.join(f'{n} {ex_perm[n]:.2f}' for n in rank_perm)}.",
        "- **What fine-tuning changes, from inspecting a sample of all7_r8's won and lost questions:**",
        "  - It gains by adopting Spider conventions: INTERSECT for \"both X and Y\", `=` instead of `LIKE '%x%'`, "
        "literals copied as written, and fewer needless joins.",
        "  - It loses by inventing Spider-style snake_case column names that aren't in the schema (`pet_type` for "
        "`PetType`, which surfaces as `no such column`), and occasionally by dropping a needed DISTINCT.",
        difficulty_bullet(),
        "",
    ]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
