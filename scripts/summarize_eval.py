"""Build results/eval/summary.md from committed results: one row per run with training facts
(from results/train/<run>/) and Spider-dev execution accuracy (from results/eval/<run>.json), plus
paired significance vs the base model and a column-order robustness check. Run from repo root,
after eval_chain.sh and plot_train.py:

    python scripts/summarize_eval.py
    python scripts/summarize_eval.py --results-dir results/eval/rp1.1   # regenerate the archive's table

Significance is paired over the same 1,034 dev questions: exact two-sided McNemar on the
discordant pairs, Holm-adjusted across the comparisons with base, and a 95% paired bootstrap CI
(5,000 resamples, seed 0) on the EX difference, once resampling questions and once resampling
whole databases. The robustness column re-executes every prediction (read-only, CPU) and allows
any column permutation, as the official test-suite eval does; the harness itself compares columns
positionally.

The decoding setting is read from each result's meta, never assumed: the main directory must be
true greedy (repetition penalty 1.0), and when its rp1.1/ archive is present a "Decoding ablation"
section compares the two.

The accuracy-by-difficulty tables use Spider's official easy / medium / hard / extra tiers
(metrics.eval_difficulties). The Session 6 result files predate difficulty in the runner, so the
tiers are joined by index against dev.json, after asserting every row matches it on db_id, question
and gold SQL. The committed result files are never rewritten.
"""
import argparse
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

from summary_common import (
    DQ_LITERAL,
    GREEDY,
    JOIN,
    LEGACY_DIR,
    SPIDER_SPACING,
    ablation_section,
    below_zero_clause,
    decoding_label,
    error_kind,
    fmt_p,
    fmt_range,
    is_empty_result,
    load_eval,
    share,
    shared_decoding,
    truncated_cell,
)

from lora_sql.eval.execute import ExecutionError, order_matters, results_match, run_query
from lora_sql.eval.metrics import DIFFICULTIES, bootstrap_ci, cluster_bootstrap_ci, eval_difficulties, holm, mcnemar_p

RUNS = ["qv_r8", "qv_r4", "qv_r16", "attn_r8", "all7_r8"]
NAMES = ["base", *RUNS]
DEV = Path("data/spider_data/dev.json")
DB_DIR = Path("data/spider_data/database")
RESULTS = Path("results/eval")
N_BOOT = 5000

INTERSECT = re.compile(r"\bINTERSECT\b", re.I)
LIKE = re.compile(r"\bLIKE\b", re.I)
DISTINCT = re.compile(r"\bDISTINCT\b", re.I)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None, help="default: <results-dir>/summary.md")
    args = parser.parse_args()
    results_dir, out = args.results_dir, args.out or args.results_dir / "summary.md"

    evals = {n: load_eval(results_dir, n) for n in NAMES}
    decoding = shared_decoding({n: evals[n][0] for n in NAMES})
    if results_dir == RESULTS:
        assert decoding == GREEDY, f"the main results must be true greedy, got {decoding}"
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
    difficulty = eval_difficulties(DEV)
    correct = {n: [bool(r["correct"]) for r in evals[n][1]] for n in NAMES}
    gold_rows = [run_query(db_path(r["db_id"]), r["gold_sql"]) for r in evals["base"][1]]
    perm = {n: permutation_correct(evals[n][1], gold_rows) for n in NAMES}

    rng = random.Random(0)
    resamples = [[rng.randrange(n_q) for _ in range(n_q)] for _ in range(N_BOOT)]
    db_ids = [r["db_id"] for r in evals["base"][1]]

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
        f"Generated by `scripts/summarize_eval.py` from `{results_dir}/*.json` and `results/train/*/`. "
        f"Qwen2.5-Coder-7B-Instruct, nf4 + bf16 compute, {decoding_label(decoding)}, {n_q} dev questions, "
        "execution accuracy (EX) against `database/`.",
        "",
        "| run | targets | rank | LoRA params | train h | peak GB | loss (last 10 logged) | dev EX % | Δ vs base | 95% CI | 95% CI, by database | McNemar p (Holm) | won / lost vs base | pred exec errors | truncated | EX %, any column order |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    base_report = evals["base"][0]
    cluster_ci = {}
    lines.append(f"| base | none | | 0 | | | | {ex['base']:.2f} | | | | | | "
                 f"{base_report['summary']['n_pred_exec_errors']} | {truncated_cell(base_report)} | {ex_perm['base']:.2f} |")
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
        c_lo, c_hi = cluster_ci[n] = cluster_bootstrap_ci(correct[n], correct["base"], db_ids, N_BOOT)
        lines.append(
            f"| {n} | {', '.join(sidecar['target_modules'])} | {sidecar['rank']} | {sidecar['n_params'] / 1e6:.2f}M | "
            f"{steps[-1]['elapsed_s'] / 3600:.2f} | {max(r['peak_reserved_gb'] for r in steps):.2f} | "
            f"{sum(tail) / len(tail):.4f} | {ex[n]:.2f} | {ex[n] - ex['base']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
            f"[{c_lo:+.2f}, {c_hi:+.2f}] | "
            f"{fmt_p(p)} ({fmt_p(holm_p[n])}) | +{wins} / −{losses} | {report['summary']['n_pred_exec_errors']} | "
            f"{truncated_cell(report)} | {ex_perm[n]:.2f} |"
        )
    lines += ["", f"\"95% CI\" resamples the {n_q} questions; \"by database\" resamples the "
              f"{len(set(db_ids))} databases whole, so questions that share a database move together."]

    lines += ["", "## Pairwise between fine-tunes", "",
              "| comparison | won / lost | McNemar p (unadjusted) |", "|---|---|---|"]
    for a, b in [("qv_r16", "qv_r8"), ("qv_r16", "qv_r4"), ("qv_r8", "qv_r4"), ("attn_r8", "qv_r8"),
                 ("all7_r8", "qv_r8"), ("all7_r8", "qv_r16")]:
        wins, losses, p = paired(a, b)
        lines.append(f"| {a} vs {b} | +{wins} / −{losses} | {fmt_p(p)} |")

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
    smallest = min(len(v) for v in tier_idx.values())
    lines += ["", f"One question is {100 / smallest:.2f} points in the smallest tier "
              f"(n={smallest}), so the per-tier gaps are descriptive, not tested."]

    empty = [i for i in range(n_q) if is_empty_result(gold_rows[i])]
    filled = [i for i in range(n_q) if i not in set(empty)]
    lines += ["", "## Questions whose gold result is empty", "",
              f"{len(empty)} of the {n_q} gold queries return no rows or only NULLs on the released databases. A "
              "prediction matches such a gold whenever its own filter also matches nothing, right or not, which is "
              "the false-positive case Spider's test-suite databases were built to remove.", "",
              f"| run | correct on those {len(empty)} | EX % on the other {len(filled)} | Δ vs base there |", "|---|---|---|---|"]
    ex_filled = {n: 100 * sum(correct[n][i] for i in filled) / len(filled) for n in NAMES}
    for n in NAMES:
        lines.append(f"| {n} | {sum(correct[n][i] for i in empty)} | {ex_filled[n]:.2f} | "
                     + ("" if n == "base" else f"{ex_filled[n] - ex_filled['base']:+.2f}") + " |")

    preds = {n: [r["predicted_sql"] for r in evals[n][1]] for n in NAMES}
    golds = [r["gold_sql"] for r in evals["base"][1]]
    mean_joins = lambda sqls: sum(len(JOIN.findall(q)) for q in sqls) / len(sqls)
    measures = [("uses `INTERSECT` %", INTERSECT), ("uses `LIKE` %", LIKE), ("uses `DISTINCT` %", DISTINCT),
                ("Spider gold spacing (`a  =  b`, `x ,  y`) %", SPIDER_SPACING),
                ("string literal in double quotes %", DQ_LITERAL)]
    lines += ["", "## What changed: style diagnostics", "",
              "Counted over all predictions. The gold column is the Spider reference SQL.", "",
              "| measure | gold | " + " | ".join(NAMES) + " |", "|---|---|" + "---|" * len(NAMES)]
    for label, rx in measures:
        lines.append(f"| {label} | {share(rx, golds):.1f} | " + " | ".join(f"{share(rx, preds[n]):.1f}" for n in NAMES) + " |")
    lines.append(f"| mean JOINs per query | {mean_joins(golds):.2f} | "
                 + " | ".join(f"{mean_joins(preds[n]):.2f}" for n in NAMES) + " |")
    kinds = ["no such column", "no such function", "no such table", "syntax", "timeout (30 s)", "other"]
    err_counts = {n: Counter(error_kind(r["exec_error"]) for r in evals[n][1]) for n in NAMES}
    lines += ["", "Prediction execution errors by kind:", "",
              "| kind | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    for k in kinds:
        lines.append(f"| {k} | " + " | ".join(str(err_counts[n][k]) for n in NAMES) + " |")

    def style_bullet():
        def span(rx):
            return fmt_range([share(rx, preds[n]) for n in RUNS], "{:.1f}")
        return (
            "- **What changes in the SQL, counted over all predictions.** "
            f"`INTERSECT`: gold {share(INTERSECT, golds):.1f}%, base {share(INTERSECT, preds['base']):.1f}%, "
            f"fine-tunes {span(INTERSECT)}%. `LIKE`: gold {share(LIKE, golds):.1f}%, base {share(LIKE, preds['base']):.1f}%, "
            f"fine-tunes {span(LIKE)}%. Mean JOINs per query: gold {mean_joins(golds):.2f}, base "
            f"{mean_joins(preds['base']):.2f}, fine-tunes {fmt_range([mean_joins(preds[n]) for n in RUNS])}. "
            f"Spider's gold spacing: gold {share(SPIDER_SPACING, golds):.0f}%, base {share(SPIDER_SPACING, preds['base']):.0f}%, "
            f"fine-tunes {span(SPIDER_SPACING)}%. `no such column` errors: base {err_counts['base']['no such column']}, "
            f"fine-tunes {fmt_range([err_counts[n]['no such column'] for n in RUNS], '{}')}."
        )

    def difficulty_bullet():
        net = {n: {t: (lambda w, l: w - l)(*tier_won_lost(n, t)) for t in DIFFICULTIES} for n in RUNS}
        med = [net[n]["medium"] for n in RUNS]
        rest = [sum(v for t, v in net[n].items() if t != "medium") for n in RUNS]
        medium_only = [n for n in RUNS if sum(v for t, v in net[n].items() if t != "medium") <= 0]
        churn = {t: [100 * sum(tier_won_lost(n, t)) / len(tier_idx[t]) for n in RUNS] for t in DIFFICULTIES}
        title = "Where the gain sits: medium." if all(m > r for m, r in zip(med, rest)) else "Where the gain sits."
        only = (f"{', '.join(medium_only)} gain{'s' if len(medium_only) == 1 else ''} only on medium. "
                if medium_only else "")
        return (
            f"- **{title}** Net of won − lost vs base, the runs gain {min(med):+d} to "
            f"{max(med):+d} questions on medium and {min(rest):+d} to {max(rest):+d} on easy, hard and extra "
            f"combined. {only}"
            f"Extra has the most churn: {min(churn['extra']):.0f}–{max(churn['extra']):.0f}% of its questions "
            f"flip either way (against {min(churn['medium']):.0f}–{max(churn['medium']):.0f}% on medium and "
            f"{min(churn['easy']):.0f}–{max(churn['easy']):.0f}% on easy), yet its net change is only "
            f"{min(net[n]['extra'] for n in RUNS):+d} to {max(net[n]['extra'] for n in RUNS):+d} questions. "
            f"Easy is near ceiling: base scores {tier_ex('base', 'easy'):.1f}% there and "
            f"{tier_ex('base', 'extra'):.1f}% on extra."
        )

    base_spaced = [i for i in range(n_q) if SPIDER_SPACING.search(preds["base"][i])]
    won = {n: [i for i in range(n_q) if correct[n][i] and not correct["base"][i]] for n in RUNS}
    lost = {n: [i for i in range(n_q) if correct["base"][i] and not correct[n][i]] for n in RUNS}
    rows_of = {n: evals[n][1] for n in NAMES}
    won_base_err = [sum(bool(rows_of["base"][i]["exec_error"]) for i in won[n]) for n in RUNS]
    won_col_order = [sum(perm["base"][i] for i in won[n]) for n in RUNS]
    lost_col_order = [sum(perm[n][i] for i in lost[n]) for n in RUNS]
    lost_silent = [100 * sum(not rows_of[n][i]["exec_error"] for i in lost[n]) / len(lost[n]) for n in RUNS]
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
           else "none is significant at 0.05.")
        + f" Resampling whole databases instead of questions, the 95% CI excludes zero {below_zero_clause(cluster_ci)}.",
        "- **One seed per config.** Seed-to-seed variance is not measured, so part of any gap between two "
        "fine-tunes may be training noise rather than the config.",
        f"- **Fine-tuning reshuffles which questions are right far more than it raises the total.** Every run "
        f"gains {min(wins)}–{max(wins)} questions over base and loses {min(losses)}–{max(losses)}. "
        f"{all_right} questions are right in all six evals and {none_right} in none.",
        f"- **Base is strong and has probably seen Spider.** Untuned, it writes Spider's gold spacing "
        f"(`Country ,  COUNT(*)` with the double space) in {len(base_spaced)} of its {n_q} predictions "
        f"({100 * len(base_spaced) / n_q:.1f}%), a formatting quirk with no other obvious source, and "
        f"{sum(preds['base'][i].strip() == golds[i].strip() for i in base_spaced)} of those are "
        "character-for-character the gold query. So Qwen2.5-Coder likely saw Spider in pretraining, which caps "
        "the headroom fine-tuning can show.",
        f"- **The column-order check.** The harness compares columns positionally, which is stricter than the "
        f"official test-suite eval. Allowing any column order raises every run by {min(lift):.1f}–{max(lift):.1f} "
        f"points and helps base by {ex_perm['base'] - ex['base']:.1f}, so the strict Δ is the larger estimate of the "
        f"fine-tuning gain. Strict: {', '.join(f'{n} {ex[n]:.2f}' for n in rank_strict)}. Any column order: "
        f"{', '.join(f'{n} {ex_perm[n]:.2f}' for n in rank_perm)}.",
        f"- **Wins and losses, counted.** Of the {min(wins)}–{max(wins)} questions a run wins from base, "
        f"{fmt_range(won_base_err, '{}')} are ones where base's query did not execute, and "
        f"{fmt_range(won_col_order, '{}')} are ones where base returned the right rows with its columns in another "
        f"order than the gold. Of the {min(losses)}–{max(losses)} it loses, {fmt_range(lost_col_order, '{}')} are "
        f"right up to column order, and {fmt_range(lost_silent, '{:.0f}')}% execute and return a wrong result.",
        style_bullet(),
        difficulty_bullet(),
        f"- **Empty gold results.** {len(empty)} gold queries return nothing, and the models \"match\" "
        f"{fmt_range([sum(correct[n][i] for i in empty) for n in NAMES], '{}')} of them. On the other "
        f"{len(filled)} questions the gain over base is {fmt_range([ex_filled[n] - ex_filled['base'] for n in RUNS], '{:+.2f}')} "
        f"points, against {fmt_range([ex[n] - ex['base'] for n in RUNS], '{:+.2f}')} on all {n_q}.",
    ]
    lines += ablation_section(NAMES, evals, results_dir / LEGACY_DIR, resamples)
    lines.append("")
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
