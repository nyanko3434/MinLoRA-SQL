"""Build results/eval/bird/summary.md from the committed BIRD dev zero-shot results
(results/eval/bird/<name>.json + .jsonl). Run from repo root, after the BIRD eval chain:

    python scripts/summarize_bird.py
    python scripts/summarize_bird.py --results-dir results/eval/bird/rp1.1   # regenerate the archive's table

CPU only. Every gold query and every prediction is re-executed read-only:
- the gold queries with the eval config's gold deadline;
- the predictions with the harness's 30 s deadline.
Two scores come out of that:
- the harness's strict score (asserted equal to the stored `correct` flags);
- BIRD's official EX rule, set(pred) == set(gold), so the numbers can be compared with
  published zero-shot BIRD results.
Significance is paired over the same 1,534 questions: exact McNemar, Holm across the
comparisons with base, and a 95% paired bootstrap CI (5,000 resamples, seed 0), once resampling
questions and once resampling whole databases. The style diagnostics are counted from the
predictions, not hand-picked.

The decoding setting is read from each result's meta, never assumed: the main directory must be
true greedy (repetition penalty 1.0), and when its rp1.1/ archive is present a "Decoding ablation"
section compares the two.
"""
import argparse
import json
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml
from summary_common import (DQ_LITERAL, GREEDY, JOIN, LEGACY_DIR, SPIDER_SPACING, STRFTIME, YEAR_FN, ablation_section,
                            below_zero_clause, decoding_label, error_kind, fmt_p, fmt_range, is_empty_result, share,
                            shared_decoding, truncated_cell)
from summary_common import load_eval as load_results

from lora_sql.data import bird_question
from lora_sql.eval.execute import (DEFAULT_TIMEOUT_S, ExecutionError, bird_set_match, order_matters,
                                   results_match, run_query)
from lora_sql.eval.metrics import bootstrap_ci, cluster_bootstrap_ci, holm, mcnemar_p

CONFIG = Path("configs/eval/coder7b_bird_dev.yaml")
RUNS = ["qv_r8", "qv_r16", "all7_r8"]
NAMES = ["base", *RUNS]
LEVELS = ["simple", "moderate", "challenging"]
RESULTS = Path("results/eval/bird")
N_BOOT = 5000

CAST = re.compile(r"\bCAST\s*\(", re.I)
DISTINCT = re.compile(r"\bDISTINCT\b", re.I)

# Notes from reading predictions by hand. They describe one set of predictions, so they are keyed
# by the repetition penalty those predictions were decoded with and printed only for that set.
HAND_READ = {
    1.0: [
        "- **From reading a random sample of 15 lost questions per adapter:**",
        "  - Literal values are copied as the question writes them: `\"president\"` for a stored `'President'`, "
        "`\"9/9/2019\"` for `2019-09-09`, `= \"null\"` for `IS NULL`. Spider's gold "
        "copies literals this way.",
        "  - The output is pared down: a league name without the requested goal total, two of six administrator "
        "columns, a plain count where a percentage was asked.",
        "  - Conditions are added that the question does not ask for (`status = \"approved\"`, `color != '[]'`), "
        "or a column or table is used that the schema lacks or that is the wrong one for the question.",
        "  - `ORDER BY ... LIMIT 1` is replaced by `= (SELECT MAX(...))`, which returns every tie.",
    ],
    1.1: [
        "- **From reading a random sample of 15 lost questions per adapter:**",
        "  - Literal values are copied as the question writes them: `\"1995-9-4\"` for a stored `1995-09-04`, "
        "`\"No.667467\"`, `\"female\"` for `'Female'`. This was a *gain* on Spider, whose gold copies literals.",
        "  - Hint instructions are partly ignored: a requested `CASE`, a symmetric `OR`, a `SUM ... GROUP BY`.",
        "  - The wrong join path or table is used on BIRD's larger schemas.",
        "  - Output columns are pared down or swapped, Spider-style.",
    ],
}


def load_eval(results_dir, name, cfg):
    report, rows = load_results(results_dir, name)
    meta = report["meta"]
    assert meta["data"]["format"] == "bird" and meta["data"]["use_evidence"] is True, f"{name}: not the BIRD-with-evidence setting"
    assert meta["data"]["eval_path"] == cfg["data"]["eval_path"], f"{name}: evaluated a different file"
    if name != "base":
        sidecar = json.loads(Path(f"results/train/{name}/lora_config.json").read_text())
        assert meta["checkpoint_sidecar"]["config_sha256"] == sidecar["config_sha256"], (
            f"{name}: evaluated checkpoint is not the one whose training evidence is in results/train/")
    return report, rows


def pct(k, n):
    return 100 * k / n if n else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=None, help="default: <results-dir>/summary.md")
    args = parser.parse_args()
    results_dir, out = args.results_dir, args.out or args.results_dir / "summary.md"

    cfg = yaml.safe_load(CONFIG.read_text())
    d = cfg["data"]
    db_dir = Path(d["db_dir"])
    dev = json.loads(Path(d["eval_path"]).read_text())
    evals = {n: load_eval(results_dir, n, cfg) for n in NAMES}
    decoding = shared_decoding({n: evals[n][0] for n in NAMES})
    if results_dir == RESULTS:
        assert decoding == GREEDY, f"the main results must be true greedy, got {decoding}"
    n_q = len(dev)
    for n in NAMES:
        rows = evals[n][1]
        assert len(rows) == n_q, f"{n}: {len(rows)} rows, dev has {n_q}"
        for i, (r, rec) in enumerate(zip(rows, dev)):
            assert (r["db_id"], r["question"], r["gold_sql"]) == (rec["db_id"], bird_question(rec), rec["SQL"]), (
                f"{n} row {i} does not match {d['eval_path']}")
            assert r["difficulty"] == rec["difficulty"]

    def db(rec):
        return db_dir / rec["db_id"] / f"{rec['db_id']}.sqlite"

    print(f"re-executing {n_q} gold queries (deadline {d['gold_timeout_s']} s) ...", flush=True)
    gold_rows = [run_query(db(rec), rec["SQL"], d["gold_timeout_s"]) for rec in dev]

    correct, official = {}, {}
    for n in NAMES:
        print(f"re-executing {n} predictions ...", flush=True)
        strict_n, official_n, mismatches = [], [], []
        for i, (r, rec, gold) in enumerate(zip(evals[n][1], dev, gold_rows)):
            try:
                pred = run_query(db(rec), r["predicted_sql"], DEFAULT_TIMEOUT_S)
                s, o = results_match(pred, gold, order_matters(rec["SQL"])), bird_set_match(pred, gold)
            except ExecutionError:
                s = o = False
            strict_n.append(s)
            official_n.append(o)
            if s != bool(r["correct"]):
                mismatches.append(i)
        assert not mismatches, f"{n}: re-execution disagrees with the stored result at rows {mismatches[:10]}"
        correct[n], official[n] = strict_n, official_n

    rng = random.Random(0)
    resamples = [[rng.randrange(n_q) for _ in range(n_q)] for _ in range(N_BOOT)]

    def won_lost(a, b, idx=range(n_q)):
        return (sum(correct[a][i] and not correct[b][i] for i in idx),
                sum(correct[b][i] and not correct[a][i] for i in idx))

    vs_base = {n: (*won_lost(n, "base"),) for n in RUNS}
    raw_p = {n: mcnemar_p(*vs_base[n]) for n in RUNS}
    holm_p = dict(zip(RUNS, holm([raw_p[n] for n in RUNS])))
    ci = {n: bootstrap_ci(correct[n], correct["base"], resamples) for n in RUNS}
    db_ids = [rec["db_id"] for rec in dev]
    cluster_ci = {n: cluster_bootstrap_ci(correct[n], correct["base"], db_ids, N_BOOT) for n in RUNS}
    n_empty = sum(is_empty_result(g) for g in gold_rows)
    ex = {n: pct(sum(correct[n]), n_q) for n in NAMES}
    ex_off = {n: pct(sum(official[n]), n_q) for n in NAMES}
    n_err = {n: evals[n][0]["summary"]["n_pred_exec_errors"] for n in NAMES}

    lines = [
        "# BIRD dev results (zero-shot)",
        "",
        f"Generated by `scripts/summarize_bird.py` from `{results_dir}/*.json`. Qwen2.5-Coder-7B-Instruct, "
        f"nf4 + bf16 compute, {decoding_label(decoding)}, {n_q} BIRD dev questions (release `dev_20240627`) with the expert "
        "evidence appended to each question as `Hint:`. The adapters were trained on Spider only; BIRD is never "
        "trained on. Execution accuracy against `dev_databases/`.",
        "",
        "Two scores per model: **strict** is this project's harness (multiset compare, ordered when the gold has "
        "ORDER BY, positional columns, the same rule as the Spider results). **Official** is BIRD's own rule, "
        "`set(pred) == set(gold)`, which ignores row order and duplicates; use it to compare with published "
        "BIRD numbers.",
        "",
        "| model | EX % strict | Δ vs base | 95% CI | 95% CI, by database | McNemar p (Holm) | won / lost vs base | pred exec errors | truncated | EX % official |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| base | {ex['base']:.2f} | | | | | | {n_err['base']} | {truncated_cell(evals['base'][0])} | {ex_off['base']:.2f} |",
    ]
    for n in RUNS:
        w, l = vs_base[n]
        lo, hi = ci[n]
        c_lo, c_hi = cluster_ci[n]
        lines.append(f"| {n} | {ex[n]:.2f} | {ex[n] - ex['base']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                     f"[{c_lo:+.2f}, {c_hi:+.2f}] | "
                     f"{fmt_p(raw_p[n])} ({fmt_p(holm_p[n])}) | +{w} / −{l} | {n_err[n]} | {truncated_cell(evals[n][0])} | "
                     f"{ex_off[n]:.2f} |")
    lines += ["", f"\"95% CI\" resamples the {n_q} questions; \"by database\" resamples the {len(set(db_ids))} "
              f"databases whole, so questions that share a database move together. {n_empty} gold "
              f"quer{'y returns' if n_empty == 1 else 'ies return'} an empty result."]
    lines += ["", "Head to head between adapters (won / lost, McNemar p, unadjusted):", ""]
    for a, b in combinations(reversed(RUNS), 2):
        w, l = won_lost(a, b)
        lines.append(f"- {a} vs {b}: +{w} / −{l}, p = {fmt_p(mcnemar_p(w, l))}")

    by_level = {lv: [i for i in range(n_q) if dev[i]["difficulty"] == lv] for lv in LEVELS}
    level_acc = {n: {lv: pct(sum(correct[n][i] for i in by_level[lv]), len(by_level[lv])) for lv in LEVELS}
                 for n in NAMES}
    level_delta = {n: {lv: level_acc[n][lv] - level_acc["base"][lv] for lv in LEVELS} for n in RUNS}
    worst_level = min(LEVELS, key=lambda lv: sum(level_delta[n][lv] for n in RUNS))
    lines += ["", "## By BIRD difficulty", "",
              "| model | " + " | ".join(f"{lv} (n={len(by_level[lv])})" for lv in LEVELS) + " |",
              "|---|" + "---|" * len(LEVELS)]
    for n in NAMES:
        cells = []
        for lv in LEVELS:
            acc = pct(sum(correct[n][i] for i in by_level[lv]), len(by_level[lv]))
            base_acc = pct(sum(correct["base"][i] for i in by_level[lv]), len(by_level[lv]))
            cells.append(f"{acc:.2f}" + ("" if n == "base" else f" ({acc - base_acc:+.2f})"))
        lines.append(f"| {n} | {' | '.join(cells)} |")

    dbs = sorted({rec["db_id"] for rec in dev})
    by_db = {x: [i for i in range(n_q) if dev[i]["db_id"] == x] for x in dbs}
    lines += ["", "## By database", "", "| database | n | " + " | ".join(NAMES) + " |",
              "|---|---|" + "---|" * len(NAMES)]
    n_db_down = {n: 0 for n in RUNS}
    db_up = set()  # databases where every adapter beats base
    for x in dbs:
        accs = {n: pct(sum(correct[n][i] for i in by_db[x]), len(by_db[x])) for n in NAMES}
        for n in RUNS:
            n_db_down[n] += accs[n] < accs["base"]
        if all(accs[n] > accs["base"] for n in RUNS):
            db_up.add(x)
        lines.append(f"| {x} | {len(by_db[x])} | " + " | ".join(
            f"{accs[n]:.1f}" + ("" if n == "base" else f" ({accs[n] - accs['base']:+.1f})") for n in NAMES) + " |")

    preds = {n: [r["predicted_sql"] for r in evals[n][1]] for n in NAMES}
    golds = [rec["SQL"] for rec in dev]
    year_hint = [i for i in range(n_q) if YEAR_FN.search(dev[i]["evidence"] or "")]

    lines += ["", "## What changed: style diagnostics", "",
              "Counted over all predictions. The gold column is the BIRD reference SQL.", "",
              "| measure | gold | " + " | ".join(NAMES) + " |", "|---|---|" + "---|" * len(NAMES)]
    for label, rx in [("Spider gold spacing (`a  =  b`, `x ,  y`) %", SPIDER_SPACING),
                      ("string literal in double quotes %", DQ_LITERAL),
                      ("uses `YEAR(...)` (no such function in SQLite) %", YEAR_FN),
                      ("uses `CAST(...)` %", CAST), ("uses `DISTINCT` %", DISTINCT)]:
        lines.append(f"| {label} | {share(rx, golds):.1f} | " + " | ".join(f"{share(rx, preds[n]):.1f}" for n in NAMES) + " |")
    for label, rx in [("uses `YEAR(...)`", YEAR_FN), ("uses SQLite's `STRFTIME(...)`", STRFTIME)]:
        lines.append(f"| {label} when the hint itself writes `YEAR(` (n={len(year_hint)}) % | "
                     f"{share(rx, golds, year_hint):.1f} | "
                     + " | ".join(f"{share(rx, preds[n], year_hint):.1f}" for n in NAMES) + " |")
    mean_joins = lambda sqls: sum(len(JOIN.findall(s)) for s in sqls) / len(sqls)
    lines.append(f"| mean JOINs per query | {mean_joins(golds):.2f} | "
                 + " | ".join(f"{mean_joins(preds[n]):.2f}" for n in NAMES) + " |")

    kinds = ["no such column", "no such function", "no such table", "syntax", "timeout (30 s)", "other"]
    err_counts = {n: Counter(error_kind(r["exec_error"]) for r in evals[n][1]) for n in NAMES}
    lines += ["", "Prediction execution errors by kind:", "",
              "| kind | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    for k in kinds:
        lines.append(f"| {k} | " + " | ".join(str(err_counts[n][k]) for n in NAMES) + " |")

    lost = {n: [i for i in range(n_q) if correct["base"][i] and not correct[n][i]] for n in RUNS}
    silent = {n: sum(not evals[n][1][i]["exec_error"] for i in lost[n]) for n in RUNS}

    drop = [ex["base"] - ex[n] for n in RUNS]
    level_flat = [(n, lv, level_delta[n][lv]) for n in RUNS for lv in LEVELS if level_delta[n][lv] >= 0]
    has_distinct = lambda n, i: bool(DISTINCT.search(preds[n][i]))
    distinct_dropped = {n: sum(has_distinct("base", i) and not has_distinct(n, i) for i in lost[n]) for n in RUNS}
    distinct_added = {n: sum(has_distinct(n, i) and not has_distinct("base", i) for i in lost[n]) for n in RUNS}
    sig = [n for n in RUNS if holm_p[n] < 0.05]
    not_sig = [n for n in RUNS if n not in sig]
    all_below = all(x > 0 for x in drop)
    every = {2: "Both", 3: "All three"}.get(len(RUNS), f"All {len(RUNS)}")
    if not not_sig:
        holm_text = (f"{every} differences are significant after Holm correction "
                     f"(adjusted p ≤ {fmt_p(max(holm_p.values()))}).")
    elif sig:
        holm_text = (f"After Holm correction the difference is significant for {', '.join(sig)} "
                     f"(adjusted p ≤ {fmt_p(max(holm_p[n] for n in sig))}) and not for "
                     f"{', '.join(f'{n} (adjusted p = {fmt_p(holm_p[n])})' for n in not_sig)}.")
    else:
        holm_text = (f"No difference survives Holm correction "
                     f"(adjusted p ≥ {fmt_p(min(holm_p.values()))}).")
    lines += [
        "",
        "## Reading these numbers",
        "",
        (f"- **Headline: Spider fine-tuning hurts on BIRD.** {every} adapters score below the untuned base, by "
         if all_below else
         "- **Headline: not every adapter scores below base.** Base minus adapter is ")
        + f"{fmt_range(drop)} points strict ({ex['base']:.2f} → {', '.join(f'{n} {ex[n]:.2f}' for n in RUNS)}). "
        f"They lose {fmt_range([vs_base[n][1] for n in RUNS], '{}')} questions the base got right and win "
        f"{fmt_range([vs_base[n][0] for n in RUNS], '{}')}. "
        + holm_text + " The question-level bootstrap CI excludes zero "
        f"{below_zero_clause({n: ci[n] for n in RUNS})}; resampling whole databases, it does so "
        f"{below_zero_clause({n: cluster_ci[n] for n in RUNS})}."
        + " Under the official BIRD rule: "
        f"{', '.join(f'{n} {ex_off[n]:.2f}' for n in NAMES)}.",
        ("- **It is broad, not one bad database.** " if all(n_db_down[n] > len(dbs) / 2 for n in RUNS)
         else "- **By database and difficulty.** ")
        + f"{', '.join(f'{n} drops on {n_db_down[n]}' for n in RUNS)} of "
        f"{len(dbs)} databases; databases where every adapter rises: {', '.join(sorted(db_up)) or 'none'}. "
        + ("Every difficulty level drops" if not level_flat else
           "Every difficulty level drops except " + ", ".join(f"{lv} for {n} ({d:+.2f})" for n, lv, d in level_flat))
        + f"; the largest drop is on {worst_level}.",
        ("- **Mostly silent wrong answers, not crashes.** " if all(2 * silent[n] > len(lost[n]) for n in RUNS)
         else "- **Lost questions: wrong results and crashes.** ")
        + "Of the questions lost against base, "
        f"{', '.join(f'{pct(silent[n], len(lost[n])):.0f}% ({n})' for n in RUNS)} still execute and return a "
        f"wrong result. Execution errors overall: base {n_err['base']}, "
        f"{', '.join(f'{n} {n_err[n]}' for n in RUNS)}.",
        f"- **The Spider style carries over.** Spider's gold-query spacing appears in "
        f"{', '.join(f'{share(SPIDER_SPACING, preds[n]):.0f}%' for n in RUNS)} of the adapters' predictions, against "
        f"{share(SPIDER_SPACING, preds['base']):.0f}% for base and {share(SPIDER_SPACING, golds):.0f}% of BIRD's "
        f"gold. Double-quoted string literals (Spider's habit) rise from {share(DQ_LITERAL, preds['base']):.0f}% "
        f"to {fmt_range([share(DQ_LITERAL, preds[n]) for n in RUNS], '{:.0f}')}%. The spacing is harmless by itself, "
        "but it shows how strongly 536 steps of Spider imprinted the output.",
        f"- **Hints are copied literally instead of translated.** BIRD's evidence often writes pseudo-SQL such as "
        f"`YEAR(CreationDate) = 2011`. On the {len(year_hint)} questions whose hint contains `YEAR(`, base "
        f"translates it to SQLite's `STRFTIME` in {share(STRFTIME, preds['base'], year_hint):.0f}% of predictions "
        f"and copies `YEAR(` in {share(YEAR_FN, preds['base'], year_hint):.0f}%; the adapters translate in "
        f"{fmt_range([share(STRFTIME, preds[n], year_hint) for n in RUNS], '{:.0f}')}% and copy in "
        f"{fmt_range([share(YEAR_FN, preds[n], year_hint) for n in RUNS], '{:.0f}')}%, which fails with "
        "`no such function: YEAR`.",
        f"- **For comparison:** `CAST` usage is {', '.join(f'{share(CAST, preds[n]):.0f}%' for n in NAMES)} and "
        f"`no such column` errors number {', '.join(str(err_counts[n]['no such column']) for n in NAMES)} "
        f"({', '.join(NAMES)}). Among the questions lost against base, the adapter omits a `DISTINCT` the base "
        f"wrote in {', '.join(f'{distinct_dropped[n]} ({n})' for n in RUNS)} and adds one the base did not write in "
        f"{', '.join(str(distinct_added[n]) for n in RUNS)}.",
        *HAND_READ.get(decoding["repetition_penalty"], []),
        "- **Caveats.** One training seed per adapter. Qwen2.5-Coder (2024) may also have seen BIRD (2023), so the "
        "base is less likely contaminated, not certainly clean. Evidence was included (the standard BIRD setting).",
    ]
    lines += ablation_section(NAMES, evals, results_dir / LEGACY_DIR, resamples)
    lines.append("")
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
