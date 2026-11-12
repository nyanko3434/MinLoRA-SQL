"""Build results/eval/seeds/summary.md: seed repeats of one configuration. qv_r16 was trained again
with other seeds (configs/train/qv_r16_s<N>.yaml; the seed sets the LoRA init, the dropout masks
and the data order together) and each run scored on Spider dev and on BIRD dev. Run from repo root
(CPU, seconds; it reads the stored per-question results and executes nothing):

    python scripts/summarize_seeds.py

Reads results/eval/{base,qv_r16,...}.json and results/eval/seeds/qv_r16_s<N>.json for Spider, the
same under results/eval/bird/ for BIRD, and the training evidence in results/train/<run>/.
Statistics are paired over the same questions, with the same bootstrap seed and resample count as
summarize_eval.py / summarize_bird.py.

Seed 0 is the run in the main tables and stays the reported qv_r16. No seed here is picked as a
result; the point is the spread. p-values are unadjusted.
"""
import json
import random
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

from summary_common import (
    DQ_LITERAL,
    SPIDER_SPACING,
    STRFTIME,
    YEAR_FN,
    decoding_label,
    error_kind,
    fmt_p,
    fmt_range,
    load_eval,
    share,
    shared_decoding,
    truncated_cell,
)

from lora_sql.config import load_config
from lora_sql.eval.metrics import bootstrap_ci, cluster_bootstrap_ci, mcnemar_p
from lora_sql.train import config_sha256

RUN = "qv_r16"
N_BOOT = 5000
OUT = Path("results/eval/seeds/summary.md")
BENCHMARKS = {  # name -> where the main results and the seed repeats live, and the other configs scored there
    "Spider dev": {"main": Path("results/eval"), "seeds": Path("results/eval/seeds"), "kind": "in-domain",
                   "others": ["qv_r4", "qv_r8", "attn_r8", "all7_r8"]},
    "BIRD dev": {"main": Path("results/eval/bird"), "seeds": Path("results/eval/bird/seeds"), "kind": "zero-shot",
                 "others": ["qv_r8", "all7_r8"]},
}
HINT = "\nHint: "  # data.load_bird_examples appends the evidence to the question this way


def seed_runs(directory: Path) -> dict[int, str]:
    """{seed: run name}: seed 0 is the main run, the rest are whatever repeats have been scored."""
    found = {int(p.stem.removeprefix(f"{RUN}_s")): p.stem for p in directory.glob(f"{RUN}_s*.json")}
    return {0: RUN, **dict(sorted(found.items()))}


def load_bench(bench: str):
    """(base, {seed: (name, report, rows)}, {other config: (report, rows)}), all on the same questions."""
    where = BENCHMARKS[bench]
    base = load_eval(where["main"], "base")
    seeds = {s: (name, *load_eval(where["main"] if s == 0 else where["seeds"], name))
             for s, name in seed_runs(where["seeds"]).items()}
    others = {n: load_eval(where["main"], n) for n in where["others"]}

    key = lambda rows: [(r["db_id"], r["question"], r["gold_sql"]) for r in rows]
    shape = lambda report: {k: report["meta"]["checkpoint_sidecar"][k] for k in ("target_modules", "rank", "alpha")}
    for s, (name, report, rows) in seeds.items():
        assert key(rows) == key(base[1]), f"{bench} {name}: not the same questions as base"
        assert shape(report) == shape(seeds[0][1]), f"{bench} {name}: not the same adapter shape as seed 0"
        cfg = load_config(Path("configs/train") / f"{name}.yaml")
        assert cfg.seed == s, f"{name}: config seed is {cfg.seed}, expected {s}"
        assert report["meta"]["checkpoint_sidecar"]["config_sha256"] == config_sha256(cfg), \
            f"{bench} {name}: the checkpoint was not trained from configs/train/{name}.yaml as committed"
    for field in ("config_sha256", "data_order_sha256"):
        values = [report["meta"]["checkpoint_sidecar"][field] for _, report, _ in seeds.values()]
        assert len(set(values)) == len(values), f"{bench}: two seeds share a {field}"
    for n, (_, rows) in others.items():
        assert key(rows) == key(base[1]), f"{bench} {n}: not the same questions as base"
    shared_decoding({"base": base[0], **{name: report for name, report, _ in seeds.values()},
                     **{n: report for n, (report, _) in others.items()}})
    return base, seeds, others


def compute(bench: str) -> dict:
    """Every number the section and the figure need for one benchmark."""
    base, seeds, others = load_bench(bench)
    flags = lambda rows: [bool(r["correct"]) for r in rows]
    b = flags(base[1])
    dbs = [r["db_id"] for r in base[1]]
    n = len(b)
    rng = random.Random(0)  # same resamples as the other summary scripts
    resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(N_BOOT)]

    def paired(x, y):
        won = sum(p and not q for p, q in zip(x, y))
        lost = sum(q and not p for p, q in zip(x, y))
        return {"delta": 100 * (sum(x) - sum(y)) / n, "ci": bootstrap_ci(x, y, resamples),
                "cluster_ci": cluster_bootstrap_ci(x, y, dbs, N_BOOT), "won": won, "lost": lost,
                "p": mcnemar_p(won, lost), "differs": 100 * (won + lost) / n}

    same_text = lambda r1, r2: 100 * sum(p["predicted_sql"] == q["predicted_sql"] for p, q in zip(r1, r2)) / n
    c = {s: flags(rows) for s, (_, _, rows) in seeds.items()}
    vs_base = {s: {"ex": 100 * sum(c[s]) / n, **paired(c[s], b)} for s in seeds}
    seed_pairs = {(s, t): {**paired(c[t], c[s]), "same_text": same_text(seeds[t][2], seeds[s][2])}
                  for s, t in combinations(seeds, 2)}
    other = {}
    for name, (_, rows) in others.items():
        o = flags(rows)
        other[name] = {"ex": 100 * sum(o) / n, "vs_base": paired(o, b),
                       "vs_seed0": {**paired(o, c[0]), "same_text": same_text(rows, seeds[0][2])}}

    mean_flags = [sum(c[s][i] for s in seeds) / len(seeds) for i in range(n)]
    gained = Counter(sum(c[s][i] for s in seeds) for i in range(n) if not b[i])      # base wrong: seeds right
    lost = Counter(sum(not c[s][i] for s in seeds) for i in range(n) if b[i])        # base right: seeds wrong
    return {
        "n": n, "base": base, "seeds": seeds, "base_ex": 100 * sum(b) / n, "vs_base": vs_base,
        "seed_pairs": seed_pairs, "other": other,
        "mean_delta": 100 * (sum(mean_flags) - sum(b)) / n, "mean_ci": bootstrap_ci(mean_flags, b, resamples),
        "mean_cluster_ci": cluster_bootstrap_ci(mean_flags, b, dbs, N_BOOT),
        "gained": gained, "lost": lost,
    }


ci = lambda pair: f"[{pair[0]:+.2f}, {pair[1]:+.2f}]"


def span(values: list[float], f: str = "{:.2f}") -> str:
    """'lowest to highest': fmt_range's dash reads badly between signed numbers."""
    return f"{f.format(min(values))} to {f.format(max(values))}"


def spread(values: list[float], f: str = "{:.2f}") -> str:
    """mean, range and sample sd of a handful of values, as text."""
    out = f"mean {f.format(statistics.mean(values))}, range {span(values, f)}"
    return out + (f", sample sd {statistics.stdev(values):.2f}" if len(values) > 1 else "")


def benchmark_section(bench: str, r: dict) -> list[str]:
    seeds, n_seeds = r["seeds"], len(r["seeds"])
    lines = [
        "", f"## {bench} ({BENCHMARKS[bench]['kind']}, n={r['n']:,})", "",
        "| seed | run | EX % | Δ vs base | 95% CI | 95% CI, by database | won / lost vs base | McNemar p vs base |",
        "|---|---|---|---|---|---|---|---|",
        f"| | base | {r['base_ex']:.2f} | | | | | |",
    ]
    for s, (name, _, _) in seeds.items():
        e = r["vs_base"][s]
        lines.append(f"| {s} | {name} | {e['ex']:.2f} | {e['delta']:+.2f} | {ci(e['ci'])} | {ci(e['cluster_ci'])} | "
                     f"+{e['won']} / −{e['lost']} | {fmt_p(e['p'])} |")
    if n_seeds > 1:
        lines += [
            "",
            f"Across the {n_seeds} seeds: EX {spread([r['vs_base'][s]['ex'] for s in seeds])}; Δ vs base "
            f"{spread([r['vs_base'][s]['delta'] for s in seeds], '{:+.2f}')}. Averaging the seeds' correctness per question, "
            f"Δ vs base is {r['mean_delta']:+.2f} points, 95% CI {ci(r['mean_ci'])} (by database "
            f"{ci(r['mean_cluster_ci'])}). That interval resamples questions with the seeds held fixed; it does "
            "not cover seeds that were not run.",
            "", "Seed against seed, and seed 0 against the other configurations (one seed each):", "",
            "| pair | Δ EX | won / lost | McNemar p | questions where correctness differs % | same SQL text % |",
            "|---|---|---|---|---|---|",
        ]
        for (s, t), e in r["seed_pairs"].items():
            lines.append(f"| seed {t} vs seed {s} | {e['delta']:+.2f} | +{e['won']} / −{e['lost']} | {fmt_p(e['p'])} | "
                         f"{e['differs']:.1f} | {e['same_text']:.1f} |")
        for name, o in r["other"].items():
            e = o["vs_seed0"]
            lines.append(f"| {name} vs {RUN} seed 0 | {e['delta']:+.2f} | +{e['won']} / −{e['lost']} | {fmt_p(e['p'])} | "
                         f"{e['differs']:.1f} | {e['same_text']:.1f} |")

        counts = lambda counter: " | ".join(str(counter[k]) for k in range(n_seeds, 0, -1))
        head = " | ".join(f"by {'all ' if k == n_seeds else 'exactly '}{k}" for k in range(n_seeds, 0, -1))
        lines += [
            "", "How many seeds agree on each question the base gets wrong (gained) or right (lost):", "",
            f"| questions | {head} |", "|" + "---|" * (n_seeds + 1),
            f"| gained (base wrong, seed right) | {counts(r['gained'])} |",
            f"| lost (base right, seed wrong) | {counts(r['lost'])} |",
        ]

    base_rows = r["base"][1]
    is_bird = any(HINT in row["question"] for row in base_rows)
    year_hint = [i for i, row in enumerate(base_rows)
                 if HINT in row["question"] and YEAR_FN.search(row["question"].split(HINT, 1)[1])]
    head = ["model", "Spider gold spacing %", "string literal in double quotes %"]
    if is_bird:
        head += [f"`YEAR(` copied where the hint writes it (n={len(year_hint)}) %",
                 f"`STRFTIME(` there (n={len(year_hint)}) %"]
    head += ["`no such column` errors", "`no such function` errors", "exec errors", "truncated"]
    lines += ["", "What the predictions look like:", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, report, rows in [("base", *r["base"]), *((f"seed {s}", rep, rows) for s, (_, rep, rows) in seeds.items())]:
        sqls = [row["predicted_sql"] for row in rows]
        kinds = Counter(error_kind(row["exec_error"]) for row in rows)
        cells = [label, f"{share(SPIDER_SPACING, sqls):.1f}", f"{share(DQ_LITERAL, sqls):.1f}"]
        if is_bird:
            cells += [f"{share(YEAR_FN, sqls, year_hint):.1f}", f"{share(STRFTIME, sqls, year_hint):.1f}"]
        cells += [str(kinds["no such column"]), str(kinds["no such function"]),
                  str(report["summary"]["n_pred_exec_errors"]), truncated_cell(report)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def training_section(seeds: dict) -> list[str]:
    lines = ["", "## Training", "",
             "| seed | run | steps | final logged loss | mean of last 10 logged | hours | peak reserved GB | "
             "data order sha256 | config sha256 |", "|---|---|---|---|---|---|---|---|---|"]
    for s, (name, _, _) in seeds.items():
        evidence = Path("results/train") / name
        assert evidence.is_dir(), f"no training evidence at {evidence}; run scripts/plot_train.py for {name} first"
        rows = [json.loads(l) for l in (evidence / "log.jsonl").read_text().splitlines()]
        rows = [row for row in rows if "event" not in row]
        sidecar = json.loads((evidence / "lora_config.json").read_text())
        tail = [row["loss"] for row in rows[-10:]]
        lines.append(f"| {s} | {name} | {rows[-1]['step']} | {rows[-1]['loss']:.4f} | {sum(tail) / len(tail):.4f} | "
                     f"{rows[-1]['elapsed_s'] / 3600:.2f} | {max(row['peak_reserved_gb'] for row in rows):.2f} | "
                     f"`{sidecar['data_order_sha256'][:12]}` | `{sidecar['config_sha256'][:12]}` |")
    lines += ["", "The logged loss is one accumulation window every 10 steps, so single values are noisy."]
    return lines


def reading(results: dict) -> list[str]:
    """Bullets generated from the numbers; every branch states what was found, not what was hoped."""
    sp, bd = results["Spider dev"], results["BIRD dev"]
    n_seeds = len(sp["seeds"])
    lines = ["", "## Reading these numbers", ""]
    if n_seeds < 2 or len(bd["seeds"]) != n_seeds:
        return lines + ["- Seed repeats are not scored on both benchmarks yet; there is no spread to read."]

    def excl(r, sign):  # seeds whose question-level CI excludes zero on the given side
        return sum((e["ci"][0] > 0) if sign > 0 else (e["ci"][1] < 0) for e in r["vs_base"].values())

    for name, r, sign, word in (("Spider", sp, 1, "above"), ("BIRD", bd, -1, "below")):
        deltas = [r["vs_base"][s]["delta"] for s in r["seeds"]]
        same_side = sum(d * sign > 0 for d in deltas)
        lines.append(
            f"- **{name}:** {same_side} of {n_seeds} seeds are {word} the base (Δ {span(deltas, '{:+.2f}')}); the "
            f"question-level 95% CI lies {word} zero for {excl(r, sign)} of {n_seeds}, the by-database one for "
            f"{sum((e['cluster_ci'][0] > 0) if sign > 0 else (e['cluster_ci'][1] < 0) for e in r['vs_base'].values())}. "
            f"Averaged over the seeds: {r['mean_delta']:+.2f} {ci(r['mean_ci'])}.")
    for name, r in (("Spider", sp), ("BIRD", bd)):
        exs = [r["vs_base"][s]["ex"] for s in r["seeds"]]
        lo, hi = min(exs), max(exs)
        first = exs[0]  # seed 0, the run in the main tables
        place = "the highest" if first == hi else "the lowest" if first == lo else "neither the highest nor the lowest"
        inside = [f"{o} ({v['ex']:.2f})" for o, v in r["other"].items() if lo <= v["ex"] <= hi]
        gaps = {o: v["ex"] - (hi if v["ex"] > hi else lo) for o, v in r["other"].items() if not lo <= v["ex"] <= hi}
        outside = [f"{o} ({r['other'][o]['ex']:.2f}, {abs(g):.2f} {'above' if g > 0 else 'below'} the nearest seed)"
                   for o, g in gaps.items()]
        beyond = [o for o, g in gaps.items() if abs(g) > hi - lo]
        if not gaps:
            verdict = ""
        elif beyond:
            verdict = (f" Further from the nearest seed than the seeds are from each other: {', '.join(beyond)} "
                       f"({len(beyond)} of {len(gaps)}).")
        else:
            verdict = " None of them is further from the nearest seed than the seeds are from each other."
        lines.append(
            f"- **{name}, seed spread against configuration gaps:** the {RUN} seeds span {hi - lo:.2f} points "
            f"({lo:.2f} to {hi:.2f}); seed 0, the run in the main tables, is {place} of the {n_seeds}. Inside that "
            f"range: {', '.join(inside) or 'no other configuration'}. Outside it: {', '.join(outside) or 'none'}."
            f"{verdict} The other configurations have one seed each, so this compares their gap with {RUN}'s own "
            "spread and nothing more.")
        pair_p = [e["p"] for e in r["seed_pairs"].values()]
        seed_diff = [e["differs"] for e in r["seed_pairs"].values()]
        cfg_diff = [v["vs_seed0"]["differs"] for v in r["other"].values()]
        lines.append(
            f"- **{name}, question level:** two seeds of the same configuration disagree on "
            f"{fmt_range(seed_diff, '{:.1f}')}% of the questions (pairwise McNemar p {fmt_range(pair_p, '{:.3f}')}); "
            f"seed 0 and another configuration disagree on {fmt_range(cfg_diff, '{:.1f}')}%.")
        all_k, some = r["gained"][n_seeds], sum(v for k, v in r["gained"].items() if 0 < k < n_seeds)
        all_l, some_l = r["lost"][n_seeds], sum(v for k, v in r["lost"].items() if 0 < k < n_seeds)
        lines.append(
            f"- **{name}, which questions move:** {all_k} questions are gained by all {n_seeds} seeds and {some} by "
            f"only some; {all_l} are lost by all {n_seeds} and {some_l} by only some.")

    sqls = {s: [row["predicted_sql"] for row in rows] for s, (_, _, rows) in bd["seeds"].items()}
    base_sqls = [row["predicted_sql"] for row in bd["base"][1]]
    base_rows = bd["base"][1]
    year_hint = [i for i, row in enumerate(base_rows)
                 if HINT in row["question"] and YEAR_FN.search(row["question"].split(HINT, 1)[1])]
    habit = lambda rx, idx=None: (fmt_range([share(rx, q, idx) for q in sqls.values()], "{:.1f}"),
                                  f"{share(rx, base_sqls, idx):.1f}")
    (sp_r, sp_b), (dq_r, dq_b), (yr_r, yr_b) = habit(SPIDER_SPACING), habit(DQ_LITERAL), habit(YEAR_FN, year_hint)
    lines.append(
        f"- **The style counts on BIRD, across seeds:** Spider's gold spacing {sp_r}% (base {sp_b}%), double-quoted "
        f"string literals {dq_r}% (base {dq_b}%), `YEAR(` copied from the hint {yr_r}% (base {yr_b}%, "
        f"n={len(year_hint)}).")
    lines.append(
        f"- **Caveats.** {n_seeds} seeds give a rough spread, not a variance estimate. Only {RUN} was repeated. A "
        "seed changes the LoRA initialisation, the dropout masks and the data order together, so their separate "
        "effects are not measured. Seed 0 stays the run in the main tables; no seed is picked as a result. p-values "
        "are unadjusted. Spider's held-out test set was not used.")
    return lines


def main():
    results = {bench: compute(bench) for bench in BENCHMARKS}
    reports = {f"{bench} {name}": report for bench, r in results.items() for name, report, _ in r["seeds"].values()}
    decoding = decoding_label(shared_decoding(reports))
    scored = {bench: ", ".join(str(s) for s in r["seeds"]) for bench, r in results.items()}

    lines = [
        "# Seed repeats", "",
        f"Generated by `scripts/summarize_seeds.py`. `{RUN}` (the best adapter on Spider) was trained again from "
        f"`configs/train/{RUN}_s<N>.yaml`, which differ from `{RUN}.yaml` only in the seed, the run name and the "
        "output directory. The seed sets the LoRA initialisation, the dropout masks and the data order together. "
        f"Seeds scored: Spider {scored['Spider dev']}; BIRD {scored['BIRD dev']}. Seed 0 is the run in the main "
        f"tables. Qwen2.5-Coder-7B-Instruct, nf4 + bf16 compute, {decoding}.", "",
        "\"95% CI\" resamples questions and \"by database\" resamples whole databases (5,000 resamples, seed 0). "
        "p-values are exact two-sided McNemar, unadjusted. With this few seeds the sample sd is a rough guide only.",
    ]
    for bench, r in results.items():
        lines += benchmark_section(bench, r)
    lines += training_section(results["Spider dev"]["seeds"])
    lines += reading(results)
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
