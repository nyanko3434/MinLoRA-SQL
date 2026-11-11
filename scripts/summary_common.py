"""Pieces shared by summarize_eval.py and summarize_bird.py: loading a results directory, reading
the decoding setting out of a result's meta, style / error counters, and the decoding-ablation
section that compares a results directory with its rp1.1/ archive.

The rp1.1/ archives are the Session 6 and 8 results. They predate `meta.decoding`: the runner then
left repetition_penalty unset, so generate() applied the 1.1 from Qwen's generation_config.json.
"""
import json
import re
from pathlib import Path

from lora_sql.eval.metrics import bootstrap_ci, cluster_bootstrap_ci, mcnemar_p

LEGACY_DECODING = {"do_sample": False, "repetition_penalty": 1.1}
GREEDY = {"do_sample": False, "repetition_penalty": 1.0}
LEGACY_DIR = "rp1.1"

SPIDER_SPACING = re.compile(r"\S {2}(?:=|>=|<=|!=|<|>) {2}\S| ,  ")  # Spider gold's `a  =  b`, `x ,  y`
DQ_LITERAL = re.compile(r"(?:=|!=|<>|\bLIKE)\s*\"", re.I)            # string literal in double quotes
JOIN = re.compile(r"\bJOIN\b", re.I)
YEAR_FN = re.compile(r"\bYEAR\s*\(", re.I)                          # MySQL-style; SQLite has no YEAR()
STRFTIME = re.compile(r"\bSTRFTIME\s*\(", re.I)                    # SQLite's way to take a year


def load_eval(directory: Path, name: str, lora_scale: float = 1.0):
    """(report, per-question rows) for one model, refusing --limit smoke runs and any result whose
    adapter ran at another strength than `lora_scale` (results from before the key count as 1.0)."""
    report = json.loads((directory / f"{name}.json").read_text())
    assert report["meta"]["limit"] is None, f"{directory / name} was a --limit smoke run"
    got = report["meta"].get("lora_scale", 1.0)
    assert got == lora_scale, f"{directory / name} ran the adapter at scale {got}, expected {lora_scale}"
    rows = [json.loads(l) for l in (directory / f"{name}.jsonl").read_text().splitlines()]
    assert len(rows) == len(report["results"]) == report["summary"]["total"]
    return report, rows


def decoding_of(report: dict) -> dict:
    return report["meta"].get("decoding", LEGACY_DECODING)


def shared_decoding(reports: dict) -> dict:
    """The one decoding setting every model in a summary was scored with."""
    settings = {n: decoding_of(r) for n, r in reports.items()}
    first = next(iter(settings.values()))
    assert all(s == first for s in settings.values()), f"models were decoded differently: {settings}"
    return first


def decoding_label(decoding: dict) -> str:
    assert decoding["do_sample"] is False
    penalty = decoding["repetition_penalty"]
    if penalty == 1.0:
        return "greedy decoding (repetition penalty 1.0, i.e. off)"
    return (f"greedy search with repetition penalty {penalty} (the default in Qwen's "
            "`generation_config.json`, applied over the prompt as well as the output)")


def is_empty_result(rows: list[tuple]) -> bool:
    """No rows, or nothing but NULLs: a result any query with an unsatisfied filter also returns."""
    return all(all(v is None for v in row) for row in rows)


def error_kind(err: str | None) -> str | None:
    if not err:
        return None
    if err.startswith("no such column"):
        return "no such column"
    if err.startswith("no such function"):
        return "no such function"
    if err.startswith("no such table"):
        return "no such table"
    if err == "interrupted":
        return "timeout (30 s)"
    if "syntax error" in err or err.startswith("near "):
        return "syntax"
    return "other"


def share(rx: re.Pattern, sqls: list[str], idx=None) -> float:
    idx = range(len(sqls)) if idx is None else idx
    return 100 * sum(bool(rx.search(sqls[i])) for i in idx) / len(idx) if len(idx) else 0.0


def fmt_range(vals, f="{:.2f}") -> str:
    return f"{f.format(min(vals))}–{f.format(max(vals))}" if min(vals) != max(vals) else f.format(vals[0])


def fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def below_zero_clause(cis: dict) -> str:
    """Which models' intervals exclude zero, as a phrase: 'for both', 'for all three', 'for none',
    'only for a, b'."""
    out = [n for n, (lo, hi) in cis.items() if hi < 0 or lo > 0]
    if len(out) == len(cis):
        return {2: "for both", 3: "for all three"}.get(len(cis), "for every one")
    return f"only for {', '.join(out)}" if out else "for none"


def truncated_cell(report: dict) -> str:
    n = report["summary"].get("n_truncated")
    return "n/a" if n is None else str(n)


def ablation_section(names: list[str], main: dict, legacy_dir: Path, resamples: list[list[int]]) -> list[str]:
    """Markdown lines comparing `main` ({name: (report, rows)}, scored at one decoding setting)
    with the same models in `legacy_dir`. A model the archive lacks is left out and named in a
    closing line. Empty when the archive has none of them."""
    missing = [n for n in names if not (legacy_dir / f"{n}.json").exists()]
    names = [n for n in names if n not in missing]
    if not names:
        return []
    legacy = {n: load_eval(legacy_dir, n) for n in names}
    new_dec, old_dec = shared_decoding({n: main[n][0] for n in names}), shared_decoding({n: legacy[n][0] for n in names})
    assert new_dec != old_dec, "the archive was decoded the same way; there is nothing to compare"
    new_p, old_p = new_dec["repetition_penalty"], old_dec["repetition_penalty"]

    stats = {}
    for n in names:
        (new_report, new_rows), (old_report, old_rows) = main[n], legacy[n]
        assert [(r["db_id"], r["question"], r["gold_sql"]) for r in new_rows] == \
               [(r["db_id"], r["question"], r["gold_sql"]) for r in old_rows], f"{n}: the archive has other questions"
        new_side, old_side = new_report["meta"]["checkpoint_sidecar"], old_report["meta"]["checkpoint_sidecar"]
        assert (new_side or {}).get("config_sha256") == (old_side or {}).get("config_sha256"), (
            f"{n}: the archive scored a different checkpoint")
        a, b = [bool(r["correct"]) for r in new_rows], [bool(r["correct"]) for r in old_rows]
        won, lost = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
        stats[n] = {
            "new": 100 * sum(a) / len(a), "old": 100 * sum(b) / len(b), "won": won, "lost": lost,
            "p": mcnemar_p(won, lost), "ci": bootstrap_ci(a, b, resamples),
            "cluster_ci": cluster_bootstrap_ci(a, b, [r["db_id"] for r in new_rows], len(resamples)),
            "changed": 100 * sum(x["predicted_sql"] != y["predicted_sql"] for x, y in zip(new_rows, old_rows)) / len(a),
            "err_new": new_report["summary"]["n_pred_exec_errors"], "err_old": old_report["summary"]["n_pred_exec_errors"],
        }

    lines = [
        "", f"## Decoding ablation: repetition penalty {old_p} vs {new_p}", "",
        f"`{legacy_dir}/` holds the same checkpoints and questions scored with {decoding_label(old_dec)}. That is "
        "what `generate()` does when `repetition_penalty` is not passed, and how these evals first ran. "
        f"Won / lost counts questions right at {new_p} and wrong at {old_p}, and the reverse.", "",
        f"| model | EX % at {old_p} | EX % at {new_p} | Δ | 95% CI | 95% CI, by database | won / lost | McNemar p | "
        f"predictions changed % | exec errors {old_p} → {new_p} |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        s = stats[n]
        lines.append(
            f"| {n} | {s['old']:.2f} | {s['new']:.2f} | {s['new'] - s['old']:+.2f} | "
            f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] | [{s['cluster_ci'][0]:+.2f}, {s['cluster_ci'][1]:+.2f}] | "
            f"+{s['won']} / −{s['lost']} | {fmt_p(s['p'])} | {s['changed']:.1f} | {s['err_old']} → {s['err_new']} |")

    runs = [n for n in names if n != "base"]
    if "base" in names and runs:
        lines += ["", "The fine-tuning effect under each setting (Δ EX vs base, points):", "",
                  f"| model | Δ vs base at {old_p} | Δ vs base at {new_p} |", "|---|---|---|"]
        for n in runs:
            lines.append(f"| {n} | {stats[n]['old'] - stats['base']['old']:+.2f} | "
                         f"{stats[n]['new'] - stats['base']['new']:+.2f} |")

    deltas = [stats[n]["new"] - stats[n]["old"] for n in names]
    sig = [n for n in names if stats[n]["p"] < 0.05]
    lines += ["", f"Removing the penalty changes {fmt_range([stats[n]['changed'] for n in names], '{:.0f}')}% of the "
              f"predictions and moves EX by {fmt_range(deltas, '{:+.2f}')} points. "
              + (f"The change is nominally significant (unadjusted p < 0.05) for {', '.join(sig)}."
                 if sig else "No model's change is nominally significant (unadjusted p < 0.05).")]
    if missing:
        lines += ["", f"Not in this comparison: {', '.join(missing)}, never scored at {old_p}."]
    return lines
