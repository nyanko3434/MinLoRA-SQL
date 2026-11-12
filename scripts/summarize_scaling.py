"""Build results/eval/scaling/summary.md: the adapter-scaling probe. One adapter (qv_r16) is scored
at fractions of its trained LoRA scaling (eval --lora-scale), on Spider dev and on BIRD dev, and
set between the base model (scale 0) and the adapter as trained (scale 1). Run from repo root
(CPU, seconds; it reads the stored per-question results and executes nothing):

    python scripts/summarize_scaling.py

Reads results/eval/{base,qv_r16}.json and results/eval/scaling/qv_r16_x<S>.json for Spider, and
the same under results/eval/bird/ for BIRD. Statistics are paired over the same questions, with
the same bootstrap seed and resample count as summarize_eval.py / summarize_bird.py.

This is a diagnostic. The scales are read off the dev sets, so no scale here is a tuned result
and none of these files enters the main tables. p-values are unadjusted.
"""
import random
from collections import Counter
from pathlib import Path

from summary_common import (
    DQ_LITERAL,
    SPIDER_SPACING,
    STRFTIME,
    YEAR_FN,
    decoding_label,
    error_kind,
    fmt_p,
    load_eval,
    share,
    shared_decoding,
    truncated_cell,
)

from lora_sql.eval.metrics import bootstrap_ci, cluster_bootstrap_ci, mcnemar_p

RUN = "qv_r16"
N_BOOT = 5000
OUT = Path("results/eval/scaling/summary.md")
BENCHMARKS = {  # name -> where its main results and its scaled results live
    "Spider dev": {"main": Path("results/eval"), "scaled": Path("results/eval/scaling"), "kind": "in-domain"},
    "BIRD dev": {"main": Path("results/eval/bird"), "scaled": Path("results/eval/bird/scaling"), "kind": "zero-shot"},
}
HINT = "\nHint: "  # data.load_bird_examples appends the evidence to the question this way


def scale_names(directory: Path) -> list[str]:
    """The scales scored so far, as written in the file names, ascending."""
    return sorted((p.stem.removeprefix(f"{RUN}_x") for p in directory.glob(f"{RUN}_x*.json")), key=float)


def load_points(bench: str) -> list[tuple[float, str, dict, list[dict]]]:
    """[(scale, label, report, rows)] from scale 0 (base) to 1 (the adapter as trained)."""
    where = BENCHMARKS[bench]
    base, full = load_eval(where["main"], "base"), load_eval(where["main"], RUN)
    points = [(0.0, "base", *base)]
    for s in scale_names(where["scaled"]):
        assert 0 < float(s) < 1, f"{bench}: scale {s} is outside (0, 1)"
        points.append((float(s), f"{RUN}_x{s}", *load_eval(where["scaled"], f"{RUN}_x{s}", lora_scale=float(s))))
    points.append((1.0, RUN, *full))

    key = lambda rows: [(r["db_id"], r["question"], r["gold_sql"]) for r in rows]
    sha = full[0]["meta"]["checkpoint_sidecar"]["config_sha256"]
    for _, label, report, rows in points[1:]:
        assert key(rows) == key(base[1]), f"{bench} {label}: not the same questions as base"
        assert report["meta"]["checkpoint_sidecar"]["config_sha256"] == sha, f"{bench} {label}: another checkpoint"
    shared_decoding({label: report for _, label, report, _ in points})
    return points


def effects(points) -> tuple[dict, float]:
    """({scale: paired stats vs base, and vs the adapter as trained}, base EX %)."""
    flags = lambda rows: [bool(r["correct"]) for r in rows]
    base, full = flags(points[0][3]), flags(points[-1][3])
    dbs = [r["db_id"] for r in points[0][3]]
    n = len(base)
    rng = random.Random(0)  # same resamples as the other summary scripts
    resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(N_BOOT)]
    wl = lambda a, b: (sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b)))
    out = {}
    for scale, _, _, rows in points[1:]:
        c = flags(rows)
        won, lost = wl(c, base)
        won_f, lost_f = wl(c, full)
        out[scale] = {
            "ex": 100 * sum(c) / n, "delta": 100 * (sum(c) - sum(base)) / n,
            "ci": bootstrap_ci(c, base, resamples), "cluster_ci": cluster_bootstrap_ci(c, base, dbs, N_BOOT),
            "won": won, "lost": lost, "p": mcnemar_p(won, lost),
            "delta_full": 100 * (sum(c) - sum(full)) / n, "won_full": won_f, "lost_full": lost_f,
            "p_full": mcnemar_p(won_f, lost_f),
        }
    return out, 100 * sum(base) / n


def scale_label(scale: float) -> str:
    return {0.0: "0 (base)", 1.0: "1 (as trained)"}.get(scale, f"{scale:g}")


def benchmark_section(bench: str, points, eff, base_ex) -> list[str]:
    n = len(points[0][3])
    base_rows, full_rows = points[0][3], points[-1][3]
    lines = [
        "", f"## {bench} ({BENCHMARKS[bench]['kind']}, n={n:,})", "",
        "| adapter scale | EX % | Δ vs base | 95% CI | 95% CI, by database | won / lost vs base | McNemar p vs base | "
        "share of the full adapter's Δ % | Δ vs as trained | won / lost vs as trained | McNemar p vs as trained |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {scale_label(0.0)} | {base_ex:.2f} | | | | | | | | | |",
    ]
    full_delta = eff[1.0]["delta"]
    for scale, _, _, _ in points[1:]:
        e = eff[scale]
        vs_full = ("| | | |" if scale == 1.0 else
                   f"| {e['delta_full']:+.2f} | +{e['won_full']} / −{e['lost_full']} | {fmt_p(e['p_full'])} |")
        lines.append(
            f"| {scale_label(scale)} | {e['ex']:.2f} | {e['delta']:+.2f} | [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}] | "
            f"[{e['cluster_ci'][0]:+.2f}, {e['cluster_ci'][1]:+.2f}] | +{e['won']} / −{e['lost']} | {fmt_p(e['p'])} | "
            f"{100 * e['delta'] / full_delta:.0f} {vs_full}")

    is_bird = any(HINT in r["question"] for r in base_rows)
    year_hint = [i for i, r in enumerate(base_rows) if HINT in r["question"] and YEAR_FN.search(r["question"].split(HINT, 1)[1])]
    head = ["adapter scale", "Spider gold spacing %", "string literal in double quotes %"]
    if is_bird:
        head += [f"`YEAR(` copied where the hint writes it (n={len(year_hint)}) %",
                 f"`STRFTIME(` there (n={len(year_hint)}) %"]
    head += ["`no such column` errors", "`no such function` errors", "exec errors", "truncated",
             "same text as base %", "same text as trained %"]
    lines += ["", "What the predictions look like at each scale:", "",
              "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    same = lambda a, b: 100 * sum(x["predicted_sql"] == y["predicted_sql"] for x, y in zip(a, b)) / n
    for scale, _, report, rows in points:
        sqls = [r["predicted_sql"] for r in rows]
        kinds = Counter(error_kind(r["exec_error"]) for r in rows)
        cells = [scale_label(scale), f"{share(SPIDER_SPACING, sqls):.1f}", f"{share(DQ_LITERAL, sqls):.1f}"]
        if is_bird:
            cells += [f"{share(YEAR_FN, sqls, year_hint):.1f}", f"{share(STRFTIME, sqls, year_hint):.1f}"]
        cells += [str(kinds["no such column"]), str(kinds["no such function"]),
                  str(report["summary"]["n_pred_exec_errors"]), truncated_cell(report),
                  f"{same(rows, base_rows):.1f}", f"{same(rows, full_rows):.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def reading(results: dict) -> list[str]:
    """Bullets generated from the numbers; every branch states what was found, not what was hoped."""
    (sp_points, sp), (bd_points, bd) = (results[b][:2] for b in ("Spider dev", "BIRD dev"))
    common = sorted(set(sp) & set(bd) - {1.0})
    ci = lambda e: f"[{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"
    lines = ["", "## Reading these numbers", ""]
    lines.append(f"- **As trained (scale 1):** Spider {sp[1.0]['delta']:+.2f} points vs base {ci(sp[1.0])}, "
                 f"BIRD {bd[1.0]['delta']:+.2f} {ci(bd[1.0])}.")
    for s in common:
        lines.append(
            f"- **At {s:g}×:** Spider {sp[s]['delta']:+.2f} points vs base {ci(sp[s])}, "
            f"{100 * sp[s]['delta'] / sp[1.0]['delta']:.0f}% of the full adapter's change. "
            f"BIRD {bd[s]['delta']:+.2f} {ci(bd[s])}, {100 * bd[s]['delta'] / bd[1.0]['delta']:.0f}% of the full "
            "adapter's change.")

    good = [s for s in common if sp[s]["ci"][0] > 0 and bd[s]["ci"][1] >= 0]
    if good:
        lines.append(f"- **Trade-off.** At {', '.join(f'{s:g}×' for s in good)} the Spider gain is still "
                     "distinguishable from zero (question-level 95% CI above zero) and the BIRD change is not "
                     "(its CI includes zero). The interval including zero does not show that the loss is gone, "
                     "only that this sample cannot tell it from none.")
    else:
        lines.append("- **Trade-off.** No tested scale keeps a Spider gain distinguishable from zero while bringing "
                     "the BIRD change within noise of zero (question-level 95% CIs).")
    cci = lambda e: f"[{e['cluster_ci'][0]:+.2f}, {e['cluster_ci'][1]:+.2f}]"
    by_db = [s for s in common if sp[s]["cluster_ci"][0] > 0 and bd[s]["cluster_ci"][1] >= 0]
    intervals = "; ".join(f"{s:g}×: Spider {cci(sp[s])}, BIRD {cci(bd[s])}" for s in common)
    lines.append("- **Resampling whole databases instead of questions,** "
                 + (f"that pattern holds at {', '.join(f'{s:g}×' for s in by_db)}" if by_db else
                    "no tested scale has a Spider interval above zero together with a BIRD interval that includes "
                    "zero") + f" ({intervals}). As trained: Spider {cci(sp[1.0])}, BIRD {cci(bd[1.0])}.")

    for name, eff in (("Spider", sp), ("BIRD", bd)):
        seq = [0.0] + [eff[s]["delta"] for s in sorted(eff)]
        one_way = _monotone(seq) or _monotone([-x for x in seq])
        by_scale = ", ".join(f"{eff[s]['delta']:+.2f} at {s:g}×" for s in sorted(eff))
        lines.append(f"- **{name} {'moves' if one_way else 'does not move'} one way with scale:** "
                     f"Δ vs base is {by_scale}.")

    vs_full = "; ".join(
        f"at {s:g}× Spider {sp[s]['delta_full']:+.2f} points (+{sp[s]['won_full']} / −{sp[s]['lost_full']}, "
        f"p {fmt_p(sp[s]['p_full'])}) and BIRD {bd[s]['delta_full']:+.2f} (+{bd[s]['won_full']} / "
        f"−{bd[s]['lost_full']}, p {fmt_p(bd[s]['p_full'])})" for s in common)
    lines.append(f"- **Against the adapter as trained** (paired, unadjusted p): {vs_full}.")

    # Which of the adapter's habits are already at their trained level at the largest scaled point?
    bird_sqls = {scale: [r["predicted_sql"] for r in rows] for scale, _, _, rows in bd_points}
    base_rows = bd_points[0][3]
    year_hint = [i for i, r in enumerate(base_rows)
                 if HINT in r["question"] and YEAR_FN.search(r["question"].split(HINT, 1)[1])]
    habits = {
        "Spider's gold spacing": lambda q: share(SPIDER_SPACING, q),
        "double-quoted string literals": lambda q: share(DQ_LITERAL, q),
        f"`YEAR(` copied from the hint (n={len(year_hint)})": lambda q: share(YEAR_FN, q, year_hint),
    }
    for name, fn in habits.items():
        by_scale = ", ".join(f"{fn(bird_sqls[scale]):.1f}% at {scale_label(scale)}" for scale in sorted(bird_sqls))
        lines.append(f"- **On BIRD, {name}:** {by_scale}.")
    if common:
        top = max(common)
        level = lambda fn: 100 * (fn(bird_sqls[top]) - fn(bird_sqls[0.0])) / (fn(bird_sqls[1.0]) - fn(bird_sqls[0.0]))
        levels = ", ".join(f"{name.split(' (n=')[0]} {level(fn):.0f}%" for name, fn in habits.items())
        lines.append(f"- **How far each habit has come at {top:g}×,** as a share of its base-to-trained change on BIRD: "
                     f"{levels}. The BIRD accuracy change at {top:g}× is "
                     f"{100 * bd[top]['delta'] / bd[1.0]['delta']:.0f}% of the trained one. These are counts of "
                     "surface patterns; they show what moves with the accuracy, not what causes it.")
    lines.append(f"- **Caveats.** One adapter ({RUN}), one training seed, and scales chosen in advance (not "
                 "searched). Reading a best scale off these dev sets would be tuning on dev, so none is reported as "
                 "a tuned result. p-values are unadjusted. Spider's held-out test set was not used.")
    return lines


def _monotone(xs: list[float]) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:]))


def main():
    results = {}
    for bench in BENCHMARKS:
        points = load_points(bench)
        results[bench] = (points, *effects(points))
    decodings = {decoding_label(shared_decoding({label: r for _, label, r, _ in results[b][0]})) for b in results}
    assert len(decodings) == 1, f"the two benchmarks were decoded differently: {decodings}"
    scales = {b: [f"{s:g}" for s, *_ in results[b][0] if 0 < s < 1] for b in results}

    lines = [
        "# Adapter-scaling probe", "",
        f"Generated by `scripts/summarize_scaling.py`. `{RUN}` (the best adapter on Spider) is scored with its LoRA "
        "scaling `alpha / r` multiplied by a factor below 1 at inference (`--lora-scale`), with no retraining. "
        "Scale 0 is the base model and scale 1 the adapter as trained; both rows are the committed main results. "
        f"Scaled runs: Spider {', '.join(scales['Spider dev']) or 'none'}; BIRD {', '.join(scales['BIRD dev']) or 'none'}. "
        f"Qwen2.5-Coder-7B-Instruct, nf4 + bf16 compute, {decodings.pop()}.", "",
        "This is a diagnostic, not a tuned result: both benchmarks are dev sets, so picking a scale from these "
        "tables would be tuning on dev. \"95% CI\" resamples "
        "questions and \"by database\" resamples whole databases (5,000 resamples, seed 0). p-values are exact "
        "two-sided McNemar, unadjusted. \"Share of the full adapter's Δ\" is Δ vs base at that scale divided by Δ "
        "vs base at scale 1; it is a ratio of two noisy numbers.",
    ]
    for bench, (points, eff, base_ex) in results.items():
        lines += benchmark_section(bench, points, eff, base_ex)
    lines += reading(results)
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
