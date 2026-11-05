"""The write-up's key figure: change in execution accuracy vs the base model, with 95% paired
bootstrap CIs, on Spider dev (in-domain) and BIRD dev (zero-shot), side by side on one shared
x-scale. Run from repo root (CPU, reads only committed per-question results):

    python scripts/plot_results.py

Writes results/spider_vs_bird.png. The CIs use metrics.bootstrap_ci with the same seed and resample
count as summarize_eval.py / summarize_bird.py, so they equal the numbers in both summary tables.
"""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from lora_sql.eval.metrics import bootstrap_ci

# Same fixed per-run slots and surface/ink tokens as plot_train.py (dataviz reference palette).
RUN_COLORS = {"qv_r4": "#eb6834", "qv_r8": "#2a78d6", "qv_r16": "#1baf7a", "attn_r8": "#eda100", "all7_r8": "#e87ba4"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ROWS = ["qv_r4", "qv_r8", "qv_r16", "attn_r8", "all7_r8"]   # top to bottom, README order
LABELLED = {"qv_r16", "all7_r8"}                             # the two runs scored on both benchmarks
N_BOOT = 5000
OUT = Path("results/spider_vs_bird.png")


def flags(path):
    return [bool(json.loads(line)["correct"]) for line in Path(path).read_text().splitlines()]


def deltas(result_dir, names):
    base = flags(f"{result_dir}/base.jsonl")
    rng = random.Random(0)  # same resamples as the summary scripts
    resamples = [[rng.randrange(len(base)) for _ in range(len(base))] for _ in range(N_BOOT)]
    out = {}
    for n in names:
        c = flags(f"{result_dir}/{n}.jsonl")
        assert len(c) == len(base)
        out[n] = (100 * (sum(c) - sum(base)) / len(base), *bootstrap_ci(c, base, resamples))
    return out, 100 * sum(base) / len(base), len(base)


def main():
    spider, spider_base, n_spider = deltas("results/eval", ROWS)
    bird_names = [n for n in ROWS if Path(f"results/eval/bird/{n}.jsonl").exists()]
    bird, bird_base, n_bird = deltas("results/eval/bird", bird_names)

    lo = min(v[1] for d in (spider, bird) for v in d.values())
    hi = max(v[2] for d in (spider, bird) for v in d.values())
    pad = 0.8
    xlim = (lo - pad, hi + pad)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, facecolor=SURFACE)
    panels = [
        (axes[0], spider, f"Spider dev (in-domain, n={n_spider:,})", f"base {spider_base:.2f}%"),
        (axes[1], bird, f"BIRD dev (zero-shot, n={n_bird:,})", f"base {bird_base:.2f}%"),
    ]
    ys = {n: len(ROWS) - 1 - i for i, n in enumerate(ROWS)}
    for ax, data, title, base_label in panels:
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
        ax.text(0, len(ROWS) - 0.35, f"  {base_label}", color=INK_2, fontsize=9, va="bottom", ha="left")
        for n in ROWS:
            y = ys[n]
            if n not in data:
                ax.text(0, y, "  not evaluated", color=MUTED, fontsize=9, va="center", ha="left", style="italic")
                continue
            d, c_lo, c_hi = data[n]
            strong = n in LABELLED
            alpha = 1.0 if strong else 0.45
            ax.plot([c_lo, c_hi], [y, y], color=RUN_COLORS[n], linewidth=2, alpha=alpha, solid_capstyle="round", zorder=2)
            ax.plot(d, y, "o", markersize=8, color=RUN_COLORS[n], alpha=alpha,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            if strong:
                ax.text(c_hi + 0.15, y, f"{d:+.2f}".replace("-", "−"), color=INK, fontsize=9, va="center", ha="left")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(ROWS) - 0.1)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=14)
        ax.set_xlabel("Δ execution accuracy vs base (points), 95% paired bootstrap CI", color=INK_2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[0].set_yticks([ys[n] for n in ROWS], ROWS)
    for tick in axes[0].get_yticklabels():
        tick.set_color(INK if tick.get_text() in LABELLED else INK_2)

    fig.tight_layout(w_pad=3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")
    for name, data in (("spider", spider), ("bird", bird)):
        for n, (d, c_lo, c_hi) in data.items():
            print(f"{name:6s} {n:8s} {d:+.2f} [{c_lo:+.2f}, {c_hi:+.2f}]")


if __name__ == "__main__":
    main()
