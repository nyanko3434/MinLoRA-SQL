"""Seed repeats as one figure: change in execution accuracy vs the base model for each seed of
qv_r16, with the other configurations (one seed each) as muted context, on Spider dev and BIRD
dev side by side on one shared x-scale. Run from repo root (CPU, reads only stored results):

    python scripts/plot_seeds.py

Writes results/seed_spread.png. Points and intervals come from summarize_seeds.compute, so they
equal the numbers in results/eval/seeds/summary.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_seeds import BENCHMARKS, RUN, compute

# qv_r16 keeps its color from plot_train.py / plot_results.py (reference palette slot 3). The seeds
# are one entity, told apart by their row labels; the other configurations are context, in neutral ink.
SEED_COLOR = "#1baf7a"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OTHERS = ["qv_r4", "qv_r8", "attn_r8", "all7_r8"]   # README order
OUT = Path("results/seed_spread.png")


def main():
    results = {bench: compute(bench) for bench in BENCHMARKS}
    seeds = sorted(set().union(*(r["seeds"] for r in results.values())))
    rows = [f"{RUN} seed {s}" for s in seeds] + OTHERS
    ys = {label: len(rows) - 1 - i - (0.5 if label in OTHERS else 0) for i, label in enumerate(rows)}

    points = {}  # bench -> {row label: (delta, lo, hi)}
    for bench, r in results.items():
        points[bench] = {f"{RUN} seed {s}": (e["delta"], *e["ci"]) for s, e in r["vs_base"].items()}
        points[bench] |= {n: (o["vs_base"]["delta"], *o["vs_base"]["ci"]) for n, o in r["other"].items()}
    lo = min(v[1] for d in points.values() for v in d.values())
    hi = max(v[2] for d in points.values() for v in d.values())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=SURFACE)
    for ax, (bench, r) in zip(axes, results.items()):
        data = points[bench]
        ax.set_facecolor(SURFACE)
        seed_deltas = [data[f"{RUN} seed {s}"][0] for s in r["seeds"]]
        if len(seed_deltas) > 1:  # the range the seeds span, to read the other rows against
            ax.axvspan(min(seed_deltas), max(seed_deltas), color=SEED_COLOR, alpha=0.12, linewidth=0, zorder=0)
            ax.text((min(seed_deltas) + max(seed_deltas)) / 2, len(rows) - 0.42, "seed range", color=INK_2,
                    fontsize=8, ha="center", va="bottom")
        ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
        ax.text(0, -1.25, f"  base {r['base_ex']:.2f}%", color=INK_2, fontsize=8, va="center", ha="left")
        for label in rows:
            y = ys[label]
            if label not in data:
                ax.text(0, y, "  not evaluated", color=MUTED, fontsize=9, va="center", ha="left", style="italic")
                continue
            d, c_lo, c_hi = data[label]
            color = MUTED if label in OTHERS else SEED_COLOR
            ax.plot([c_lo, c_hi], [y, y], color=color, linewidth=2, alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot(d, y, "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.text(c_hi + 0.15, y, f"{d:+.2f}".replace("-", "−"), color=INK if label not in OTHERS else INK_2,
                    fontsize=9, va="center", ha="left")
            print(f"{bench:10s} {label:16s} {d:+.2f} [{c_lo:+.2f}, {c_hi:+.2f}]")
        ax.set_xlim(lo - 0.8, hi + 1.4)
        ax.set_ylim(-1.5, len(rows) - 0.1)
        ax.set_title(f"{bench} ({BENCHMARKS[bench]['kind']}, n={r['n']:,})", color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("Δ execution accuracy vs base (points), 95% paired bootstrap CI", color=INK_2, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=INK_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[0].set_yticks([ys[label] for label in rows], rows)
    for tick in axes[0].get_yticklabels():
        tick.set_color(INK_2 if tick.get_text() in OTHERS else INK)
    fig.text(0.012, 0.012, f"Green: {RUN} trained with different seeds. Grey: the other configurations, one seed "
             "each. Bars resample questions; they do not include seed-to-seed variation.",
             color=INK_2, fontsize=8, ha="left", va="bottom")

    fig.tight_layout(w_pad=3, rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
