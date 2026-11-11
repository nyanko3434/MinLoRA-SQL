"""The adapter-scaling probe as one figure: change in execution accuracy vs the base model against
the fraction of its trained LoRA scaling the adapter runs at, for Spider dev (in-domain) and BIRD
dev (zero-shot), on one shared axis. Run from repo root (CPU, reads only stored results):

    python scripts/plot_scaling.py

Writes results/scaling_curve.png. Points and intervals come from summarize_scaling.effects, so
they equal the numbers in results/eval/scaling/summary.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_scaling import BENCHMARKS, RUN, effects, load_points

# Same surface / ink tokens as plot_results.py; categorical slots 1 and 2 of the reference palette,
# in fixed order. Identity is also carried by the direct labels, never by color alone.
SERIES_COLORS = {"Spider dev": "#2a78d6", "BIRD dev": "#eb6834"}
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
OUT = Path("results/scaling_curve.png")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK_2, linewidth=1, zorder=1)

    ticks = {0.0, 1.0}
    # The two series share x positions and their intervals overlap near zero, so each is nudged
    # 0.004 off the tick: enough to keep the bars apart, too little to misplace a point.
    for (bench, color), dodge in zip(SERIES_COLORS.items(), (-0.004, 0.004)):
        eff, base_ex = effects(load_points(bench))
        scales = sorted(eff)
        ticks |= set(scales)
        xs, ys = [0.0, *(s + dodge for s in scales)], [0.0, *(eff[s]["delta"] for s in scales)]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=2, label=f"{bench} ({BENCHMARKS[bench]['kind']})")
        for s in scales:
            lo, hi = eff[s]["ci"]
            ax.plot([s + dodge, s + dodge], [lo, hi], color=color, linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot(xs[1:], ys[1:], "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        for s in scales:
            d = eff[s]["delta"]
            ax.annotate(f"{d:+.2f}".replace("-", "−"), (s + dodge, d), xytext=(-8, 7 if d >= 0 else -7),
                        textcoords="offset points", color=INK, fontsize=9, ha="right",
                        va="bottom" if d >= 0 else "top")
            print(f"{bench:10s} x{s:<5g} {d:+.2f} [{eff[s]['ci'][0]:+.2f}, {eff[s]['ci'][1]:+.2f}]")
        ax.annotate(f"{bench}\nbase {base_ex:.2f}%", (1.0, eff[1.0]["delta"]), xytext=(44, 0),
                    textcoords="offset points", color=INK_2, fontsize=9, ha="left", va="center")

    # Both series start at the base model, which is one point, not one per series.
    ax.plot(0, 0, "o", markersize=8, color=INK_2, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ordered = sorted(ticks)
    ax.set_xticks(ordered, [{0.0: "0\n(base)", 1.0: "1\n(as trained)"}.get(s, f"{s:g}") for s in ordered])
    ax.set_xlim(-0.05, 1.32)
    ax.set_xlabel(f"adapter scale: fraction of the trained alpha / r ({RUN})", color=INK_2, fontsize=9)
    ax.set_ylabel("Δ execution accuracy vs base (points)", color=INK_2, fontsize=9)
    ax.set_title("Turning the adapter down: Spider gain and BIRD loss by scale", color=INK, fontsize=11,
                 loc="left", pad=30)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_bounds(0, 1)
    legend = ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, fontsize=9,
                       handlelength=1.6, columnspacing=1.8, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.text(0.012, 0.012, "Vertical bars: 95% paired bootstrap CI (resampling questions). Diagnostic, not a tuned result.",
             color=INK_2, fontsize=8, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
