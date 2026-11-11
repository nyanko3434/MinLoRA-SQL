"""Plot training loss + LR schedule for finished runs and copy their small evidence files into
results/ (runs/ is gitignored; adapter weights stay there). Run from repo root:

    python scripts/plot_train.py --runs qv_r8 qv_r4 qv_r16 attn_r8 all7_r8
    python scripts/plot_train.py --runs qv_r16 qv_r16_s1 qv_r16_s2 --out results/train_loss_seeds.png

Writes results/train_loss.png and results/train/<run>/{log.jsonl, data_report.json,
lora_config.json}. The logged loss is ONE window's token-weighted loss every log_every steps, so
it is noisy; the bold line is a rolling mean over logged points.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

# Color follows the run, never its position in --runs (fixed categorical slots, validated
# reference palette order: blue, orange, aqua, yellow, magenta, green, violet). The last two are
# the seed repeats of qv_r16; plot them with --out so results/train_loss.png keeps the five configs.
RUN_COLORS = {
    "qv_r8": "#2a78d6",
    "qv_r4": "#eb6834",
    "qv_r16": "#1baf7a",
    "attn_r8": "#eda100",
    "all7_r8": "#e87ba4",
    "qv_r16_s1": "#008300",
    "qv_r16_s2": "#6250d6",
}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
WINDOW = 5  # logged points (= 50 steps at log_every 10)

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--out", default="results/train_loss.png")
args = parser.parse_args()
unknown = [r for r in args.runs if r not in RUN_COLORS]
assert not unknown, f"no fixed color for {unknown}; add it to RUN_COLORS"


def rolling_mean(values, window):
    return [sum(values[max(0, i - window + 1):i + 1]) / (i + 1 - max(0, i - window + 1))
            for i in range(len(values))]


fig, (ax_loss, ax_lr) = plt.subplots(
    2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE
)
for name in args.runs:
    run_dir = Path("runs") / name
    rows = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    events = [r for r in rows if "event" in r]
    rows = [r for r in rows if "event" not in r]
    sidecar = json.loads((run_dir / "final" / "lora_config.json").read_text())
    steps = [r["step"] for r in rows]
    loss = [r["loss"] for r in rows]
    color = RUN_COLORS[name]

    ax_loss.plot(steps, loss, color=color, linewidth=1, alpha=0.3)
    ax_loss.plot(steps, rolling_mean(loss, WINDOW), color=color, linewidth=2,
                 label=f"{name}  ({sidecar['n_params'] / 1e6:.2f}M LoRA params)")
    ax_lr.plot(steps, [r["lr"] for r in rows], color=color, linewidth=2)

    tail = loss[-10:]
    print(f"{name}: steps {steps[-1]}, final logged loss {loss[-1]:.4f}, mean of last {len(tail)} logged "
          f"{sum(tail) / len(tail):.4f}, {rows[-1]['elapsed_s'] / 3600:.2f} h, peak reserved "
          f"{max(r['peak_reserved_gb'] for r in rows):.2f} GB{', EVENTS ' + str(events) if events else ''}")

    evidence = Path("results/train") / name
    evidence.mkdir(parents=True, exist_ok=True)
    for src in (run_dir / "log.jsonl", run_dir / "data_report.json", run_dir / "final" / "lora_config.json"):
        shutil.copy2(src, evidence / src.name)

for ax in (ax_loss, ax_lr):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)

ax_loss.set_ylim(bottom=0)
ax_loss.set_ylabel("completion loss (token-weighted)", color=INK_2)
ax_loss.set_title(f"Coder-7B QLoRA on Spider train: loss per logged step (faint) and {WINDOW}-point rolling mean",
                  color=INK, fontsize=11, loc="left")
ax_loss.legend(frameon=False, labelcolor=INK_2)
ax_lr.set_ylabel("learning rate", color=INK_2)
ax_lr.set_xlabel("optimizer step", color=INK_2)
ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

fig.tight_layout()
fig.savefig(args.out, dpi=150, facecolor=SURFACE)
print(f"saved {args.out}")
