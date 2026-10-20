"""Figure 1 of the risk-advice paper: P(risky option) against how much the gamble is worth
relative to the sure option, money vs each people domain, per open model and regime.

x: ln(EV of gamble / sure amount), binned; right of 0 the gamble pays more on average.
y: mean P(risky), self and advise, gain and mixed frames and both orders pooled.
An expected-value maximizer is a step from 0 to 1 at x = 0.

Usage: python analysis/make_paper_figures.py  ->  writeups/figures/fig1_money_vs_lives.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from stage_a import DATA, DOMAINS, MODELS, load  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "writeups" / "figures"
# categorical slots 1-4 of the dataviz reference palette (validated: CVD and normal-vision pass);
# markers are the secondary encoding for print and for the two low-contrast hues
STYLE = {"money": ("#2a78d6", "o"), "civilians": ("#eb6834", "s"),
         "troops": ("#1baf7a", "^"), "detainees": ("#eda100", "D")}
BINS = np.array([-np.inf, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, np.inf])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "sans-serif"})
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 7.2), sharex=True, sharey=True)
    for i, (label, stem) in enumerate(MODELS):
        for j, (suffix, regime) in enumerate((("", "Immediate answer"), ("_reason", "Reason first"))):
            ax = axes[i, j]
            d = load(DATA / f"stageA_{stem}{suffix}.csv")
            d = d[(d.excluded == 0) & d.frame.isin(["gain", "mixed"])].copy()
            d["x"] = np.log(d.ev / d.sure)
            d["bin"] = np.digitize(d.x, BINS) - 1
            ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for dom in DOMAINS:
                g = d[d.domain == dom].groupby("bin").agg(x=("x", "mean"), y=("p_gamble", "mean"),
                                                         n=("x", "size"))
                g = g[g.n >= 8]
                color, marker = STYLE[dom]
                ax.plot(g.x, g.y, color=color, lw=2, marker=marker, ms=5,
                        mec="white", mew=0.8, label=dom.capitalize(), zorder=3)
            ax.set_ylim(-0.03, 1.03)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(regime, fontsize=9.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"{label.split(' (')[0]}\nP(risky option)")
            if i == 2:
                ax.set_xlabel("ln(gamble's expected value / sure amount)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_money_vs_lives.{ext}", dpi=200, facecolor="white")
    print(f"wrote {OUT}/fig1_money_vs_lives.pdf/.png")


if __name__ == "__main__":
    main()
