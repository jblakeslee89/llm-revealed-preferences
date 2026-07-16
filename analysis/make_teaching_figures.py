"""Generate the pedagogical figures for the student-facing Phase 1-2 walkthrough.

Outputs vector PDFs to writeups/figures/. Formula-based curves use the estimated
values from the project; empirical panels read the real elicitation CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

MOSS = "#758C58"
INK = "#1C1C1E"
HUMAN = "#C0692E"   # warm accent for the human curve
RAT = "#8A8A93"     # gray for the rational benchmark

rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})

FIG = Path(__file__).resolve().parent.parent / "writeups" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = Path(__file__).resolve().parent.parent / "data"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Fig 1: what "risk aversion" is -- CRRA utility curves
# ---------------------------------------------------------------------------
def fig_crra():
    x = np.linspace(1, 200, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for r, ls, lab in [(0.0, ":", "r = 0  (risk-neutral, a straight line)"),
                       (0.5, "-", "r = 0.5  (mild risk aversion)"),
                       (0.9, "--", "r = 0.9  (strong risk aversion)")]:
        u = x if r == 0 else (x ** (1 - r) - 1) / (1 - r)
        u = (u - u.min()) / (u.max() - u.min())
        ax.plot(x, u, ls, color=MOSS if r == 0.5 else INK, lw=2 if r == 0.5 else 1.4, label=lab)
    ax.set_xlabel("money ($)")
    ax.set_ylabel("happiness from the money (utility)")
    ax.set_title("Risk aversion is a curved happiness line", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.annotate("the more it curves,\nthe more the AI\ndislikes gambling",
                xy=(120, 0.78), xytext=(70, 0.35), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=RAT))
    save(fig, "fig1_crra")


# ---------------------------------------------------------------------------
# Fig 2: probability weighting -- Claude vs human vs rational (Prelec)
# ---------------------------------------------------------------------------
def prelec(p, g):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.exp(-((-np.log(p)) ** g))


def fig_weighting():
    p = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(p, p, ":", color=RAT, lw=1.6, label="rational: weight = probability")
    ax.plot(p, prelec(p, 0.65), "-", color=HUMAN, lw=2.2,
            label="typical human (about 0.65): overweights long shots")
    ax.plot(p, prelec(p, 2.69), "-", color=MOSS, lw=2.6,
            label="Claude (2.69): underweights long shots")
    ax.set_xlabel("actual chance of winning")
    ax.set_ylabel("weight the decider puts on it")
    ax.set_title("How much a small chance 'feels' worth", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("a 20% chance barely\nregisters for Claude",
                xy=(0.2, prelec(0.2, 2.69)), xytext=(0.42, 0.12), fontsize=9,
                color=INK, arrowprops=dict(arrowstyle="->", color=MOSS))
    save(fig, "fig2_weighting")


# ---------------------------------------------------------------------------
# Fig 3: empirical -- choice tracks probability, not payoff (real Phase 1 data)
# ---------------------------------------------------------------------------
def fig_empirical_p1():
    df = pd.read_csv(DATA / "phase1_claude-haiku-4-5-20251001.csv")
    df = df[df["discarded"] == 0].copy()
    df["chose_gamble"] = df["chose_gamble"].astype(int)
    df["pbin"] = pd.cut(df["p"], [0.15, 0.3, 0.45, 0.6, 0.85])
    g = df.groupby("pbin", observed=True).agg(
        take=("chose_gamble", "mean"), ev=("ev_ratio", "mean")).reset_index()
    centers = [0.225, 0.375, 0.525, 0.725]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(centers, g["take"], width=0.12, color=MOSS, label="how often Claude took the gamble")
    ax2 = ax.twinx()
    ax2.plot(centers, g["ev"], "o--", color=INK, lw=1.6,
             label="how good the gamble actually was (payoff vs. sure thing)")
    ax2.set_ylim(0.9, 1.3)
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("chance of winning the gamble")
    ax.set_ylabel("share of times the gamble was chosen", color=MOSS)
    ax2.set_ylabel("actual value of the gamble", color=INK)
    ax.set_title("Claude follows the odds, not the money", color=INK, fontsize=13)
    ax.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left")
    save(fig, "fig3_empirical_p1")


# ---------------------------------------------------------------------------
# Fig 4: loss aversion value function -- Claude vs human vs rational
# ---------------------------------------------------------------------------
def value(x, alpha, lam):
    return np.where(x >= 0, np.abs(x) ** alpha, -lam * np.abs(x) ** alpha)


def fig_lossaversion():
    x = np.linspace(-100, 100, 500)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for a, l, c, lw, lab in [
        (1.0, 1.0, RAT, 1.6, "rational: a gain and an equal loss cancel"),
        (0.88, 2.25, HUMAN, 2.2, "typical human (fear 2.25)"),
        (0.50, 3.23, MOSS, 2.6, "Claude (fear 3.23)")]:
        y = value(x, a, l)
        y = y / np.abs(value(np.array([100.0]), a, l))[0]
        ax.plot(x, y, color=c, lw=lw, label=lab)
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("outcome relative to the starting point  (losses  <-  0  ->  gains)")
    ax.set_ylabel("subjective value")
    ax.set_title("The steeper the drop, the more losses sting", color=INK, fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.annotate("Claude's loss side\nfalls fastest",
                xy=(-70, value(np.array([-70.0]), 0.5, 3.23)[0] / value(np.array([100.0]), 0.5, 3.23)[0]),
                xytext=(-95, -1.6), fontsize=9, color=MOSS,
                arrowprops=dict(arrowstyle="->", color=MOSS))
    save(fig, "fig4_lossaversion")


# ---------------------------------------------------------------------------
# Fig 5: the framing effect (real Phase 2 data)
# ---------------------------------------------------------------------------
def fig_framing():
    df = pd.read_csv(DATA / "phase2_claude-haiku-4-5-20251001.csv")
    df = df[df["discarded"] == 0].copy()
    df["chose_gamble"] = df["chose_gamble"].astype(int)
    order = ["gain", "neutral", "mixed", "dominant"]
    labels = ["worded as\na GAIN", "neutral\nwording", "downside worded\nas a LOSS",
              "control:\none is clearly\nbigger"]
    means = [df[df.frame == f]["chose_gamble"].mean() for f in order]
    colors = [MOSS, MOSS, MOSS, RAT]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    bars = ax.bar(range(4), means, color=colors, width=0.62)
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m*100:.0f}%", ha="center", fontsize=11, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("share of times Claude took the gamble")
    ax.set_ylim(0, 1.1)
    ax.set_title("Same gamble, different word, different choice", color=INK, fontsize=13)
    ax.annotate("", xy=(2, 0.30), xytext=(0, 0.44),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(1.0, 0.52, "just renaming the\ndownside 'a loss'\ncuts risk-taking 4x",
            ha="center", fontsize=9, color=INK)
    save(fig, "fig5_framing")


if __name__ == "__main__":
    fig_crra()
    fig_weighting()
    fig_empirical_p1()
    fig_lossaversion()
    fig_framing()
    print("all figures ->", FIG)
