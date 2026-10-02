"""Figures for writeups/phase3-teaching-paper.tex.

Re-renders the five Phase 3 briefing charts at paper aspect ratios, in the same
Upshot house style: white ground, no chart junk, direct labels on the marks,
conversational annotations on thin leader lines, RAND purple for the story
series, warm gray for context, red reserved for negatives and reference lines.

Every plotted number is recomputed here from the CSVs in data/. Nothing is
typed in by hand except the Phase 2 Haiku reference values, which come from
data/phase2_claude-haiku-4-5-20251001.csv and are also recomputed below.

Usage:
    python analysis/make_teaching_paper_figures.py
Writes writeups/figures/teach_*.pdf
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "writeups", "figures")
os.makedirs(OUT, exist_ok=True)

PURPLE = "#761DDB"
PURPLE_PALE = "#F0E6FB"
GRAY = "#A39A94"          # warm gray, context series
GRAY_TEXT = "#6E6660"
INK = "#1C1C1E"
RED = "#C00000"
RULE = "#D9D5D2"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "text.color": INK,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "xtick.color": GRAY_TEXT,
    "ytick.color": GRAY_TEXT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


# ----------------------------------------------------------------- data layer
def load(tag: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, f"phase3_{tag}.csv"))
    return d[(d["excluded"] == 0) & d["p_gamble"].notna()]


def dominance(tag: str) -> float:
    d = load(tag)
    return d.loc[d["frame"] == "dominant", "p_gamble"].mean()


def frame_gap(tag: str) -> float:
    d = load(tag)
    return (d.loc[d["frame"] == "mixed", "p_gamble"].mean()
            - d.loc[d["frame"] == "gain", "p_gamble"].mean())


def haiku_phase2() -> tuple[float, float]:
    d = pd.read_csv(os.path.join(DATA, "phase2_claude-haiku-4-5-20251001.csv"))
    d = d[d["discarded"] == 0]
    dom = d.loc[d["frame"] == "dominant", "chose_gamble"].mean()
    gap = (d.loc[d["frame"] == "mixed", "chose_gamble"].mean()
           - d.loc[d["frame"] == "gain", "chose_gamble"].mean())
    return dom, gap


def induced_scores(tag: str, rule: str = "riskneutral", margin: float = 0.05):
    """Mirror of analysis/score_induced.py."""
    df = load(tag)
    s = df["sure"].values.astype(float)
    h = df["hi"].values.astype(float)
    p = df["p"].values.astype(float)
    if rule == "riskneutral":
        eu_g, eu_s = p * h, s
    else:
        eu_g, eu_s = p * np.sqrt(h), np.sqrt(s)
    opt = eu_g > eu_s
    pg = df["p_gamble"].values.astype(float)
    mass = np.where(opt, pg, 1 - pg)
    hard = (pg > 0.5) == opt
    dist = np.abs(np.log(np.clip(eu_g, 1e-9, None) / np.clip(eu_s, 1e-9, None)))
    sc = pd.DataFrame({"frame": df["frame"].values, "mass": mass,
                       "hard": hard, "clear": dist >= margin})
    core_clear = sc[(sc["frame"] != "dominant") & sc["clear"]]
    dom = sc[sc["frame"] == "dominant"]
    return core_clear["mass"].mean() * 100, dom["hard"].mean() * 100


# ----------------------------------------------------------------- style bits
def bare(ax, keep_bottom=False, keep_left=False):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if keep_bottom:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(RULE)
    if keep_left:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(RULE)
    ax.tick_params(length=0)


def leader(ax, xy_text, xy_point, text, color=GRAY_TEXT, ha="left", va="center",
           size=7.6, lw=0.6):
    ax.annotate(text, xy=xy_point, xytext=xy_text, color=color, fontsize=size,
                ha=ha, va=va,
                arrowprops=dict(arrowstyle="-", color=color, lw=lw,
                                shrinkA=2, shrinkB=2))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------- Fig 1: free-money bars
def fig_freemoney():
    haiku_dom, _ = haiku_phase2()
    rows = [
        ("Claude Haiku", "a finished assistant", haiku_dom * 100, PURPLE),
        ("Qwen2.5", "fresh from pre-training", dominance("qwen-base_fewshot") * 100, GRAY),
        ("Llama-3.1", "fresh from pre-training", dominance("llama-base_fewshot") * 100, GRAY),
        ("OLMo-2", "fresh from pre-training", dominance("olmo-base_fewshot") * 100, GRAY),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.55))
    y = np.arange(len(rows))[::-1]
    for yi, (name, sub, val, col) in zip(y, rows):
        ax.barh(yi, val, height=0.52, color=col, zorder=3)
        ax.text(val + 1.4, yi, f"{val:.1f}%", va="center", ha="left",
                fontsize=9, color=col if col == PURPLE else GRAY_TEXT)
        ax.text(-2.0, yi + 0.16, name, va="center", ha="right", fontsize=9, color=INK)
        ax.text(-2.0, yi - 0.20, sub, va="center", ha="right", fontsize=7.4, color=GRAY_TEXT)

    ax.axvline(50, color=RED, ls=(0, (2, 2)), lw=0.9, zorder=4)
    ax.text(50, len(rows) - 0.42, "a coin flip", color=RED, fontsize=7.6,
            ha="center", va="bottom")
    leader(ax, (86, 0.95), (65, 0.62),
           "All three take the free money\nbarely more often than chance")

    ax.set_xlim(-30, 150)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_yticks([])
    ax.set_xticks([])
    bare(ax)
    save(fig, "teach_freemoney.pdf")


# ------------------------------------------------------- Fig 2: the staircase
def fig_staircase():
    stages = ["olmo-base_fewshot", "olmo-sft_fewshot",
              "olmo-dpo_fewshot", "olmo-inst_fewshot"]
    labels = ["Pre-trained\nbase", "After supervised\nfine-tuning",
              "After preference\ntraining", "Finished\nassistant"]
    dom = np.array([dominance(t) for t in stages]) * 100
    gap = np.array([frame_gap(t) for t in stages]) * 100
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    ax.axvspan(-0.35, 1.0, color=PURPLE_PALE, zorder=0)
    for yy in (25, 50, 75, 100):
        ax.axhline(yy, color=RULE, lw=0.6, zorder=1)

    ax.plot(x, dom, color=PURPLE, lw=2.4, zorder=3)
    ax.scatter(x, dom, s=48, color=PURPLE, zorder=4)
    ax.plot(x, gap, color=GRAY, lw=2.4, zorder=3)
    ax.scatter(x, gap, s=48, color=GRAY, zorder=4)

    for xi, v in zip(x, dom):
        ax.text(xi, v + 4.2, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color=PURPLE)
    for xi, v in zip(x, gap):
        off = -5.0 if xi in (0, 1) else 5.0
        va = "top" if off < 0 else "bottom"
        ax.text(xi, v + off, f"{v:.1f}", ha="center", va=va,
                fontsize=9, color=GRAY_TEXT)

    ax.text(3.14, dom[-1], "Picks the larger\nsure amount", color=PURPLE,
            fontsize=8.4, va="center", ha="left")
    ax.text(3.14, gap[-1], "Gambles this much more\nwhen a bet is called a loss",
            color=GRAY_TEXT, fontsize=8.4, va="center", ha="left")
    ax.text(0.5, 101, "Most of the move\nhappens here", color=PURPLE,
            fontsize=8.4, ha="center", va="bottom")
    ax.plot([0.5, 0.5], [84, 97], color=PURPLE, lw=0.6, zorder=3)

    ax.set_xlim(-0.4, 4.75)
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4, color=INK)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
    ax.set_ylabel("percent", fontsize=8, color=GRAY_TEXT)
    bare(ax, keep_bottom=True)
    save(fig, "teach_staircase.pdf")


# ---------------------------------------------------------- Fig 3: format gap
def fig_format():
    rows = [
        ("Qwen2.5", dominance("qwen-inst_fewshot") * 100,
         dominance("qwen-inst_chat") * 100, "Chat wakes it up"),
        ("OLMo-2", dominance("olmo-inst_fewshot") * 100,
         dominance("olmo-inst_chat") * 100, "Chat puts it to sleep"),
        ("Llama-3.1", dominance("llama-inst_fewshot") * 100,
         dominance("llama-inst_chat") * 100, "The one steady model"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    y = np.arange(len(rows))[::-1]
    for yi, (name, fs, ch, note) in zip(y, rows):
        ax.plot([fs, ch], [yi, yi], color=RULE, lw=2.6, zorder=2,
                solid_capstyle="round")
        ax.scatter([fs], [yi], s=95, color=GRAY, zorder=3)
        ax.scatter([ch], [yi], s=95, color=PURPLE, zorder=3)
        lo, hi = (fs, ch) if fs <= ch else (ch, fs)
        ax.text(lo - 1.6, yi, f"{lo:.1f}", ha="right", va="center",
                fontsize=8.6, color=GRAY_TEXT)
        ax.text(hi + 1.6, yi, f"{hi:.1f}", ha="left", va="center",
                fontsize=8.6, color=GRAY_TEXT)
        ax.text(45, yi, name, ha="left", va="center", fontsize=9.5, color=INK)
        ax.text(101, yi, note, ha="left", va="center", fontsize=8.4, color=GRAY_TEXT)

    ax.text(70, len(rows) - 0.42, "asked as a\nfill-in-the-blank", color=GRAY_TEXT,
            fontsize=8, ha="center", va="bottom")
    ax.text(93, len(rows) - 0.42, "asked as a\nchat message", color=PURPLE,
            fontsize=8, ha="center", va="bottom")

    ax.set_xlim(44, 128)
    ax.set_ylim(-0.6, len(rows) + 0.45)
    ax.set_yticks([])
    ax.set_xticks([])
    bare(ax)
    save(fig, "teach_format.pdf")


# --------------------------------------------------------- Fig 4: frame gaps
def fig_framing():
    _, haiku_gap = haiku_phase2()
    rows = [
        ("Claude Haiku", haiku_gap * 100),
        ("Qwen2.5", frame_gap("qwen-inst_chat") * 100),
        ("OLMo-2", frame_gap("olmo-inst_fewshot") * 100),
        ("Llama-3.1", frame_gap("llama-inst_fewshot") * 100),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = np.arange(len(rows))
    for yy in (-25, 25, 50):
        ax.axhline(yy, color=RULE, lw=0.6, zorder=1)
    for xi, (name, v) in zip(x, rows):
        col = RED if v < 0 else PURPLE
        ax.bar(xi, v, width=0.5, color=col, zorder=3)
        ax.text(xi, v + (3.5 if v > 0 else -3.5), f"{v:+.1f}",
                ha="center", va="bottom" if v > 0 else "top",
                fontsize=10, color=INK)
    ax.axhline(0, color=INK, lw=1.1, zorder=4)

    ax.text(-0.62, 62, "MORE GAMBLING", color=PURPLE, fontsize=7.6,
            ha="left", va="center")
    ax.text(-0.62, -40, "LESS", color=RED, fontsize=7.6, ha="left", va="center")
    leader(ax, (1.42, 63), (1.28, 55),
           "Qwen nearly doubles its\ngambling on the same bet")
    ax.text(3.0, 22, "Llama hardly notices", color=GRAY_TEXT, fontsize=7.6,
            ha="center", va="bottom")
    leader(ax, (0.52, -30), (0.28, -22), "Claude Haiku\ngoes the other way",
           color=RED)

    ax.set_xlim(-0.75, 3.6)
    ax.set_ylim(-48, 74)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_yticks([-25, 0, 25, 50])
    ax.set_yticklabels(["-25", "0", "+25", "+50"], fontsize=8)
    ax.set_ylabel("change in gambling, points", fontsize=8, color=GRAY_TEXT)
    bare(ax)
    save(fig, "teach_framing.pdf")


# -------------------------------------------------------- Fig 5: induced arm
def fig_induced():
    pairs = [
        ("Qwen2.5\nassistant", "qwen-inst_chat", "qwen-inst_chat_riskneutral"),
        ("OLMo-2\nbase", "olmo-base_fewshot", "olmo-base_fewshot_riskneutral"),
        ("OLMo-2\nafter\nfine-tuning", "olmo-sft_fewshot", "olmo-sft_fewshot_riskneutral"),
        ("OLMo-2\nafter preference\ntraining", "olmo-dpo_fewshot", "olmo-dpo_fewshot_riskneutral"),
        ("OLMo-2\nfinished\nassistant", "olmo-inst_fewshot", "olmo-inst_fewshot_riskneutral"),
    ]
    before, after, domline = [], [], []
    for _, base_tag, ind_tag in pairs:
        b, _ = induced_scores(base_tag)
        a, d = induced_scores(ind_tag)
        before.append(b)
        after.append(a)
        domline.append(d)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(pairs))
    w = 0.33
    bx = dict(facecolor="white", edgecolor="none", pad=0.6)
    for yy in (25, 50, 75, 100):
        ax.axhline(yy, color=RULE, lw=0.6, zorder=1)
    ax.axhline(50, color=RED, ls=(0, (2, 2)), lw=0.9, zorder=2)
    ax.bar(x - w / 2, before, width=w, color=GRAY, zorder=3)
    ax.bar(x + w / 2, after, width=w, color=PURPLE, zorder=3)
    for xi, (b, a) in zip(x, zip(before, after)):
        ax.text(xi - w / 2, b + 1.8, f"{b:.1f}", ha="center", va="bottom",
                fontsize=8.4, color=GRAY_TEXT, bbox=bx, zorder=5)
        ax.text(xi + w / 2, a + 1.8, f"{a:.1f}", ha="center", va="bottom",
                fontsize=8.4, color=PURPLE, bbox=bx, zorder=5)

    ax.plot(x, domline, color=INK, lw=1.7, zorder=4)
    ax.scatter(x, domline, s=34, color=INK, zorder=5)
    for xi, v in zip(x, domline):
        ax.text(xi, v + 3.0, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8.6, color=INK)

    ax.text(4.55, 50, "a coin flip", color=RED, fontsize=7.6, ha="left",
            va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(4.55, domline[-1], "Same instruction, on questions\nneeding no arithmetic",
            color=INK, fontsize=8, ha="left", va="center")
    ax.text(-0.52, 63, "before", color=GRAY_TEXT, fontsize=8, ha="left")
    ax.text(0.60, 63, "after the instruction", color=PURPLE, fontsize=8, ha="left")
    leader(ax, (2.05, 69), (3.17, 55),
           "The instruction does almost nothing\nwhere the rule requires arithmetic",
           color=PURPLE, ha="left", va="bottom")

    ax.set_xlim(-0.6, 6.9)
    ax.set_ylim(0, 116)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=7.8, color=INK)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
    ax.set_ylabel("percent following the instructed rule", fontsize=8, color=GRAY_TEXT)
    bare(ax, keep_bottom=True)
    save(fig, "teach_induced.pdf")


def reason_scores(stem: str, suffix: str):
    """EV hard accuracy on clear core cells and frame gap, on the reasoning arm's cells."""
    key = ["instrument", "trial_id"]
    cells = load(f"{stem}_reason")[key]
    d = load(f"{stem}{suffix}").merge(cells, on=key)
    ev = d["p"] * d["hi"]
    opt = ev > d["sure"]
    clear = (np.abs(np.log(ev / d["sure"])) >= 0.05) & (d["frame"] != "dominant")
    hit = ((d["p_gamble"] > 0.5) == opt)[clear].mean() * 100
    gap = (d[d["frame"] == "mixed"]["p_gamble"].mean()
           - d[d["frame"] == "gain"]["p_gamble"].mean()) * 100
    return hit, gap


def fig_reason():
    subjects = [
        ("Qwen2.5\nassistant", "qwen-inst_chat"),
        ("OLMo-2\nafter\nfine-tuning", "olmo-sft_fewshot"),
        ("OLMo-2\nafter\npreference\ntraining", "olmo-dpo_fewshot"),
        ("OLMo-2\nfinished\nassistant", "olmo-inst_fewshot"),
        ("Llama-3.1\nassistant", "llama-inst_chat"),
    ]

    def at_once_score(stem):
        # Llama has no immediate-answer run under the instruction: leave the bar out
        try:
            return reason_scores(stem, "_riskneutral")[0]
        except FileNotFoundError:
            return float("nan")

    at_once = [at_once_score(s) for _, s in subjects]
    reasoned = [reason_scores(s, "_reason_riskneutral")[0] for _, s in subjects]
    gap_once = [reason_scores(s, "")[1] for _, s in subjects]
    gap_reason = [reason_scores(s, "_reason")[1] for _, s in subjects]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"wspace": 0.38})
    x = np.arange(len(subjects))
    w = 0.38
    bx = dict(facecolor="white", edgecolor="none", pad=0.6)

    for yy in (25, 50, 75, 100):
        a.axhline(yy, color=RULE, lw=0.6, zorder=1)
    a.axhline(50, color=RED, ls=(0, (2, 2)), lw=0.9, zorder=2)
    a.bar(x - w / 2, at_once, width=w, color=GRAY, zorder=3)
    a.bar(x + w / 2, reasoned, width=w, color=PURPLE, zorder=3)
    for xi, (o, r) in zip(x, zip(at_once, reasoned)):
        if np.isnan(o):
            a.text(xi - w / 2, 3, "not\nrun", ha="center", va="bottom", fontsize=6.2,
                   color=GRAY_TEXT, zorder=5)
        else:
            a.text(xi - w / 2, o + 1.8, f"{o:.0f}", ha="center", va="bottom", fontsize=7.4,
                   color=GRAY_TEXT, bbox=bx, zorder=5)
        a.text(xi + w / 2, r + 1.8, f"{r:.0f}", ha="center", va="bottom", fontsize=7.4,
               color=PURPLE, bbox=bx, zorder=5)
    a.text(-0.55, 108, "answer at once", color=GRAY_TEXT, fontsize=7.8, ha="left")
    a.text(1.35, 108, "reason first", color=PURPLE, fontsize=7.8, ha="left")
    a.set_title("Told to maximize expected value:\npercent following the rule", fontsize=8.4,
                color=INK, loc="left")
    a.set_ylim(0, 116)
    a.set_yticks([25, 50, 75, 100])
    a.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
    a.set_xticks(x)
    a.set_xticklabels([s[0] for s in subjects], fontsize=6.3, color=INK)
    bare(a, keep_bottom=True)

    b.axhline(0, color=INK, lw=0.7, zorder=2)
    b.bar(x - w / 2, gap_once, width=w, color=GRAY, zorder=3)
    b.bar(x + w / 2, gap_reason, width=w, color=PURPLE, zorder=3)
    for xi, (o, r) in zip(x, zip(gap_once, gap_reason)):
        b.text(xi - w / 2, o + (1.5 if o >= 0 else -6.5), f"{o:+.0f}", ha="center",
               va="bottom", fontsize=7.4, color=GRAY_TEXT, zorder=5)
        b.text(xi + w / 2, r + (1.5 if r >= 0 else -6.5), f"{r:+.0f}", ha="center",
               va="bottom", fontsize=7.4, color=PURPLE, zorder=5)
    b.set_title("No instruction: extra gambling when\nthe bet is called a loss (points)",
                fontsize=8.4, color=INK, loc="left")
    b.set_ylim(-22, 72)
    b.set_yticks([-20, 0, 20, 40, 60])
    b.set_yticklabels(["-20", "0", "20", "40", "60"], fontsize=8)
    b.set_xticks(x)
    b.set_xticklabels([s[0] for s in subjects], fontsize=6.3, color=INK)
    bare(b, keep_bottom=True)
    save(fig, "teach_reason.pdf")


if __name__ == "__main__":
    fig_freemoney()
    fig_staircase()
    fig_format()
    fig_framing()
    fig_induced()
    fig_reason()
