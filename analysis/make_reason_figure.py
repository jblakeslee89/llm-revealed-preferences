"""Figure for the reason-then-answer arm in writeups/phase3-writeup.tex.

Panel A: agreement with risk-neutral EV on clear core cells, four conditions per subject
(immediate vs reason-first, uninduced vs induced). Panel B: the frame gap (mean P(gamble),
mixed minus gain frame) with and without reasoning, uninduced.

Every number is recomputed from data/ on the cells the reasoning runs cover (20 of 40 gambles
per instrument, matched on instrument and trial_id), using the same scoring as
analysis/score_induced.py.

Usage:
    python analysis/make_reason_figure.py      # writes writeups/figures/phase3_reason.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_induced import load, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "writeups" / "figures" / "phase3_reason.pdf"
KEY = ["instrument", "trial_id"]

MOSS, MOSS_PALE = "#758C58", "#B9C6A6"
GRAY, GRAY_PALE = "#8E8E96", "#C8C8CE"
RED, INK = "#C00000", "#1C1C1E"

SUBJECTS = [  # label, file stem
    ("Qwen Instruct\n(chat)", "qwen-inst_chat"),
    ("OLMo SFT\n(few-shot)", "olmo-sft_fewshot"),
    ("OLMo DPO\n(few-shot)", "olmo-dpo_fewshot"),
    ("OLMo Instruct\n(few-shot)", "olmo-inst_fewshot"),
]
CONDITIONS = [  # label, suffix, color
    ("immediate, uninduced", "", GRAY_PALE),
    ("immediate, induced", "_riskneutral", GRAY),
    ("reason first, uninduced", "_reason", MOSS_PALE),
    ("reason first, induced", "_reason_riskneutral", MOSS),
]


def summarize(stem):
    cells = load(DATA / f"phase3_{stem}_reason.csv")[KEY]
    out = {}
    for label, suffix, _ in CONDITIONS:
        d = load(DATA / f"phase3_{stem}{suffix}.csv").merge(cells, on=KEY)
        sc = score(d, "riskneutral", 0.05)
        clear = sc[(sc["frame"] != "dominant") & sc["clear"]]
        gap = d[d["frame"] == "mixed"]["p_gamble"].mean() - d[d["frame"] == "gain"]["p_gamble"].mean()
        out[label] = {"ev": clear["hard_correct"].mean(), "gap": gap}
    return out


def main():
    stats = {stem: summarize(stem) for _, stem in SUBJECTS}
    plt.rcParams.update({"font.size": 8, "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"],
                         "axes.edgecolor": "#B0B0B0", "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(SUBJECTS))
    w = 0.2
    for k, (label, _, color) in enumerate(CONDITIONS):
        vals = [stats[stem][label]["ev"] for _, stem in SUBJECTS]
        bars = a.bar(x + (k - 1.5) * w, vals, w, color=color, label=label)
        for bar, v in zip(bars, vals):
            a.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                   fontsize=6.5, color=INK)
    a.axhline(0.5, color=RED, lw=0.8, ls=":")
    a.text(len(SUBJECTS) - 0.58, 0.51, "chance", color=RED, fontsize=6.5)
    a.set_xticks(x, [s for s, _ in SUBJECTS])
    a.set_ylim(0, 1.08)
    a.set_ylabel("Agreement with expected value\n(clear cells)")
    a.set_title("A. Reasoning lifts expected-value agreement", loc="left", fontsize=8.5)
    a.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    gi = [stats[stem]["immediate, uninduced"]["gap"] for _, stem in SUBJECTS]
    gr = [stats[stem]["reason first, uninduced"]["gap"] for _, stem in SUBJECTS]
    w2 = 0.35
    for off, vals, color, label in [(-w2 / 2, gi, GRAY, "immediate answer"), (w2 / 2, gr, MOSS, "reason first")]:
        bars = b.bar(x + off, vals, w2, color=color, label=label)
        for bar, v in zip(bars, vals):
            y = v + (0.02 if v >= 0 else -0.06)
            b.text(bar.get_x() + bar.get_width() / 2, y, f"{v:+.2f}", ha="center", fontsize=6.5, color=INK)
    b.axhline(0, color=INK, lw=0.6)
    b.set_xticks(x, [s for s, _ in SUBJECTS])
    b.set_ylim(-0.25, 0.75)
    b.set_ylabel("Frame gap (mixed minus gain)")
    b.set_title("B. Loss wording, with and without reasoning (uninduced)", loc="left", fontsize=8.5)
    b.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    for _, stem in SUBJECTS:
        print(stem, {k: {m: round(v, 3) for m, v in d.items()} for k, d in stats[stem].items()})
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()
