"""Figures 2-4 of the risk-advice paper, from results/paper_frontier_tables.csv
(python analysis/paper_frontier_tables.py first). Point estimates, roles pooled.
Usage: python analysis/make_frontier_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "writeups" / "figures"
t = pd.read_csv(ROOT / "results" / "paper_frontier_tables.csv")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"civilians": "#3b6ea5", "troops": "#5a9e6f", "detainees": "#b5473a", "release": "#d08c3c"}

# Fig 2: shift from money/self by model and arm, main grid, low effort
rows = [("Sonnet", "snap", "-", "Sonnet, snap"),
        ("Sonnet", "written reasoning", "-", "Sonnet, written reasoning"),
        ("Sonnet", "extended thinking", "low", "Sonnet, extended thinking"),
        ("Fable", "extended thinking", "low", "Fable, extended thinking"),
        ("Haiku", "snap", "-", "Haiku, snap"),
        ("Haiku", "written reasoning", "-", "Haiku, written reasoning"),
        ("Haiku", "extended thinking", "low", "Haiku, extended thinking"),
        ("Opus", "extended thinking", "low", "Opus, extended thinking"),
        ("Opus", "ext. thinking + written reasoning", "low", "Opus, ext. thinking + written")]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for i, (m, a, e, lab) in enumerate(rows):
    for k, dom in enumerate(["civilians", "troops", "detainees"]):
        v = t[(t.model == m) & (t.arm == a) & (t.effort == e) & (t.domain == dom)]["shift"].iloc[0]
        ax.plot(v, -i + (k - 1) * 0.18, "o", color=COL[dom], ms=5, label=dom if i == 0 else None)
ax.axvline(0, color="0.4", lw=0.8)
ax.set_yticks([-i for i in range(len(rows))], [r[3] for r in rows])
ax.set_xlabel("Change in P(take the gamble) relative to money, own decision")
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_xlim(-0.75, 0.2)
fig.tight_layout()
fig.savefig(OUT / "fig2_frontier_shift.pdf"); fig.savefig(OUT / "fig2_frontier_shift.png", dpi=200)

# Fig 3: effort dose
eff = ["low", "medium", "high"]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
mk = {"Sonnet": ("o-", "#3b3b8f"), "Fable": ("s-", "#8f3b6e"), "Haiku": ("^--", "#888888"),
      "Opus": ("d--", "#444444")}
for ax, doms, title in [(axes[0], ["civilians", "troops"], "Civilians and soldiers"),
                        (axes[1], ["detainees"], "Detained aid workers")]:
    for m, (style, c) in mk.items():
        xs, ys = [], []
        for j, e in enumerate(eff):
            s = t[(t.model == m) & (t.arm == "extended thinking") & (t.effort == e) & t.domain.isin(doms)]
            if len(s):
                xs.append(j); ys.append(s["shift"].mean())
        ax.plot(xs, ys, style, color=c, label=m, ms=5, lw=1.2)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(3), eff); ax.set_title(title, fontsize=9); ax.set_xlabel("Effort setting")
axes[0].set_ylabel("Change in P(gamble) from money")
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig3_effort_dose.pdf"); fig.savefig(OUT / "fig3_effort_dose.png", dpi=200)

# Fig 4: detainees vs release, second wording
runs = [("Sonnet", "A2 snap", "Sonnet\nsnap"), ("Sonnet", "A2 written reasoning (84%)", "Sonnet\nwritten"),
        ("Sonnet", "A2 extended thinking", "Sonnet\nthinking"), ("Fable", "A2 extended thinking", "Fable\nthinking"),
        ("Haiku", "A2 written reasoning", "Haiku\nwritten"), ("Haiku", "A2 extended thinking", "Haiku\nthinking"),
        ("Opus", "A2 extended thinking", "Opus\nthinking")]
fig, ax = plt.subplots(figsize=(6.2, 2.9))
w = 0.38
for i, (m, a, lab) in enumerate(runs):
    for k, dom in enumerate(["detainees", "release"]):
        v = t[(t.model == m) & (t.arm == a) & (t.domain == dom)]["shift"].iloc[0]
        ax.bar(i + (k - 0.5) * w, v, w, color=COL[dom],
               label=("with escalation clause" if dom == "detainees" else "release, no escalation") if i == 0 else None)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(range(len(runs)), [r[2] for r in runs], fontsize=8)
ax.set_ylabel("Change in P(gamble)\nfrom money")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig4_release.pdf"); fig.savefig(OUT / "fig4_release.png", dpi=200)
print("wrote fig2-fig4 to", OUT)
