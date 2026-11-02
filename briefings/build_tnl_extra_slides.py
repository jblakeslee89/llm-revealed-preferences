"""Build briefings/tnl-extra-slides-2026-11.pptx: two data slides for the Nov 2 TNL breakout.

Slide 1 (in class): which Claude models keep the extra caution with lives.
Slide 2 (pitched for Wednesday): the thinking-effort dose response.

Numbers are copied from docs/frontier-results-2026-10-20.md (Oct 23 table: thinking arm,
one-letter answer, shift in P(risky) from money/self, ranges across roles and frames).
Styling copies briefings/project-overview-5min.pptx (white, Georgia, RAND purple 761DDB).

Usage:
    python briefings/build_tnl_extra_slides.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "tnl-extra-slides-2026-11.pptx"
FIG = HERE / "figures"
LOGO = HERE / "figures" / "rand_logo.png"

PURPLE = "#761DDB"
LIGHT = "#C9A8F2"
GREY = "#9A948E"
INK = "#1C1C1E"

# (lo, hi) shift in P(risky) vs money/self; thinking + one letter
LOW = {
    "Haiku 5.5": {"civ": (-0.05, 0.05), "det": (-0.05, 0.03), "r_civ": "0.87–0.97", "r_det": "0.89–0.96"},
    "Opus 5.5": {"civ": (0.01, 0.07), "det": (-0.07, -0.05), "r_civ": "0.88–0.93", "r_det": "0.91–0.94"},
    "Fable 5.1": {"civ": (-0.25, -0.21), "det": (-0.40, -0.36), "r_civ": "0.63–0.69", "r_det": "0.34–0.43"},
    "Sonnet 5.5": {"civ": (-0.28, -0.24), "det": (-0.48, -0.46), "r_civ": "0.56–0.67", "r_det": "0.28"},
}
EFFORT = {  # model -> effort -> (civ range, det range)
    "Sonnet 5.5": {"low": ((-0.28, -0.24), (-0.48, -0.46)),
                   "medium": ((-0.27, -0.21), (-0.47, -0.47)),
                   "high": ((-0.12, -0.10), (-0.40, -0.39))},
    "Fable 5.1": {"low": ((-0.25, -0.21), (-0.40, -0.36)),
                  "high": ((-0.16, -0.03), (-0.35, -0.27))},
    "Opus 5.5": {"low": ((0.01, 0.07), (-0.07, -0.05)),
                 "high": ((0.03, 0.08), (-0.09, -0.05))},
    "Haiku 5.5": {"low": ((-0.05, 0.05), (-0.05, 0.03)),
                  "high": ((-0.03, 0.00), (-0.04, -0.01))},
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=13)


def fig_models(path):
    fig, ax = plt.subplots(figsize=(10.4, 6.0), dpi=200)
    names = list(LOW)
    for i, m in enumerate(names):
        for key, dy, col, lab in (("civ", 0.14, LIGHT, "Civilians, troops"), ("det", -0.14, PURPLE, "Detainees")):
            lo, hi = LOW[m][key]
            ax.plot([lo, hi], [i + dy] * 2, color=col, lw=9, solid_capstyle="round",
                    label=lab if i == 0 else None)
    ax.axvline(0, color=GREY, lw=1, ls="--")
    ax.text(0.005, len(names) - 0.45, "same as money", color=GREY, fontsize=12)
    ax.set_yticks(range(len(names)), names, fontsize=15)
    ax.set_xlim(-0.55, 0.15)
    ax.set_ylim(-0.6, len(names) - 0.3)
    ax.set_xlabel("Change in how often the model picks the gamble, lives vs. money", fontsize=13, color=INK)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.1f}" if v else "0")
    style(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def fig_effort(path):
    fig, ax = plt.subplots(figsize=(10.4, 6.0), dpi=200)
    xs = {"low": 0, "medium": 1, "high": 2}
    for m, mk, alpha in (("Sonnet 5.5", "o", 1.0), ("Fable 5.1", "s", 1.0),
                         ("Opus 5.5", "^", 0.35), ("Haiku 5.5", "D", 0.35)):
        for k, col, lab in ((0, LIGHT, "civilians, troops"), (1, PURPLE, "detainees")):
            pts = sorted(EFFORT[m].items(), key=lambda kv: xs[kv[0]])
            x = [xs[e] for e, _ in pts]
            mid = [sum(r[k]) / 2 for _, r in pts]
            err = [[mm - r[k][0] for mm, (_, r) in zip(mid, pts)], [r[k][1] - mm for mm, (_, r) in zip(mid, pts)]]
            ax.errorbar(x, mid, yerr=err, color=col if alpha == 1 else GREY, alpha=alpha, marker=mk,
                        ms=9, lw=2.5, capsize=4)
            if alpha == 1:
                dy = 9 if (m.startswith("Fable") and k == 0) else (-7 if k == 0 else 0)
                ax.annotate(f"{m.split()[0]}, {lab}", (x[-1], mid[-1]), xytext=(12, dy),
                            textcoords="offset points", va="center", fontsize=12, color=INK)
    ax.text(2.06, 0.05, "Opus, Haiku (all domains)", fontsize=12, color=GREY, va="center")
    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.set_xticks([0, 1, 2], ["low", "medium", "high"], fontsize=14)
    ax.set_xlim(-0.2, 3.1)
    ax.set_ylim(-0.55, 0.15)
    ax.set_xlabel("Thinking effort setting", fontsize=13, color=INK)
    ax.set_ylabel("Change in P(gamble), lives vs. money", fontsize=13, color=INK)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.1f}" if v else "0")
    style(ax)
    fig.tight_layout()
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def text(slide, x, y, w, h, s, size, font="Helvetica Neue", color=INK, bold=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    paras = s if isinstance(s, list) else [s]
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(8)
        r = p.add_run()
        r.text = ptxt
        r.font.name, r.font.size, r.font.bold = font, Pt(size), bold
        r.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
    return tb


def line(slide, x, y, w, color, pt):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + w), Inches(y))
    c.line.color.rgb = RGBColor.from_string(color.lstrip("#"))
    c.line.width = Pt(pt)


def frame(prs, title, n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    line(s, 0.5, 0.75, 16.0, PURPLE, 0.5)
    if LOGO.exists():
        s.shapes.add_picture(str(LOGO), Inches(15.99), Inches(0.18), Inches(0.5), Inches(0.5))
    text(s, 0.5, 0.98, 15.1, 0.9, title, 36, font="Georgia")
    for k, lab in enumerate(["LLM revealed preferences", "TNL breakout", "November 2026", f"Extra slide {n}"]):
        x = 0.5 + k * 4.17
        line(s, x, 9.21, 3.5, "#CFCAC4", 0.75)
        text(s, x, 9.29, 3.5, 0.3, lab, 12, color="#6E6862")
    return s


def sidebar(s, blocks, y0=2.1):
    y = y0
    for head, body, h in blocks:
        line(s, 11.6, y, 4.9, PURPLE, 1)
        text(s, 11.6, y + 0.15, 4.9, 0.5, head, 20, font="Georgia")
        text(s, 11.6, y + 0.7, 4.9, h, body, 14)
        y += 0.85 + h


def main():
    FIG.mkdir(exist_ok=True)
    f1, f2 = FIG / "tnl_models.png", FIG / "tnl_effort.png"
    fig_models(f1)
    fig_effort(f2)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(17), Inches(11)

    s = frame(prs, "Two models treat lives like money; two turn cautious", 1)
    s.shapes.add_picture(str(f1), Inches(0.5), Inches(2.0), Inches(10.6))
    sidebar(s, [
        ("Same bet, new stakes",
         "Each model sees the same sure-thing-or-gamble choice with dollars, then with civilians, "
         "troops or detainees. Bars span roles (deciding vs. advising) and gain/loss frames.", 1.5),
        ("Opus and Haiku: no gap",
         "With thinking on they pick by expected value everywhere. Money answers predict lives "
         "answers (r ≈ 0.9).", 1.1),
        ("Sonnet and Fable: caution",
         "They gamble 21–48 points less with lives, most with detainees. Money predicts lives "
         "only weakly (r 0.28–0.69).", 1.1),
    ])
    text(s, 0.5, 8.55, 10.6, 0.5,
         "Extended thinking at low effort, one-letter answer, 3 samples per question; ranges from "
         "95% gamble-clustered estimates. Source: docs/frontier-results-2026-10-20.md.", 11, color="#6E6862")
    s.notes_slide.notes_text_frame.text = (
        "Here is the frontier extension. Same bets as before, now on four Claude models with thinking on. "
        "Zero means the model treats lives exactly like money. Opus and Haiku sit on zero: they choose by "
        "expected value no matter what is at stake, so a money test would predict their advice well. "
        "Sonnet and Fable move left: they gamble twenty to forty-eight points less when lives are at stake, "
        "and most of all with detainees. For those two, the money test says little about the advice. "
        "So whether a cheap lottery screens advisory risk depends on the model.")

    s = frame(prs, "More thinking closes the gap, except for detainees", 2)
    s.shapes.add_picture(str(f2), Inches(0.5), Inches(2.0), Inches(10.6))
    sidebar(s, [
        ("A dial, not a trait",
         "Turning Sonnet's effort from medium to high halves its caution with civilians and troops "
         "(−0.24 to −0.11). Fable moves the same way.", 1.1),
        ("Detainees hold",
         "Caution with detained people stays at −0.27 to −0.47 at every effort and in a second "
         "wording.", 1.0),
        ("Not about escalation",
         "Dropping the escalation clause ('release' scenario) leaves it intact: Sonnet −0.39 to −0.44, "
         "Fable −0.26 to −0.32. The issue is giving up a sure release.", 1.4),
    ])
    text(s, 0.5, 8.55, 10.6, 0.5,
         "Points are midpoints of the range across lives domains, roles and frames; bars show the range. "
         "Sonnet ran at low, medium and high; the others at low and high.", 11, color="#6E6862")
    s.notes_slide.notes_text_frame.text = (
        "This is what I would build out for Wednesday. Effort is a setting the user controls. For Sonnet, "
        "going from medium to high effort halves the extra caution with civilians and troops, and Fable "
        "moves most of the way to money-like choices. But caution about detainees barely moves, holds "
        "in a second wording, and survives removing the escalation clause. The models cite escalation, "
        "but what drives the choice is trading a sure release for a gamble. So an evaluation of a model's "
        "risk posture has to report the effort setting, and some domains behave differently from all others.")

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
