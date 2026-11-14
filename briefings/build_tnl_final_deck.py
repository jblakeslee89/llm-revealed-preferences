"""Build briefings/tnl-final-deck-2026-11.pptx: the 9-minute TNL residency final presentation.

Follows Phil Armour's "Plan for the Endgame" slides (Nov 9 2026):
  * 9-minute talk, then 3 minutes for a discussant and 1-2 audience questions.
  * First slide: motivation, research question and general approach, one sentence each.
  * Last slide: one pivot (1 sentence), significance (2 sentences), next steps (1 sentence).
  * In between: past research and results, conveyed as a narrative.
  * Graded on clarity and design, narrative, and finishing on time.

Numbers come from docs/paper-risk-advice-draft.md (Tables 1-4, section 4 table).
Styling copies briefings/project-overview-5min.pptx (white, Georgia, RAND purple 761DDB).

Usage:
    python briefings/build_tnl_final_deck.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUT = HERE / "tnl-final-deck-2026-11.pptx"
FIG = HERE / "figures"
LOGO = FIG / "rand_logo.png"

PURPLE = "#761DDB"
MID = "#A56BEA"
LIGHT = "#C9A8F0"
PALE = "#E4D4F8"
INK = "#1C1C1E"
GREY = "#6E6862"
RULE = "#CFCAC4"
AXIS = "#9A948E"

N_SLIDES = 8
FOOTER = ["LLM revealed preferences", "TNL residency final presentation", "November 2026"]

if any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Inter"


# ----------------------------------------------------------------------------- data
# Open models, reason first, mean P(risky) when the gamble pays more (paper section 4).
OPEN = {
    "Qwen2.5-7B": (0.81, 0.17, 0.32, 0.18),
    "Llama-3.1-8B": (0.88, 0.63, 0.64, 0.53),
    "OLMo-2-7B": (0.51, 0.43, 0.40, 0.14),
}

# Claude, low effort, main grid: shift from money/self (est, lo, hi) for civilians, troops, detainees (Table 2).
CLAUDE = [
    ("Sonnet, snap", (-0.24, -0.28, -0.19), (-0.23, -0.27, -0.19), (-0.63, -0.69, -0.58)),
    ("Haiku, snap", (-0.09, -0.15, -0.04), (0.12, 0.07, 0.17), (-0.37, -0.42, -0.33)),
    ("Sonnet, written", (-0.23, -0.36, -0.11), (-0.28, -0.43, -0.14), (-0.44, -0.63, -0.25)),
    ("Haiku, written", (-0.28, -0.40, -0.17), (-0.23, -0.36, -0.12), (-0.45, -0.61, -0.28)),
    ("Sonnet, thinking", (-0.26, -0.41, -0.12), (-0.25, -0.40, -0.13), (-0.47, -0.65, -0.29)),
    ("Fable, thinking", (-0.23, -0.32, -0.14), (-0.23, -0.34, -0.12), (-0.38, -0.53, -0.24)),
    ("Haiku, thinking", (0.01, -0.03, 0.05), (0.04, 0.02, 0.08), (-0.01, -0.05, 0.02)),
    ("Opus, thinking", (0.05, -0.00, 0.12), (0.03, -0.00, 0.07), (-0.06, -0.14, -0.01)),
]

# Effort ladder (Table 3): model -> effort -> (civilians, troops, detainees) point estimates.
EFFORT = {
    "Sonnet": {"low": (-0.26, -0.25, -0.47), "medium": (-0.24, -0.23, -0.47), "high": (-0.12, -0.10, -0.39)},
    "Fable": {"low": (-0.23, -0.23, -0.38), "high": (-0.12, -0.04, -0.31)},
    "Haiku": {"low": (0.01, 0.04, -0.01), "high": (-0.01, -0.00, -0.03)},
    "Opus": {"low": (0.05, 0.03, -0.06), "high": (0.05, 0.04, -0.07)},
}

# Detainees, Stage A2 (Table 4): (second wording est, lo, hi), (release without escalation est, lo, hi).
DETAIN = [
    ("Sonnet, snap", (-0.41, -0.46, -0.37), (-0.54, -0.60, -0.48)),
    ("Sonnet, written", (-0.45, -0.65, -0.25), (-0.40, -0.58, -0.23)),
    ("Sonnet, thinking", (-0.46, -0.62, -0.31), (-0.43, -0.58, -0.29)),
    ("Fable, thinking", (-0.34, -0.48, -0.19), (-0.29, -0.41, -0.18)),
    ("Haiku, snap", (-0.42, -0.44, -0.39), (-0.31, -0.35, -0.28)),
    ("Haiku, written", (-0.41, -0.56, -0.25), (-0.40, -0.55, -0.25)),
    ("Haiku, thinking", (-0.10, -0.17, -0.03), (-0.04, -0.07, -0.01)),
    ("Opus, thinking", (-0.10, -0.20, -0.04), (-0.04, -0.10, 0.00)),
]


# ----------------------------------------------------------------------------- charts
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK, labelsize=14)


def signed(v, _=None):
    return "0" if abs(v) < 1e-9 else f"{v:+.1f}".replace("-", "−")


def fig_open(path):
    fig, ax = plt.subplots(figsize=(10.4, 5.9), dpi=200)
    cols = [GREY, LIGHT, MID, PURPLE]
    labs = ["Money", "Civilians", "Soldiers", "Detained aid workers"]
    w = 0.19
    for i, (m, vals) in enumerate(OPEN.items()):
        for k, v in enumerate(vals):
            x = i + (k - 1.5) * (w + 0.02)
            ax.bar(x, v, w, color=cols[k], label=labs[k] if i == 0 else None)
            ax.text(x, v + 0.015, f"{v:.2f}".lstrip("0"), ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(len(OPEN)), list(OPEN), fontsize=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of answers taking the gamble", fontsize=14, color=INK)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_claude(path):
    fig, ax = plt.subplots(figsize=(10.4, 6.1), dpi=200)
    cols = [LIGHT, MID, PURPLE]
    labs = ["Civilians", "Soldiers", "Detained aid workers"]
    n = len(CLAUDE)
    ys = [n - i + (0 if i < 4 else -0.6) for i in range(n)]  # gap between thinking off and on
    for y, (name, *doms) in zip(ys, CLAUDE):
        for k, (est, lo, hi) in enumerate(doms):
            yy = y + (1 - k) * 0.22
            ax.plot([lo, hi], [yy, yy], color=cols[k], lw=2.2)
            ax.plot(est, yy, "o", color=cols[k], ms=8, label=labs[k] if name == CLAUDE[0][0] else None)
    ax.axvline(0, color=AXIS, lw=1, ls="--")
    ax.set_yticks(ys, [c[0] for c in CLAUDE], fontsize=14)
    ax.set_xlim(-0.75, 0.25)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axhline((ys[3] + ys[4]) / 2, color=RULE, lw=1)
    ax.text(0.24, ys[0] + 0.35, "thinking off", ha="right", fontsize=12, color=GREY)
    ax.text(0.24, ys[4] + 0.35, "extended thinking on", ha="right", fontsize=12, color=GREY)
    ax.text(0.01, min(ys) - 0.45, "same as money", fontsize=12, color=GREY)
    ax.set_xlabel("Change in how often the model takes the gamble, lives vs. money", fontsize=13, color=INK)
    ax.xaxis.set_major_formatter(signed)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.1), ncol=3, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_effort(path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.9), dpi=200, sharey=True)
    xs = {"low": 0, "medium": 1, "high": 2}
    for ax, title, pick in ((axes[0], "Civilians and soldiers", lambda v: (v[0] + v[1]) / 2),
                            (axes[1], "Detained aid workers", lambda v: v[2])):
        for m, mk, col, lw in (("Opus", "^", AXIS, 1.5), ("Haiku", "D", AXIS, 1.5),
                               ("Fable", "s", MID, 3), ("Sonnet", "o", PURPLE, 3)):
            pts = sorted(EFFORT[m].items(), key=lambda kv: xs[kv[0]])
            x = [xs[e] for e, _ in pts]
            y = [pick(v) for _, v in pts]
            ax.plot(x, y, marker=mk, color=col, lw=lw, ms=9)
            dy = {"Sonnet": -10, "Fable": 10, "Opus": 9, "Haiku": -9}[m]
            if title.startswith("Det"):
                dy = {"Sonnet": -9, "Fable": 9, "Opus": -9, "Haiku": 9}[m]
            ax.annotate(m, (x[-1], y[-1]), xytext=(10, dy), textcoords="offset points",
                        va="center", fontsize=13, color=col if col != AXIS else GREY)
        ax.axhline(0, color=AXIS, lw=1, ls="--")
        ax.set_xticks([0, 1, 2], ["low", "medium", "high"], fontsize=14)
        ax.set_xlim(-0.25, 2.75)
        ax.set_title(title, fontsize=15, color=INK, loc="left", pad=12)
        ax.set_xlabel("Thinking effort", fontsize=13, color=INK)
        style(ax)
    axes[0].set_ylim(-0.55, 0.12)
    axes[0].set_ylabel("Change in P(gamble), lives vs. money", fontsize=13, color=INK)
    axes[0].yaxis.set_major_formatter(signed)
    fig.tight_layout(w_pad=3)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_detain(path):
    fig, ax = plt.subplots(figsize=(10.4, 6.1), dpi=200)
    n = len(DETAIN)
    for i, (name, a, b) in enumerate(DETAIN):
        y = n - 1 - i
        for (est, lo, hi), dy, col, lab in ((a, 0.17, PURPLE, "Detainees, with escalation clause"),
                                            (b, -0.17, LIGHT, "Release, escalation clause removed")):
            ax.barh(y + dy, est, 0.32, color=col, label=lab if i == 0 else None)
            ax.plot([lo, hi], [y + dy] * 2, color=INK, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(n - 1, -1, -1), [d[0] for d in DETAIN], fontsize=14)
    ax.set_xlim(-0.7, 0.05)
    ax.set_xlabel("Change in how often the model takes the gamble, vs. money", fontsize=13, color=INK)
    ax.xaxis.set_major_formatter(signed)
    style(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- slide helpers
def rgb(c):
    return RGBColor.from_string(c.lstrip("#"))


def text(slide, x, y, w, h, s, size, font="Helvetica Neue", color=INK, bold=False, after=8, spacing=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    paras = s if isinstance(s, list) else [s]
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(after)
        if spacing:
            p.line_spacing = spacing
        r = p.add_run()
        r.text = ptxt
        r.font.name, r.font.size, r.font.bold = font, Pt(size), bold
        r.font.color.rgb = rgb(color)
    return tb


def line(slide, x, y, w, color, pt):
    # A thin filled rectangle renders the same in PowerPoint and LibreOffice (connectors pick up a shadow).
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Pt(pt))
    r.fill.solid()
    r.fill.fore_color.rgb = rgb(color)
    r.line.fill.background()
    r.shadow.inherit = False


def frame(prs, n, title=None, sub=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    line(s, 0.5, 0.75, 15.35, PALE, 1.5)
    if LOGO.exists():
        s.shapes.add_picture(str(LOGO), Inches(16.0), Inches(0.25), Inches(0.5), Inches(0.5))
    if title:
        text(s, 0.5, 0.98, 15.5, 0.9, title, 36, font="Georgia")
    if sub:
        text(s, 0.5, 1.82, 15.5, 0.5, sub, 17, color=GREY)
    for k, lab in enumerate(FOOTER + [f"Slide {n} of {N_SLIDES}"]):
        x = 0.5 + k * 4.17
        line(s, x, 9.21, 3.5, RULE, 0.75)
        text(s, x, 9.29, 3.5, 0.3, lab, 12, color=GREY)
    for k in range(N_SLIDES):  # progress dots, bottom right
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(15.2 - (N_SLIDES - 1 - k) * 0.24), Inches(10.3),
                               Inches(0.14), Inches(0.14))
        d.fill.solid()
        d.fill.fore_color.rgb = rgb(PURPLE if k == n - 1 else PALE)
        d.line.fill.background()
    return s


def sidebar(s, blocks, y0=2.55, x=11.55, w=4.95):
    y = y0
    for head, body, h in blocks:
        line(s, x, y, w, PURPLE, 1)
        text(s, x, y + 0.15, w, 0.5, head, 20, font="Georgia")
        text(s, x, y + 0.7, w, h, body, 14.5, spacing=1.1)
        y += 0.9 + h


def source(s, msg, y=8.55):
    text(s, 0.5, y, 10.7, 0.5, msg, 11, color=GREY)


def labeled_sentences(s, items, y0, label_w=3.0, body_w=12.0, size=24, gap=0.35, heights=None):
    y = y0
    for i, (lab, body) in enumerate(items):
        h = heights[i] if heights else 0.9
        line(s, 0.5, y, 15.5, RULE, 0.75)
        text(s, 0.5, y + 0.2, label_w, 0.5, lab.upper(), 13, color=PURPLE, bold=True)
        text(s, 0.5 + label_w + 0.3, y + 0.12, body_w, h, body, size, font="Georgia", spacing=1.05)
        y += h + gap


# ----------------------------------------------------------------------------- deck
def main():
    FIG.mkdir(exist_ok=True)
    f_open, f_claude = FIG / "tnl_final_open.png", FIG / "tnl_final_claude.png"
    f_effort, f_det = FIG / "tnl_final_effort.png", FIG / "tnl_final_detainees.png"
    fig_open(f_open)
    fig_claude(f_claude)
    fig_effort(f_effort)
    fig_detain(f_det)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(17), Inches(11)

    # 1. First slide: motivation, research question, approach (one sentence each).
    s = frame(prs, 1)
    text(s, 0.5, 1.05, 14.5, 1.5, ["When AI Risk Benchmarks Fail to Transfer", "to Life or Death Stakes"], 40,
         font="Georgia", after=0, spacing=0.95)
    text(s, 0.5, 2.6, 15.5, 0.5,
         "Does an AI's attitude to risk on money bets predict its advice when lives are at stake?",
         18, color=PURPLE)
    labeled_sentences(s, [
        ("Motivation",
         "Defense and crisis teams are starting to ask AI models for advice, yet the standard test "
         "of a model's attitude to risk uses bets over money."),
        ("Research question",
         "Does the way a model gambles with money predict the advice it gives when lives are at stake?"),
        ("Approach",
         "I restated 40 money bets, number for number, as choices about civilians, soldiers and "
         "detained aid workers, and put them to three open models and four Claude models."),
    ], y0=3.4, heights=[1.05, 1.05, 1.45], size=23)
    text(s, 0.5, 8.25, 10, 0.4, "John Blakeslee", 14)
    text(s, 0.5, 8.6, 10, 0.4, "RAND School of Public Policy, TNL residency", 13, color=GREY)
    s.notes_slide.notes_text_frame.text = (
        "Defense and crisis teams are starting to use AI models as advisers. When people want to know how "
        "a model handles risk, the standard tool comes from experimental economics. You offer the model a "
        "sure amount of money or a gamble, and read off its risk attitude. That test is cheap and "
        "repeatable. My question is whether it tells you anything about the advice a model gives when the "
        "stakes are lives. To find out, I took 40 money bets and restated each one with the same numbers "
        "as a choice about civilians, soldiers, or detained aid workers. Then I ran them on three open "
        "models and four Claude models.")

    # 2. Design.
    s = frame(prs, 2, "The same bet, four ways",
              "Only the units change. The numbers, the odds and the structure stay fixed.")
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(2.6), Inches(10.5), Inches(4.3))
    box.fill.background()
    box.line.color.rgb = rgb(RULE)
    box.line.width = Pt(1)
    rows = [
        ("Money", "a sure $186", "or a 26% chance at $984"),
        ("Civilians", "evacuate 186 for certain", "or a 26% chance to evacuate 984"),
        ("Soldiers", "bring 186 out for certain", "or a 26% chance to bring 984 out"),
        ("Detained aid workers", "accept a deal freeing 186", "or press for full release, 26% chance 984 freed"),
    ]
    for i, (dom, sure, gamble) in enumerate(rows):
        y = 2.85 + i * 1.02
        text(s, 0.8, y, 2.9, 0.5, dom.upper(), 12, color=PURPLE, bold=True)
        text(s, 0.8, y + 0.35, 4.1, 0.5, sure, 17, font="Courier New")
        text(s, 4.9, y + 0.35, 6.0, 0.5, gamble, 17, font="Courier New")
    text(s, 0.5, 7.15, 10.5, 1.3, [
        "Each bet is asked as the model's own decision and as advice to an official, in gain and loss "
        "wording, with each option listed first. Control questions offer two sure amounts to check that the "
        "model prefers saving more people.",
    ], 14.5, spacing=1.1)
    stats = [("40", "bets, each in 4 domains"), ("7", "models, 3 open and 4 Claude"),
             ("3", "ways of answering, from a snap letter to extended thinking"), ("1,440", "questions per model in the main grid")]
    for i, (big, lab) in enumerate(stats):
        x = 11.55 + (i % 2) * 2.6
        y = 2.6 + (i // 2) * 2.35
        line(s, x, y, 2.3, LIGHT, 1.5)
        text(s, x, y + 0.2, 2.3, 0.9, big, 44, font="Georgia", color=PURPLE)
        text(s, x, y + 1.15, 2.3, 1.0, lab, 13, color=GREY)
    text(s, 11.55, 7.15, 4.95, 1.4,
         "Claude models answer three ways. A one-letter snap answer, written reasoning before the answer, "
         "and hidden extended thinking at low, medium or high effort.", 14.5, spacing=1.1)
    s.notes_slide.notes_text_frame.text = (
        "Here is the design. Take one bet, a sure 186 or a 26 percent chance at 984. In money, the gamble "
        "is worth more. I ask the identical choice about civilians in a besieged city, soldiers in an "
        "isolated unit, and aid workers held in a standoff. Each question comes in two roles, the model "
        "deciding or advising an official, in gain and loss wording, and with each option listed first, so "
        "order effects show up. Control questions offer two sure amounts so I can check the model is paying "
        "attention. The Claude models also answer in three ways, which turns out to matter a great deal.")

    # 3. Open models.
    s = frame(prs, 3, "Open models turn cautious when lives are at stake",
              "Reasoning first, on questions where the gamble pays more")
    s.shapes.add_picture(str(f_open), Inches(0.5), Inches(2.5), Inches(10.6))
    sidebar(s, [
        ("Qwen does the math for money",
         "It computes expected value in every money answer and in 6% of detainee answers. In every "
         "lives answer it argues for certainty.", 1.3),
        ("Money predicts little",
         "With reasoning, Qwen's money answers explain at most a quarter of the variation in its lives "
         "answers, and OLMo's at most 15%.", 1.3),
        ("Snap advice follows the list",
         "Llama's one-letter answers on lives pick whichever option is listed first.", 0.8),
    ])
    source(s, "Mean P(risky) when the gamble's expected value is higher, self and advise roles pooled. "
              "Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, OLMo-2-1124-7B-Instruct.")
    s.notes_slide.notes_text_frame.text = (
        "Start with the open models, reasoning before they answer. In grey is money. When the gamble pays "
        "more, Qwen takes it 81 percent of the time. Put lives on the line and that drops to between 17 and "
        "32 percent. Qwen's reasoning shows why. It computes expected value for every money question and "
        "almost never for detainees. Instead it argues for certainty. And the money answers barely predict "
        "the lives answers, at most a quarter of the variation. Llama is a warning about quick answers. Its "
        "one-letter advice on lives simply follows whichever option comes first.")

    # 4. Claude by mode.
    s = frame(prs, 4, "In Claude, the gap depends on how the model is run",
              "Zero means the model treats lives exactly as it treats money")
    s.shapes.add_picture(str(f_claude), Inches(0.5), Inches(2.4), Inches(10.6))
    sidebar(s, [
        ("Thinking off, a large gap",
         "In snap answers and written reasoning, Sonnet is 0.23 to 0.63 less likely to gamble with lives, "
         "and Haiku's written reasoning shows the same gap. Money answers predict lives answers weakly.", 1.55),
        ("Thinking on, the models split",
         "Haiku and Opus treat every domain like money, and money predicts lives with r near 0.9. Sonnet "
         "and Fable keep the gap.", 1.3),
        ("Length is not the cause",
         "Haiku's written reasoning runs about 244 tokens and keeps the gap. Its extended thinking runs "
         "about 87 and closes it.", 1.3),
    ], y0=2.45)
    source(s, "Shift in P(risky) from money, with 95% intervals from resampling gambles. Low effort, main "
              "grid, roles pooled. Opus and Fable cannot turn thinking off.")
    s.notes_slide.notes_text_frame.text = (
        "Now the four Claude models. Each dot is how much less often the model gambles with lives than with "
        "money. Zero means no difference. With thinking off, in snap answers or written reasoning, Claude "
        "looks like the open models. It is far more cautious with lives, most of all with detainees. Turn "
        "on extended thinking and the models split. Haiku and Opus sit on zero. They choose by expected "
        "value everywhere, and a money test would predict their advice well. Sonnet and Fable keep the gap. "
        "This is not about how long they think. Haiku's written reasoning is three times longer than its "
        "thinking and keeps the gap.")

    # 5. Effort.
    s = frame(prs, 5, "Higher effort halves the caution for civilians and soldiers",
              "Sonnet and Fable at each extended-thinking effort setting")
    s.shapes.add_picture(str(f_effort), Inches(0.5), Inches(2.5), Inches(10.6))
    sidebar(s, [
        ("A step at high effort",
         "Sonnet barely changes from low to medium. At high effort its caution with civilians and "
         "soldiers falls from about 0.25 to 0.11, and money predicts lives with r of 0.85.", 1.55),
        ("Little extra thinking",
         "High effort adds only 18 tokens per answer for Sonnet and 45 for Fable.", 0.8),
        ("Detainees barely move",
         "Sonnet stays 0.39 less likely to gamble for detainees at high effort, and Fable 0.31.", 0.8),
    ], y0=2.45)
    source(s, "Shift in P(risky) from each model's own money answers at the same effort, main grid. Sonnet "
              "ran at low, medium and high, the others at low and high.")
    s.notes_slide.notes_text_frame.text = (
        "Effort is a setting the deployer picks and the user usually never sees. For Sonnet and Fable it "
        "acts like a dose. Low to medium changes nothing. At high effort, Sonnet's extra caution with "
        "civilians and soldiers halves, and its money answers suddenly predict its lives answers well. "
        "Fable moves the same way, and its caution with soldiers almost disappears. The extra thinking is "
        "tiny, 18 tokens for Sonnet. But on the right, look at detainees. Effort barely touches them.")

    # 6. Detainees.
    s = frame(prs, 6, "Caution about detainees survives every test",
              "A second wording, and a version with the escalation clause removed")
    s.shapes.add_picture(str(f_det), Inches(0.5), Inches(2.4), Inches(10.6))
    sidebar(s, [
        ("The stated reason is wrong",
         "Sonnet's written reasoning cites escalation in every detainee answer. Deleting the escalation "
         "clause leaves its caution almost unchanged.", 1.3),
        ("Three possible causes",
         "Trading a sure release for a gamble may feel taboo. Pressing for more is an act, while accepting "
         "is the default. Or the models learned hostage-negotiation norms.", 1.85),
    ], y0=2.45)
    source(s, "Detainee shifts from money, second-wording run (Stage A2), low effort, 95% intervals. "
              "Removing escalation changes the shift by 0.13 or less in every row.")
    s.notes_slide.notes_text_frame.text = (
        "Detainees are the exception. The caution holds in a second, independently written wording. The "
        "original question says that if talks collapse, the crisis escalates, and Sonnet cites escalation "
        "in every single detainee answer. So I deleted that clause. Purple is with escalation, light purple "
        "without. Almost nothing changes. The model's stated reason does not drive its choice. What does? "
        "Possibly a protected value, possibly a bias toward accepting the default, possibly learned "
        "negotiation norms. A variant that flips act and default would separate the first two.")

    # 7. Implications.
    s = frame(prs, 7, "What this means for evaluating AI advisers",
              "A money test screens advice on lives only for some models at some settings, and only the "
              "advisory questions reveal which.")
    cards = [
        ("Test the real task", "Measure risk posture on advisory questions in the domain where the model "
                               "will be used."),
        ("Report the setting", "Report snap, written and extended-thinking answers separately, with the "
                               "effort level, the way a trial reports a dose."),
        ("Control the format", "Include controls and both option orders, since quick advice on lives can "
                               "follow the order of the list."),
        ("Test the stated reason", "Remove the reason a model gives and check whether its choice moves."),
    ]
    for i, (head, body) in enumerate(cards):
        x = 0.5 + i * 4.0
        line(s, x, 3.2, 3.6, PURPLE, 1.5)
        text(s, x, 3.35, 3.6, 0.4, f"0{i + 1}", 34, font="Georgia", color=PURPLE)
        text(s, x, 4.3, 3.6, 0.6, head, 21, font="Georgia")
        text(s, x, 5.0, 3.6, 2.2, body, 15, spacing=1.1)
    text(s, 0.5, 6.9, 15.5, 1.0,
         "Neither posture is clearly correct. Expected value is the impartial rule, and many decision makers "
         "would take the certain partial outcome. The posture should be known, measured and disclosed.",
         15, color=GREY, spacing=1.1)
    s.notes_slide.notes_text_frame.text = (
        "So what should evaluators do? First, test on the advisory task itself. The lottery works for Opus "
        "and for Haiku with thinking, but you only learn that by running the advisory questions, and then "
        "the lottery adds little. Second, report the thinking setting like a dose, because it can flip the "
        "answer. Third, randomize option order and include controls. Fourth, test stated reasons by removing "
        "them. I am not claiming expected value is the right rule for lives. My point is that a deployer "
        "should know which posture they are getting.")

    # 8. Last slide: pivot, significance (2 sentences), next steps.
    s = frame(prs, 8)
    text(s, 0.5, 1.15, 15.5, 1.0, "Pivot, significance and next steps", 40, font="Georgia")
    labeled_sentences(s, [
        ("Pivot",
         "I started by measuring where AI models get their risk attitudes over money, and pivoted to asking "
         "whether those measurements carry over to advice about lives."),
        ("Significance",
         "A cheap money test predicts a model's advice on lives only for some models at some settings, and "
         "nothing in the test says which. The same model can give a cautious or an expected-value "
         "recommendation depending on a thinking setting the user never sees."),
        ("Next steps",
         "Test a model from another developer with an adjustable thinking setting, add the act-versus-accept "
         "detainee variant, and post the paper to arXiv."),
    ], y0=2.45, heights=[1.5, 1.95, 1.1], size=23)
    s.notes_slide.notes_text_frame.text = (
        "To wrap up. I started this project measuring where models pick up their risk attitudes over money. "
        "The pivot was to ask whether those measurements carry over to the decisions people actually use "
        "these models for. They often do not, and the same model can go either way depending on a setting "
        "the user never sees. Next I want a model from another developer with a thinking dial, the "
        "act-versus-accept detainee test, and then the paper goes to arXiv. Thank you.")

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
