"""Clarification check: does naming the right comparison remove the loss-frame error?

Compares four reasoning-first conditions on the same gain and mixed cells (20 gambles per
instrument): uninduced, clarify only, EV rule, EV rule + clarify. Reports EV agreement on
clear cells by frame, mixed-frame agreement where the gamble is optimal (the direction of
the reference-point slip), and the frame gap (mean P(gamble), mixed minus gain).

Usage:
    python analysis/score_clarify.py qwen-inst_chat
    python analysis/score_clarify.py olmo-inst_fewshot
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_induced import load, score  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
KEY = ["instrument", "trial_id"]
CONDITIONS = [
    ("uninduced", "_reason"),
    ("clarify only", "_reason_clarify"),
    ("EV rule", "_reason_riskneutral"),
    ("EV rule + clarify", "_reason_riskneutral_clarify"),
]


def main(stem):
    cells = load(DATA / f"phase3_{stem}_reason_riskneutral_clarify.csv")[KEY]
    print(f"{stem}, reasoning first, gain + mixed cells ({len(cells)})")
    print(f"{'condition':20s} {'gain EV':>8s} {'mixed EV':>9s} {'(misses)':>9s} {'mixed, gamble opt':>18s} {'frame gap':>10s}")
    for label, suffix in CONDITIONS:
        path = DATA / f"phase3_{stem}{suffix}.csv"
        if not path.exists():
            print(f"{label:20s} (missing {path.name})")
            continue
        d = load(path).merge(cells, on=KEY)
        sc = score(d, "riskneutral", 0.05)
        d = d.assign(correct=sc["hard_correct"].values, clear=sc["clear"].values)
        c = d[d["clear"]]
        gain, mixed = c[c["frame"] == "gain"], c[c["frame"] == "mixed"]
        mg = mixed[mixed.p * mixed.hi > mixed.sure]
        gap = d[d["frame"] == "mixed"]["p_gamble"].mean() - d[d["frame"] == "gain"]["p_gamble"].mean()
        print(f"{label:20s} {gain['correct'].mean():8.3f} {mixed['correct'].mean():9.3f} "
              f"{int((~mixed['correct']).sum()):9d} {mg['correct'].mean():18.3f} {gap:+10.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen-inst_chat")
