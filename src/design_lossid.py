"""Phase 3 follow-up: a mixed-gamble block that separates loss aversion from a constant lean.

Why (docs/phase3-robustness-2026-10-03.md): in the Phase 3 grid, lambda enters the choice index
only through the mixed frame, where the loss side is the sure amount s and the gain side is h - s.
Across the 40 gambles the two sides are correlated (log-log r = 0.50) and the fitted value function
is nearly flat in payoffs, so a constant lean toward the safe option fits the gain/mixed contrast
about as well as lambda does. Adding that constant sends lambda to zero for most subjects.

What identifies lambda against a constant is the response to the size of the loss, holding the
gain fixed (the mixed-gamble acceptance design of Tom et al. 2007 and Gachter, Johnson and
Herrmann 2022). A constant shifts acceptance equally everywhere; loss aversion lowers it more as the
loss grows. So this block crosses gain size G and loss size L independently on a grid.

Each item reuses the Phase 2 mixed-frame wording exactly, with s = L and h = L + G:
    "You have been given $L up front."
    keep your $L with no change   vs   p% chance to rise to $(L+G) (a gain of $G)
                                       and (1-p)% chance to drop to $0 (a loss of $L)
so the existing estimator (reference = anchor = s, low outcome 0) needs no change. Each item also
gets its terminal-wealth twin in the gain frame (certain $L vs p chance of $(L+G)), which keeps the
frame-invariance contrast and gives a within-item baseline for the constant.

Probabilities sit near one half (jittered to 0.44-0.58, never 0.50) so probability weighting
scales both sides alike and cannot stand in for lambda. Payoffs are non-round (not multiples of
5). Five templates x two orders per cell, plus dominance controls, as in the main grid.

Usage:
    python src/design_lossid.py --export data/phase3_lossid_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design import Gamble, _non_round  # noqa: E402
from design_phase2 import PHASE2_TEMPLATES, Phase2Trial  # noqa: E402

LOSS_LEVELS = (14, 31, 58, 97, 163, 271)          # roughly log-spaced, none a multiple of 5
GAIN_RATIOS = (0.45, 0.8, 1.3, 2.1, 3.4, 5.5)     # G / L, log-spaced around the human break-even ~2
GAMBLE_ID_OFFSET = 1000                            # keeps ids distinct from the main grid's 0-39


def generate_lossid_gambles(seed: int = 7) -> list[Gamble]:
    rng = random.Random(seed)
    out = []
    gid = GAMBLE_ID_OFFSET
    for L in LOSS_LEVELS:
        for r in GAIN_RATIOS:
            G = max(int(round(L * r * rng.uniform(0.94, 1.06))), 2)
            if G % 5 == 0:
                G += 1
            p = round(rng.uniform(0.44, 0.58), 2)
            if p == 0.50:
                p = 0.51
            out.append(Gamble(gamble_id=gid, sure=L, hi=L + G, p=p))
            gid += 1
    return out


def build_lossid_trials(gambles: list[Gamble], seed: int = 7) -> list[Phase2Trial]:
    rng = random.Random(seed + 2)
    trials, tid = [], 50000
    for g in gambles:
        for frame in ("gain", "mixed"):
            for template_id in PHASE2_TEMPLATES:
                for safe_first in (True, False):
                    trials.append(Phase2Trial(
                        trial_id=tid, gamble_id=g.gamble_id, frame=frame,
                        anchor=g.sure if frame == "mixed" else 0,
                        sure=g.sure, hi=g.hi, p=g.p, ev_ratio=round(g.ev_ratio, 4),
                        template_id=template_id, safe_first=safe_first, rep=0, response_mode="tool"))
                    tid += 1
        k = _non_round(rng, 12, 60)  # dominance control per item, both orders
        for safe_first in (True, False):
            trials.append(Phase2Trial(
                trial_id=tid, gamble_id=g.gamble_id, frame="dominant", anchor=0,
                sure=g.sure, hi=g.sure + k, p=1.0, ev_ratio=round((g.sure + k) / g.sure, 4),
                template_id=rng.randrange(len(PHASE2_TEMPLATES)), safe_first=safe_first,
                rep=0, response_mode="tool"))
            tid += 1
    rng.shuffle(trials)
    return trials


def rows(trials):
    for t in trials:
        yield {"instrument": "p3", "trial_id": t.trial_id, "gamble_id": t.gamble_id,
               "frame": t.frame, "anchor": t.anchor, "sure": t.sure, "hi": t.hi, "p": t.p,
               "ev_ratio": t.ev_ratio, "template_id": t.template_id, "safe_first": t.safe_first,
               "gamble_letter": t.gamble_letter, "prompt": t.prompt()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", default="data/phase3_lossid_grid.csv")
    args = ap.parse_args()
    out = list(rows(build_lossid_trials(generate_lossid_gambles())))
    Path(args.export).parent.mkdir(parents=True, exist_ok=True)
    with open(args.export, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Exported {len(out)} cells -> {args.export}")


if __name__ == "__main__":
    main()
