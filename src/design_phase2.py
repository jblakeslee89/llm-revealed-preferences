"""Phase 2 design: gain/loss framing to identify loss aversion and the reference point.

The core manipulation presents the SAME terminal-wealth lottery under two frames:

    gain frame  (anchor R = 0):   a certain gain of $s   vs   p chance to gain $h, else $0
    loss frame  (anchor R = h):   a certain loss of $(h-s)  vs  p chance to lose $0, else lose $h

Both frames leave identical final wealth: $s for sure, versus {$h w.p. p, $0 w.p. 1-p}.
A frame-invariant expected-utility agent chooses identically across frames. A
reference-dependent, loss-averse agent does not: in the loss frame the sure option
reads as a certain loss and the gamble as a chance to avoid it, so gamble-taking rises
(the reflection effect). The gap between frames identifies loss aversion; varying the
stated anchor identifies how much the reference point tracks the frame.

A 'neutral' frame reproduces the Phase 1 wording (baseline, comparable).
A 'dominant' positive control offers a certain $s versus a certain larger amount: an
agent that reads magnitudes at all should pick the larger with probability ~1. This makes
a weak framing result interpretable (proposal §3.4), and it directly re-probes the Phase 1
finding that the model barely weighs payoff magnitude.

Contamination safety, counterbalancing, and the 5 paraphrase templates carry over from
Phase 1 unchanged.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from design import Gamble, generate_gambles, _non_round


# Five framing-aware paraphrases: {endow} states the starting position (may be empty),
# {first}/{second} are the counterbalanced option descriptions.
PHASE2_TEMPLATES: dict[int, str] = {
    0: ("{endow}You must pick one of two payment options.\n"
        "Option A: {first}\n"
        "Option B: {second}\n"
        "Pick the option you prefer."),
    1: ("{endow}Two arrangements are available and you take exactly one.\n"
        "Arrangement A: {first}\n"
        "Arrangement B: {second}\n"
        "Which arrangement do you take?"),
    2: ("{endow}Consider the following pair and select one.\n"
        "Alternative A: {first}\n"
        "Alternative B: {second}\n"
        "Select the alternative you would rather have."),
    3: ("{endow}You are entitled to one of two outcomes.\n"
        "Outcome A: {first}\n"
        "Outcome B: {second}\n"
        "Choose the outcome you want."),
    4: ("{endow}A counterparty offers a one-time decision between two terms.\n"
        "Term A: {first}\n"
        "Term B: {second}\n"
        "Decide which term you take."),
}


def _pct(p: float) -> int:
    return round(p * 100)


def framed_options(frame: str, s: int, h: int, p: float) -> tuple[str, str, str]:
    """Return (endowment_line, safe_description, target_description) for a frame.

    The 'target' is the non-safe slot: the gamble in neutral/gain/loss frames, or the
    larger certain amount in the dominant control.
    """
    if frame == "neutral":
        endow = ""
        safe = f"receive ${s} for certain"
        target = (f"receive ${h} with {_pct(p)}% probability, "
                  f"and $0 with {100 - _pct(p)}% probability")
    elif frame == "gain":
        endow = "You are starting from $0. "
        safe = f"a certain gain of ${s}"
        target = (f"a {_pct(p)}% chance to gain ${h} "
                  f"and a {100 - _pct(p)}% chance to gain nothing")
    elif frame == "mixed":
        # anchor at the sure amount: the gamble straddles the reference (a gain of
        # $(h-s) above it or a loss of $s below it), which is what identifies loss
        # aversion. Terminal wealth still matches the gain frame exactly.
        endow = f"You have been given ${s} up front. "
        safe = f"keep your ${s} with no change"
        target = (f"a {_pct(p)}% chance to rise to ${h} (a gain of ${h - s}) "
                  f"and a {100 - _pct(p)}% chance to drop to $0 (a loss of ${s})")
    elif frame == "dominant":  # positive control: h is a strictly larger certain amount
        endow = ""
        safe = f"receive ${s} for certain"
        target = f"receive ${h} for certain"
    else:
        raise ValueError(f"unknown frame: {frame}")
    return endow, safe, target


@dataclass
class Phase2Trial:
    trial_id: int
    gamble_id: int
    frame: str          # neutral | gain | loss | dominant
    anchor: int         # stated starting wealth R (0 for gain/neutral, h for loss)
    sure: int
    hi: int
    p: float
    ev_ratio: float
    template_id: int
    safe_first: bool
    rep: int
    response_mode: str

    def prompt(self) -> str:
        endow, safe, target = framed_options(self.frame, self.sure, self.hi, self.p)
        first, second = (safe, target) if self.safe_first else (target, safe)
        return PHASE2_TEMPLATES[self.template_id].format(endow=endow, first=first, second=second)

    @property
    def gamble_letter(self) -> str:
        """Letter mapping to the target (gamble, or larger certain amount) slot."""
        return "B" if self.safe_first else "A"


def build_phase2_trials(gambles: list[Gamble], reps: int = 2, seed: int = 42,
                        frames: tuple[str, ...] = ("neutral", "gain", "mixed"),
                        text_arm_share: float = 0.08,
                        dominant_controls: bool = True) -> list[Phase2Trial]:
    rng = random.Random(seed + 2)
    trials: list[Phase2Trial] = []
    tid = 0
    for g in gambles:
        for frame in frames:
            anchor = g.sure if frame == "mixed" else 0
            for template_id in PHASE2_TEMPLATES:
                for safe_first in (True, False):
                    for rep in range(reps):
                        mode = "text" if rng.random() < text_arm_share else "tool"
                        trials.append(Phase2Trial(
                            trial_id=tid, gamble_id=g.gamble_id, frame=frame, anchor=anchor,
                            sure=g.sure, hi=g.hi, p=g.p, ev_ratio=round(g.ev_ratio, 4),
                            template_id=template_id, safe_first=safe_first,
                            rep=rep, response_mode=mode))
                        tid += 1

    if dominant_controls:
        # one dominant-choice control per gamble: certain $s vs a strictly larger certain amount
        for g in gambles:
            k = _non_round(rng, 12, 60)
            for safe_first in (True, False):
                trials.append(Phase2Trial(
                    trial_id=tid, gamble_id=g.gamble_id, frame="dominant", anchor=0,
                    sure=g.sure, hi=g.sure + k, p=1.0, ev_ratio=round((g.sure + k) / g.sure, 4),
                    template_id=rng.randrange(len(PHASE2_TEMPLATES)),
                    safe_first=safe_first, rep=0, response_mode="tool"))
                tid += 1

    rng.shuffle(trials)
    return trials


def trial_row(t: Phase2Trial) -> dict:
    return asdict(t)


if __name__ == "__main__":  # quick visual check of one trial per frame
    g = generate_gambles(3, seed=42)[0]
    for frame in ("neutral", "gain", "mixed", "dominant"):
        anchor = g.sure if frame == "mixed" else 0
        h = g.sure + 30 if frame == "dominant" else g.hi
        t = Phase2Trial(0, 0, frame, anchor, g.sure, h, g.p if frame != "dominant" else 1.0,
                        1.0, 0, True, 0, "tool")
        print(f"\n===== {frame} =====\n{t.prompt()}")
