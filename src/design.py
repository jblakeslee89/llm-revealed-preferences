"""Experimental design: procedural gamble generation, prompt templates, trial construction.

Design commitments implemented here (see proposal §3.4):
- Contamination safety: every gamble is procedurally generated with non-round payoffs
  and probabilities; no canonical Holt-Laury rows, no textbook vignette wording.
- Counterbalancing: every (gamble, template) cell is presented in BOTH orders
  (safe first / risky first), so position bias can be estimated as a nuisance
  parameter rather than assumed away.
- Prompt template as an experimental factor: 5 semantically equivalent templates,
  logged per trial, so template enters the choice model as a random effect.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Gambles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gamble:
    """One binary choice: a sure amount vs. a two-outcome gamble (hi with prob p, else 0)."""
    gamble_id: int
    sure: int      # sure payoff, dollars
    hi: int        # gamble's high payoff, dollars
    p: float       # probability of the high payoff

    @property
    def ev_gamble(self) -> float:
        return self.p * self.hi

    @property
    def ev_ratio(self) -> float:
        """EV(gamble) / sure. >1 means the gamble is actuarially favorable."""
        return self.ev_gamble / self.sure


def _non_round(rng: random.Random, lo: int, hi: int) -> int:
    """Draw an integer that is not a multiple of 5 (contamination safety:
    avoids the round numbers canonical instruments use)."""
    while True:
        x = rng.randint(lo, hi)
        if x % 5 != 0:
            return x


def generate_gambles(n: int = 40, seed: int = 42) -> list[Gamble]:
    """Procedurally generate a grid of gambles spanning EV ratios ~0.75-1.7.

    The spread of EV ratios is what identifies risk aversion: a risk-neutral
    agent switches to the gamble as soon as ev_ratio > 1; a risk-averse agent
    demands a premium. Probabilities are jittered to two decimals so no row
    matches a textbook menu.
    """
    rng = random.Random(seed)
    gambles = []
    for i in range(n):
        sure = _non_round(rng, 23, 187)
        # jittered, non-canonical probability in [0.19, 0.83]
        p = round(rng.uniform(0.19, 0.83), 2)
        if p in (0.25, 0.50, 0.75):  # avoid the textbook values exactly
            p += 0.01
        # target EV ratio spread; log-uniform so both sides of 1.0 are covered
        ratio = 0.75 * (1.7 / 0.75) ** rng.random()
        hi = max(int(round(sure * ratio / p)), sure + 1)
        if hi % 5 == 0:
            hi += 1
        gambles.append(Gamble(gamble_id=i, sure=sure, hi=hi, p=p))
    return gambles


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
# Five semantically equivalent phrasings. Novel wording; none reproduces the
# Holt-Laury / textbook lottery register. {first} and {second} are option
# descriptions; the harness controls which is safe vs. risky.

TEMPLATES: dict[int, str] = {
    0: ("You must pick one of two payment options.\n"
        "Option A: {first}\n"
        "Option B: {second}\n"
        "Pick the option you prefer."),
    1: ("Two offers are on the table and you can accept exactly one.\n"
        "Offer A: {first}\n"
        "Offer B: {second}\n"
        "Which offer do you accept?"),
    2: ("Consider the following pair of alternatives and select one.\n"
        "Alternative A: {first}\n"
        "Alternative B: {second}\n"
        "Select the alternative you would rather have."),
    3: ("You are entitled to one of two payouts.\n"
        "Payout A: {first}\n"
        "Payout B: {second}\n"
        "Choose the payout you want."),
    4: ("A counterparty gives you a one-time decision between two deals.\n"
        "Deal A: {first}\n"
        "Deal B: {second}\n"
        "Decide which deal you take."),
}


def describe_safe(g: Gamble) -> str:
    return f"receive ${g.sure} with certainty."


def describe_risky(g: Gamble) -> str:
    pct = round(g.p * 100)
    return (f"receive ${g.hi} with {pct}% probability, "
            f"and $0 with {100 - pct}% probability.")


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    trial_id: int
    gamble_id: int
    sure: int
    hi: int
    p: float
    ev_ratio: float
    template_id: int
    safe_first: bool     # counterbalanced: True -> A is the sure amount
    rep: int
    response_mode: str   # 'tool' (forced structured output) or 'text' (validation arm)

    def prompt(self) -> str:
        g = Gamble(self.gamble_id, self.sure, self.hi, self.p)
        safe, risky = describe_safe(g), describe_risky(g)
        first, second = (safe, risky) if self.safe_first else (risky, safe)
        return TEMPLATES[self.template_id].format(first=first, second=second)

    @property
    def gamble_letter(self) -> str:
        """Which letter maps to the risky option in this trial."""
        return "B" if self.safe_first else "A"


def build_trials(gambles: list[Gamble], reps: int = 3, seed: int = 42,
                 text_arm_share: float = 0.10) -> list[Trial]:
    """Full factorial: gambles x templates x both orders x reps.

    ~10% of trials use plain-text elicitation instead of forced tool output,
    so the two response modes can be compared on the same instruments
    (design commitment: validate that output forcing does not change choices).
    """
    rng = random.Random(seed + 1)
    trials = []
    tid = 0
    for g in gambles:
        for template_id in TEMPLATES:
            for safe_first in (True, False):
                for rep in range(reps):
                    mode = "text" if rng.random() < text_arm_share else "tool"
                    trials.append(Trial(
                        trial_id=tid, gamble_id=g.gamble_id,
                        sure=g.sure, hi=g.hi, p=g.p,
                        ev_ratio=round(g.ev_ratio, 4),
                        template_id=template_id, safe_first=safe_first,
                        rep=rep, response_mode=mode,
                    ))
                    tid += 1
    rng.shuffle(trials)  # randomize run order so drift/rate-limits don't confound conditions
    return trials


def trial_row(t: Trial) -> dict:
    return asdict(t)
