"""Phase 3 elicitation: read choice probabilities from next-token log-probabilities.

Base (non-instruct) models do not follow "reply A or B", so the Phase 1-2 forced-tool
method cannot run on them. Instead we wrap each trial in a few-shot completion, ask for
one token, and read P(A) vs P(B) directly from the token log-probabilities. One call per
design cell replaces ~25 sampled repetitions: the probability IS the quantity the logit
likelihood wants.

Provider-agnostic: any OpenAI-compatible endpoint that returns `logprobs` on the
completions (or chat) API works. base_url and model id are supplied at construction, so
switching providers or the base/instruct variant is a one-line change.

Diagnostics logged per cell:
  p_gamble : renormalized P(choose the gamble) = massG / (massG + massSafe)
  ab_mass  : exp(lpA)+exp(lpB), the share of next-token mass on the two answer letters.
             Low mass means the model is not treating this as an A/B question (a base-
             model compliance signal); cells below a threshold are excluded and the rate
             reported.
"""

from __future__ import annotations

import math
import os

# Few-shot exemplars teach the FORMAT (one letter, read then pick) without teaching a
# preference: each answer is the unambiguously better option, balanced 2 A / 2 B, and
# the "safe" option sits in each letter slot equally. Payoffs are unrelated to the test
# grid so they cannot anchor it.
FEWSHOT = """You must pick one of two payment options.
Option A: receive $50 for certain
Option B: receive $6 for certain
Answer: A

You must pick one of two payment options.
Option A: receive $4 for certain
Option B: receive $63 for certain
Answer: B

You must pick one of two payment options.
Option A: an 8% chance of $70 and a 92% chance of $0
Option B: receive $47 for certain
Answer: B

You must pick one of two payment options.
Option A: receive $71 for certain
Option B: a 15% chance of $90 and an 85% chance of $0
Answer: A

"""

A_TOKENS = ("A", " A", "A)", " A)", "A.", " A.")
B_TOKENS = ("B", " B", "B)", " B)", "B.", " B.")


def fewshot_prompt(trial_prompt: str) -> str:
    return FEWSHOT + trial_prompt.rstrip() + "\nAnswer:"


def _mass(top_logprobs: dict, tokens) -> float:
    """Sum exp(logprob) over any candidate token spelling present in the top-k."""
    total = 0.0
    for tok, lp in top_logprobs.items():
        if tok.strip() in (t.strip() for t in tokens):
            total += math.exp(lp)
    return total


def probs_from_toplogprobs(top_logprobs: dict) -> tuple[float, float]:
    """Return (mass_A, mass_B) from a {token: logprob} dict at the answer position."""
    return _mass(top_logprobs, A_TOKENS), _mass(top_logprobs, B_TOKENS)


class LogprobAgent:
    """Real elicitation against an OpenAI-compatible provider.

    fmt='fewshot' uses the raw completions endpoint (works for base AND instruct).
    fmt='chat' uses the chat endpoint with the model's own template (instruct only);
    running an instruct model under BOTH formats is the format-confound control.
    """

    def __init__(self, model: str, base_url: str, api_key_env: str,
                 fmt: str = "fewshot", top_logprobs: int = 20):
        from openai import OpenAI
        key = os.environ.get(api_key_env)
        if not key:
            raise SystemExit(f"{api_key_env} is not set (put it in .env).")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.base_url = base_url
        self.fmt = fmt
        self.top_logprobs = top_logprobs
        self.temperature = 0.0  # deterministic; the distribution is read, not sampled

    def _top_at_answer(self, trial) -> dict:
        if self.fmt == "chat":
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=1, temperature=0.0,
                logprobs=True, top_logprobs=self.top_logprobs,
                messages=[{"role": "user",
                           "content": trial.prompt() + "\nAnswer with only A or B."}])
            content = resp.choices[0].logprobs.content
            if not content:
                return {}
            return {t.token: t.logprob for t in content[0].top_logprobs}
        # raw completion (base-model path)
        resp = self.client.completions.create(
            model=self.model, prompt=fewshot_prompt(trial.prompt()),
            max_tokens=1, temperature=0.0, logprobs=self.top_logprobs)
        lp = resp.choices[0].logprobs
        if not lp or not lp.top_logprobs:
            return {}
        return dict(lp.top_logprobs[0])

    def choose_prob(self, trial) -> tuple[float, float, str]:
        top = self._top_at_answer(trial)
        mass_a, mass_b = probs_from_toplogprobs(top)
        ab = mass_a + mass_b
        if ab <= 0:
            return float("nan"), 0.0, "no A/B mass in top logprobs"
        mass_gamble = mass_b if trial.gamble_letter == "B" else mass_a
        p_gamble = mass_gamble / ab
        return p_gamble, ab, f"massA={mass_a:.3g} massB={mass_b:.3g}"


class SimulatedLogprobAgent:
    """Emits EXACT model probabilities (no sampling) from known parameters, so the
    fractional-MLE estimator can be validated to near-perfect recovery. Uses the same
    reference-dependent + Prelec structure as the Phase 2 simulated agent."""

    def __init__(self, alpha=0.8, lam=2.0, gamma=1.0, phi=1.0, mu=0.15,
                 delta=0.0, base_mass=0.9, model="simulated-logprob"):
        self.alpha, self.lam, self.gamma = alpha, lam, gamma
        self.phi, self.mu, self.delta = phi, mu, delta
        self.base_mass = base_mass
        self.model = model
        self.base_url = "simulated"
        self.fmt = "fewshot"

    def _w(self, p):
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        return math.exp(-((-math.log(p)) ** self.gamma))

    def _v(self, x, rho):
        return (x - rho) ** self.alpha if x >= rho else -self.lam * (rho - x) ** self.alpha

    def choose_prob(self, trial) -> tuple[float, float, str]:
        anchor = getattr(trial, "anchor", 0)
        rho = self.phi * anchor
        v_safe = self._v(trial.sure, rho)
        w = self._w(trial.p)
        v_risky = w * self._v(trial.hi, rho) + (1 - w) * self._v(0, rho)
        scale = max(max(trial.hi, trial.sure), 1.0) ** self.alpha
        z = (v_risky - v_safe) / (self.mu * scale) + self.delta * float(trial.safe_first)
        p_gamble = 1 / (1 + math.exp(-max(-35, min(35, z))))
        return p_gamble, self.base_mass, "simulated"
