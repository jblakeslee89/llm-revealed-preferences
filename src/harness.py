"""Phase 1 elicitation harness: run the trial list against a model, write a tidy CSV.

Usage:
    # dry run against a simulated CRRA agent (no API key needed; validates the pipeline)
    python src/harness.py --dry-run --reps 3

    # small live pilot against Claude Haiku (needs ANTHROPIC_API_KEY)
    python src/harness.py --pilot 40

    # full run
    python src/harness.py --reps 3

Every design condition is logged per row, plus model snapshot, temperature,
seed, raw response, latency, and a discard flag, so the released dataset is
self-documenting (proposal §3.4, reproducibility commitment).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design import Trial, build_trials, generate_gambles, trial_row

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_dotenv() -> None:
    """Read KEY=VALUE lines from the project-root .env (gitignored) into the
    environment, so the API key never lives in code or shell profiles."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

CHOICE_TOOL = {
    "name": "submit_choice",
    "description": "Submit your choice between the two options.",
    "input_schema": {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
        "required": ["choice"],
    },
}

# Run-metadata columns appended to whatever design columns the trial carries.
# Design columns are read dynamically from the trial dataclass, so Phase 1 and
# Phase 2 (which adds frame/anchor) share the same runner.
RUN_FIELDS = [
    "model", "temperature", "seed",
    "letter", "chose_gamble", "raw", "discarded",
    "latency_s", "ts_utc",
]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class ClaudeAgent:
    """Live elicitation against the Anthropic API. Fresh context per call
    (independence); forced tool output on 'tool' trials, first-token parse
    on 'text' trials (the validation arm)."""

    def __init__(self, model: str, temperature: float):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature

    def choose(self, trial: Trial) -> tuple[str | None, str]:
        prompt = trial.prompt()
        if trial.response_mode == "tool":
            resp = self.client.messages.create(
                model=self.model, max_tokens=64, temperature=self.temperature,
                tools=[CHOICE_TOOL],
                tool_choice={"type": "tool", "name": "submit_choice"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return block.input.get("choice"), str(block.input)
            return None, str(resp.content)
        # plain-text validation arm
        resp = self.client.messages.create(
            model=self.model, max_tokens=5, temperature=self.temperature,
            messages=[{"role": "user", "content":
                       prompt + "\nRespond with only the letter A or B."}],
        )
        text = resp.content[0].text.strip() if resp.content else ""
        letter = text[:1].upper()
        return (letter if letter in ("A", "B") else None), text


class SimulatedCRRAAgent:
    """Synthetic subject for --dry-run: a CRRA expected-utility maximizer with
    logit choice noise and known parameters. Running the pipeline against it
    and recovering (r, mu) in analysis/ validates the whole loop end to end."""

    def __init__(self, r: float = 0.5, mu: float = 0.15, seed: int = 7):
        self.r, self.mu = r, mu
        self.rng = random.Random(seed)
        self.model = f"simulated-crra(r={r},mu={mu})"
        self.temperature = float("nan")

    def _u(self, x: float) -> float:
        if x <= 0:
            return 0.0
        return math.log(x) if abs(self.r - 1) < 1e-9 else x ** (1 - self.r) / (1 - self.r)

    def choose(self, trial: Trial) -> tuple[str, str]:
        eu_safe = self._u(trial.sure)
        eu_gamble = trial.p * self._u(trial.hi)
        # scale utilities so mu is comparable across stake levels
        scale = self._u(max(trial.hi, trial.sure))
        p_gamble = 1 / (1 + math.exp(-(eu_gamble - eu_safe) / (self.mu * scale)))
        chose_gamble = self.rng.random() < p_gamble
        letter = trial.gamble_letter if chose_gamble else ("A" if trial.gamble_letter == "B" else "B")
        return letter, f"simulated p_gamble={p_gamble:.3f}"


class SimulatedReferenceAgent:
    """Synthetic Phase 2 subject: reference-dependent, loss-averse, Prelec-weighting
    chooser with known (alpha, lam, gamma, phi, mu, delta). Recovering these in
    analysis/estimate_reference.py validates the Phase 2 estimator before spending
    on live calls."""

    def __init__(self, alpha=0.8, lam=2.0, gamma=1.0, phi=1.0, mu=0.15,
                 delta=0.0, seed=7):
        self.alpha, self.lam, self.gamma = alpha, lam, gamma
        self.phi, self.mu, self.delta = phi, mu, delta
        self.rng = random.Random(seed)
        self.model = f"simulated-ref(alpha={alpha},lam={lam},phi={phi})"
        self.temperature = float("nan")

    def _w(self, p):
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        return math.exp(-((-math.log(p)) ** self.gamma))

    def _v(self, x, rho):
        if x >= rho:
            return (x - rho) ** self.alpha
        return -self.lam * (rho - x) ** self.alpha

    def choose(self, trial) -> tuple[str, str]:
        anchor = getattr(trial, "anchor", 0)
        rho = self.phi * anchor
        v_safe = self._v(trial.sure, rho)
        w = self._w(trial.p)
        v_risky = w * self._v(trial.hi, rho) + (1 - w) * self._v(0, rho)
        scale = max(max(trial.hi, trial.sure), 1.0) ** self.alpha  # lambda-free, matches estimator
        z = (v_risky - v_safe) / (self.mu * scale) + self.delta * float(trial.safe_first)
        p_gamble = 1 / (1 + math.exp(-max(-35, min(35, z))))
        chose_gamble = self.rng.random() < p_gamble
        letter = trial.gamble_letter if chose_gamble else ("A" if trial.gamble_letter == "B" else "B")
        return letter, f"simulated p_gamble={p_gamble:.3f}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(trials, agent, seed: int, out_path: Path, row_fn,
        max_retries: int = 3, sleep_s: float = 0.05) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    n_done = n_discard = 0
    fieldnames = list(row_fn(trials[0]).keys()) + RUN_FIELDS

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for i, t in enumerate(trials):
            letter, raw, latency = None, "", float("nan")
            for attempt in range(max_retries):
                t0 = time.time()
                try:
                    letter, raw = agent.choose(t)
                    latency = time.time() - t0
                    break
                except Exception as e:  # rate limits, transient API errors
                    raw = f"ERROR: {e}"
                    wait = 2 ** attempt
                    print(f"  trial {t.trial_id}: {e} (retry in {wait}s)", file=sys.stderr)
                    time.sleep(wait)

            discarded = letter not in ("A", "B")
            chose_gamble = "" if discarded else int(letter == t.gamble_letter)
            row = row_fn(t) | {
                "model": agent.model, "temperature": agent.temperature, "seed": seed,
                "letter": letter or "", "chose_gamble": chose_gamble,
                "raw": (raw or "")[:200], "discarded": int(discarded),
                "latency_s": round(latency, 3) if latency == latency else "",
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            writer.writerow(row)
            f.flush()  # interrupted runs keep their data

            n_done += 1
            n_discard += int(discarded)
            if n_done % 25 == 0 or n_done == len(trials):
                print(f"{n_done}/{len(trials)} trials | discards: {n_discard}")
            time.sleep(sleep_s)

    print(f"\nWrote {n_done} rows -> {out_path}")
    print(f"Discard rate: {n_discard / max(n_done, 1):.1%}"
          + ("  <-- investigate if above ~2%" if n_discard / max(n_done, 1) > 0.02 else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, choices=[1, 2], default=1)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per design cell (default 3 for phase 1, 2 for phase 2)")
    ap.add_argument("--n-gambles", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", type=int, default=None, metavar="N",
                    help="run only the first N trials (after shuffling)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use a simulated agent instead of the API")
    ap.add_argument("--sim-r", type=float, default=0.5, help="phase-1 dry-run agent's true r")
    ap.add_argument("--sim-mu", type=float, default=0.15, help="dry-run agent's true mu")
    ap.add_argument("--sim-lam", type=float, default=2.0, help="phase-2 dry-run agent's true lambda")
    ap.add_argument("--sim-phi", type=float, default=1.0, help="phase-2 dry-run reference tracking")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--redo", default=None, metavar="CSV",
                    help="re-run only the discarded trials from an existing CSV "
                         "(appends valid rows to the same file)")
    args = ap.parse_args()

    load_dotenv()
    gambles = generate_gambles(n=args.n_gambles, seed=args.seed)

    if args.phase == 2:
        from design_phase2 import build_phase2_trials, trial_row as row_fn
        reps = args.reps if args.reps is not None else 2
        trials = build_phase2_trials(gambles, reps=reps, seed=args.seed)
        tag = "phase2"
    else:
        from design import trial_row as row_fn
        reps = args.reps if args.reps is not None else 3
        trials = build_trials(gambles, reps=reps, seed=args.seed)
        tag = "phase1"

    if args.redo:
        import pandas as pd
        prev = pd.read_csv(args.redo)
        done_ok = set(prev.loc[prev["discarded"] == 0, "trial_id"])
        trials = [t for t in trials if t.trial_id not in done_ok]
        args.out = args.redo
        print(f"Redo mode: {len(trials)} trials still need a valid response.")
    if args.pilot:
        trials = trials[:args.pilot]

    if args.dry_run:
        if args.phase == 2:
            agent = SimulatedReferenceAgent(lam=args.sim_lam, phi=args.sim_phi, seed=args.seed)
        else:
            agent = SimulatedCRRAAgent(r=args.sim_r, mu=args.sim_mu, seed=args.seed)
        default_out = DATA_DIR / f"{tag}_dryrun.csv"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set. Export it, or use --dry-run.")
        agent = ClaudeAgent(model=args.model, temperature=args.temperature)
        default_out = DATA_DIR / f"{tag}_{args.model.replace('/', '_')}.csv"

    out_path = Path(args.out) if args.out else default_out
    print(f"Phase {args.phase} | agent: {agent.model} | trials: {len(trials)} | out: {out_path}\n")
    run(trials, agent, seed=args.seed, out_path=out_path, row_fn=row_fn,
        sleep_s=0.0 if args.dry_run else 0.05)


if __name__ == "__main__":
    main()
