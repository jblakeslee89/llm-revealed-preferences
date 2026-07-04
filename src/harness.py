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

FIELDNAMES = [
    "trial_id", "gamble_id", "sure", "hi", "p", "ev_ratio",
    "template_id", "safe_first", "rep", "response_mode",
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(trials: list[Trial], agent, seed: int, out_path: Path,
        max_retries: int = 3, sleep_s: float = 0.05) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    n_done = n_discard = 0

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
            row = trial_row(t) | {
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
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3, help="repetitions per design cell")
    ap.add_argument("--n-gambles", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", type=int, default=None, metavar="N",
                    help="run only the first N trials (after shuffling)")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the simulated CRRA agent instead of the API")
    ap.add_argument("--sim-r", type=float, default=0.5, help="dry-run agent's true r")
    ap.add_argument("--sim-mu", type=float, default=0.15, help="dry-run agent's true mu")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    load_dotenv()
    gambles = generate_gambles(n=args.n_gambles, seed=args.seed)
    trials = build_trials(gambles, reps=args.reps, seed=args.seed)
    if args.pilot:
        trials = trials[:args.pilot]

    if args.dry_run:
        agent = SimulatedCRRAAgent(r=args.sim_r, mu=args.sim_mu, seed=args.seed)
        default_out = DATA_DIR / "phase1_dryrun.csv"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set. Export it, or use --dry-run.")
        agent = ClaudeAgent(model=args.model, temperature=args.temperature)
        default_out = DATA_DIR / f"phase1_{args.model.replace('/', '_')}.csv"

    out_path = Path(args.out) if args.out else default_out
    print(f"Agent: {agent.model} | trials: {len(trials)} | out: {out_path}\n")
    run(trials, agent, seed=args.seed, out_path=out_path,
        sleep_s=0.0 if args.dry_run else 0.05)


if __name__ == "__main__":
    main()
