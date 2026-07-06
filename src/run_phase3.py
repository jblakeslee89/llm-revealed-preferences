"""Phase 3 runner: elicit choice probabilities for a base or instruct model.

Runs the Phase 1 (risk/weighting) and Phase 2 (framing) instruments once per design
cell via log-probabilities, writing one row per cell with p_gamble and the A/B mass
diagnostic. Reps are unnecessary: logprobs give the probability directly.

Examples:
    # validation gate: recover known params from a simulated logprob agent
    python src/run_phase3.py --dry-run --sim-lam 2.0 --sim-gamma 1.5

    # live base model (provider + id filled in after the catalog check)
    python src/run_phase3.py --model meta-llama/Llama-3.1-8B \\
        --base-url https://api.together.xyz/v1 --key-env TOGETHER_API_KEY --tag llama-base

    # instruct model under BOTH formats (format-confound control)
    python src/run_phase3.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --base-url ... --key-env TOGETHER_API_KEY --tag llama-inst-fewshot --fmt fewshot
    python src/run_phase3.py --model ... --tag llama-inst-chat --fmt chat

    # export the trial grid to CSV for the Colab fallback (no calls)
    python src/run_phase3.py --export-grid data/phase3_grid.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design import generate_gambles, build_trials
from design_phase2 import build_phase2_trials
from harness import load_dotenv, DATA_DIR


def build_cells(n_gambles=40, seed=42):
    """Phase 1 and Phase 2 instruments, one cell each (reps=1)."""
    gambles = generate_gambles(n=n_gambles, seed=seed)
    p1 = build_trials(gambles, reps=1, seed=seed, text_arm_share=0.0)
    p2 = build_phase2_trials(gambles, reps=1, seed=seed, text_arm_share=0.0)
    cells = [("p1", t) for t in p1] + [("p2", t) for t in p2]
    return cells


def cell_row(instrument, t) -> dict:
    return {
        "instrument": instrument,
        "trial_id": t.trial_id,
        "gamble_id": t.gamble_id,
        "frame": getattr(t, "frame", "risk"),
        "anchor": getattr(t, "anchor", 0),
        "sure": t.sure, "hi": t.hi, "p": t.p, "ev_ratio": t.ev_ratio,
        "template_id": t.template_id, "safe_first": t.safe_first,
        "gamble_letter": t.gamble_letter,
    }


RUN_FIELDS = ["model", "fmt", "p_gamble", "ab_mass", "excluded", "note", "latency_s", "ts_utc"]


def export_grid(path: Path):
    cells = build_cells()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [cell_row(inst, t) | {"prompt": t.prompt()} for inst, t in cells]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Exported {len(rows)} cells (with prompts) -> {path}")


def run(agent, out_path: Path, mass_threshold=0.20, sleep_s=0.05):
    cells = build_cells()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(cell_row(*cells[0]).keys()) + RUN_FIELDS
    n_excl = 0
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (inst, t) in enumerate(cells):
            p_gamble, ab_mass, note, latency = float("nan"), 0.0, "", float("nan")
            for attempt in range(3):
                t0 = time.time()
                try:
                    p_gamble, ab_mass, note = agent.choose_prob(t)
                    latency = time.time() - t0
                    break
                except Exception as e:
                    note = f"ERROR: {e}"
                    time.sleep(2 ** attempt)
            excluded = int(not (ab_mass >= mass_threshold) or p_gamble != p_gamble)
            n_excl += excluded
            w.writerow(cell_row(inst, t) | {
                "model": agent.model, "fmt": agent.fmt,
                "p_gamble": "" if p_gamble != p_gamble else round(p_gamble, 5),
                "ab_mass": round(ab_mass, 5), "excluded": excluded, "note": note[:120],
                "latency_s": round(latency, 3) if latency == latency else "",
                "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            f.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(cells):
                print(f"{i+1}/{len(cells)} cells | excluded: {n_excl}")
            time.sleep(sleep_s)
    rate = n_excl / max(len(cells), 1)
    print(f"\nWrote {len(cells)} cells -> {out_path}")
    print(f"Excluded (low A/B mass): {rate:.1%}"
          + ("  <-- base-model non-compliance; inspect" if rate > 0.15 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--fmt", choices=["fewshot", "chat"], default="fewshot")
    ap.add_argument("--tag", default=None, help="label for the output filename")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sim-lam", type=float, default=2.0)
    ap.add_argument("--sim-gamma", type=float, default=1.0)
    ap.add_argument("--sim-phi", type=float, default=1.0)
    ap.add_argument("--export-grid", default=None, metavar="CSV")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.export_grid:
        export_grid(Path(args.export_grid))
        return

    load_dotenv()
    if args.dry_run:
        from logprob_agent import SimulatedLogprobAgent
        agent = SimulatedLogprobAgent(lam=args.sim_lam, gamma=args.sim_gamma, phi=args.sim_phi)
        out = Path(args.out or DATA_DIR / "phase3_dryrun.csv")
    else:
        if not args.model or not args.base_url:
            sys.exit("Live run needs --model and --base-url (see docs/phase3-scope.md).")
        from logprob_agent import LogprobAgent
        agent = LogprobAgent(model=args.model, base_url=args.base_url,
                             api_key_env=args.key_env, fmt=args.fmt)
        tag = args.tag or args.model.replace("/", "_")
        out = Path(args.out or DATA_DIR / f"phase3_{tag}_{args.fmt}.csv")

    print(f"Phase 3 | agent: {agent.model} | fmt: {agent.fmt} | out: {out}\n")
    run(agent, out, sleep_s=0.0 if args.dry_run else 0.05)


if __name__ == "__main__":
    main()
