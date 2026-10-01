"""Reason-then-answer arm across the OLMo-2 staircase (SFT, DPO, Instruct), fewshot.

For each stage and condition, on the cells the reasoning runs cover (matched on instrument
and trial_id): dominance accuracy, hard EV agreement on clear core cells (|ln EV ratio| >=
0.05), and the frame gap (mean P(gamble), mixed minus gain). Then cluster-bootstrap 95%
intervals (resampling the 40 gamble clusters, instrument x gamble_id) for the stage-to-stage
contrasts under reasoning, which split the SFT-to-Instruct change between DPO and the final
stage.

Usage:
    python analysis/score_reason_staircase.py [--B 2000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_induced import load, score  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
KEY = ["instrument", "trial_id"]
STAGES = ["sft", "dpo", "inst"]
CONDITIONS = [  # label, file suffix
    ("immediate, uninduced", ""),
    ("immediate, induced", "_riskneutral"),
    ("reason, uninduced", "_reason"),
    ("reason, induced", "_reason_riskneutral"),
]


def cells(stage, suffix, keep):
    d = load(DATA / f"phase3_olmo-{stage}_fewshot{suffix}.csv").merge(keep, on=KEY)
    sc = score(d, "riskneutral", 0.05)
    d = d.assign(correct=sc["hard_correct"].values, clear=sc["clear"].values)
    d["cluster"] = d["instrument"] + ":" + d["gamble_id"].astype(str)
    return d


def stats(d):
    core = d[d["frame"] != "dominant"]
    clear = core[core["clear"]]
    dom = d[d["frame"] == "dominant"]
    gap = d[d["frame"] == "mixed"]["p_gamble"].mean() - d[d["frame"] == "gain"]["p_gamble"].mean()
    return {"dom": dom["correct"].mean(), "ev": clear["correct"].mean(), "gap": gap}


def boot_contrast(a, b, metric, B, rng):
    """95% interval for stats(b)[metric] - stats(a)[metric], resampling shared gamble clusters."""
    clusters = np.array(sorted(set(a["cluster"]) & set(b["cluster"])))
    ga, gb = dict(tuple(a.groupby("cluster"))), dict(tuple(b.groupby("cluster")))
    draws = []
    for _ in range(B):
        pick = rng.choice(clusters, len(clusters), replace=True)
        da = pd.concat([ga[c] for c in pick])
        db = pd.concat([gb[c] for c in pick])
        draws.append(stats(db)[metric] - stats(da)[metric])
    return np.percentile(draws, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    keep = load(DATA / "phase3_olmo-sft_fewshot_reason.csv")[KEY]
    data = {(s, lab): cells(s, suf, keep) for s in STAGES for lab, suf in CONDITIONS}

    print(f"{'condition':24s}" + "".join(f"{s:>22s}" for s in STAGES))
    print(f"{'':24s}" + "   dom    EV     gap  " * len(STAGES))
    for lab, _ in CONDITIONS:
        row = f"{lab:24s}"
        for s in STAGES:
            r = stats(data[(s, lab)])
            row += f"  {r['dom']:.3f} {r['ev']:.3f} {r['gap']:+.3f}"
        print(row)

    rng = np.random.default_rng(1)
    print(f"\nContrasts under reasoning (95% cluster bootstrap, B={args.B}):")
    for lab in ("reason, uninduced", "reason, induced"):
        for a, b in (("sft", "dpo"), ("dpo", "inst"), ("sft", "inst")):
            for m in ("ev", "gap"):
                da, db = data[(a, lab)], data[(b, lab)]
                est = stats(db)[m] - stats(da)[m]
                lo, hi = boot_contrast(da, db, m, args.B, rng)
                print(f"  {lab:18s} {b}-{a:4s} {m:3s} {est:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
