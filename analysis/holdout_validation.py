"""Held-out validation for Phase 3 fractional data (Armour point 4, Aug 2026).

Answers the overfitting objection from the Phase 1 M2/M3 near-tie: does the
structural model predict choices on gambles it never saw, and does it beat a
probability-only reduced form out of sample?

Procedure: split the 40 gamble ids into train/test (default 30/10), fit each
candidate on the train cells, evaluate mean per-cell binomial deviance and RMSE
on the held-out cells, repeat over random splits.

Candidates:
  structural  reference-dependent + Prelec + position (phi=1), via the Phase 3
              fractional MLE (estimate_from_probs.fit)
  p-only      logistic in the win probability alone (2 params): the reduced
              form that nearly tied the structural model in Phase 1
  constant    train-mean probability (floor benchmark)

Usage:
    python analysis/holdout_validation.py data/phase3_qwen-inst_chat.csv
    python analysis/holdout_validation.py data/phase3_olmo-inst_fewshot.csv --splits 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
from estimate_from_probs import load, fit, frac_deviance  # noqa: E402
from estimate_reference import choice_prob  # noqa: E402


def dev_cells(y, pm):
    pm = np.clip(pm, 1e-6, 1 - 1e-6)
    return y * np.log(y / pm) + (1 - y) * np.log((1 - y) / (1 - pm))


def fit_p_only(df):
    y = df["p_obs"].values
    p = df["p"].values.astype(float)

    def obj(th):
        pm = 1 / (1 + np.exp(-(th[0] + th[1] * p)))
        return dev_cells(y, pm).sum()

    best = None
    for s in ([0.0, 0.0], [-2.0, 4.0], [2.0, -4.0]):
        r = optimize.minimize(obj, np.array(s), method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--test-gambles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(args.csv)
    core = df[df["frame"].isin(["risk", "neutral", "gain", "mixed"])].copy()
    gids = np.array(sorted(core["gamble_id"].unique()))
    rng = np.random.default_rng(args.seed)
    print(f"{args.csv}: {len(core)} core cells over {len(gids)} gambles; "
          f"{args.splits} splits, {args.test_gambles} held-out gambles each")

    rows = []
    for split in range(args.splits):
        test_g = rng.choice(gids, args.test_gambles, replace=False)
        tr = core[~core["gamble_id"].isin(test_g)]
        te = core[core["gamble_id"].isin(test_g)]
        y = te["p_obs"].values

        theta = fit(tr, fix_phi=1.0, n_starts=4, seed=split).x
        pm_s = np.clip(choice_prob(te, theta, 1.0), 1e-6, 1 - 1e-6)

        b0, b1 = fit_p_only(tr)
        pm_p = 1 / (1 + np.exp(-(b0 + b1 * te["p"].values.astype(float))))

        pm_c = np.full(len(te), tr["p_obs"].mean())

        for name, pm in (("structural", pm_s), ("p-only", pm_p), ("constant", pm_c)):
            rows.append({"split": split, "model": name,
                         "dev": dev_cells(y, pm).mean(),
                         "rmse": float(np.sqrt(np.mean((y - pm) ** 2)))})

    import pandas as pd
    res = pd.DataFrame(rows)
    print("\nheld-out mean per-cell deviance (lower = better):")
    agg = res.groupby("model")[["dev", "rmse"]].agg(["mean", "std"])
    for m in ("structural", "p-only", "constant"):
        d, ds = agg.loc[m, ("dev", "mean")], agg.loc[m, ("dev", "std")]
        r, rs = agg.loc[m, ("rmse", "mean")], agg.loc[m, ("rmse", "std")]
        print(f"  {m:>10}: dev {d:.4f} (sd {ds:.4f})   rmse {r:.4f} (sd {rs:.4f})")

    piv = res.pivot(index="split", columns="model", values="dev")
    wins = int((piv["structural"] < piv["p-only"]).sum())
    print(f"\nstructural beats p-only on {wins}/{args.splits} splits")
    print("(Phase 1's in-sample near-tie is the objection; an out-of-sample win here is"
          " the answer, and an out-of-sample loss is worth reporting honestly.)")


if __name__ == "__main__":
    main()
