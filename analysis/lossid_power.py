"""Does the crossed gain x loss block (src/design_lossid.py) separate lambda from a constant lean?

Simulation on the OLMo SFT fit. Two data-generating truths that fit the current Phase 3 grid
almost equally well (fractional deviance 228.8 vs 224.6 on the 1,600 core cells):
    A  loss aversion, no lean:   the profile_refit.py fit (lambda 1.14, c = 0)
    B  lean, no loss aversion:   the intercept fit (lambda ~ 0, c = -0.80)
For each truth, simulate elicited probabilities on
    D1  the current core grid (1,600 cells)
    D2  the current grid plus the crossed block (720 more cells, instrument p3)
with noise shaped like the real residuals (template intercepts, gamble intercepts, cell noise on the
logit scale; scales from template_effects.py at SFT), then fit the intercept model (lambda and c both
free) and record lambda-hat. The block works if, on D2, lambda-hat lands near 1.1 under A and near 0
under B, while on D1 the two truths give overlapping lambda-hats.

Usage:
    python analysis/lossid_power.py --R 40
Output: results/lossid_power/{sims.csv,summary.csv}
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from estimate_from_probs import load  # noqa: E402
from profile_refit import BOX, full_fit, minimize_box, natural  # noqa: E402
from template_effects import index_z, kl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "lossid_power"
CORE = ["risk", "neutral", "gain", "mixed"]
SD_TEMPLATE, SD_GAMBLE, SD_CELL = 0.6, 0.15, 1.0
B6 = np.vstack([BOX, [[-10, 10]]])


def fit_intercept(df, n_starts=8, seed=0):
    th, _ = full_fit(df, n_starts=6, seed=seed)
    rng = np.random.default_rng(seed)
    f = lambda x: kl(index_z(df, x[:5]) + x[5], df["p_obs"].values).sum()
    best = None
    for s in range(n_starts):
        x0 = np.r_[th, 0.0] + (0 if s == 0 else rng.normal(0, [2, 2, 1, 2, 2, 1]))
        cand = minimize_box(f, np.clip(x0, B6[:, 0], B6[:, 1]), [tuple(b) for b in B6])
        if best is None or cand[1] < best[1]:
            best = cand
    return best


def designs():
    real = load(ROOT / "data" / "phase3_olmo-sft_fewshot.csv")
    d1 = real[real["frame"].isin(CORE)].reset_index(drop=True)
    g = pd.read_csv(ROOT / "data" / "phase3_lossid_grid.csv")
    g = g[g["frame"].isin(CORE)].copy()
    g["safe_first"] = g["safe_first"].astype(bool)
    cols = ["instrument", "gamble_id", "frame", "anchor", "sure", "hi", "p", "template_id", "safe_first"]
    d2 = pd.concat([d1[cols], g[cols]], ignore_index=True)
    return d1, d2[cols].copy()


def simulate(df, z_true, rng):
    t = rng.normal(0, SD_TEMPLATE, 5)[df["template_id"].astype(int).values]
    key = df["instrument"] + "_" + df["gamble_id"].astype(str)
    codes, uniq = pd.factorize(key)
    gm = rng.normal(0, SD_GAMBLE, len(uniq))[codes]
    z = z_true + t + gm + rng.normal(0, SD_CELL, len(df))
    return np.clip(1 / (1 + np.exp(-np.clip(z, -35, 35))), 1e-4, 1 - 1e-4)


def one(args):
    truth, x_true, design, r = args
    d1, d2 = designs()
    df = (d1 if design == "D1" else d2).copy()
    rng = np.random.default_rng(1000 * r + (0 if truth == "A" else 1) + (0 if design == "D1" else 7))
    df["p_obs"] = simulate(df, index_z(df, x_true[:5]) + x_true[5], rng)
    x, dev = fit_intercept(df, seed=r)
    # constrained fit at lambda = floor, for an LR-type discrimination check
    f0 = lambda xf: kl(index_z(df, np.r_[xf[0], BOX[1, 0], xf[1:4]]) + xf[4], df["p_obs"].values).sum()
    bounds0 = [tuple(B6[k]) for k in (0, 2, 3, 4, 5)]
    x0, dev0 = minimize_box(f0, np.r_[x[0], x[2:6]], bounds0)
    return {"truth": truth, "design": design, "r": r, "n": len(df),
            "lambda_hat": natural(1, x[1]), "c_hat": x[5], "gamma_hat": natural(2, x[2]),
            "alpha_hat": natural(0, x[0]), "dev": dev,
            "ddev_lambda0": dev0 - dev}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--R", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d1, _ = designs()
    thA, devA = full_fit(d1, n_starts=16)
    xA = np.r_[thA, 0.0]
    xB, devB = fit_intercept(d1, n_starts=16)
    print(f"truth A (no c):   lambda {natural(1, xA[1]):.3f} gamma {natural(2, xA[2]):.3f} "
          f"alpha {natural(0, xA[0]):.3f} c 0        dev {devA:.1f}")
    print(f"truth B (with c): lambda {natural(1, xB[1]):.4f} gamma {natural(2, xB[2]):.3f} "
          f"alpha {natural(0, xB[0]):.3f} c {xB[5]:+.3f} dev {devB:.1f}")

    jobs = [(t, x, d, r) for t, x in (("A", xA), ("B", xB)) for d in ("D1", "D2") for r in range(args.R)]
    with ProcessPoolExecutor(args.workers) as ex:
        sims = pd.DataFrame(list(ex.map(one, jobs)))
    sims.to_csv(OUT / "sims.csv", index=False)

    q = lambda s: pd.Series({"lambda_med": s.lambda_hat.median(),
                             "lambda_q10": s.lambda_hat.quantile(.1), "lambda_q90": s.lambda_hat.quantile(.9),
                             "share_lambda_gt_0.5": (s.lambda_hat > 0.5).mean(),
                             "c_med": s.c_hat.median(),
                             "ddev_lambda0_med": s.ddev_lambda0.median()})
    summ = sims.groupby(["truth", "design"]).apply(q).reset_index()
    summ.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nR = {args.R} simulations per cell; noise sd template {SD_TEMPLATE}, gamble {SD_GAMBLE}, "
          f"cell {SD_CELL} (logit scale)")
    print(summ.round(3).to_string(index=False))
    print("\nReading: the block separates the truths if lambda_hat is high under A and near 0 under B on D2.")


if __name__ == "__main__":
    main()
