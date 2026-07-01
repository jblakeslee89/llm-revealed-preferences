"""Phase 1 estimation: CRRA + Fechner-logit maximum likelihood on elicited choices.

Model (proposal §3.1):
    u(x) = x^(1-r) / (1-r)                      (CRRA; log utility at r=1)
    Pr(choose gamble) = logistic((EU_g - EU_s) / (mu * scale))

where scale = u(max payoff in the pair) normalizes the Fechner noise across
stake levels. Estimated by MLE; standard errors from the numerical Hessian.

Also reports the sanity checks that must pass before the estimates are
trusted (proposal §3.4): label balance, EV-monotonicity, discard rate, and
the tool-vs-text response-mode comparison.

Usage:
    python analysis/estimate_crra.py data/phase1_dryrun.csv
    python analysis/estimate_crra.py data/phase1_claude-haiku-4-5-20251001.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def crra_u(x: np.ndarray, r: float) -> np.ndarray:
    x = np.maximum(x, 1e-12)
    if abs(r - 1.0) < 1e-9:
        return np.log(x)
    return x ** (1.0 - r) / (1.0 - r)


def choice_prob(df: pd.DataFrame, r: float, mu: float) -> np.ndarray:
    """Pr(choose gamble) for each row under (r, mu)."""
    eu_safe = crra_u(df["sure"].values.astype(float), r)
    eu_gamble = df["p"].values * crra_u(df["hi"].values.astype(float), r)
    scale = np.abs(crra_u(np.maximum(df["hi"].values, df["sure"].values).astype(float), r))
    z = (eu_gamble - eu_safe) / (mu * np.maximum(scale, 1e-12))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def neg_loglik(theta: np.ndarray, df: pd.DataFrame) -> float:
    r, log_mu = theta
    mu = np.exp(log_mu)  # mu > 0 via log transform
    pg = choice_prob(df, r, mu)
    y = df["chose_gamble"].values.astype(float)
    ll = y * np.log(np.clip(pg, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - pg, 1e-12, 1))
    return -ll.sum()


def numerical_hessian(f, theta: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = eps, eps
            H[i, j] = (f(theta + ei + ej) - f(theta + ei - ej)
                       - f(theta - ei + ej) + f(theta - ei - ej)) / (4 * eps * eps)
    return H


def estimate(df: pd.DataFrame) -> dict:
    best = None
    for r0 in (0.1, 0.5, 0.9, 1.5):  # multi-start; the likelihood can be flat in r
        res = optimize.minimize(neg_loglik, x0=np.array([r0, np.log(0.2)]),
                                args=(df,), method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
        if best is None or res.fun < best.fun:
            best = res

    r_hat, log_mu_hat = best.x
    mu_hat = float(np.exp(log_mu_hat))

    H = numerical_hessian(lambda th: neg_loglik(th, df), best.x)
    try:
        cov = np.linalg.inv(H)
        se_r = float(np.sqrt(max(cov[0, 0], 0)))
        # delta method for mu = exp(log_mu)
        se_mu = float(np.sqrt(max(cov[1, 1], 0)) * mu_hat)
    except np.linalg.LinAlgError:
        se_r = se_mu = float("nan")

    return {"r": float(r_hat), "se_r": se_r, "mu": mu_hat, "se_mu": se_mu,
            "loglik": -float(best.fun), "n": len(df), "converged": bool(best.success)}


# ---------------------------------------------------------------------------
# Sanity checks (must pass before estimates are trusted)
# ---------------------------------------------------------------------------

def sanity_checks(df: pd.DataFrame) -> None:
    print("── Sanity checks " + "─" * 43)

    # 1. Discard rate
    if "discarded" in df.columns:
        rate = df["discarded"].mean()
        print(f"discard rate:        {rate:.1%}" + ("   FLAG (>2%)" if rate > 0.02 else "   ok"))

    kept = df[df.get("discarded", 0) == 0].copy()
    kept["chose_gamble"] = kept["chose_gamble"].astype(int)

    # 2. Label/position balance: chose_gamble should not depend on where the safe
    #    option sat. A significant gap = position bias masquerading as preference.
    by_pos = kept.groupby("safe_first")["chose_gamble"].agg(["mean", "count"])
    if len(by_pos) == 2:
        a = kept[kept["safe_first"] == True]["chose_gamble"]
        b = kept[kept["safe_first"] == False]["chose_gamble"]
        _, pval = stats.ttest_ind(a, b)
        gap = abs(a.mean() - b.mean())
        print(f"position gap:        {gap:.3f} (p={pval:.3f})"
              + ("   FLAG: position bias" if pval < 0.01 and gap > 0.05 else "   ok"))

    # 3. Monotonicity: P(gamble) should rise with the gamble's EV ratio.
    kept["ev_bin"] = pd.qcut(kept["ev_ratio"], 4, labels=False, duplicates="drop")
    means = kept.groupby("ev_bin")["chose_gamble"].mean()
    mono = means.is_monotonic_increasing
    print("P(gamble) by EV quartile: " + "  ".join(f"{m:.2f}" for m in means)
          + ("   ok (monotone)" if mono else "   FLAG: non-monotone"))

    # 4. Tool vs. text response mode (the validation arm)
    if "response_mode" in kept.columns and kept["response_mode"].nunique() == 2:
        by_mode = kept.groupby("response_mode")["chose_gamble"].agg(["mean", "count"])
        print("response-mode means: "
              + "  ".join(f"{m}={r['mean']:.3f} (n={int(r['count'])})"
                          for m, r in by_mode.iterrows()))

    # 5. Template spread (preview of the random-effects question)
    by_t = kept.groupby("template_id")["chose_gamble"].mean()
    print(f"template range:      {by_t.min():.3f}–{by_t.max():.3f} "
          f"(spread {by_t.max() - by_t.min():.3f})")
    print("─" * 60)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="choices CSV produced by src/harness.py")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    if "model" in df.columns:
        for m in df["model"].unique():
            print(f"  model: {m} ({(df['model'] == m).sum()} rows)")
    print()

    sanity_checks(df)

    kept = df[df.get("discarded", 0) == 0].copy()
    kept = kept[kept["chose_gamble"].notna()]
    kept["chose_gamble"] = kept["chose_gamble"].astype(int)

    print("\n── CRRA + Fechner-logit MLE " + "─" * 32)
    est = estimate(kept)
    ci_lo, ci_hi = est["r"] - 1.96 * est["se_r"], est["r"] + 1.96 * est["se_r"]
    print(f"r  (risk aversion) = {est['r']:.4f}  (se {est['se_r']:.4f}, "
          f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")
    print(f"mu (Fechner noise) = {est['mu']:.4f}  (se {est['se_mu']:.4f})")
    print(f"log-likelihood     = {est['loglik']:.2f}   n = {est['n']}"
          f"   converged = {est['converged']}")
    print()
    verdict = ("risk-averse" if ci_lo > 0 else
               "risk-seeking" if ci_hi < 0 else
               "not distinguishable from risk-neutral")
    print(f"Verdict: the agent is {verdict} (r=0 is risk-neutral, r>0 risk-averse).")


if __name__ == "__main__":
    main()
