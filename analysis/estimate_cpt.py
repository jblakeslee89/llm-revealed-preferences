"""Extended structural estimation: CRRA + Prelec probability weighting + position nuisance.

Motivated by the Phase 1 diagnostic: choice is driven by win probability far more
than by EV ratio, which plain CRRA cannot produce. This model adds:

    w(p) = exp(-(-ln p)^gamma)        (Prelec 1998; gamma=1 -> no distortion,
                                       gamma>1 -> S-shape: long shots underweighted)
    EU_gamble = w(p) * u(hi),  EU_safe = u(sure),  u = CRRA(r)
    z = (EU_gamble - EU_safe) / (mu * scale) + delta * 1[gamble listed second]
    Pr(choose gamble) = logistic(z)

Estimates (r, gamma, mu, delta) by MLE. Reports three-model comparison:
  M1: CRRA + Fechner            (the Phase 1 baseline; nested at gamma=1, delta=0)
  M2: CRRA + Prelec + position  (this model)
  M3: p-only logit              (the "probability heuristic" reduced form)
with likelihood-ratio test M1 vs M2 and AIC/BIC across all three, plus a
fitted-vs-observed check by probability bin.

Usage:
    python analysis/estimate_cpt.py data/phase1_claude-haiku-4-5-20251001.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import optimize, stats


def crra_u(x: np.ndarray, r: float) -> np.ndarray:
    x = np.maximum(x, 1e-12)
    if abs(r - 1.0) < 1e-9:
        return np.log(x)
    return x ** (1.0 - r) / (1.0 - r)


def prelec_w(p: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.exp(-((-np.log(p)) ** gamma))


# ---------------------------------------------------------------------------
# Models: each returns Pr(choose gamble) per row
# ---------------------------------------------------------------------------

def prob_m1(df: pd.DataFrame, theta) -> np.ndarray:
    """M1: CRRA + Fechner logit (r, log_mu)."""
    r, log_mu = theta
    mu = np.exp(log_mu)
    eu_s = crra_u(df["sure"].values.astype(float), r)
    eu_g = df["p"].values * crra_u(df["hi"].values.astype(float), r)
    scale = np.abs(crra_u(np.maximum(df["hi"], df["sure"]).values.astype(float), r))
    z = (eu_g - eu_s) / (mu * np.maximum(scale, 1e-12))
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def prob_m2(df: pd.DataFrame, theta) -> np.ndarray:
    """M2: CRRA + Prelec weighting + position (r, log_gamma, log_mu, delta)."""
    r, log_gamma, log_mu, delta = theta
    gamma, mu = np.exp(log_gamma), np.exp(log_mu)
    eu_s = crra_u(df["sure"].values.astype(float), r)
    eu_g = prelec_w(df["p"].values, gamma) * crra_u(df["hi"].values.astype(float), r)
    scale = np.abs(crra_u(np.maximum(df["hi"], df["sure"]).values.astype(float), r))
    gamble_second = df["safe_first"].values.astype(float)  # safe first => gamble is option B
    z = (eu_g - eu_s) / (mu * np.maximum(scale, 1e-12)) + delta * gamble_second
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def prob_m3(df: pd.DataFrame, theta) -> np.ndarray:
    """M3: p-only logit (a, b) - the pure probability heuristic, ignores payoffs."""
    a, b = theta
    z = a + b * df["p"].values
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


MODELS = {
    "M1 CRRA":            (prob_m1, ["r", "log_mu"],                      [0.3, np.log(0.2)]),
    "M2 CRRA+Prelec+pos": (prob_m2, ["r", "log_gamma", "log_mu", "delta"],[0.3, 0.3, np.log(0.2), 0.5]),
    "M3 p-only":          (prob_m3, ["a", "b"],                           [-4.0, 8.0]),
}


def neg_loglik(theta, df, prob_fn):
    pg = prob_fn(df, theta)
    y = df["chose_gamble"].values.astype(float)
    ll = y * np.log(np.clip(pg, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - pg, 1e-12, 1))
    return -ll.sum()


def numerical_hessian(f, theta, eps=1e-4):
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = eps, eps
            H[i, j] = (f(theta + ei + ej) - f(theta + ei - ej)
                       - f(theta - ei + ej) + f(theta - ei - ej)) / (4 * eps * eps)
    return H


def fit(df, prob_fn, x0, n_starts=6, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    starts = [np.array(x0)] + [np.array(x0) + rng.normal(0, 0.5, len(x0))
                               for _ in range(n_starts - 1)]
    for s in starts:
        res = optimize.minimize(neg_loglik, s, args=(df, prob_fn), method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 8000})
        if best is None or res.fun < best.fun:
            best = res
    H = numerical_hessian(lambda th: neg_loglik(th, df, prob_fn), best.x)
    try:
        cov = np.linalg.inv(H)
        ses = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        ses = np.full(len(best.x), np.nan)
    return best, ses


def transform_report(names, theta, ses):
    """Report log-parameterized quantities on their natural scale (delta method)."""
    rows = []
    for nm, th, se in zip(names, theta, ses):
        if nm.startswith("log_"):
            rows.append((nm[4:], np.exp(th), np.exp(th) * se))
        else:
            rows.append((nm, th, se))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["discarded"] == 0].copy()
    df = df[df["chose_gamble"].notna()]
    df["chose_gamble"] = df["chose_gamble"].astype(int)
    df["safe_first"] = df["safe_first"].astype(bool)
    n = len(df)
    print(f"n = {n} valid choices\n")

    results = {}
    for name, (fn, pnames, x0) in MODELS.items():
        res, ses = fit(df, fn, x0)
        k = len(x0)
        ll = -res.fun
        results[name] = {"ll": ll, "k": k, "aic": 2 * k - 2 * ll,
                         "bic": k * np.log(n) - 2 * ll,
                         "theta": res.x, "ses": ses, "pnames": pnames, "fn": fn}

    print("── Model comparison " + "─" * 40)
    print(f"{'model':<22}{'k':>3}{'logLik':>10}{'AIC':>10}{'BIC':>10}")
    for name, r in results.items():
        print(f"{name:<22}{r['k']:>3}{r['ll']:>10.2f}{r['aic']:>10.1f}{r['bic']:>10.1f}")

    m1, m2 = results["M1 CRRA"], results["M2 CRRA+Prelec+pos"]
    lr = 2 * (m2["ll"] - m1["ll"])
    p_lr = stats.chi2.sf(lr, df=m2["k"] - m1["k"])
    print(f"\nLR test M1 vs M2: chi2({m2['k'] - m1['k']}) = {lr:.1f}, p = {p_lr:.2e}")

    print("\n── M2 estimates (natural scale) " + "─" * 28)
    for nm, val, se in transform_report(m2["pnames"], m2["theta"], m2["ses"]):
        note = {"r": "  (0=risk-neutral)",
                "gamma": "  (1=no distortion; >1 = S-shape, long shots underweighted)",
                "mu": "  (Fechner noise)",
                "delta": "  (shift toward gamble when listed second)"}.get(nm, "")
        print(f"  {nm:>6} = {val:8.4f}  (se {se:.4f}){note}")

    print("\n── Fit by probability bin: observed vs M1 vs M2 " + "─" * 12)
    df["p_bin"] = pd.cut(df["p"], [0.15, 0.3, 0.45, 0.6, 0.85])
    df["pg_m1"] = prob_m1(df, m1["theta"])
    df["pg_m2"] = prob_m2(df, m2["theta"])
    tab = df.groupby("p_bin", observed=True).agg(
        observed=("chose_gamble", "mean"), M1=("pg_m1", "mean"),
        M2=("pg_m2", "mean"), n=("chose_gamble", "size")).round(3)
    print(tab.to_string())


if __name__ == "__main__":
    main()
