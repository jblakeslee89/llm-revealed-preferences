"""Phase 2 estimation: reference-dependent value + Prelec weighting + loss aversion.

Model (carries gamma and the position nuisance forward from Phase 1):

    reference   rho = phi * anchor          (anchor = 0 gain/neutral, h loss frame)
    value       v(x) = (x - rho)^alpha             if x >= rho
                     = -lambda * (rho - x)^alpha    if x <  rho
    weighting   w(p) = exp(-(-ln p)^gamma)          (Prelec)
    index       z = (V_risky - V_safe)/(mu*scale) + delta * 1[gamble listed second]
                Pr(choose gamble) = logistic(z)

Parameters: alpha (curvature), lambda (loss aversion), gamma (weighting),
mu (noise), delta (position), phi (how much the reference tracks the frame).

Two headline tests:
  1. Loss aversion:  H0 lambda = 1   (Wald)
  2. Frame invariance: H0 phi = 0    (LR test; phi=0 makes gain and loss frames
     identical, so choices cannot depend on the frame). This is the operational
     frame-invariance test the proposal promises.

Also reports a model-free frame gap (gain vs loss uptake) and the dominant-choice
positive control (should be ~1.0 if the model reads payoff magnitudes at all).

Usage:
    python analysis/estimate_reference.py data/phase2_claude-haiku-4-5-20251001.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import optimize, stats


def prelec_w(p, gamma):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.exp(-((-np.log(p)) ** gamma))


def value(x, rho, alpha, lam):
    x = np.asarray(x, float)
    gain = np.maximum(x - rho, 0.0) ** alpha
    loss = -lam * np.maximum(rho - x, 0.0) ** alpha
    return np.where(x >= rho, gain, loss)


def unpack(theta, fix_phi=None):
    # transforms keep parameters in valid ranges
    a_raw, log_lam, log_gamma, log_mu, delta = theta[:5]
    alpha = 1 / (1 + np.exp(-a_raw))            # (0,1)
    lam = np.exp(np.clip(log_lam, -4, 4))        # >0, bounded to [0.018, 54.6]
    gamma = np.exp(log_gamma)                    # >0
    mu = np.exp(log_mu)                          # >0
    if fix_phi is not None:
        phi = fix_phi
    else:
        phi = 1 / (1 + np.exp(-theta[5]))        # (0,1)
    return alpha, lam, gamma, mu, delta, phi


def choice_prob(df, theta, fix_phi=None):
    alpha, lam, gamma, mu, delta, phi = unpack(theta, fix_phi)
    anchor = df["anchor"].values.astype(float)
    rho = phi * anchor
    s = df["sure"].values.astype(float)
    h = df["hi"].values.astype(float)
    p = df["p"].values.astype(float)
    w = prelec_w(p, gamma)
    v_safe = value(s, rho, alpha, lam)
    v_risky = w * value(h, rho, alpha, lam) + (1 - w) * value(np.zeros_like(h), rho, alpha, lam)
    # lambda-free, rho-free magnitude normalizer: keeps mu comparable across stakes
    # without absorbing loss aversion (which a value-based scale would).
    scale = np.maximum(np.maximum(h, s), 1.0) ** alpha
    gamble_second = df["safe_first"].values.astype(float)
    z = (v_risky - v_safe) / (mu * scale) + delta * gamble_second
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def neg_loglik(theta, df, fix_phi=None):
    pg = choice_prob(df, theta, fix_phi)
    y = df["chose_gamble"].values.astype(float)
    ll = y * np.log(np.clip(pg, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - pg, 1e-12, 1))
    return -ll.sum()


def num_hessian(f, theta, eps=1e-4):
    k = len(theta)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = eps, eps
            H[i, j] = (f(theta + ei + ej) - f(theta + ei - ej)
                       - f(theta - ei + ej) + f(theta - ei - ej)) / (4 * eps * eps)
    return H


def fit(df, fix_phi=None, n_starts=8, seed=0):
    rng = np.random.default_rng(seed)
    k = 5 if fix_phi is not None else 6
    x0 = np.array([0.5, np.log(2.0), np.log(1.5), np.log(0.2), 0.5] + ([] if fix_phi is not None else [0.0]))
    best = None
    for j in range(n_starts):
        s = x0 + (0 if j == 0 else rng.normal(0, 0.5, k))
        res = optimize.minimize(neg_loglik, s, args=(df, fix_phi), method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 12000})
        if best is None or res.fun < best.fun:
            best = res
    return best


def se_from_hessian(df, theta, fix_phi=None):
    H = num_hessian(lambda th: neg_loglik(th, df, fix_phi), theta)
    try:
        cov = np.linalg.inv(H)
        return np.sqrt(np.maximum(np.diag(cov), 0)), cov
    except np.linalg.LinAlgError:
        return np.full(len(theta), np.nan), None


def report_params(theta, ses, fix_phi=None):
    alpha, lam, gamma, mu, delta, phi = unpack(theta, fix_phi)
    # delta-method SEs on transformed params
    a_raw = theta[0]
    d_alpha = np.exp(-a_raw) / (1 + np.exp(-a_raw)) ** 2
    out = {
        "alpha": (alpha, ses[0] * d_alpha),
        "lambda": (lam, ses[1] * lam),
        "gamma": (gamma, ses[2] * gamma),
        "mu": (mu, ses[3] * mu),
        "delta": (delta, ses[4]),
    }
    if fix_phi is None:
        d_phi = np.exp(-theta[5]) / (1 + np.exp(-theta[5])) ** 2
        out["phi"] = (phi, ses[5] * d_phi)
    else:
        out["phi"] = (phi, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["discarded"] == 0].copy()
    df = df[df["chose_gamble"].notna()]
    df["chose_gamble"] = df["chose_gamble"].astype(int)
    df["safe_first"] = df["safe_first"].astype(bool)

    core = df[df["frame"].isin(["neutral", "gain", "mixed"])].copy()
    dom = df[df["frame"] == "dominant"].copy()
    print(f"core choices: {len(core)}   dominant-control choices: {len(dom)}\n")

    # ---- positive control ----
    print("── Positive control (dominant choice) " + "─" * 22)
    if len(dom):
        rate = dom["chose_gamble"].mean()  # chose_gamble == chose the larger certain amount
        print(f"picked the strictly larger sure amount: {rate:.1%} of {len(dom)} trials"
              + ("   ok" if rate > 0.9 else "   FLAG: magnitude-insensitive (echoes Phase 1)"))
    print()

    # ---- model-free frame gap ----
    print("── Model-free frame gap " + "─" * 36)
    fg = core.groupby("frame")["chose_gamble"].agg(["mean", "count"])
    for fr in ("neutral", "gain", "mixed"):
        if fr in fg.index:
            print(f"  {fr:>7}: gamble uptake {fg.loc[fr, 'mean']:.3f}  (n={int(fg.loc[fr,'count'])})")
    if {"gain", "mixed"} <= set(fg.index):
        g = core[core["frame"] == "gain"]["chose_gamble"]
        m = core[core["frame"] == "mixed"]["chose_gamble"]
        t, pval = stats.ttest_ind(m, g)
        print(f"  mixed - gain gap: {m.mean() - g.mean():+.3f}  (p={pval:.2e})")
        print("  (a negative gap = loss aversion: naming the downside a 'loss' deters the gamble)")
    print()

    # ---- structural: unrestricted vs frame-invariant (phi=0) ----
    print("── Structural estimation " + "─" * 35)
    full = fit(core, fix_phi=None)      # free reference tracking
    restr = fit(core, fix_phi=0.0)      # frame-invariant null
    stated = fit(core, fix_phi=1.0)     # Koszegi-Rabin stated-reference assumption
    lr = 2 * (restr.fun - full.fun)     # both are neg-loglik minima
    p_lr = stats.chi2.sf(lr, df=1)

    ses, _ = se_from_hessian(core, full.x, fix_phi=None)
    params = report_params(full.x, ses, fix_phi=None)

    # lambda under the stated-reference assumption (phi=1): clean, avoids the
    # lambda-phi collinearity that inflates SEs in the free-phi fit.
    ses_s, _ = se_from_hessian(core, stated.x, fix_phi=1.0)
    params_stated = report_params(stated.x, ses_s, fix_phi=1.0)

    labels = {
        "alpha": "curvature (0-1, <1 concave over gains)",
        "lambda": "loss aversion (1 = none; human ~2.25)",
        "gamma": "Prelec weighting (1 = none; >1 S-shape)",
        "mu": "Fechner noise",
        "delta": "position (shift toward second-listed gamble)",
        "phi": "reference tracking (0 = frame-invariant, 1 = full)",
    }
    for k in ("alpha", "lambda", "gamma", "mu", "delta", "phi"):
        val, se = params[k]
        print(f"  {k:>7} = {val:8.4f}  (se {se:.4f})   {labels[k]}")

    lam_s, lam_s_se = params_stated["lambda"]
    print(f"\n  Loss aversion under stated reference (phi=1): "
          f"lambda = {lam_s:.3f} (se {lam_s_se:.3f})")

    print(f"\n  logLik full = {-full.fun:.2f}   logLik (phi=0) = {-restr.fun:.2f}   "
          f"logLik (phi=1) = {-stated.fun:.2f}")
    print(f"  Frame-invariance LR test  H0: phi=0   chi2(1) = {lr:.2f}, p = {p_lr:.2e}")
    if lam_s_se > 0:
        zlam = (lam_s - 1) / lam_s_se
        print(f"  Loss-aversion Wald test   H0: lambda=1 (phi=1)   "
              f"z = {zlam:+.2f}, p = {2*stats.norm.sf(abs(zlam)):.2e}")

    print("\n  Verdict:")
    verdict_frame = "frame-DEPENDENT (choices move with the frame)" if p_lr < 0.01 else "frame-invariant (no detectable frame effect)"
    verdict_lam = ("loss-averse" if lam_s > 1 and lam_s_se > 0 and (lam_s - 1) / lam_s_se > 1.96
                   else "loss-tolerant" if lam_s < 1 and lam_s_se > 0 and (1 - lam_s) / lam_s_se > 1.96
                   else "not distinguishable from loss-neutral (lambda=1)")
    print(f"    - the model is {verdict_frame}")
    print(f"    - loss aversion (stated reference): {verdict_lam} (lambda = {lam_s:.2f})")


if __name__ == "__main__":
    main()
