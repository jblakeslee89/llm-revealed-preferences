"""Phase 3: intercept, template and gamble effects in the structural fit.

Why: along the OLMo staircase, DPO and the final stage move both model-free quantities (dominance,
frame gap) while lambda, gamma and alpha hold still, and the structural fit gets worse (deviance on
the same 1,600 core cells 229 -> 421 -> 457). The writeup names heterogeneity across paraphrase
templates or across gambles as the first candidate for what the specification misses. This script
adds each to the choice index and asks, per subject:

  1. How much of the deviance does it absorb, and does that share grow at DPO? If it does,
     preference optimization made the model more sensitive to wording (templates) or to
     gamble-specific features the value function does not see (gambles).
  2. Do lambda, gamma and alpha keep their values once the heterogeneity has somewhere to go?

Models, nested, all on the profile_refit.py specification (phi = 1, wide box):
    none          z = (V_risky - V_safe) / (mu * scale) + delta * 1[gamble second]
    intercept     z + c                                  (a constant lean toward or away from the gamble)
    template RE   z + c + u_t,      u_t ~ N(0, s_t^2)    5 paraphrase templates, 320 cells each
    template FE   z + c_t                                five free template intercepts
    tFE+gamble RE z + c_t + v_g,    v_g ~ N(0, s_g^2)    80 gambles (10-30 cells each), crossed with
                                                          templates, so templates enter as fixed effects
Every random-effects model contains a constant, so the plain intercept model is the right baseline
for "does heterogeneity matter"; "none" vs "intercept" is a separate question about the base
specification.

The fractional objective is a quasi-likelihood: log L = -sum_c KL(y_c, p_c) / phi_hat, with phi_hat
the Pearson dispersion of the "none" fit (the binomial deviance is twice the summed KL). Random
effects are integrated out on a fixed standardized grid (u = sigma * U, U in [-6, 6], step 0.01).
LR tests of sigma = 0 are boundary tests: p = 0.5 * P(chi2_1 > LR) (Self and Liang 1987).

Usage:
    python analysis/template_effects.py                  # all eleven subjects
    python analysis/template_effects.py --subjects olmo  # the four staircase stages
Outputs: results/template_effects/{fits,stage_contrasts,template_modelfree}.csv, stdout log
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).parent))
from estimate_from_probs import load  # noqa: E402
from estimate_reference import prelec_w, value  # noqa: E402
from profile_refit import BOX, SUBJECTS, full_fit, minimize_box, natural  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "template_effects"
CORE = ["risk", "neutral", "gain", "mixed"]
U = np.arange(-6.0, 6.0 + 1e-9, 0.01)      # standardized grid: u_g = sigma * U
WU = stats.norm.logpdf(U) + np.log(0.01)   # log quadrature weights
LOG_SIG_BOX = (-7.0, 2.0)
C_BOX = (-10.0, 10.0)
STAGES = ["olmo-base", "olmo-sft", "olmo-dpo", "olmo-inst"]


def index_z(df, th):
    """Structural choice index (no logistic), wide-box parameterization of profile_refit.py."""
    a_raw, log_lam, log_gamma, log_mu, delta = th[:5]
    alpha, lam, gamma, mu = 1 / (1 + np.exp(-a_raw)), np.exp(log_lam), np.exp(log_gamma), np.exp(log_mu)
    rho = df["anchor"].values.astype(float)  # phi = 1
    s, h, p = (df[c].values.astype(float) for c in ("sure", "hi", "p"))
    w = prelec_w(p, gamma)
    v_safe = value(s, rho, alpha, lam)
    v_risky = w * value(h, rho, alpha, lam) + (1 - w) * value(np.zeros_like(h), rho, alpha, lam)
    scale = np.maximum(np.maximum(h, s), 1.0) ** alpha
    return (v_risky - v_safe) / (mu * scale) + delta * df["safe_first"].values.astype(float)


def kl(z, y):
    pm = np.clip(1 / (1 + np.exp(-np.clip(z, -35, 35))), 1e-6, 1 - 1e-6)
    return y * np.log(y / pm) + (1 - y) * np.log((1 - y) / (1 - pm))


def onehot(groups):
    M = np.zeros((len(groups), groups.max() + 1))
    M[np.arange(len(groups)), groups] = 1.0
    return M


class Subject:
    def __init__(self, core):
        self.df = core
        self.y = core["p_obs"].values
        self.tmpl = core["template_id"].astype(int).values
        self.gamb = pd.factorize(core["instrument"] + "_" + core["gamble_id"].astype(str))[0]
        self.Mt, self.Mg = onehot(self.tmpl), onehot(self.gamb)

    # offsets: "c" (one constant) or "tfe" (five template intercepts)
    def offset(self, x, kind):
        return x[5] if kind == "c" else x[5:10][self.tmpl]

    def fixed_obj(self, x, kind, phi):
        return kl(index_z(self.df, x[:5]) + self.offset(x, kind), self.y).sum() / phi

    def re_obj(self, x, kind, M, phi):
        sig = np.exp(x[-1])
        z = index_z(self.df, x[:5]) + self.offset(x, kind)
        a = -(kl(z[None, :] + sig * U[:, None], self.y) @ M) / phi + WU[:, None]
        m = a.max(0)
        return -(m + np.log(np.exp(a - m).sum(0))).sum()

    def re_blups(self, x, kind, M, phi):
        sig = np.exp(x[-1])
        z = index_z(self.df, x[:5]) + self.offset(x, kind)
        a = -(kl(z[None, :] + sig * U[:, None], self.y) @ M) / phi + WU[:, None]
        w = np.exp(a - a.max(0))
        return sig * (w * U[:, None]).sum(0) / w.sum(0)


def polish(f, x0, bounds):
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    r = optimize.minimize(f, np.clip(x0, lo, hi), method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 3000})
    r2 = optimize.minimize(lambda x: f(np.clip(x, lo, hi)), r.x, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 6000})
    x = np.clip(r2.x, lo, hi)
    fx = f(x)
    return (x, fx) if fx < r.fun else (r.x, r.fun)


def fit_intercept(S, theta0, phi, n_starts=12, seed=0):
    """Multi-start: adding c can move the structural parameters a long way (see the log)."""
    rng = np.random.default_rng(seed)
    bounds = [tuple(b) for b in BOX] + [C_BOX]
    best = None
    for s in range(n_starts):
        x0 = np.r_[theta0, 0.0]
        if s:
            x0 = x0 + rng.normal(0, [2, 2, 1, 2, 2, 1])
        cand = polish(lambda x: S.fixed_obj(x, "c", phi), x0, bounds)
        if best is None or cand[1] < best[1]:
            best = cand
    return best


def fit_tfe(S, x_int, phi):
    bounds = [tuple(b) for b in BOX] + [C_BOX] * 5
    return polish(lambda x: S.fixed_obj(x, "tfe", phi), np.r_[x_int[:5], np.full(5, x_int[5])], bounds)


def fit_re(S, x_start, kind, M, phi):
    nfix = 6 if kind == "c" else 10
    bounds = [tuple(b) for b in BOX] + [C_BOX] * (nfix - 5) + [LOG_SIG_BOX]
    best = None
    for ls0 in (-3.0, -1.5, -0.5, 0.3):
        cand = polish(lambda x: S.re_obj(x, kind, M, phi), np.r_[x_start, ls0], bounds)
        if best is None or cand[1] < best[1]:
            best = cand
    return best


def par(x):
    return {"alpha": natural(0, x[0]), "lambda": natural(1, x[1]), "gamma": natural(2, x[2]),
            "mu": np.exp(x[3]), "delta": x[4]}


def run_subject(args):
    key, fname = args
    df = load(ROOT / "data" / fname)
    core = df[df["frame"].isin(CORE)].copy().reset_index(drop=True)
    S = Subject(core)
    theta, _ = full_fit(core, n_starts=16)
    pm = np.clip(1 / (1 + np.exp(-np.clip(index_z(core, theta), -35, 35))), 1e-6, 1 - 1e-6)
    phi = ((S.y - pm) ** 2 / (pm * (1 - pm))).sum() / (len(core) - 5)
    dev = lambda z: kl(z, S.y).sum()  # same convention as profile_refit.py ("deviance" 229 at SFT)

    rows = []
    def add(model, x, nll, k, z, prev=None, sigma=np.nan, c=np.nan, extra=None):
        r = {"subject": key, "model": model, "k": k, "negll": nll, "dev": dev(z), "phi_hat": phi,
             "c": c, "sigma": sigma, **par(x)}
        if prev is not None:
            r["lr_vs_prev"] = 2 * (prev - nll)
            if model.endswith("RE"):
                r["p_vs_prev"] = 0.5 * stats.chi2.sf(max(r["lr_vs_prev"], 0), 1)
            else:
                r["p_vs_prev"] = stats.chi2.sf(max(r["lr_vs_prev"], 0), k - rows[-1]["k"] if rows else 1)
        r.update(extra or {})
        rows.append(r)

    nll0 = S.fixed_obj(np.r_[theta, 0.0], "c", phi)
    add("none", theta, nll0, 5, index_z(core, theta), c=0.0)

    xi, nlli = fit_intercept(S, theta, phi)
    add("intercept", xi, nlli, 6, index_z(core, xi[:5]) + xi[5], prev=nll0, c=xi[5])

    xr, nllr = fit_re(S, xi, "c", S.Mt, phi)
    ut = S.re_blups(xr, "c", S.Mt, phi)
    add("template RE", xr, nllr, 7, index_z(core, xr[:5]) + xr[5] + ut[S.tmpl], prev=nlli,
        sigma=np.exp(xr[-1]), c=xr[5], extra={"template_effects": ";".join(f"{v:+.3f}" for v in ut)})

    xf, nllf = fit_tfe(S, xi, phi)
    rows_k = len(rows)
    add("template FE", xf, nllf, 10, index_z(core, xf[:5]) + xf[5:10][S.tmpl], prev=nlli,
        extra={"template_effects": ";".join(f"{v:+.3f}" for v in xf[5:10])})
    rows[rows_k]["p_vs_prev"] = stats.chi2.sf(max(rows[rows_k]["lr_vs_prev"], 0), 4)

    xg, nllg = fit_re(S, xf, "tfe", S.Mg, phi)
    vg = S.re_blups(xg, "tfe", S.Mg, phi)
    add("tFE + gamble RE", xg, nllg, 11, index_z(core, xg[:5]) + xg[5:10][S.tmpl] + vg[S.gamb],
        prev=nllf, sigma=np.exp(xg[-1]),
        extra={"template_effects": ";".join(f"{v:+.3f}" for v in xg[5:10])})
    rows[-1]["p_vs_prev"] = 0.5 * stats.chi2.sf(max(rows[-1]["lr_vs_prev"], 0), 1)

    g = core[core.frame == "gain"].groupby("template_id")["p_gamble"].mean()
    m = core[core.frame == "mixed"].groupby("template_id")["p_gamble"].mean()
    up = core.groupby("template_id")["p_gamble"].mean()
    mf = {"subject": key, "frame_gap_by_template": ";".join(f"{v:+.3f}" for v in (m - g).values),
          "frame_gap_sd": float((m - g).std(ddof=1)),
          "uptake_by_template": ";".join(f"{v:.3f}" for v in up.values),
          "uptake_sd": float(up.std(ddof=1))}
    return rows, mf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subjects", default="all", help="'olmo' for the four staircase stages only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = STAGES if args.subjects == "olmo" else list(SUBJECTS)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run_subject, [(k, SUBJECTS[k]) for k in keys]))
    fits = pd.DataFrame([r for rows, _ in res for r in rows])
    mf = pd.DataFrame([m for _, m in res])
    base = fits[fits.model == "intercept"].set_index("subject")["dev"]
    fits["dev_absorbed_vs_intercept"] = 1 - fits["dev"] / fits["subject"].map(base)
    fits.to_csv(OUT / "fits.csv", index=False)
    mf.to_csv(OUT / "template_modelfree.csv", index=False)

    cons = []
    for a, b in zip(STAGES[:-1], STAGES[1:]):
        if a not in keys or b not in keys:
            continue
        for model in fits.model.unique():
            fa = fits[(fits.subject == a) & (fits.model == model)].iloc[0]
            fb = fits[(fits.subject == b) & (fits.model == model)].iloc[0]
            cons.append({"contrast": f"{a} -> {b}", "model": model,
                         **{f"d_{p}": fb[p] - fa[p] for p in ("lambda", "gamma", "alpha", "c")},
                         "d_dev": fb["dev"] - fa["dev"]})
    cons = pd.DataFrame(cons)
    cons.to_csv(OUT / "stage_contrasts.csv", index=False)

    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_rows", 200)
    cols = ["subject", "model", "k", "dev", "dev_absorbed_vs_intercept", "lr_vs_prev", "p_vs_prev",
            "c", "sigma", "lambda", "gamma", "alpha", "mu", "delta", "template_effects"]
    print("== fits (dev = summed fractional deviance on the 1,600 core cells, profile_refit convention) ==")
    print(fits[cols].round(4).to_string(index=False))
    print("\n== model-free spread across templates ==")
    print(mf.round(4).to_string(index=False))
    if len(cons):
        print("\n== OLMo stage-to-stage changes by model ==")
        print(cons.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
