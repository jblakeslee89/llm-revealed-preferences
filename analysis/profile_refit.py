"""Phase 3 refit: wide bounds + profile-likelihood intervals for lambda, gamma, alpha.

Why: the Phase 3 fits in estimate_from_probs.py hit the hard lambda clip (log lambda in
[-4, 4]) and the alpha sigmoid ceiling, and the cell-bootstrap intervals overflowed at
those bounds (worst case ~1e86). This script refits every Phase 3 subject with a much wider
box, then profiles each headline parameter: fix it on a grid, re-optimize the nuisances,
and read the interval off the objective. A parameter whose profile stays flat out to the
wide bound is reported as not identified in that direction, instead of as a number.

Objective and model are unchanged from estimate_from_probs.py (fractional binomial
deviance, reference-dependent value, Prelec weighting, position nuisance, phi fixed at 1),
fit on the core frames (risk, neutral, gain, mixed).

Two interval calibrations, because the fractional deviance is a quasi-likelihood:
  quasi    2*Delta(dev) / phi_hat <= 3.84, phi_hat = Pearson X^2 / (n - k)
  cluster  2*Delta(dev) / c_j     <= 3.84, c_j = robust/naive variance ratio for the
           parameter, sandwich clustered by gamble (80 clusters: 5 templates x 2 orders
           x frames of one gamble are not independent). Primary where the Hessian is usable.

Contrasts (instruct - base, or stage-to-stage) use the fact that the two subjects share
no nuisance parameters: the profile for a difference d is min_x S_b(x) + S_i(x + d).

Usage:
    python analysis/profile_refit.py            # all subjects and contrasts
    python analysis/profile_refit.py --quick    # coarse grid, for a smoke test
Outputs: results/profile_refit/{fits,intervals,contrasts,profiles}.csv
         writeups/figures/phase3_profiles.pdf
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).parent))
from estimate_from_probs import load  # noqa: E402
from estimate_reference import choice_prob, num_hessian  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "profile_refit"
FIG = ROOT / "writeups" / "figures" / "phase3_profiles.pdf"

LAM_CLIP = 50.0  # effectively unclipped; the box below does the bounding
NAMES = ["a_raw", "log_lam", "log_gamma", "log_mu", "delta"]
BOX = np.array([[-12, 12], [-9, 9], [-4, 4], [-15, 8], [-25, 25]], float)
PROFILED = {"alpha": 0, "lambda": 1, "gamma": 2}
CRIT = 3.841  # chi2(1) 95%

SUBJECTS = {
    "qwen-base": "phase3_qwen-base_fewshot.csv",
    "qwen-inst": "phase3_qwen-inst_fewshot.csv",
    "qwen-inst-chat": "phase3_qwen-inst_chat.csv",
    "llama-base": "phase3_llama-base_fewshot.csv",
    "llama-inst": "phase3_llama-inst_fewshot.csv",
    "llama-inst-chat": "phase3_llama-inst_chat.csv",
    "olmo-base": "phase3_olmo-base_fewshot.csv",
    "olmo-sft": "phase3_olmo-sft_fewshot.csv",
    "olmo-dpo": "phase3_olmo-dpo_fewshot.csv",
    "olmo-inst": "phase3_olmo-inst_fewshot.csv",
    "olmo-inst-chat": "phase3_olmo-inst_chat.csv",
}
CONTRASTS = [  # (label, from, to)
    ("Qwen base -> instruct (fewshot)", "qwen-base", "qwen-inst"),
    ("Llama base -> instruct (fewshot)", "llama-base", "llama-inst"),
    ("OLMo base -> SFT", "olmo-base", "olmo-sft"),
    ("OLMo SFT -> DPO", "olmo-sft", "olmo-dpo"),
    ("OLMo DPO -> Instruct", "olmo-dpo", "olmo-inst"),
    ("OLMo base -> Instruct", "olmo-base", "olmo-inst"),
    ("Qwen instruct: fewshot -> chat", "qwen-inst", "qwen-inst-chat"),
    ("Llama instruct: fewshot -> chat", "llama-inst", "llama-inst-chat"),
    ("OLMo instruct: fewshot -> chat", "olmo-inst", "olmo-inst-chat"),
]


def natural(j, t):
    """Map transformed coordinate j to the reported parameter."""
    t = np.asarray(t, float)
    return 1 / (1 + np.exp(-t)) if j == 0 else np.exp(t)


# ---------------------------------------------------------------- objective

def cell_dev(theta, df):
    pm = np.clip(choice_prob(df, theta, fix_phi=1.0, lam_clip=LAM_CLIP), 1e-6, 1 - 1e-6)
    y = df["p_obs"].values
    return y * np.log(y / pm) + (1 - y) * np.log((1 - y) / (1 - pm))


def dev(theta, df):
    return cell_dev(theta, df).sum()


def minimize_box(f, x0, bounds):
    res = optimize.minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8})
    # Nelder-Mead polish inside the box guards against L-BFGS-B stalling on flat ridges
    clip = lambda x: np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
    nm = optimize.minimize(lambda x: f(clip(x)), res.x, method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
    x = clip(nm.x)
    fx = f(x)
    return (x, fx) if fx < res.fun else (res.x, res.fun)


def full_fit(df, n_starts=16, seed=0):
    rng = np.random.default_rng(seed)
    x0 = np.array([0.5, np.log(2.0), np.log(1.5), np.log(0.2), 0.0])
    bounds = [tuple(b) for b in BOX]
    best = None
    for s in range(n_starts):
        start = x0 if s == 0 else np.clip(x0 + rng.normal(0, [2, 2, 1, 2, 2]), BOX[:, 0], BOX[:, 1])
        x, fx = minimize_box(lambda th: dev(th, df), start, bounds)
        if best is None or fx < best[1]:
            best = (x, fx)
    return best


def profile(df, j, theta_hat, dev_hat, n_side=30):
    """Profile transformed parameter j across the full box, continuing outward from the MLE."""
    lo, hi = BOX[j]
    free = [k for k in range(5) if k != j]
    bounds = [tuple(BOX[k]) for k in free]
    rows = [(theta_hat[j], dev_hat)]
    for edge in (lo, hi):
        span = edge - theta_hat[j]
        if abs(span) < 1e-9:
            continue
        warm = theta_hat[free].copy()
        for u in np.linspace(0, 1, n_side + 1)[1:] ** 2:
            tj = theta_hat[j] + u * span

            def f(xf, tj=tj):
                th = np.empty(5)
                th[j], th[free] = tj, xf
                return dev(th, df)

            cands = [minimize_box(f, warm, bounds), minimize_box(f, theta_hat[free], bounds)]
            xf, fx = min(cands, key=lambda c: c[1])
            warm = xf
            rows.append((tj, fx))
    prof = pd.DataFrame(rows, columns=["t", "dev"]).sort_values("t").drop_duplicates("t")
    return prof


# ---------------------------------------------------------------- calibration

def calibration(df, theta):
    n, k = len(df), 5
    pm = np.clip(choice_prob(df, theta, fix_phi=1.0, lam_clip=LAM_CLIP), 1e-6, 1 - 1e-6)
    y = df["p_obs"].values
    phi_hat = ((y - pm) ** 2 / (pm * (1 - pm))).sum() / (n - k)

    H = num_hessian(lambda th: dev(th, df), theta)
    eps = 1e-5
    scores = np.empty((n, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = eps
        scores[:, i] = (cell_dev(theta + e, df) - cell_dev(theta - e, df)) / (2 * eps)
    cl = (df["instrument"] + "_" + df["gamble_id"].astype(str)).values
    S = pd.DataFrame(scores).groupby(cl).sum().values
    B = S.T @ S
    cond = np.linalg.cond(H)
    c = {}
    if np.isfinite(cond) and cond < 1e10:
        Hinv = np.linalg.inv(H)
        V = Hinv @ B @ Hinv
        for name, j in PROFILED.items():
            c[name] = V[j, j] / Hinv[j, j] if Hinv[j, j] > 0 else np.nan
    else:
        c = {name: np.nan for name in PROFILED}
    return phi_hat, c, cond


def scaled(prof, scale):
    return 2 * (prof["dev"].values - prof["dev"].min()) / scale


def interval(t, S, t_hat):
    """Where S (sorted by t) crosses CRIT on each side of t_hat; None = open to the box edge."""
    def side(mask, reverse):
        tt, ss = t[mask], S[mask]
        if reverse:
            tt, ss = tt[::-1], ss[::-1]
        for a in range(1, len(tt)):
            if ss[a] > CRIT >= ss[a - 1]:
                return tt[a - 1] + (CRIT - ss[a - 1]) * (tt[a] - tt[a - 1]) / (ss[a] - ss[a - 1])
            if ss[a - 1] > CRIT:  # MLE side already above crit: numerical edge case
                return tt[a - 1]
        return None
    lo = side(t <= t_hat, reverse=True)
    hi = side(t >= t_hat, reverse=False)
    return lo, hi


# ---------------------------------------------------------------- per-subject driver

def run_subject(args):
    key, fname, quick = args
    df = load(ROOT / "data" / fname)
    core = df[df["frame"].isin(["risk", "neutral", "gain", "mixed"])].copy()
    theta, dev_hat = full_fit(core, n_starts=6 if quick else 16)
    phi_hat, c, cond = calibration(core, theta)
    out = {"key": key, "theta": theta, "dev": dev_hat, "phi_hat": phi_hat, "c": c,
           "cond": cond, "n": len(core), "profiles": {}}
    for name, j in PROFILED.items():
        out["profiles"][name] = profile(core, j, theta, dev_hat, n_side=10 if quick else 30)
    return out


def at_edge(theta):
    return [NAMES[k] for k in range(5) if min(abs(theta[k] - BOX[k, 0]), abs(theta[k] - BOX[k, 1])) < 1e-3]


def contrast(pb, pi_, jn, sb, si, ratio):
    """Profile of d = x_i - x_b (level) or log(x_i / x_b) (ratio), from two independent profiles.

    Returns the 95% set's endpoints on the grid. Whether an endpoint is real or an artifact
    of the box is decided by the caller from the two subjects' own intervals."""
    tb, ti = pb["t"].values, pi_["t"].values
    if ratio:  # transformed coords are logs for lambda, gamma
        xb, xi = tb, ti
    else:
        xb, xi = natural(jn, tb), natural(jn, ti)
    ob, oi = np.argsort(xb), np.argsort(xi)
    xb, Sb, xi, Si = xb[ob], sb[ob], xi[oi], si[oi]
    d_grid = np.unique((xi[:, None] - xb[None, :]).ravel())
    if len(d_grid) > 4000:
        d_grid = np.unique(np.quantile(d_grid, np.linspace(0, 1, 4000)))
    f = np.empty(len(d_grid))
    for a, d in enumerate(d_grid):
        v1 = Sb + np.interp(xb + d, xi, Si, left=np.inf, right=np.inf)
        v2 = np.interp(xi - d, xb, Sb, left=np.inf, right=np.inf) + Si
        f[a] = min(v1.min(), v2.min())
    ok = d_grid[f <= CRIT]
    if not len(ok):
        return np.nan, np.nan
    return ok.min(), ok.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from profiles.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(pd.read_csv(OUT / "profiles.csv"))
        return

    jobs = [(k, f, args.quick) for k, f in SUBJECTS.items()]
    with ProcessPoolExecutor(args.workers) as ex:
        res = {r["key"]: r for r in ex.map(run_subject, jobs)}

    fits, ints, profs = [], [], []
    for key in SUBJECTS:
        r = res[key]
        th = r["theta"]
        edge = at_edge(th)
        fits.append({"subject": key, "n": r["n"], "dev": r["dev"], "phi_hat": r["phi_hat"],
                     "hess_cond": r["cond"], "alpha": natural(0, th[0]), "lambda": natural(1, th[1]),
                     "gamma": natural(2, th[2]), "mu": np.exp(th[3]), "delta": th[4],
                     "at_wide_bound": ",".join(edge)})
        for name, j in PROFILED.items():
            p = r["profiles"][name]
            t = p["t"].values
            # the sandwich is only valid at an interior optimum with a usable Hessian
            cal = "cluster" if (not edge and np.isfinite(r["c"][name])) else "quasi"
            p = p.assign(subject=key, param=name, value=natural(j, t),
                         S_quasi=scaled(p, r["phi_hat"]),
                         S_cluster=scaled(p, r["c"][name]) if np.isfinite(r["c"][name]) else np.nan,
                         calibration=cal)
            p["S"] = p[f"S_{cal}"]
            profs.append(p)
            lo, hi = interval(t, p["S"].values, th[j])
            ints.append({"subject": key, "param": name, "mle": natural(j, th[j]), "calibration": cal,
                         "lo": natural(j, lo if lo is not None else BOX[j, 0]),
                         "hi": natural(j, hi if hi is not None else BOX[j, 1]),
                         "lo_open": lo is None, "hi_open": hi is None,
                         "phi_hat": r["phi_hat"], "c_cluster": r["c"][name],
                         # >0 means the profile found a lower deviance than the full fit
                         "profile_improves_fit": r["dev"] - p["dev"].min()})

    fits, ints, profs = pd.DataFrame(fits), pd.DataFrame(ints), pd.concat(profs)
    iv = ints.set_index(["subject", "param"])

    cons = []
    for label, a, b in CONTRASTS:
        for name, j in PROFILED.items():
            pa = profs[(profs.subject == a) & (profs.param == name)]
            pb = profs[(profs.subject == b) & (profs.param == name)]
            ia, ib = iv.loc[(a, name)], iv.loc[(b, name)]
            ma, mb = ia["mle"], ib["mle"]
            for ratio in ([False, True] if name != "alpha" else [False]):
                lo, hi = contrast(pa, pb, j, pa["S"].values, pb["S"].values, ratio)
                # d = x_b - x_a is open above if b is open above or a open below, and vice versa.
                # On the level scale the lower box edges (lambda 1e-4, gamma 0.018) and both alpha
                # edges sit at the parameter's natural limit, so only upper edges of lambda and
                # gamma leave a level contrast open. On the log-ratio scale every edge is arbitrary.
                if ratio:
                    up, down = ib["hi_open"] or ia["lo_open"], ib["lo_open"] or ia["hi_open"]
                elif name == "alpha":
                    up = down = False
                else:
                    up, down = bool(ib["hi_open"]), bool(ia["hi_open"])
                cons.append({"contrast": label, "param": name,
                             "scale": "log ratio" if ratio else "level",
                             "calibration": f"{ia['calibration']}/{ib['calibration']}",
                             "estimate": np.log(mb / ma) if ratio else mb - ma,
                             "lo": -np.inf if down else lo, "hi": np.inf if up else hi,
                             "excludes_0": bool((lo > 0 and not down) or (hi < 0 and not up))})
    cons = pd.DataFrame(cons)

    fits.to_csv(OUT / "fits.csv", index=False)
    ints.to_csv(OUT / "intervals.csv", index=False)
    cons.to_csv(OUT / "contrasts.csv", index=False)
    profs.to_csv(OUT / "profiles.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 200)
    print("\n== wide-bound fits ==")
    print(fits.round(4).to_string(index=False))
    print("\n== per-subject 95% profile intervals ==")
    print(ints.round(4).to_string(index=False))
    print("\n== contrasts (to - from) ==")
    print(cons.round(4).to_string(index=False))
    plot(profs)


def plot(profs):
    """Profile curves per family. Solid: cluster-calibrated; dashed: dispersion fallback.
    Red line: 95% cutoff. Redraw from saved output with --plot-only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("OLMo-2 staircase (fewshot)", ["olmo-base", "olmo-sft", "olmo-dpo", "olmo-inst"]),
              ("Llama-3.1 (fewshot)", ["llama-base", "llama-inst"]),
              ("Qwen2.5 (fewshot)", ["qwen-base", "qwen-inst"])]
    colors = ["#A3A3AC", "#758C58", "#3F5230", "#1C1C1E"]
    fig, axes = plt.subplots(len(groups), 3, figsize=(11, 8.5), sharey=True)
    for r, (title, keys) in enumerate(groups):
        for c, name in enumerate(PROFILED):
            ax = axes[r, c]
            for k, col in zip(keys, colors):
                p = profs[(profs.subject == k) & (profs.param == name)].sort_values("value")
                ls = "-" if p["calibration"].iloc[0] == "cluster" else "--"
                ax.plot(p["value"], np.minimum(p["S"], 40), ls, color=col, lw=1.2, label=k)
            ax.axhline(CRIT, color="#C00000", lw=0.5)
            if name != "alpha":
                ax.set_xscale("log")
            ax.set_ylim(0, 40)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{title}\nscaled profile deviance", fontsize=7)
            if c == 2:
                ax.legend(fontsize=6, frameon=False)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
