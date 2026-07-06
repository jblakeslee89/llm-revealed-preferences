"""Phase 3 estimation: fractional MLE on elicited probabilities, base-vs-instruct contrast.

The Phase 3 data are choice PROBABILITIES (one per cell), not Bernoulli draws. This
estimator fits the same structural models as Phases 1-2 by minimizing binomial deviance
with the observed probability as the target (fractional-response MLE), then contrasts the
recovered parameters across a base/instruct pair.

Reuses the Phase 2 reference-dependent + Prelec likelihood (the Phase 1 risk cells are the
gain/neutral 'risk' rows with anchor 0, so a single model covers both instruments).

Usage:
    # recover truth from the simulated dry run (validation gate 1)
    python analysis/estimate_from_probs.py data/phase3_dryrun.csv

    # attribution: pass base then instruct
    python analysis/estimate_from_probs.py data/phase3_llama-base_fewshot.csv \\
        --instruct data/phase3_llama-inst_fewshot.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# reuse the Phase 2 structural machinery
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from estimate_reference import choice_prob, unpack  # noqa: E402
from scipy import optimize  # noqa: E402


def load(path):
    df = pd.read_csv(path)
    df = df[df["excluded"] == 0].copy()
    df = df[df["p_gamble"].notna()]
    df["safe_first"] = df["safe_first"].astype(bool)
    # clip away exact 0/1 so the deviance is finite
    df["p_obs"] = df["p_gamble"].clip(1e-4, 1 - 1e-4)
    return df


def frac_deviance(theta, df, fix_phi=None):
    """Binomial deviance with a continuous target (fractional MLE objective)."""
    pm = choice_prob(df, theta, fix_phi)
    pm = np.clip(pm, 1e-6, 1 - 1e-6)
    y = df["p_obs"].values
    dev = y * np.log(y / pm) + (1 - y) * np.log((1 - y) / (1 - pm))
    return dev.sum()


def fit(df, fix_phi=1.0, n_starts=8, seed=0):
    rng = np.random.default_rng(seed)
    k = 5 if fix_phi is not None else 6
    x0 = np.array([0.5, np.log(2.0), np.log(1.5), np.log(0.2), 0.0]
                  + ([] if fix_phi is not None else [0.0]))
    best = None
    for j in range(n_starts):
        s = x0 + (0 if j == 0 else rng.normal(0, 0.5, k))
        res = optimize.minimize(frac_deviance, s, args=(df, fix_phi), method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 12000})
        if best is None or res.fun < best.fun:
            best = res
    return best


def params_of(theta, fix_phi=1.0):
    alpha, lam, gamma, mu, delta, phi = unpack(theta, fix_phi)
    return {"alpha": alpha, "lambda": lam, "gamma": gamma, "mu": mu, "delta": delta, "phi": phi}


def bootstrap(df, fix_phi=1.0, B=300, seed=1):
    """Resample design cells; return param draws for CIs and contrasts."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    draws = []
    for b in range(B):
        samp = df.iloc[rng.choice(idx, len(idx), replace=True)]
        try:
            draws.append(params_of(fit(samp, fix_phi, n_starts=3, seed=b).x, fix_phi))
        except Exception:
            pass
    return pd.DataFrame(draws)


def frame_gap(df):
    """Model-free gain-vs-mixed mean-probability gap (Phase 2 headline, in prob units)."""
    g = df[df["frame"] == "gain"]["p_gamble"].mean()
    m = df[df["frame"] == "mixed"]["p_gamble"].mean()
    return g, m, m - g


def summarize(name, df, do_boot=True):
    print(f"\n===== {name}  (n={len(df)} cells) =====")
    dom = df[df["frame"] == "dominant"]
    if len(dom):
        print(f"  dominant control uptake: {dom['p_gamble'].mean():.3f}")
    core = df[df["frame"].isin(["risk", "neutral", "gain", "mixed"])].copy()
    g, m, gap = frame_gap(df)
    print(f"  frame gap (mixed - gain): {gap:+.3f}   [gain {g:.3f}, mixed {m:.3f}]")
    fit_res = fit(core, fix_phi=1.0)
    pars = params_of(fit_res.x, 1.0)
    print(f"  structural (phi=1):  lambda={pars['lambda']:.3f}  gamma={pars['gamma']:.3f}  "
          f"alpha={pars['alpha']:.3f}  delta={pars['delta']:+.3f}")
    boot = bootstrap(core, B=200) if do_boot else None
    return {"params": pars, "gap": gap, "boot": boot, "core": core}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="base-model CSV (or the dry-run CSV)")
    ap.add_argument("--instruct", default=None, help="instruct-model CSV for the contrast")
    ap.add_argument("--label-base", default="BASE")
    ap.add_argument("--label-instruct", default="INSTRUCT")
    args = ap.parse_args()

    b = summarize(args.label_base, load(args.base))

    if not args.instruct:
        print("\n(no --instruct given: single-model summary only. For a dry run, compare"
              " the printed params to the --sim values used to generate the data.)")
        return

    i = summarize(args.label_instruct, load(args.instruct))

    print("\n===== ATTRIBUTION: instruct - base =====")
    for k in ("lambda", "gamma", "alpha"):
        d = i["params"][k] - b["params"][k]
        line = f"  Delta {k:>6} = {d:+.3f}"
        if b["boot"] is not None and i["boot"] is not None and k in b["boot"] and k in i["boot"]:
            bd = i["boot"][k].sample(len(i["boot"]), replace=True, random_state=0).values \
                 - b["boot"][k].sample(len(b["boot"]), replace=True, random_state=1).values
            lo, hi = np.percentile(bd, [2.5, 97.5])
            line += f"   95% CI [{lo:+.3f}, {hi:+.3f}]" + ("  *" if lo * hi > 0 else "")
        print(line)
    dgap = i["gap"] - b["gap"]
    print(f"  Delta frame-gap = {dgap:+.3f}")
    print("\n  (* = bootstrap CI excludes 0: post-training moved this parameter.)")
    print("  Reading: a large negative Delta lambda or Delta frame-gap in the instruct"
          " direction means post-training INSTILLED loss aversion / framing sensitivity.")


if __name__ == "__main__":
    main()
