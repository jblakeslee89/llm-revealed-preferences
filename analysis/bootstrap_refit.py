"""Phase 3: model-based bootstrap, clustered by gamble, for the wide-bounds structural fits.

Why: the profile-likelihood refit (profile_refit.py) scales its intervals by a gamble-clustered
sandwich, but where a parameter sits at the edge of the wide box the sandwich is invalid and the
script falls back to Pearson dispersion, which ignores the correlation among a gamble's ten
presentations. Five subjects are affected (qwen-inst, qwen-inst-chat, llama-base, llama-inst-chat,
olmo-base), including the OLMo base model that anchors the base -> SFT contrast. This script gives
every subject, boundary or not, an interval calibrated the same way, and checks the clustered
profile intervals of the interior subjects against it.

Scheme (wild cluster bootstrap around the fitted model, on the response scale):
    r_c   = y_c - p_hat_c                               residual of cell c
    y*_c  = p_hat_c + w_g(c) * r_c                      w_g = +/-1 (Rademacher), one per gamble
clipped to [1e-4, 1 - 1e-4] (the share of clipped cells is logged). E[y*] = p_hat, so the fitted
structural model supplies the mean and the gamble's own residual pattern, flipped as a block,
supplies the noise. (A logit-scale version was tried first and dropped: it is not mean-preserving
on the probability scale the fractional objective works on, and its replicates were visibly
off-center.) Within-gamble correlation (templates, orders, frames) is preserved
exactly and no distribution is assumed for it. Each replicate is refit with the same objective,
box and starts as profile_refit.py. Percentile intervals.

Contrasts: the 80 gambles are the same in every subject, and a gamble that is hard for one checkpoint
is plausibly hard for the next. So one weight vector w is drawn per replicate and applied to every
subject, and a contrast's interval is the percentile interval of the paired replicate differences.
(The profile contrasts treat subjects as independent; pairing is the less conservative choice only
when residuals correlate positively across subjects, which the script reports.)

Caveat: at a boundary MLE the bootstrap is itself not consistent (Andrews 2000); the replicate
distribution piles up at the edge. Intervals for boundary parameters are reported with the share of
replicates at the edge, and read as a check on the dispersion intervals, not as exact coverage.

Specifications:
    --spec base        the profile_refit.py model (default)
    --spec intercept   the same plus a constant c on the choice index; template_effects.py shows
                       that c and lambda are close to collinear, so this spec gets its own intervals
                       (c is bootstrapped too and reported alongside lambda, gamma, alpha)

Usage:
    python analysis/bootstrap_refit.py                        # B = 400, base spec
    python analysis/bootstrap_refit.py --spec intercept       # results/bootstrap_refit_intercept/
    python analysis/bootstrap_refit.py --B 20                 # smoke test
Outputs: results/bootstrap_refit[_intercept]/{draws,intervals,contrasts,compare}.csv
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
from estimate_reference import choice_prob  # noqa: E402
from profile_refit import (BOX, CONTRASTS, LAM_CLIP, NAMES, PROFILED, SUBJECTS,  # noqa: E402
                           full_fit, minimize_box, dev, natural)
from template_effects import index_z, kl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "bootstrap_refit"
PREV = ROOT / "results" / "profile_refit"
CORE = ["risk", "neutral", "gain", "mixed"]


def logit(p):
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def core_of(fname):
    df = load(ROOT / "data" / fname)
    core = df[df["frame"].isin(CORE)].copy().reset_index(drop=True)
    core["cluster"] = core["instrument"] + "_" + core["gamble_id"].astype(str)
    return core


SPEC = "base"
C_BOX = (-10.0, 10.0)


def box():
    return BOX if SPEC == "base" else np.vstack([BOX, [C_BOX]])


def objective(th, df):
    if SPEC == "base":
        return dev(th, df)
    return kl(index_z(df, th[:5]) + th[5], df["p_obs"].values).sum()


def fitted_p(df, th):
    if SPEC == "base":
        return choice_prob(df, th, fix_phi=1.0, lam_clip=LAM_CLIP)
    return expit(np.clip(index_z(df, th[:5]) + th[5], -35, 35))


def fit_spec(df, n_starts=16, seed=0):
    theta, d = full_fit(df, n_starts=n_starts)
    if SPEC == "base":
        return theta, d
    rng = np.random.default_rng(seed)
    B_ = box()
    best = None
    for s in range(n_starts):
        x0 = np.r_[theta, 0.0] + (0 if s == 0 else rng.normal(0, [2, 2, 1, 2, 2, 1]))
        cand = minimize_box(lambda x: objective(x, df), np.clip(x0, B_[:, 0], B_[:, 1]),
                            [tuple(b) for b in B_])
        if best is None or cand[1] < best[1]:
            best = cand
    return best


def refit(df, theta_hat, rng):
    """Warm start at the original MLE, plus two jittered starts; keep the best."""
    B_ = box()
    bounds = [tuple(b) for b in B_]
    f = lambda th: objective(th, df)
    best = minimize_box(f, theta_hat, bounds)
    sd = [1.5, 1.5, 0.7, 1.0, 1.0] + ([] if SPEC == "base" else [0.7])
    for _ in range(2):
        s = np.clip(theta_hat + rng.normal(0, sd), B_[:, 0], B_[:, 1])
        cand = minimize_box(f, s, bounds)
        if cand[1] < best[1]:
            best = cand
    return best


def run_subject(args):
    key, fname, W, clusters, seed, spec = args
    global SPEC
    SPEC = spec
    core = core_of(fname)
    theta, dev_hat = fit_spec(core)
    p_hat = np.clip(fitted_p(core, theta), 1e-4, 1 - 1e-4)
    r = core["p_obs"].values - p_hat
    idx = pd.Index(clusters).get_indexer(core["cluster"])
    assert (idx >= 0).all()
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(W.shape[0]):
        y_raw = p_hat + W[b, idx] * r
        y = y_raw.clip(1e-4, 1 - 1e-4)
        star = core.assign(p_obs=y)
        th, fx = refit(star, theta, rng)
        rows.append({"subject": key, "b": b, "dev": fx, "clip_share": float(np.mean(y != y_raw)),
                     **{n: natural(j, th[j]) for n, j in PROFILED.items()},
                     **{f"t_{n}": th[k] for k, n in enumerate(NAMES)},
                     **({"c": th[5]} if SPEC != "base" else {})})
    return key, theta, r, core["cluster"].values, pd.DataFrame(rows)


def edge_share(t, j):
    lo, hi = BOX[j]
    return float(np.mean((np.abs(t - lo) < 1e-2) | (np.abs(t - hi) < 1e-2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--spec", choices=["base", "intercept"], default="base")
    ap.add_argument("--subjects", default="all", help="comma-separated subject keys, or 'all'")
    args = ap.parse_args()
    global SPEC, OUT, SUBJ
    SPEC = args.spec
    if SPEC != "base":
        OUT = OUT.parent / f"bootstrap_refit_{SPEC}"
    SUBJ = SUBJECTS if args.subjects == "all" else {k: SUBJECTS[k] for k in args.subjects.split(",")}
    OUT.mkdir(parents=True, exist_ok=True)

    clusters = sorted(core_of(SUBJECTS["olmo-sft"])["cluster"].unique())
    rng = np.random.default_rng(args.seed)
    W = rng.choice([-1.0, 1.0], size=(args.B, len(clusters)))  # shared across subjects

    jobs = [(k, f, W, clusters, args.seed + i, SPEC) for i, (k, f) in enumerate(SUBJ.items())]
    with ProcessPoolExecutor(args.workers) as ex:
        res = {k: (th, r, cl, d) for k, th, r, cl, d in ex.map(run_subject, jobs)}

    draws = pd.concat([res[k][3] for k in SUBJ], ignore_index=True)
    params = dict(PROFILED, **({"c": 5} if SPEC != "base" else {}))
    nat = lambda j, t: t if j == 5 else natural(j, t)
    draws.to_csv(OUT / "draws.csv", index=False)

    ints = []
    for key in SUBJ:
        th, d = res[key][0], draws[draws.subject == key]
        for name, j in params.items():
            v = d[name].values
            ints.append({"subject": key, "param": name, "mle": nat(j, th[j]),
                         "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5),
                         "boot_median": np.median(v),
                         "edge_share": edge_share(d[f"t_{NAMES[j]}"].values, j) if j < 5 else np.nan,
                         "B": len(v)})
    ints = pd.DataFrame(ints)
    ints.to_csv(OUT / "intervals.csv", index=False)

    # cross-subject residual correlation within gambles (justifies pairing)
    def gamble_resid(key):
        _, r, cl, _ = res[key]
        return pd.Series(r).groupby(cl).mean()

    cons = []
    for label, a, b in CONTRASTS:
        if a not in SUBJ or b not in SUBJ:
            continue
        da = draws[draws.subject == a].set_index("b")
        db = draws[draws.subject == b].set_index("b")
        rho = np.corrcoef(gamble_resid(a), gamble_resid(b).reindex(gamble_resid(a).index))[0, 1]
        for name, j in params.items():
            ma = nat(j, res[a][0][j])
            mb = nat(j, res[b][0][j])
            diff = (db[name] - da[name]).values
            # unpaired version for comparison with the profile contrasts
            unp = db[name].values - np.random.default_rng(0).permutation(da[name].values)
            cons.append({"contrast": label, "param": name, "estimate": mb - ma,
                         "lo": np.percentile(diff, 2.5), "hi": np.percentile(diff, 97.5),
                         "lo_unpaired": np.percentile(unp, 2.5), "hi_unpaired": np.percentile(unp, 97.5),
                         "excludes_0": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0),
                         "resid_corr": rho})
    cons = pd.DataFrame(cons)
    cons.to_csv(OUT / "contrasts.csv", index=False)

    # side by side with the profile intervals
    comp = []
    if SPEC == "base" and (PREV / "intervals.csv").exists():
        prof = pd.read_csv(PREV / "intervals.csv").set_index(["subject", "param"])
        for _, row in ints.iterrows():
            p = prof.loc[(row.subject, row.param)]
            comp.append({"subject": row.subject, "param": row.param, "mle": row.mle,
                         "profile_cal": p["calibration"], "profile_lo": p["lo"], "profile_hi": p["hi"],
                         "boot_lo": row.lo, "boot_hi": row.hi, "edge_share": row.edge_share,
                         "width_ratio": (row.hi - row.lo) / (p["hi"] - p["lo"]) if p["hi"] > p["lo"] else np.nan})
        comp = pd.DataFrame(comp)
        comp.to_csv(OUT / "compare.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 300)
    print(f"spec = {SPEC}; B = {args.B} wild-cluster replicates, {len(clusters)} gamble clusters, weights shared across subjects")
    print("\n== per-subject 95% percentile intervals ==")
    print(ints.round(4).to_string(index=False))
    print("\n== contrasts (to - from), paired replicates ==")
    print(cons.round(4).to_string(index=False))
    if len(comp):
        print("\n== bootstrap vs profile intervals ==")
        print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
