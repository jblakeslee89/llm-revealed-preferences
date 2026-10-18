"""Bounds that turn the Stage A "money doesn't predict advice" result into a finding.

For each open model and regime, money/self against each people domain (self and advise pooled):
  r        cell-level correlation of P(risky) across matched cells (gamble x frame x order)
  r CI     95% gamble-cluster bootstrap interval; r_hi^2 bounds the share of cell-to-cell
           variation in lives answers that the money answers can explain
  shift    P(risky) in the domain minus money/self, with its 95% interval
  ev_gap   agreement with expected value, domain minus money (clear cells)

Usage: python analysis/stage_a_bounds.py [--B 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import DATA, KEY, MODELS, load  # noqa: E402

PEOPLE = ["civilians", "troops", "detainees"]


def core(d):
    return d[(d.excluded == 0) & d.frame.isin(["gain", "mixed"])]


def summarize(base, x):
    a = base.groupby(KEY).p_gamble.mean()
    b = x.groupby(KEY + ["role"]).p_gamble.mean().groupby(KEY).mean()  # pool roles per cell
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    r = j.a.corr(j.b)
    shift = x.p_gamble.mean() - base.p_gamble.mean()
    ca, cx = base[base.clear], x[x.clear]
    ev = ((cx.p_gamble > 0.5) == cx.gamble_better).mean() - ((ca.p_gamble > 0.5) == ca.gamble_better).mean()
    return r, shift, ev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    rows = []
    for label, stem in MODELS:
        for suffix, regime in (("", "immediate"), ("_reason", "reason first")):
            path = DATA / f"stageA_{stem}{suffix}.csv"
            if not path.exists():
                continue
            d = core(load(path))
            base = d[(d.domain == "money") & (d.role == "self")]
            for dom in PEOPLE:
                x = d[d.domain == dom]
                est = summarize(base, x)
                ids = np.array(sorted(set(base.gamble_id) & set(x.gamble_id)))
                gb, gx = dict(tuple(base.groupby("gamble_id"))), dict(tuple(x.groupby("gamble_id")))
                draws = []
                for _ in range(args.B):
                    pick = rng.choice(ids, len(ids), replace=True)
                    # relabel resampled gambles so duplicates stay distinct cells
                    bb = pd.concat([gb[g].assign(gamble_id=k) for k, g in enumerate(pick)])
                    xx = pd.concat([gx[g].assign(gamble_id=k) for k, g in enumerate(pick)])
                    draws.append(summarize(bb, xx))
                lo, hi = np.nanpercentile(np.array(draws), [2.5, 97.5], axis=0)
                rows.append(dict(model=label.split()[0], regime=regime, domain=dom,
                                 r=est[0], r_lo=lo[0], r_hi=hi[0], r2_max=max(hi[0], 0) ** 2,
                                 shift=est[1], shift_lo=lo[1], shift_hi=hi[1],
                                 ev_gap=est[2], ev_lo=lo[2], ev_hi=hi[2]))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


if __name__ == "__main__":
    main()
