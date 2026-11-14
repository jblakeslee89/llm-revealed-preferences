"""Compact tables for section 5 of the risk-advice paper (frontier Claude runs).

Roles pooled. The 95% interval on the shift resamples gambles (same cluster bootstrap as
analysis/stage_a.py, vectorized; 2,000 draws, seed 1).
For each run (file stem) and domain: P(risky) on core cells, agreement with
expected value on clear cells, free-outcome controls passed, excluded share,
order effect, and the cell-level correlation with money/self.
Usage: python analysis/paper_frontier_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import DATA, KEY, load, stats  # noqa: E402

RUNS = [
    ("Haiku", "snap", "-", "haiku55_api"),
    ("Haiku", "written reasoning", "-", "haiku55_api_reason"),
    ("Haiku", "extended thinking", "low", "haiku55_api_think"),
    ("Haiku", "extended thinking", "high", "haiku55_api_think_high"),
    ("Sonnet", "snap", "-", "sonnet55_api"),
    ("Sonnet", "written reasoning", "-", "sonnet55_api_reason"),
    ("Sonnet", "extended thinking", "low", "sonnet55_api_think"),
    ("Sonnet", "extended thinking", "medium", "sonnet55_api_think_medium"),
    ("Sonnet", "extended thinking", "high", "sonnet55_api_think_high"),
    ("Opus", "extended thinking", "low", "opus55_api_think"),
    ("Opus", "ext. thinking + written reasoning", "low", "opus55_api_reason"),
    ("Opus", "extended thinking", "high", "opus55_api_think_high"),
    ("Fable", "extended thinking", "low", "fable51_api_think"),
    ("Fable", "extended thinking", "high", "fable51_api_think_high"),
    # second wording + release domain (Stage A2)
    ("Haiku", "A2 snap", "-", "haiku55_a2"),
    ("Haiku", "A2 written reasoning", "-", "haiku55_a2_reason"),
    ("Haiku", "A2 extended thinking", "low", "haiku55_a2_think"),
    ("Sonnet", "A2 snap", "-", "sonnet55_a2"),
    ("Sonnet", "A2 written reasoning (84%)", "-", "sonnet55_a2_reason"),
    ("Sonnet", "A2 extended thinking", "low", "sonnet55_a2_think"),
    ("Opus", "A2 extended thinking", "low", "opus55_a2_think"),
    ("Fable", "A2 extended thinking", "low", "fable51_a2_think"),
]


def boot_shift(base, x, B=2000, seed=1):
    a = base.groupby("gamble_id").p_gamble.agg(["sum", "count"])
    b = x.groupby("gamble_id").p_gamble.agg(["sum", "count"])
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids].to_numpy(), b.loc[ids].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(B, len(ids)))
    sa, sb = a[pick].sum(axis=1), b[pick].sum(axis=1)
    draws = sb[:, 0] / sb[:, 1] - sa[:, 0] / sa[:, 1]
    return np.percentile(draws, [2.5, 97.5])


def main():
    rows = []
    for model, arm, effort, stem in RUNS:
        path = DATA / f"stageA_{stem}.csv"
        if not path.exists():
            continue
        d = load(path)
        base = d[(d.domain == "money") & (d.role == "self") & (d.excluded == 0) & d.frame.isin(["gain", "mixed"])]
        bc = base.groupby(KEY).p_gamble.mean()
        for dom in [x for x in ["money", "civilians", "troops", "detainees", "release"] if x in set(d.domain)]:
            x = d[d.domain == dom]
            s = stats(x)
            core = x[(x.excluded == 0) & x.frame.isin(["gain", "mixed"])]
            cc = core.groupby(KEY).p_gamble.mean()
            j = pd.concat([bc, cc], axis=1, keys=["a", "b"]).dropna()
            lo, hi = boot_shift(base, core)
            rows.append(dict(model=model, arm=arm, effort=effort, domain=dom, n=s["n"],
                             risky=s["risky"], shift=s["risky"] - stats(base)["risky"], lo=lo, hi=hi,
                             ev=s["ev"], controls=s["dominance"], excluded=s["excluded"],
                             order=s["order"], gap=s["gap"], r=j.a.corr(j.b)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    t.to_csv(Path(__file__).resolve().parent.parent / "results" / "paper_frontier_tables.csv", index=False)


if __name__ == "__main__":
    main()
