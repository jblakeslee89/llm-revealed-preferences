"""Stage A of the advice extension: do the money-lottery habits carry into advisory decisions?

Reads data/stageA_<model>.csv (immediate answer) and data/stageA_<model>_reason.csv
(reason first), produced by colab/phase3_elicit.py on data/stageA_grid*.csv. The instrument
column is A_<domain>_<role>.

For each model, regime, domain and role:
  - dominance: share of free-outcome controls answered for the larger amount (> 0.5)
  - excluded: share of cells with too little probability on A or B (refusal or hedging)
  - risky: mean P(risky option) on core (gain + mixed) cells
  - ev: agreement with expected value on clear core cells (|ln EV ratio| >= 0.05)
  - gap: frame gap, mean P(risky) mixed minus gain
  - order: mean P(risky) when the risky option is listed first minus when it is second

Then, against the money / self cells of the same model and regime:
  - cell-level correlation of P(risky) across matched cells (gamble, frame, order)
  - differences in risky, gap and order with 95% cluster-bootstrap intervals over gambles

Usage:
    python analysis/stage_a.py [--B 1000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = [("Qwen Instruct (chat)", "qwen-inst_chat"), ("Llama Instruct (chat)", "llama-inst_chat"),
          ("OLMo Instruct (few-shot)", "olmo-inst_fewshot")]
DOMAINS = ["money", "civilians", "troops", "detainees"]
ROLES = ["self", "advise"]
KEY = ["gamble_id", "frame", "safe_first"]


def load(path):
    d = pd.read_csv(path)
    d[["_", "domain", "role"]] = d["instrument"].str.split("_", expand=True)
    d["safe_first"] = d["safe_first"].astype(str) == "True"
    d["ev"] = d.p * d.hi
    d["clear"] = np.abs(np.log(d.ev / d.sure)) >= 0.05
    d["gamble_better"] = d.ev > d.sure
    return d


def stats(d):
    ok = d[d.excluded == 0]
    core = ok[ok.frame.isin(["gain", "mixed"])]
    dom = ok[ok.frame == "dominant"]
    clear = core[core.clear]
    return dict(
        n=len(d),
        excluded=(d.excluded != 0).mean(),
        dominance=(dom.p_gamble > 0.5).mean() if len(dom) else np.nan,
        risky=core.p_gamble.mean(),
        ev=((clear.p_gamble > 0.5) == clear.gamble_better).mean(),
        gap=core[core.frame == "mixed"].p_gamble.mean() - core[core.frame == "gain"].p_gamble.mean(),
        order=core[~core.safe_first].p_gamble.mean() - core[core.safe_first].p_gamble.mean(),
    )


def boot_diff(a, b, metric, B, rng):
    """95% interval for stats(b)[metric] - stats(a)[metric], resampling gambles."""
    ids = np.array(sorted(set(a.gamble_id) & set(b.gamble_id)))
    ga, gb = dict(tuple(a.groupby("gamble_id"))), dict(tuple(b.groupby("gamble_id")))
    draws = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        draws.append(stats(pd.concat([gb[i] for i in pick]))[metric]
                     - stats(pd.concat([ga[i] for i in pick]))[metric])
    return np.percentile(draws, [2.5, 97.5])


def report(label, d, B, rng):
    print(f"\n=== {label} ===")
    rows = []
    for dom in DOMAINS:
        for role in ROLES:
            s = stats(d[(d.domain == dom) & (d.role == role)])
            rows.append(dict(domain=dom, role=role, **s))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    base = d[(d.domain == "money") & (d.role == "self") & (d.excluded == 0) & d.frame.isin(["gain", "mixed"])]
    base_cells = base.groupby(KEY).p_gamble.mean()
    print("\n  vs money/self: cell correlation of P(risky); differences with 95% gamble-cluster intervals")
    for dom in DOMAINS:
        for role in ROLES:
            if dom == "money" and role == "self":
                continue
            x = d[(d.domain == dom) & (d.role == role) & (d.excluded == 0) & d.frame.isin(["gain", "mixed"])]
            cells = x.groupby(KEY).p_gamble.mean()
            j = pd.concat([base_cells, cells], axis=1, keys=["a", "b"]).dropna()
            r = j.a.corr(j.b)
            parts = []
            for m in ("risky", "gap", "order"):
                est = stats(x)[m] - stats(base)[m]
                lo, hi = boot_diff(base, x, m, B, rng)
                parts.append(f"{m} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"  {dom:10s} {role:6s} r = {r:.2f} | " + " | ".join(parts))


def phase3_check(stem):
    """Money/self in Stage A against the same instrument-2 cells of the Phase 3 run (other wordings)."""
    a = DATA / f"stageA_{stem}.csv"
    p = DATA / f"phase3_{stem}.csv"
    if not (a.exists() and p.exists()):
        return
    s = load(a)
    s = s[(s.domain == "money") & (s.role == "self") & s.frame.isin(["gain", "mixed"])]
    q = pd.read_csv(p)
    q = q[(q.instrument == "p2") & q.frame.isin(["gain", "mixed"]) & (q.excluded == 0)]
    q["safe_first"] = q["safe_first"].astype(str) == "True"
    j = pd.concat([s.groupby(KEY).p_gamble.mean(), q.groupby(KEY).p_gamble.mean()], axis=1,
                  keys=["stageA", "phase3"]).dropna()
    print(f"  Phase 3 check ({stem}): money/self vs Phase 3 instrument-2 cells, r = {j.stageA.corr(j.phase3):.2f}, "
          f"mean P(risky) {j.stageA.mean():.3f} vs {j.phase3.mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    for label, stem in MODELS:
        for suffix, regime in (("", "immediate answer"), ("_reason", "reason first")):
            path = DATA / f"stageA_{stem}{suffix}.csv"
            if path.exists():
                report(f"{label}, {regime}", load(path), args.B, rng)
        phase3_check(stem)


if __name__ == "__main__":
    main()
