"""Are the stated comparisons in reasoning traces true? (reason-then-answer arm)

Extracts every explicit numeric comparison from each trace ("$152.04, which is less than the
guaranteed $117") and checks it against the numbers. A comparison is classed EV-vs-sure when
one number is within 2% of the gamble's expected value p*hi and the other within 2% of the
sure amount. For those, the question is directional: when the truth favors the gamble
(EV > sure), how often is the statement reversed to favor the safe option, and vice versa?
Random arithmetic slips would reverse both directions at similar rates; a reasoning process
that bends the stated comparison toward a preferred answer would reverse one direction only.

Second check (traces with a comparison against the sure amount, regex on the wording, which
covers more phrasings than the numeric parser): in cells where the gamble is better, a
clustered logit of "stated less than" on the model's immediate-answer P(gamble) for the same
cell, the frame, and log EV ratio.

Coverage caveat: the parser only sees comparisons written with both numbers; traces that
compare in words ("higher than the certain amount") are covered by the second check only.

Usage:
    python analysis/trace_comparisons.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))
from score_induced import load, score  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
LESS = {"less", "lower", "smaller"}
NUM = r"\$?\s?(\d[\d,]*(?:\.\d+)?)"
FILLER = (r"(?:(?:the|a|an|that|of|in|from|for|this)\s+|(?:guaranteed|certain|sure|fixed|amount|"
          r"option|alternative|payout|value|expected|outcome|term|arrangement|choice|[A-B])\b[\s,]*){0,6}")
NUMERIC = re.compile(NUM + r"\)?,?\s*(?:\([^)]{0,20}\)\s*)?(?:which\s+)?(?:is\s+|are\s+)?"
                     r"(?:slightly\s+|significantly\s+|much\s+|far\s+|still\s+)?"
                     r"(less|lower|smaller|greater|higher|larger|more)\s+than\s+" + FILLER + NUM, re.I)
WORDED = re.compile(r"(less|lower|greater|higher|more)\s+than\s+(?:the\s+)?(?:guaranteed|certain|sure)", re.I)

RUNS = [
    "qwen-inst_chat_reason", "qwen-inst_chat_reason_riskneutral",
    "qwen-inst_chat_reason_clarify", "qwen-inst_chat_reason_riskneutral_clarify",
    "olmo-sft_fewshot_reason", "olmo-sft_fewshot_reason_riskneutral",
    "olmo-dpo_fewshot_reason", "olmo-dpo_fewshot_reason_riskneutral",
    "olmo-inst_fewshot_reason", "olmo-inst_fewshot_reason_riskneutral",
    "olmo-inst_fewshot_reason_clarify", "olmo-inst_fewshot_reason_riskneutral_clarify",
    "llama-inst_chat_reason", "llama-inst_chat_reason_riskneutral",
]


def _f(x):
    return float(x.replace(",", ""))


def numeric_statements(d):
    rows = []
    for _, r in d.iterrows():
        ev, s = r.p * r.hi, r.sure
        near = lambda x, t: abs(x - t) <= 0.02 * max(t, 1)  # noqa: E731
        for m in NUMERIC.finditer(r.reasoning if isinstance(r.reasoning, str) else ""):
            a, w, b = _f(m.group(1)), m.group(2).lower(), _f(m.group(3))
            if a == b:
                continue
            said_a_smaller = w in LESS
            true = (a < b) if said_a_smaller else (a > b)
            if near(a, ev) and near(b, s):
                ev_first = True
            elif near(a, s) and near(b, ev):
                ev_first = False
            else:
                rows.append(dict(frame=r.frame, kind="other", true=true, truth_favors=None))
                continue
            truth_favors = "gamble" if ev > s else "safe"
            rows.append(dict(frame=r.frame, kind="ev_vs_sure", true=true, truth_favors=truth_favors))
    return pd.DataFrame(rows, columns=["frame", "kind", "true", "truth_favors"])


def reversal_table(st):
    e = st[st["kind"] == "ev_vs_sure"]
    out = {}
    for fav in ("gamble", "safe"):
        x = e[e["truth_favors"] == fav]
        out[fav] = (int((~x["true"]).sum()), len(x))
    return out


def worded_logit(stem, immediate_stem):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = load(DATA / f"phase3_{stem}.csv")
    sc = score(d, "riskneutral", 0.05)
    d = d.assign(clear=sc["clear"].values)
    d = d[(d["frame"] != "dominant") & (d["frame"] != "mixed") & d["clear"] & (d.p * d.hi > d.sure)].copy()
    d["cmp"] = d["reasoning"].fillna("").map(lambda s: (m := WORDED.search(s)) and m.group(1).lower())
    d = d[d["cmp"].notna()].copy()
    d["false"] = d["cmp"].isin(["less", "lower"]).astype(int)
    imm = load(DATA / f"phase3_{immediate_stem}.csv")[["instrument", "trial_id", "p_gamble"]]
    d = d.merge(imm.rename(columns={"p_gamble": "pg_immediate"}), on=["instrument", "trial_id"])
    d["log_ev_ratio"] = np.log(d.p * d.hi / d.sure)
    print(f"\n{stem}: {len(d)} traces with a worded comparison against the sure amount where the gamble "
          f"is better; stated 'less' (false) in {d['false'].sum()} ({d['false'].mean():.3f})")
    print("  false rate by frame:", d.groupby("frame")["false"].mean().round(3).to_dict())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.glm("false ~ pg_immediate + log_ev_ratio + C(frame)", d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d["gamble_id"]})
    print("  clustered logit (gamble clusters):")
    for k in m.params.index:
        print(f"    {k:22s} {m.params[k]:+.3f}  se {m.bse[k]:.3f}  p {m.pvalues[k]:.3f}")


def main():
    print("Numeric EV-vs-sure comparisons: reversed / total, by which option the truth favors")
    print(f"{'run':46s} {'truth=gamble':>14s} {'truth=safe':>12s}  Fisher p   other false")
    for stem in RUNS:
        path = DATA / f"phase3_{stem}.csv"
        if not path.exists():
            continue
        d = load(path)
        d = d[d["frame"] != "dominant"]
        st = numeric_statements(d)
        t = reversal_table(st)
        (fg, ng), (fs, ns) = t["gamble"], t["safe"]
        p = fisher_exact([[fg, ng - fg], [fs, ns - fs]])[1] if ng and ns else float("nan")
        o = st[st["kind"] == "other"]
        print(f"{stem:46s} {fg:4d} / {ng:<6d}  {fs:4d} / {ns:<5d}  {p:8.2g}   {int((~o['true']).sum())} / {len(o)}")
    worded_logit("qwen-inst_chat_reason", "qwen-inst_chat")
    if (DATA / "phase3_llama-inst_chat_reason.csv").exists():
        worded_logit("llama-inst_chat_reason", "llama-inst_chat")


if __name__ == "__main__":
    main()
