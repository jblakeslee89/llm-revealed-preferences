"""Did payment change Qwen's written verdict habits? (arm 5 follow-up, Oct 2026)

Unpaid, Qwen's uninstructed reasoning often computes expected value correctly and then writes
the comparison backwards, always in favor of the sure amount, almost always through one stock
phrase: "$E, which is less than the guaranteed $S" (trace_comparisons.py, prefill test). This
script asks whether paying the model (colab/rl_induce_reason.py) removed the habit.

Two sources:
  1. The 840-cell evaluations: unpaid untold reasoning, unpaid told-the-rule reasoning, and the
     paid (untold) linear and sqrt adapters.
  2. The training monitor traces (results/paid_reasoning/<run>/monitor_traces.jsonl): 48
     held-out prompts every 25 steps, for the time course.

Measures, on non-mixed frames (the loss-worded frame has its own reference-point error):
  rev_gamble   numeric EV-vs-sure statements reversed when the truth favors the gamble
  rev_safe     the same when the truth favors the sure amount
  which_is     share of traces whose comparison with the sure amount is written "..., which is
               ___ than"; the construction that carried nearly all false verdicts
  false_less   share of gamble-better traces with a worded comparison that say "less"
  ev_word      share of traces that mention expected value

Usage:
    python analysis/verdict_habits_paid.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import trace_comparisons as tc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "results" / "paid_reasoning"
CMP = re.compile(r"\b(?:is|are)\s+(?:(?:slightly|significantly|much|far|still)\s+)?"
                 r"(less|lower|smaller|greater|higher|larger|more)\s+than\b", re.I)
WHICH = re.compile(r"which\s+(?:is|are)\s+(?:(?:slightly|significantly|much|far|still)\s+)?$", re.I)
SURE_WORDS = re.compile(r"guaranteed|certain|sure", re.I)


def comparison(trace, sure):
    """First comparison word whose object is the sure amount: (word, reached via 'which is')."""
    for m in CMP.finditer(trace or ""):
        tail = trace[m.end():m.end() + 60]
        if f"{sure:g}" in tail or SURE_WORDS.search(tail):
            return m.group(1).lower(), bool(WHICH.search(trace[:m.start(1)]))
    return None, False


def measures(d):
    d = d[d["frame"] != "mixed"].copy()
    d["reasoning"] = d["reasoning"].fillna("")
    st = tc.numeric_statements(d)
    t = tc.reversal_table(st)
    words = [comparison(r.reasoning, float(r.sure)) for r in d.itertuples()]
    d["cmp_word"] = [w for w, _ in words]
    d["via_which"] = [v for _, v in words]
    d["gamble_better"] = d.p * d.hi > d.sure
    with_cmp = d[d.cmp_word.notna()]
    gb = with_cmp[with_cmp.gamble_better]
    return {
        "traces": len(d),
        "rev_gamble": f"{t['gamble'][0]}/{t['gamble'][1]}",
        "rev_safe": f"{t['safe'][0]}/{t['safe'][1]}",
        "which_is": round(with_cmp.via_which.mean(), 3) if len(with_cmp) else float("nan"),
        "false_less": round(gb.cmp_word.isin(["less", "lower", "smaller"]).mean(), 3) if len(gb) else float("nan"),
        "ev_word": round(d.reasoning.str.contains("expected", case=False).mean(), 3),
    }


def main():
    print("840-cell evaluations, Qwen Instruct (chat), non-mixed frames")
    runs = [("unpaid, untold", "phase3_qwen-inst_chat_reason.csv"),
            ("unpaid, told EV rule", "phase3_qwen-inst_chat_reason_riskneutral.csv"),
            ("paid linear, untold", "phase4_qwen-inst_chat_reason_paid-linear.csv"),
            ("paid sqrt, untold", "phase4_qwen-inst_chat_reason_paid-sqrt.csv")]
    rows = []
    for label, f in runs:
        d = tc.load(DATA / f)
        rows.append({"run": label, **measures(d)})
    print(pd.DataFrame(rows).to_string(index=False))

    for run in ("qwen_reason_linear", "qwen_reason_sqrt"):
        p = RUNS / run / "monitor_traces.jsonl"
        if not p.exists():
            continue
        m = pd.DataFrame([json.loads(line) for line in open(p)])
        print(f"\nTraining monitor traces, {run} (48 held-out prompts per step; non-mixed frames)")
        out = [{"step": s, **measures(g)} for s, g in m.groupby("step")]
        print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
