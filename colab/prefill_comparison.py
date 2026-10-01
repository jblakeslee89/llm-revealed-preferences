"""Prefill test for the comparison step in reasoning traces (Oct 2026).

analysis/trace_comparisons.py found that Qwen's uninduced reasoning traces compute the
gamble's expected value correctly and then often misstate the comparison with the sure
amount, always in the direction of the safe option. This script measures that pull
directly instead of inferring it from generated text.

For each trace with a comparison against the sure amount ("... is $152.04, which is less
than the guaranteed $117"), cut the trace just before the comparison word and read the
model's next-token probability on the "less" group (less, lower, smaller) against the
"greater" group (greater, higher, larger, more). Conditions, all on the same cell:

    own      the prompt and the model's own trace up to the comparison word (replays the
             original run; the greedy word should match the generated one)
    rule     the same trace prefix, with the expected-value instruction prepended to the
             prompt (does the instruction act on the verdict itself?)
    minimal  the prompt, then a one-sentence assistant prefix stating the correct expected
             value: "The expected value of Option X is $E, which is" (the verdict with none
             of the model's own preceding text)

Output: one row per cell and condition with p_less = P(less group) / P(less + greater).

USAGE (Colab T4):
    !python colab/prefill_comparison.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat \
        --trace data/phase3_qwen-inst_chat_reason.csv --grid data/phase3_grid.csv \
        --out prefill_qwen-inst_chat.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_elicit import INDUCE, candidate_ids, load_model, reason_prefix  # noqa: E402

LESS_WORDS = ["less", "lower", "smaller"]
MORE_WORDS = ["greater", "higher", "larger", "more"]
CMP = re.compile(r"\b(?:is|are)\s+(?:(?:slightly|significantly|much|far|still)\s+)?"
                 r"(less|lower|smaller|greater|higher|larger|more)\s+than\b", re.I)
SURE_WORDS = re.compile(r"guaranteed|certain|sure", re.I)


def find_comparison(trace, sure):
    """First comparison word whose object is the sure amount; returns (cut index, word)."""
    sure_s = f"{sure:g}"
    for m in CMP.finditer(trace):
        tail = trace[m.end():m.end() + 60]
        if sure_s in tail or SURE_WORDS.search(tail):
            return m.start(1), m.group(1).lower()
    return None, None


def option_name(prompt, letter):
    """The label the prompt uses for the gamble ('Option A', 'Payout B', ...)."""
    m = re.search(r"\b(Option|Alternative|Payout|Outcome|Term|Arrangement|Choice)\s+" + letter + r"\b",
                  prompt)
    return m.group(0) if m else f"Option {letter}"


@torch.no_grad()
def p_less(tok, model, texts, add_special, less_ids, more_ids):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=add_special).to(model.device)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    try:
        logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1]
    except TypeError:
        logits = model(**enc, position_ids=pos).logits[:, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    pl = probs[:, less_ids].sum(-1)
    pm = probs[:, more_ids].sum(-1)
    top = probs.argmax(-1)
    return [(a, b, tok.decode([t]).strip()) for a, b, t in zip(pl.tolist(), pm.tolist(), top.tolist())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", choices=["chat", "fewshot"], default="chat")
    ap.add_argument("--trace", required=True, help="uninduced reasoning CSV from phase3_elicit.py --reason")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="own,rule,minimal")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N traces (smoke tests)")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    with open(args.grid) as f:
        prompts = {(r["instrument"], r["trial_id"]): r["prompt"] for r in csv.DictReader(f)}
    with open(args.trace) as f:
        traces = [r for r in csv.DictReader(f) if r["frame"] != "dominant" and r["excluded"] == "0"]

    items = []
    for r in traces:
        trace = (r["reasoning"] or "").lstrip()
        sure, hi, p = float(r["sure"]), float(r["hi"]), float(r["p"])
        cut, word = find_comparison(trace, sure)
        if cut is None:
            continue
        prompt = prompts[(r["instrument"], r["trial_id"])]
        items.append(dict(r=r, prompt=prompt, prefix=trace[:cut].rstrip(), word=word,
                          ev=p * hi, sure=sure))
    print(f"{len(items)} of {len(traces)} traces have a comparison against the sure amount")
    if args.limit:
        items = items[:args.limit]

    tok, model = load_model(args.model, load_4bit=not args.no_4bit)
    tok.padding_side = "left"
    less_ids = candidate_ids(tok, [" " + w for w in LESS_WORDS])
    more_ids = candidate_ids(tok, [" " + w for w in MORE_WORDS])
    print("less ids", less_ids, "more ids", more_ids)
    lead = " " if args.fmt == "fewshot" else ""

    def text_for(it, cond):
        induce = INDUCE["riskneutral"] if cond == "rule" else None
        head, add_special = reason_prefix(tok, it["prompt"], args.fmt, induce)
        if cond == "minimal":
            name = option_name(it["prompt"], it["r"]["gamble_letter"])
            body = f"The expected value of {name} is ${it['ev']:.2f}, which is"
        else:
            body = it["prefix"]
        return head + lead + body, add_special

    fields = ["instrument", "trial_id", "gamble_id", "frame", "sure", "hi", "p", "safe_first",
              "gamble_letter", "condition", "generated_word", "p_less_group", "p_more_group",
              "p_less", "top_token", "context"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond in args.conditions.split(","):
            for k in range(0, len(items), args.batch_size):
                chunk = items[k:k + args.batch_size]
                built = [text_for(it, cond) for it in chunk]
                res = p_less(tok, model, [t for t, _ in built], built[0][1], less_ids, more_ids)
                for it, (pl, pm, top) in zip(chunk, res):
                    r = it["r"]
                    w.writerow(dict(
                        instrument=r["instrument"], trial_id=r["trial_id"], gamble_id=r["gamble_id"],
                        frame=r["frame"], sure=r["sure"], hi=r["hi"], p=r["p"],
                        safe_first=r["safe_first"], gamble_letter=r["gamble_letter"], condition=cond,
                        generated_word=it["word"], p_less_group=f"{pl:.5f}", p_more_group=f"{pm:.5f}",
                        p_less=f"{pl / max(pl + pm, 1e-12):.5f}", top_token=top,
                        context=it["prefix"][-120:]))
                f.flush()
            print(f"condition {cond}: done")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
