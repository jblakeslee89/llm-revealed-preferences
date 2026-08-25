"""Phase 3 elicitation on Colab (free T4 GPU): read A/B choice probabilities from a
local HuggingFace model's full next-token logits.

This is now the PRIMARY Phase 3 path. The provider catalog check (July 2026) found no
serverless host serving BASE open models with logprobs; Together has them only on
GPU-hour dedicated endpoints, and OLMo-2 is hosted nowhere. Running the weights directly
in Colab is free, gives EXACT probabilities from the full vocabulary (no top-k
truncation), and is the only option that covers OLMo-2's staged base/SFT/DPO checkpoints.

USAGE (in a Colab notebook with a T4 GPU runtime):
    !pip install -q transformers accelerate bitsandbytes
    # upload phase3_grid.csv and this file, then:
    !python phase3_elicit.py --model Qwen/Qwen2.5-7B      --grid phase3_grid.csv --out phase3_qwen-base_fewshot.csv --fmt fewshot
    !python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --grid phase3_grid.csv --out phase3_qwen-inst_fewshot.csv --fmt fewshot
    !python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --grid phase3_grid.csv --out phase3_qwen-inst_chat.csv    --fmt chat

Then download the phase3_*.csv files and run analysis/estimate_from_probs.py locally.

MODEL PAIRS (all open, only Llama is gated and needs `huggingface-cli login`):
    Qwen/Qwen2.5-7B            <-> Qwen/Qwen2.5-7B-Instruct
    meta-llama/Llama-3.1-8B    <-> meta-llama/Llama-3.1-8B-Instruct   (gated)
    allenai/OLMo-2-1124-7B     <-> allenai/OLMo-2-1124-7B-SFT <-> ...-DPO <-> ...-Instruct
        (OLMo-2's staged checkpoints let you attribute the bias to a SPECIFIC
         post-training stage: the most distinctive version of the result.)
"""

from __future__ import annotations

import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Same few-shot preamble as src/logprob_agent.py: teaches the one-letter format with
# answers balanced 2 A / 2 B and no preference cue.
FEWSHOT = """You must pick one of two payment options.
Option A: receive $50 for certain
Option B: receive $6 for certain
Answer: A

You must pick one of two payment options.
Option A: receive $4 for certain
Option B: receive $63 for certain
Answer: B

You must pick one of two payment options.
Option A: an 8% chance of $70 and a 92% chance of $0
Option B: receive $47 for certain
Answer: B

You must pick one of two payment options.
Option A: receive $71 for certain
Option B: a 15% chance of $90 and an 85% chance of $0
Answer: A

"""

OUT_FIELDS = ["instrument", "trial_id", "gamble_id", "frame", "anchor", "sure", "hi",
              "p", "ev_ratio", "template_id", "safe_first", "gamble_letter",
              "model", "fmt", "induce", "p_gamble", "ab_mass", "excluded", "note"]

# Induced-valuation arm (Armour feedback, Aug 2026): assert a utility function and
# instruct the model to maximize it. Compliance with the induced optimum (scored by
# analysis/score_induced.py) calibrates how much to trust the uninduced elicitation,
# in the spirit of induced-value theory (Smith 1976).
INDUCE = {
    "riskneutral": (
        "Instruction: you are a risk-neutral expected-value maximizer. For each option, "
        "compute its expected dollar value (payoff times probability). Choose the option "
        "with the higher expected dollar value. Apply no risk aversion and no other "
        "consideration."),
    "sqrt": (
        "Instruction: your utility over money is u(x) = sqrt(x). For each option, compute "
        "its expected utility (probability times the square root of the payoff). Choose "
        "the option with the higher expected utility. Apply no other consideration."),
}


def candidate_ids(tok, strings):
    ids = set()
    for s in strings:
        enc = tok.encode(s, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def load_model(model_id, load_4bit=True):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="auto", torch_dtype=torch.float16)
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model


def build_input(tok, prompt, fmt, induce_text=None):
    if fmt == "chat":
        content = prompt + "\nAnswer with only A or B."
        if induce_text:
            content = induce_text + "\n\n" + content
        msgs = [{"role": "user", "content": content}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=False)
        return ids
    # fewshot: the instruction precedes the examples. All four example answers are
    # consistent with both induced rules, so the examples never contradict the instruction.
    text = FEWSHOT + prompt.rstrip() + "\nAnswer:"
    if induce_text:
        text = induce_text + "\n\n" + text
    return tok(text, return_tensors="pt").input_ids


@torch.no_grad()
def score(tok, model, prompt, fmt, a_ids, b_ids, induce_text=None):
    ids = build_input(tok, prompt, fmt, induce_text).to(model.device)
    logits = model(ids).logits[0, -1]          # next-token logits
    probs = torch.softmax(logits.float(), dim=-1)
    mass_a = float(probs[a_ids].sum())
    mass_b = float(probs[b_ids].sum())
    return mass_a, mass_b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", required=True, help="phase3_grid.csv exported by run_phase3.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", choices=["fewshot", "chat"], default="fewshot")
    ap.add_argument("--induce", choices=["none", "riskneutral", "sqrt"], default="none",
                    help="prepend an induced-utility instruction (Armour arm); score "
                         "compliance afterwards with analysis/score_induced.py")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--mass-threshold", type=float, default=0.20)
    args = ap.parse_args()
    induce_text = INDUCE.get(args.induce)

    tok, model = load_model(args.model, load_4bit=not args.no_4bit)
    a_ids = torch.tensor(candidate_ids(tok, ["A", " A"]), device=model.device)
    b_ids = torch.tensor(candidate_ids(tok, ["B", " B"]), device=model.device)
    print(f"A token ids: {a_ids.tolist()}   B token ids: {b_ids.tolist()}")

    with open(args.grid) as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} cells to score for {args.model} ({args.fmt})")

    n_excl = 0
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for i, r in enumerate(rows):
            mass_a, mass_b = score(tok, model, r["prompt"], args.fmt, a_ids, b_ids, induce_text)
            ab = mass_a + mass_b
            if ab > 0:
                mg = mass_b if r["gamble_letter"] == "B" else mass_a
                p_gamble = mg / ab
            else:
                p_gamble = float("nan")
            excluded = int(ab < args.mass_threshold or p_gamble != p_gamble)
            n_excl += excluded
            out = {k: r[k] for k in OUT_FIELDS if k in r}
            out.update(model=args.model, fmt=args.fmt, induce=args.induce,
                       p_gamble="" if p_gamble != p_gamble else round(p_gamble, 5),
                       ab_mass=round(ab, 5), excluded=excluded,
                       note=f"massA={mass_a:.3g} massB={mass_b:.3g}")
            w.writerow(out)
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"  {i+1}/{len(rows)} | excluded {n_excl}")
    print(f"Wrote {args.out}. Excluded (low A/B mass): {n_excl/len(rows):.1%}")


if __name__ == "__main__":
    main()
