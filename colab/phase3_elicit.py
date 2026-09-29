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
import os
import random
import re

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

# Reason-then-answer variant of the same preamble (--reason). The reasoning lines only
# describe each option (the certain amount, or probability x payoff); they state no
# decision rule. Every example answer is the choice any sane risk preference makes, but
# the lines do demonstrate the EV arithmetic, so compare reasoning runs to each other
# (induced vs uninduced) rather than reading uninduced fewshot reasoning as "natural".
FEWSHOT_REASON = """You must pick one of two payment options.
Option A: receive $50 for certain
Option B: receive $6 for certain
Reasoning: Option A pays $50 for certain. Option B pays $6 for certain.
Answer: A

You must pick one of two payment options.
Option A: receive $4 for certain
Option B: receive $63 for certain
Reasoning: Option A pays $4 for certain. Option B pays $63 for certain.
Answer: B

You must pick one of two payment options.
Option A: an 8% chance of $70 and a 92% chance of $0
Option B: receive $47 for certain
Reasoning: Option A is worth 0.08 x $70 = $5.60 on average. Option B pays $47 for certain.
Answer: B

You must pick one of two payment options.
Option A: receive $71 for certain
Option B: a 15% chance of $90 and an 85% chance of $0
Reasoning: Option A pays $71 for certain. Option B is worth 0.15 x $90 = $13.50 on average.
Answer: A

"""

CHAT_REASON_SUFFIX = ("\nThink it through briefly, then give your final choice on its own last "
                      "line as \"Answer: A\" or \"Answer: B\".")

OUT_FIELDS = ["instrument", "trial_id", "gamble_id", "frame", "anchor", "sure", "hi",
              "p", "ev_ratio", "template_id", "safe_first", "gamble_letter",
              "model", "fmt", "induce", "reason", "p_gamble", "ab_mass", "excluded",
              "reason_ended", "reasoning", "note"]

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
    # fp32 on CPU is only for local smoke tests with tiny models; Colab runs use the GPU
    cuda = torch.cuda.is_available()
    kw = dict(torch_dtype=torch.float16 if cuda else torch.float32)
    if cuda:
        kw["device_map"] = "auto"
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


# ---------------------------------------------------------------- reason-then-answer

def reason_prefix(tok, prompt, fmt, induce_text=None):
    """Text the model continues with its reasoning. Returns (text, add_special_tokens)."""
    if fmt == "chat":
        content = prompt + CHAT_REASON_SUFFIX
        if induce_text:
            content = induce_text + "\n\n" + content
        msgs = [{"role": "user", "content": content}]
        # the template string already carries BOS/role tokens, so don't add them again
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), False
    text = FEWSHOT_REASON + prompt.rstrip() + "\nReasoning:"
    if induce_text:
        text = induce_text + "\n\n" + text
    return text, True


_TAIL = re.compile(r"[\s*#_:-]*(final)?[\s*#_:-]*$", re.IGNORECASE)


def cut_reasoning(gen, fmt):
    """Trim generated text at the model's own answer line; report how generation ended."""
    i = gen.find("Answer:")
    if i >= 0:
        return _TAIL.sub("", gen[:i]).rstrip(), "answer"
    j = gen.find("\n\n") if fmt == "fewshot" else -1
    if j >= 0 and gen[:j].strip():  # fewshot only: the model moved on to a new example
        return gen[:j].rstrip(), "blank"
    return gen.rstrip(), "length_or_eos"


@torch.no_grad()
def reason_batch(tok, model, prompts, fmt, a_ids, b_ids, induce_text, max_new_tokens):
    """Greedy reasoning, then read A/B mass at 'Answer:' after the model's own reasoning.

    Greedy decoding keeps the arm deterministic, like the logit readout it extends: the
    randomness in the design stays in templates, orders and payoffs, not in sampling."""
    pre = [reason_prefix(tok, p, fmt, induce_text) for p in prompts]
    add_special = pre[0][1]
    texts = [t for t, _ in pre]
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=add_special).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         stop_strings=["Answer:", "\n\n"] if fmt == "fewshot" else ["Answer:"],
                         tokenizer=tok, pad_token_id=tok.pad_token_id)
    gens = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    cut = [cut_reasoning(g, fmt) for g in gens]
    lead = " " if fmt == "fewshot" else ""
    full = [t + lead + r.lstrip() + "\nAnswer:" for t, (r, _) in zip(texts, cut)]
    enc2 = tok(full, return_tensors="pt", padding=True,
               add_special_tokens=add_special).to(model.device)
    # left padding: positions must count from each row's first real token, as generate() does
    pos = (enc2.attention_mask.cumsum(-1) - 1).clamp(min=0)
    # only the last position is needed; full-sequence logits (batch x length x vocab)
    # ran a T4 out of memory on long OLMo traces
    try:
        logits = model(**enc2, position_ids=pos, logits_to_keep=1).logits[:, -1]
    except TypeError:  # older transformers without logits_to_keep
        logits = model(**enc2, position_ids=pos).logits[:, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    mass_a = probs[:, a_ids].sum(-1).tolist()
    mass_b = probs[:, b_ids].sum(-1).tolist()
    return [(ma, mb, r, how) for ma, mb, (r, how) in zip(mass_a, mass_b, cut)]


def subset_gambles(rows, k, seed=0):
    """Keep k gambles per instrument, with all their frames, templates and orders."""
    rng = random.Random(seed)
    keep = set()
    for inst in sorted({r["instrument"] for r in rows}):
        ids = sorted({r["gamble_id"] for r in rows if r["instrument"] == inst}, key=int)
        keep |= {(inst, g) for g in rng.sample(ids, min(k, len(ids)))}
    return [r for r in rows if (r["instrument"], r["gamble_id"]) in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", required=True, help="phase3_grid.csv exported by run_phase3.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", choices=["fewshot", "chat"], default="fewshot")
    ap.add_argument("--induce", choices=["none", "riskneutral", "sqrt"], default="none",
                    help="prepend an induced-utility instruction (Armour arm); score "
                         "compliance afterwards with analysis/score_induced.py")
    ap.add_argument("--reason", action="store_true",
                    help="reason-then-answer: generate reasoning greedily, then read A/B "
                         "mass at 'Answer:' (tests cannot-compute vs will-not-comply)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16, help="reason mode only")
    ap.add_argument("--gambles", type=int, default=0,
                    help="keep this many gambles per instrument (0 = full grid); "
                         "reason mode is ~10-20x slower than immediate answer")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing --out, skipping cells already scored")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--mass-threshold", type=float, default=0.20)
    args = ap.parse_args()
    induce_text = INDUCE.get(args.induce)

    tok, model = load_model(args.model, load_4bit=not args.no_4bit)
    tok.padding_side = "left"  # batched generation and last-position readout need left padding
    a_ids = torch.tensor(candidate_ids(tok, ["A", " A"]), device=model.device)
    b_ids = torch.tensor(candidate_ids(tok, ["B", " B"]), device=model.device)
    print(f"A token ids: {a_ids.tolist()}   B token ids: {b_ids.tolist()}")

    with open(args.grid) as f:
        rows = list(csv.DictReader(f))
    if args.gambles:
        rows = subset_gambles(rows, args.gambles)
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            done = {(d["instrument"], d["trial_id"]) for d in csv.DictReader(f)}
        rows = [r for r in rows if (r["instrument"], r["trial_id"]) not in done]
        print(f"resuming: {len(done)} cells already in {args.out}")
    print(f"{len(rows)} cells to score for {args.model} ({args.fmt}"
          f"{', reason' if args.reason else ''})")

    def results():
        if not args.reason:
            for r in rows:
                ma, mb = score(tok, model, r["prompt"], args.fmt, a_ids, b_ids, induce_text)
                yield r, ma, mb, "", ""
            return
        for k in range(0, len(rows), args.batch_size):
            chunk = rows[k:k + args.batch_size]
            got = reason_batch(tok, model, [r["prompt"] for r in chunk], args.fmt,
                               a_ids, b_ids, induce_text, args.max_new_tokens)
            for r, (ma, mb, text, how) in zip(chunk, got):
                yield r, ma, mb, text, how

    n_excl = 0
    ended = {}
    with open(args.out, "a" if done else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if not done:
            w.writeheader()
        for i, (r, mass_a, mass_b, reasoning, how) in enumerate(results()):
            ended[how] = ended.get(how, 0) + 1
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
                       reason=int(args.reason), reason_ended=how, reasoning=reasoning,
                       p_gamble="" if p_gamble != p_gamble else round(p_gamble, 5),
                       ab_mass=round(ab, 5), excluded=excluded,
                       note=f"massA={mass_a:.3g} massB={mass_b:.3g}")
            w.writerow(out)
            f.flush()
            if (i + 1) % (args.batch_size * 4 if args.reason else 100) == 0 or i + 1 == len(rows):
                print(f"  {i+1}/{len(rows)} | excluded {n_excl}"
                      + (f" | ended {ended}" if args.reason else ""))
    print(f"Wrote {args.out}. Excluded (low A/B mass): {n_excl/max(len(rows), 1):.1%}")


if __name__ == "__main__":
    main()
