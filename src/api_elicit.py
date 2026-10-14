"""Stage A on frontier API models: sampled answers instead of log-probabilities.

Closed frontier models expose no next-token probabilities (and reasoning models expose
none even where the API has a logprobs field), so each cell is asked n times at the
provider's default temperature and P(risky) is the share of parsed answers that pick the
gamble. Output columns match colab/phase3_elicit.py, so analysis/stage_a.py reads the
files unchanged:

    p_gamble   share of parsed samples choosing the risky option
    ab_mass    share of samples that parsed to A or B (the analogue of A/B token mass)
    excluded   1 if ab_mass < --mass-threshold (refusal or hedging)

Prompts are the open-model chat prompts, word for word:
    immediate     <prompt>\\nAnswer with only A or B.
    reason first  <prompt>\\nThink it through briefly, then give your final choice on its own
                  last line as "Answer: A" or "Answer: B".

Providers:
    anthropic   native Messages API (ANTHROPIC_API_KEY)
    openai      any OpenAI-compatible chat endpoint (OPENAI_API_KEY, or --api-key-env);
                with --base-url this covers OpenRouter, Together, and Google's
                OpenAI-compatible Gemini endpoint, so one key can reach every model
    dry         simulated agent, no network, for checking the pipeline and the estimate

NOTHING HERE SPENDS MONEY UNLESS --go IS PASSED. Without it the script prints the cell
count, call count and a cost estimate from --price-in / --price-out and exits.

Usage:
    # cost estimate only (default)
    python src/api_elicit.py --provider openai --model <id> --grid data/stageA_grid.csv \\
        --n 5 --price-in 3 --price-out 15 --thinking-tokens 0

    # pipeline check with the simulated agent, then the analysis
    python src/api_elicit.py --provider dry --model sim --grid data/stageA_grid.csv \\
        --out data/stageA_sim_api.csv --n 5 --go
    python analysis/stage_a.py --stems sim_api

    # real run (after keys are set and spend is approved)
    python src/api_elicit.py --provider anthropic --model <id> --grid data/stageA_grid.csv \\
        --out data/stageA_<stem>.csv --n 5 --go --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import random
import re
import sys
import time

# Kept identical to colab/phase3_elicit.py (not imported: that module needs torch).
IMMEDIATE_SUFFIX = "\nAnswer with only A or B."
CHAT_REASON_SUFFIX = ("\nThink it through briefly, then give your final choice on its own last "
                      "line as \"Answer: A\" or \"Answer: B\".")
OUT_FIELDS = ["instrument", "trial_id", "gamble_id", "frame", "anchor", "sure", "hi",
              "p", "ev_ratio", "template_id", "safe_first", "gamble_letter",
              "model", "fmt", "induce", "reason", "p_gamble", "ab_mass", "excluded",
              "reason_ended", "reasoning", "note"]

_ANSWER = re.compile(r"answer\s*[:\-]?\s*\**\s*\(?([AB])\b", re.IGNORECASE)
_BARE = re.compile(r"^\W*(?:option|plan|route)?\s*\(?([AB])\b", re.IGNORECASE)


def parse(text, reason):
    """Return 'A', 'B' or None. Reason-first: the last 'Answer: X'. Immediate: a leading
    letter, else an 'Answer: X' line, else a lone A/B token anywhere."""
    text = (text or "").strip()
    hits = _ANSWER.findall(text)
    if reason:
        return hits[-1].upper() if hits else None
    m = _BARE.match(text)
    if m:
        return m.group(1).upper()
    if hits:
        return hits[-1].upper()
    lone = set(re.findall(r"\b([AB])\b", text))
    return lone.pop() if len(lone) == 1 else None


# ---------------------------------------------------------------- providers

class Dry:
    """Simulated CRRA chooser with order bias and a domain-specific caution shift, so
    the analysis has something to find. Free, offline, deterministic per seed."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, row, prompt, reason):
        sure, hi, p = float(row["sure"]), float(row["hi"]), float(row["p"])
        r = 0.5 + (0.6 if "money" not in row["instrument"] else 0.0)
        u = lambda x: x ** (1 - r) if x > 0 else 0.0
        z = (p * u(hi) - u(sure)) / max(u(sure), 1e-9) * 4
        z += 0.4 if row["gamble_letter"] == "A" else -0.4
        pr = 1 / (1 + math.exp(-z)) if row["frame"] != "dominant" else 0.97
        pick_gamble = self.rng.random() < pr
        letter = row["gamble_letter"] if pick_gamble else ("B" if row["gamble_letter"] == "A" else "A")
        if self.rng.random() < 0.02:
            return "I can't make that choice for you.", {"in": 80, "out": 10}
        body = "Comparing the options by expected value.\n" if reason else ""
        return f"{body}Answer: {letter}" if reason else letter, {"in": 80, "out": 60 if reason else 1}


class Anthropic:
    def __init__(self, model, max_tokens, temperature, extra):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=8)
        self.model, self.max_tokens, self.temperature, self.extra = model, max_tokens, temperature, extra

    def __call__(self, row, prompt, reason):
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  messages=[{"role": "user", "content": prompt}], **self.extra)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        msg = self.client.messages.create(**kw)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}


class OpenAICompat:
    def __init__(self, model, max_tokens, temperature, extra, base_url, key_env):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=os.environ[key_env], max_retries=8)
        self.model, self.max_tokens, self.temperature, self.extra = model, max_tokens, temperature, extra

    def __call__(self, row, prompt, reason):
        kw = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                  max_completion_tokens=self.max_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.extra:
            kw["extra_body"] = self.extra
        r = self.client.chat.completions.create(**kw)
        u = r.usage
        return r.choices[0].message.content or "", {"in": u.prompt_tokens, "out": u.completion_tokens}


# ---------------------------------------------------------------- run

def estimate(rows, args):
    n_calls = len(rows) * args.n
    tin = sum(len(r["prompt"]) / 4 + 25 for r in rows) * args.n
    tout = n_calls * ((args.reason_tokens if args.reason else 3) + args.thinking_tokens)
    cost = tin / 1e6 * args.price_in + tout / 1e6 * args.price_out
    print(f"{len(rows)} cells x {args.n} samples = {n_calls:,} calls")
    print(f"~{tin/1e6:.2f}M input tokens, ~{tout/1e6:.2f}M output tokens "
          f"(assumes {args.reason_tokens if args.reason else 3} visible + {args.thinking_tokens} hidden reasoning tokens per call)")
    print(f"estimated cost at ${args.price_in}/M in, ${args.price_out}/M out: ${cost:,.2f}")
    return cost


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["anthropic", "openai", "dry"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out")
    ap.add_argument("--reason", action="store_true", help="reason-first prompt")
    ap.add_argument("--n", type=int, default=5, help="samples per cell")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: the provider's default (as Epoch and most evals run)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default 16 immediate / 1024 reason first; raise for thinking models")
    ap.add_argument("--extra", default="{}",
                    help='JSON passed through to the API, e.g. \'{"reasoning_effort": "low"}\' '
                         'or \'{"thinking": {"type": "enabled", "budget_tokens": 2048}}\'')
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domains", default=None, help="comma-separated, e.g. money,civilians (pilot)")
    ap.add_argument("--gambles", type=int, default=0, help="keep this many gambles per instrument (pilot)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mass-threshold", type=float, default=0.20)
    ap.add_argument("--price-in", type=float, default=3.0, help="$ per M input tokens (estimate only)")
    ap.add_argument("--price-out", type=float, default=15.0, help="$ per M output tokens (estimate only)")
    ap.add_argument("--reason-tokens", type=int, default=350, help="visible tokens per reason-first answer (estimate)")
    ap.add_argument("--thinking-tokens", type=int, default=0, help="hidden reasoning tokens per call (estimate)")
    ap.add_argument("--go", action="store_true", help="actually call the API (spends money unless --provider dry)")
    args = ap.parse_args()
    extra = json.loads(args.extra)
    max_tokens = args.max_tokens or (1024 if args.reason else 16)

    with open(args.grid) as f:
        rows = list(csv.DictReader(f))
    if args.domains:
        keep = set(args.domains.split(","))
        rows = [r for r in rows if r["instrument"].split("_")[1] in keep]
    if args.gambles:
        rng = random.Random(0)
        ids = sorted({r["gamble_id"] for r in rows if r["frame"] != "dominant"}, key=int)
        pick = set(rng.sample(ids, min(args.gambles, len(ids))))
        rows = [r for r in rows if r["frame"] == "dominant" or r["gamble_id"] in pick]

    done = set()
    if args.resume and args.out and os.path.exists(args.out):
        with open(args.out) as f:
            done = {(d["instrument"], d["trial_id"]) for d in csv.DictReader(f)}
        rows = [r for r in rows if (r["instrument"], r["trial_id"]) not in done]
        print(f"resuming: {len(done)} cells already in {args.out}")

    estimate(rows, args)
    if not args.go:
        print("estimate only; pass --go to run")
        return
    if not args.out:
        sys.exit("--out is required with --go")

    if args.provider == "dry":
        call = Dry()
    elif args.provider == "anthropic":
        call = Anthropic(args.model, max_tokens, args.temperature, extra)
    else:
        call = OpenAICompat(args.model, max_tokens, args.temperature, extra, args.base_url, args.api_key_env)

    suffix = CHAT_REASON_SUFFIX if args.reason else IMMEDIATE_SUFFIX

    def one_cell(r):
        prompt = r["prompt"] + suffix
        letters, texts, tin, tout, errs = [], [], 0, 0, 0
        for _ in range(args.n):
            try:
                text, use = call(r, prompt, args.reason)
            except Exception as e:  # one failed sample is a missing sample, not a dead run
                errs += 1
                texts.append(f"<error {type(e).__name__}: {str(e)[:120]}>")
                continue
            tin, tout = tin + use["in"], tout + use["out"]
            letters.append(parse(text, args.reason))
            texts.append(text)
        valid = [x for x in letters if x]
        ab = len(valid) / args.n
        pg = sum(x == r["gamble_letter"] for x in valid) / len(valid) if valid else float("nan")
        out = {k: r[k] for k in OUT_FIELDS if k in r}
        out.update(model=args.model, fmt="api", induce="none", reason=int(args.reason),
                   reason_ended="answer" if valid else "unparsed",
                   reasoning=json.dumps(texts) if args.reason else "",
                   p_gamble="" if pg != pg else round(pg, 5), ab_mass=round(ab, 5),
                   excluded=int(ab < args.mass_threshold or pg != pg),
                   note=json.dumps(dict(letters=letters, tok_in=tin, tok_out=tout, errors=errs,
                                        temperature=args.temperature, extra=extra)))
        return out, tin, tout

    t0, tot_in, tot_out, n_excl = time.time(), 0, 0, 0
    with open(args.out, "a" if done else "w", newline="") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if not done:
            w.writeheader()
        for i, (out, tin, tout) in enumerate(ex.map(one_cell, rows)):
            w.writerow(out)
            f.flush()
            tot_in, tot_out, n_excl = tot_in + tin, tot_out + tout, n_excl + out["excluded"]
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                spent = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
                print(f"  {i+1}/{len(rows)} | excluded {n_excl} | tokens {tot_in:,} in / {tot_out:,} out "
                      f"| ~${spent:,.2f} | {time.time()-t0:.0f}s")
    print(f"Wrote {args.out}. Excluded: {n_excl/max(len(rows), 1):.1%}")


if __name__ == "__main__":
    main()
