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
    anthropic   native Messages API (ANTHROPIC_API_KEY); --thinking off|on sets the model's
                thinking mode (snap answer, or extended thinking at --effort)
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

    # Claude runs (docs/paper-risk-advice-plan.md): snap, written reasoning, extended thinking
    python src/api_elicit.py --provider anthropic --model claude-haiku-5-5 --grid data/stageA_grid.csv \\
        --out data/stageA_haiku55_api.csv --n 5 --thinking off --max-spend 5 --go --resume
    python src/api_elicit.py --provider anthropic --model claude-sonnet-5-5 --grid data/stageA_grid_reason.csv \\
        --out data/stageA_sonnet55_api_reason.csv --n 3 --reason --thinking off --max-spend 15 --go --resume
    python src/api_elicit.py --provider anthropic --model claude-sonnet-5-5 --grid data/stageA_grid_reason.csv \\
        --out data/stageA_sonnet55_api_think.csv --n 3 --thinking on --effort low --max-spend 40 --go --resume
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


# Anthropic first-party prices, $ per million tokens (input, output), checked Oct 18 2026.
# Hidden thinking tokens are billed as output. The Batch API would halve these (not used here).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def thinking_config(model, thinking, effort):
    """Request fields for thinking off (snap answer) or on (extended thinking at an effort level).

    Claude Haiku 5.5 turns thinking off with {"type": "disabled"} (effort high or below);
    Claude Sonnet 5.5 rejects "disabled" and turns it off with {"type": "between_tools"};
    Claude Opus 5.5 cannot turn it off at all."""
    if thinking == "default":
        return {}
    if thinking == "on":
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    if "sonnet-5-5" in model:
        return {"thinking": {"type": "between_tools"}}
    if "opus-5-5" in model or "fable" in model:
        sys.exit(f"{model} cannot run with thinking off; use --thinking on --effort low")
    return {"thinking": {"type": "disabled"}}


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
        # a refusal (stop_reason "refusal") carries no text and counts as an unparsed sample
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return text, {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                      "stop": msg.stop_reason}


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
    hidden = args.thinking_tokens if args.thinking_tokens is not None else (800 if args.thinking == "on" else 0)
    tout = n_calls * ((args.reason_tokens if args.reason else 3) + hidden)
    cost = tin / 1e6 * args.price_in + tout / 1e6 * args.price_out
    print(f"{len(rows)} cells x {args.n} samples = {n_calls:,} calls")
    print(f"~{tin/1e6:.2f}M input tokens, ~{tout/1e6:.2f}M output tokens "
          f"(assumes {args.reason_tokens if args.reason else 3} visible + {hidden} hidden thinking tokens per call)")
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
                    help="default 16 immediate / 1024 reason first / 8000 with thinking on")
    ap.add_argument("--thinking", choices=["off", "on", "default"], default="off",
                    help="anthropic only: off = snap answer (the open-model comparison); on = extended "
                         "thinking at --effort; default = send nothing")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="low",
                    help="thinking effort with --thinking on (caps how much the model thinks)")
    ap.add_argument("--max-spend", type=float, default=None,
                    help="stop calling once the running cost (from token usage) reaches this many dollars")
    ap.add_argument("--extra", default="{}",
                    help='extra JSON passed through to the API, e.g. \'{"reasoning_effort": "low"}\' '
                         'for an OpenAI-compatible model')
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domains", default=None, help="comma-separated, e.g. money,civilians (pilot)")
    ap.add_argument("--gambles", type=int, default=0, help="keep this many gambles per instrument (pilot)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mass-threshold", type=float, default=0.20)
    ap.add_argument("--price-in", type=float, default=None,
                    help="$ per M input tokens (default: PRICES for known Claude models, else 3)")
    ap.add_argument("--price-out", type=float, default=None,
                    help="$ per M output tokens (default: PRICES for known Claude models, else 15)")
    ap.add_argument("--reason-tokens", type=int, default=350, help="visible tokens per reason-first answer (estimate)")
    ap.add_argument("--thinking-tokens", type=int, default=None,
                    help="hidden thinking tokens per call, for the estimate (default 800 with --thinking on, else 0)")
    ap.add_argument("--go", action="store_true", help="actually call the API (spends money unless --provider dry)")
    args = ap.parse_args()
    extra = json.loads(args.extra)
    known = PRICES.get(args.model, (3.0, 15.0))
    args.price_in = known[0] if args.price_in is None else args.price_in
    args.price_out = known[1] if args.price_out is None else args.price_out
    if args.provider == "anthropic":
        extra = {**thinking_config(args.model, args.thinking, args.effort), **extra}
    elif args.thinking != "off":
        sys.exit("--thinking applies to --provider anthropic; pass the provider's own fields in --extra")
    max_tokens = args.max_tokens or (8000 if args.thinking == "on" else 1024 if args.reason else 16)

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

    spent = [0.0]  # running cost from reported usage, for --max-spend

    def one_cell(r):
        if args.max_spend is not None and spent[0] >= args.max_spend:
            return None  # not written, so --resume picks the cell up later
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
            spent[0] += use["in"] / 1e6 * args.price_in + use["out"] / 1e6 * args.price_out
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
                                        temperature=args.temperature, thinking=args.thinking,
                                        effort=args.effort if args.thinking == "on" else None,
                                        extra=extra)))
        return out, tin, tout

    t0, tot_in, tot_out, n_excl = time.time(), 0, 0, 0
    with open(args.out, "a" if done else "w", newline="") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if not done:
            w.writeheader()
        for i, res in enumerate(ex.map(one_cell, rows)):
            if res is None:
                continue
            out, tin, tout = res
            w.writerow(out)
            f.flush()
            tot_in, tot_out, n_excl = tot_in + tin, tot_out + tout, n_excl + out["excluded"]
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                cost_now = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
                print(f"  {i+1}/{len(rows)} | excluded {n_excl} | tokens {tot_in:,} in / {tot_out:,} out "
                      f"| ~${cost_now:,.2f} | {time.time()-t0:.0f}s")
    print(f"Wrote {args.out}. Excluded: {n_excl/max(len(rows), 1):.1%}. Spent ~${spent[0]:,.2f}")
    if args.max_spend is not None and spent[0] >= args.max_spend:
        print("stopped at --max-spend; the remaining cells were not run. Rerun with --resume to continue")


if __name__ == "__main__":
    main()
