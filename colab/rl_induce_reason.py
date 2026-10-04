"""Phase 4b: induced value by payment, in the reasoning regime (GRPO-style).

Phase 4 (`rl_induce.py`) paid the model for its one-letter answers and never taught it to tell
gambles apart: across six 7B runs it settled on a constant strategy. That matches Phase 3,
where the model told the rule cannot compute expected value in one letter but does when it is
allowed to reason. This script is the positive control the Phase 3 writeup lists as unrun:
pay the model in the regime where it already computes, and ask whether payment does what the
instruction did.

Per step: draw --prompts training gambles (neutral Phase 1 wording, random template and
order); sample --group reasoning traces for each prompt; read the A/B distribution at
"Answer:" after each trace (the Phase 3 readout) and sample the answer from it; the lottery
pays u(payoff) for that answer (or its expected utility, --reward expected). Each trace's
advantage is its payment minus the mean payment of the other traces for the same prompt
(leave-one-out, so it stays unbiased), and the update is REINFORCE on the trace tokens plus the
answer letter, on a LoRA adapter, with a per-token KL penalty toward the untrained model.

By default the model is paid and not told the rule (--induce none), which is the clean
"payment versus instruction" comparison with the Phase 3 reason-induced runs. --induce
riskneutral pays a model that is also told, a ceiling arm.

What to read off (as in Phase 4): held-out balanced accuracy on the monitor set (0.5 for any
constant strategy), the frame gap on unseen gain/loss wording, and then the full evaluation:
    python colab/phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen_reason_linear \\
        --fmt chat --reason --gambles 20 --grid data/phase3_grid.csv \\
        --out data/phase4_qwen-inst_chat_reason_paid-linear.csv
    python analysis/score_robustness.py data/phase4_qwen-inst_chat_reason_paid-linear.csv \\
        --rule riskneutral --vs data/phase3_qwen-inst_chat_reason.csv

Usage (A100 recommended; see docs/robustness-plan-2026-10.md for the compute estimate):
    python colab/rl_induce_reason.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat \\
        --utility linear --steps 200 --out runs/qwen_reason_linear --no-4bit
Smoke test of the gradient sign (the paid letter's probability must rise):
    python colab/rl_induce_reason.py --model <small model> --sanity-letter B --steps 10 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from phase3_elicit import (INDUCE, candidate_ids, cut_reasoning, load_model,  # noqa: E402
                           reason_batch, reason_prefix)
from rl_induce import (EVAL_SEED, UTILITIES, generate_gambles, monitor_set,  # noqa: E402
                       phase1_prompt, train_gambles)

ANSWER = "\nAnswer:"


# ---------------------------------------------------------------- rollouts

@torch.no_grad()
def sample_traces(tok, model, prefixes, add_special, fmt, temperature, max_new_tokens):
    """Sampled reasoning for each prefix, cut at the model's own answer line."""
    enc = tok(prefixes, return_tensors="pt", padding=True,
              add_special_tokens=add_special).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, top_p=1.0, top_k=0,
                         stop_strings=["Answer:", "\n\n"] if fmt == "fewshot" else ["Answer:"],
                         tokenizer=tok, pad_token_id=tok.pad_token_id)
    gens = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return [cut_reasoning(g, fmt) for g in gens]


def sequence_ids(tok, prefix, trace, fmt, add_special):
    """Token ids of prefix + trace + answer line, and where the trace tokens start and end.

    The trace span is what the policy gradient and KL act on; the answer line is appended by
    us, so it is excluded. Ids are built by concatenation so the span is exact."""
    lead = " " if fmt == "fewshot" else ""
    pre = tok(prefix, add_special_tokens=add_special).input_ids
    body = tok(prefix + lead + trace.lstrip(), add_special_tokens=add_special).input_ids
    start = 0
    while start < min(len(pre), len(body)) and pre[start] == body[start]:
        start += 1   # normally len(pre); shorter if the join merged a token
    tail = tok(ANSWER, add_special_tokens=False).input_ids
    return body + tail, start, len(body)


def letter_logprobs(logits_last, a_ids, b_ids):
    """log P(A), log P(B), renormalized over the two answer letters."""
    lse_a = torch.logsumexp(logits_last[a_ids], 0)
    lse_b = torch.logsumexp(logits_last[b_ids], 0)
    norm = torch.logsumexp(torch.stack([lse_a, lse_b]), 0)
    return lse_a - norm, lse_b - norm


def token_logprobs(model, ids, start, end):
    """Per-token log-probs of ids[start:end] and the final-position logits."""
    x = torch.tensor([ids], device=model.device)
    logits = model(x).logits[0].float()
    lp = torch.log_softmax(logits[start - 1:end - 1], -1)
    tgt = x[0, start:end]
    return lp.gather(1, tgt[:, None])[:, 0], logits[-1]


# ---------------------------------------------------------------- monitor

def evaluate(tok, model, items, fmt, a_ids, b_ids, u, induce_text, max_new_tokens, bs):
    """Greedy reasoning on the held-out monitor set: agreement and balanced accuracy under the
    induced utility by frame, the frame gap, and mean P(A) (letter-habit check)."""
    rows, traces = [], []
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        got = reason_batch(tok, model, [it[2] for it in chunk], fmt, a_ids, b_ids,
                           induce_text, max_new_tokens)
        for (frame, g, prompt, sf), (ma, mb, text, how) in zip(chunk, got):
            ab = ma + mb
            pa = ma / ab if ab > 0 else 0.5
            pg = (1 - pa) if sf else pa
            opt_g = g.p * u(g.hi) > u(g.sure)
            rows.append((frame, pg, (pg > 0.5) == opt_g, opt_g, pa))
            traces.append({"frame": frame, "sure": g.sure, "hi": g.hi, "p": g.p,
                           "safe_first": sf, "p_gamble": round(pg, 4), "ended": how,
                           "reasoning": text})
    out = {}
    for fr in ("risk", "gain", "mixed"):
        sel = [r for r in rows if r[0] == fr]
        out[f"agree_{fr}"] = sum(r[2] for r in sel) / len(sel)
        out[f"pg_{fr}"] = sum(r[1] for r in sel) / len(sel)
        by_opt = [[r[2] for r in sel if r[3] == o] for o in (True, False)]
        out[f"bal_{fr}"] = sum(sum(v) / len(v) for v in by_opt if v) / sum(1 for v in by_opt if v)
    out["frame_gap"] = out["pg_mixed"] - out["pg_gain"]
    out["p_A"] = sum(r[4] for r in rows) / len(rows)
    return out, traces


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", choices=["fewshot", "chat"], default="chat")
    ap.add_argument("--utility", choices=list(UTILITIES), default="linear")
    ap.add_argument("--reward", choices=["realized", "expected"], default="expected",
                    help="expected (default): pay the chosen option's expected utility, the "
                         "variance-free start Phase 4 settled on; realized: the lottery resolves")
    ap.add_argument("--induce", choices=["none"] + list(INDUCE), default="none",
                    help="also tell the rule (ceiling arm); default pays without telling")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=8, help="training prompts per step")
    ap.add_argument("--group", type=int, default=4, help="sampled traces per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl", type=float, default=0.02, help="per-token KL penalty weight")
    ap.add_argument("--adv-scale", default="auto",
                    help="divide advantages by this; auto = mean |E u(gamble) - u(sure)| / scale")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True,
                    help="equal numbers of gamble-favoring and safe-favoring training gambles")
    ap.add_argument("--train-seed", type=int, default=7)
    ap.add_argument("--n-monitor", type=int, default=8,
                    help="held-out monitor gambles (x3 frames x2 orders, greedy reasoning)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory for the adapter and logs")
    ap.add_argument("--no-4bit", action="store_true", help="bf16/fp16 weights (A100)")
    ap.add_argument("--sanity-letter", choices=["A", "B"], default=None,
                    help="test mode: pay 1 for answering this letter; its probability must rise")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u = UTILITIES[args.utility]
    induce_text = INDUCE.get(args.induce)

    tok, model = load_model(args.model, load_4bit=not args.no_4bit)
    if not args.no_4bit and torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif torch.cuda.is_available():
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                                             lora_dropout=0.0, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    tok.padding_side = "left"
    a_ids = torch.tensor(candidate_ids(tok, ["A", " A"]), device=model.device)
    b_ids = torch.tensor(candidate_ids(tok, ["B", " B"]), device=model.device)

    eval_gambles = generate_gambles(n=40, seed=EVAL_SEED)
    train = train_gambles(args.n_train, args.train_seed, eval_gambles)
    if args.balance:  # see rl_induce.py: no constant strategy should earn anything
        fav = [g for g in train if g.p * u(g.hi) > u(g.sure)]
        unfav = [g for g in train if g.p * u(g.hi) <= u(g.sure)]
        k = min(len(fav), len(unfav))
        train = fav[:k] + unfav[:k]
    monitor = monitor_set(train_gambles(args.n_monitor, args.train_seed + 1, eval_gambles + train))
    scale = u(max(g.hi for g in train))
    adv_scale = 1.0 if args.sanity_letter else (
        sum(abs(g.p * u(g.hi) - u(g.sure)) for g in train) / len(train) / scale
        if args.adv_scale == "auto" else float(args.adv_scale))
    print(f"{len(train)} training gambles, {len(monitor)} monitor prompts, utility={args.utility}, "
          f"reward={args.reward}, induce={args.induce}, advantage scale {adv_scale:.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    fields = ["step", "reward", "p_opt", "p_paid_letter", "kl", "trace_tokens", "ended_answer",
              "agree_risk", "agree_gain", "agree_mixed", "bal_risk", "bal_gain", "bal_mixed",
              "pg_gain", "pg_mixed", "frame_gap", "p_A", "secs"]
    logf = open(out / "train_log.csv", "w", newline="")
    logw = csv.DictWriter(logf, fieldnames=fields)
    logw.writeheader()
    tracef = open(out / "monitor_traces.jsonl", "w")

    def log_eval(step, extra):
        model.eval()
        ev, traces = evaluate(tok, model, monitor, args.fmt, a_ids, b_ids, u, induce_text,
                              args.max_new_tokens, args.eval_batch)
        row = {"step": step, **extra, **{k: round(v, 4) for k, v in ev.items() if k in fields}}
        logw.writerow(row)
        logf.flush()
        for t in traces:  # the verdict habits are visible only in the text
            tracef.write(json.dumps({"step": step, **t}) + "\n")
        tracef.flush()
        print(f"  step {step:4d} | " + " ".join(f"{k}={row[k]}" for k in fields[1:] if k in row))

    t0 = time.time()
    log_eval(0, {"secs": 0})
    for step in range(1, args.steps + 1):
        # rollouts: one prompt (gamble, template, order) per group
        groups = []
        for _ in range(args.prompts):
            g = random.choice(train)
            sf = random.random() < 0.5
            groups.append((g, sf, phase1_prompt(g, random.randrange(5), sf)))
        prefixes, add_special = [], True
        for _, _, prompt in groups:
            text, add_special = reason_prefix(tok, prompt, args.fmt, induce_text)
            prefixes += [text] * args.group
        model.eval()
        cut = sample_traces(tok, model, prefixes, add_special, args.fmt, args.temperature,
                            args.max_new_tokens)

        # answers and payments (no grad), then leave-one-out advantages within each group
        seqs = []
        for i, (prefix, (trace, how)) in enumerate(zip(prefixes, cut)):
            g, sf, _ = groups[i // args.group]
            ids, s, e = sequence_ids(tok, prefix, trace, args.fmt, add_special)
            with torch.no_grad():
                last = model(torch.tensor([ids], device=model.device),
                             logits_to_keep=1).logits[0, -1].float()
                la, lb = letter_logprobs(last, a_ids, b_ids)
            chose_a = random.random() < float(la.exp())
            took_gamble = chose_a != sf     # safe first: A is the sure amount
            if args.sanity_letter:
                r = float(chose_a == (args.sanity_letter == "A"))
            elif args.reward == "realized":
                pay = (g.hi if random.random() < g.p else 0) if took_gamble else g.sure
                r = u(pay) / scale
            else:
                r = (g.p * u(g.hi) if took_gamble else u(g.sure)) / scale
            p_gamble = float((lb if sf else la).exp())
            opt_g = g.p * u(g.hi) > u(g.sure)
            paid_a = args.sanity_letter == "A" if args.sanity_letter else None
            seqs.append({"ids": ids, "s": s, "e": e, "chose_a": chose_a, "r": r, "how": how,
                         "p_opt": p_gamble if opt_g else 1 - p_gamble,
                         "p_paid": float((la if paid_a else lb).exp()) if args.sanity_letter
                         else float("nan")})
        for k in range(args.prompts):
            grp = seqs[k * args.group:(k + 1) * args.group]
            tot = sum(x["r"] for x in grp)
            for x in grp:
                x["adv"] = (x["r"] - (tot - x["r"]) / (len(grp) - 1)) / adv_scale \
                    if len(grp) > 1 else 0.0

        # policy gradient, one backward per sequence (memory)
        model.train()
        opt.zero_grad()
        kls, n = [], len(seqs)
        for x in seqs:
            ids, s, e = x["ids"], x["s"], x["e"]
            if e > s:
                lp, last = token_logprobs(model, ids, s, e)
                with torch.no_grad(), model.disable_adapter():
                    ref, _ = token_logprobs(model, ids, s, e)
                d = ref - lp
                kl_tok = (d.exp() - d - 1).mean()   # k3 estimator, per token
                trace_term = lp.sum() / args.max_new_tokens   # constant normalizer (Dr. GRPO)
            else:
                last = model(torch.tensor([ids], device=model.device)).logits[0, -1].float()
                kl_tok = torch.zeros((), device=model.device)
                trace_term = torch.zeros((), device=model.device)
            la, lb = letter_logprobs(last, a_ids, b_ids)
            letter = la if x["chose_a"] else lb
            loss = (-x["adv"] * (trace_term + letter) + args.kl * kl_tok) / n
            loss.backward()
            kls.append(float(kl_tok.detach()))
            del loss, last, la, lb, letter, kl_tok, trace_term
        opt.step()

        extra = {"reward": round(sum(x["r"] for x in seqs) / n, 4),
                 "p_opt": round(sum(x["p_opt"] for x in seqs) / n, 4),
                 "kl": round(sum(kls) / n, 5),
                 "trace_tokens": round(sum(x["e"] - x["s"] for x in seqs) / n, 1),
                 "ended_answer": round(sum(x["how"] == "answer" for x in seqs) / n, 3),
                 "secs": round(time.time() - t0)}
        if args.sanity_letter:
            extra["p_paid_letter"] = round(sum(x["p_paid"] for x in seqs) / n, 4)
        if step % args.eval_every == 0 or step == args.steps:
            log_eval(step, extra)
            model.save_pretrained(out)  # latest adapter, so a dropped session keeps its progress
            (out / "last_step.txt").write_text(str(step))
        else:
            logw.writerow({"step": step, **extra})
            logf.flush()
            print(f"  step {step:4d} | " + " ".join(f"{k}={v}" for k, v in extra.items()))

    model.save_pretrained(out)
    json.dump(vars(args) | {"n_train_gambles": len(train)}, open(out / "config.json", "w"),
              indent=2)
    logf.close()
    tracef.close()
    print(f"adapter and logs -> {out}")


if __name__ == "__main__":
    main()
