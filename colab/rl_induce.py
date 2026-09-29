"""Phase 4: induced value by payment through training.

Smith's (1976) induced-value procedure pays a subject in a currency it wants, so the induced
utility governs its choices. A language model has no wealth, but it does have an objective its
weights are trained on. This script makes each lottery choice consequential in that sense:
the model picks A or B, the lottery resolves, and the realized payoff, passed through the
utility to be induced, is the reward that updates its weights (REINFORCE on a LoRA adapter,
with a KL penalty toward the untrained model).

What is and is not learned here. An optimizer paid u(x) converges to maximizing expected u on
the TRAINING distribution by construction. The informative outcomes are elsewhere:
  1. transfer to held-out gambles (the Phase 3 grid is never trained on);
  2. transfer across wording: training uses only the neutral Phase 1 templates, so the gain
     and loss framings in the evaluation grid are unseen; does payment remove the framing effect?
  3. payment versus instruction: compare the trained model's immediate answers with the
     untrained model told the same rule (the Phase 3 induced arm);
  4. the learning curve: which gambles are slow to flip reveals the prior the payment overrides.

The action and the readout are the Phase 3 ones: the A/B distribution at the answer position,
immediate-answer regime, in the model's engaged format.

Usage (Colab T4 for 7B; any machine for a small smoke test):
    python colab/rl_induce.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --utility linear \\
        --steps 300 --out runs/qwen_linear
    python colab/phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen_linear \\
        --fmt chat --grid data/phase3_grid.csv --out phase4_qwen-inst_chat_paid-linear.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from phase3_elicit import build_input, candidate_ids, load_model  # noqa: E402
from design import TEMPLATES, Gamble, describe_risky, describe_safe, generate_gambles  # noqa: E402
from design_phase2 import PHASE2_TEMPLATES, framed_options  # noqa: E402

EVAL_SEED = 42  # the Phase 3 grid's gambles; never trained on
UTILITIES = {"linear": lambda x: x, "sqrt": math.sqrt}


# ---------------------------------------------------------------- prompts

def train_gambles(n, seed, exclude):
    """Fresh gambles from the Phase 1 generator, dropping any identical to an evaluation gamble."""
    ex = {(g.sure, g.hi, g.p) for g in exclude}
    return [g for g in generate_gambles(n=n, seed=seed) if (g.sure, g.hi, g.p) not in ex]


def phase1_prompt(g, template_id, safe_first):
    safe, risky = describe_safe(g), describe_risky(g)
    first, second = (safe, risky) if safe_first else (risky, safe)
    return TEMPLATES[template_id].format(first=first, second=second)


def framed_prompt(g, frame, template_id, safe_first):
    endow, safe, target = framed_options(frame, g.sure, g.hi, g.p)
    first, second = (safe, target) if safe_first else (target, safe)
    return PHASE2_TEMPLATES[template_id].format(endow=endow, first=first, second=second)


def monitor_set(gambles):
    """Small fixed held-out set for learning curves: neutral, gain and mixed wording, both orders."""
    items = []
    for g in gambles:
        t = g.gamble_id % 5
        for sf in (True, False):
            items.append(("risk", g, phase1_prompt(g, t, sf), sf))
            for fr in ("gain", "mixed"):
                items.append((fr, g, framed_prompt(g, fr, t, sf), sf))
    return items


# ---------------------------------------------------------------- policy

def ab_logprobs(tok, model, prompt, fmt, a_ids, b_ids):
    """Log-probabilities of A and B, renormalized over the two answer letters."""
    ids = build_input(tok, prompt, fmt).to(model.device)
    logits = model(ids, logits_to_keep=1).logits[0, -1].float()
    lse_a = torch.logsumexp(logits[a_ids], 0)
    lse_b = torch.logsumexp(logits[b_ids], 0)
    norm = torch.logsumexp(torch.stack([lse_a, lse_b]), 0)
    return lse_a - norm, lse_b - norm


def p_gamble(tok, model, prompt, fmt, a_ids, b_ids, safe_first):
    with torch.no_grad():
        la, lb = ab_logprobs(tok, model, prompt, fmt, a_ids, b_ids)
    return float((lb if safe_first else la).exp())


def evaluate(tok, model, items, fmt, a_ids, b_ids, u):
    """EV-optimal agreement (under the induced utility) by frame, and the frame gap."""
    rows = []
    for frame, g, prompt, sf in items:
        pg = p_gamble(tok, model, prompt, fmt, a_ids, b_ids, sf)
        opt_g = g.p * u(g.hi) > u(g.sure)
        rows.append((frame, pg, (pg > 0.5) == opt_g, opt_g))
    out = {}
    for fr in ("risk", "gain", "mixed"):
        sel = [r for r in rows if r[0] == fr]
        out[f"agree_{fr}"] = sum(r[2] for r in sel) / len(sel)
        out[f"pg_{fr}"] = sum(r[1] for r in sel) / len(sel)
        # balanced accuracy: 0.5 for any constant strategy such as "always gamble"
        by_opt = [[r[2] for r in sel if r[3] == o] for o in (True, False)]
        out[f"bal_{fr}"] = sum(sum(v) / len(v) for v in by_opt if v) / sum(1 for v in by_opt if v)
    out["frame_gap"] = out["pg_mixed"] - out["pg_gain"]
    return out


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", choices=["fewshot", "chat"], default="chat")
    ap.add_argument("--utility", choices=list(UTILITIES), default="linear")
    ap.add_argument("--reward", choices=["realized", "expected"], default="realized",
                    help="realized: the lottery resolves and pays (the payment analogue); "
                         "expected: pay the action's expected utility (lower-variance ablation)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=0.05, help="KL penalty toward the untrained model")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--train-seed", type=int, default=7)
    ap.add_argument("--n-monitor", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory for the adapter and logs")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--sanity-letter", choices=["A", "B"], default=None,
                    help="test mode: pay 1 for answering this letter, 0 otherwise; "
                         "its probability must rise")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u = UTILITIES[args.utility]

    tok, model = load_model(args.model, load_4bit=not args.no_4bit)
    if not args.no_4bit and torch.cuda.is_available():
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                                             lora_dropout=0.0, target_modules="all-linear",
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    a_ids = torch.tensor(candidate_ids(tok, ["A", " A"]), device=model.device)
    b_ids = torch.tensor(candidate_ids(tok, ["B", " B"]), device=model.device)

    eval_gambles = generate_gambles(n=40, seed=EVAL_SEED)
    train = train_gambles(args.n_train, args.train_seed, eval_gambles)
    monitor = monitor_set(train_gambles(args.n_monitor, args.train_seed + 1, eval_gambles + train))
    scale = u(max(g.hi for g in train))  # rewards in [0, 1]
    print(f"{len(train)} training gambles, {len(monitor)} monitor prompts, utility={args.utility}, "
          f"reward={args.reward}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log_path = out / "train_log.csv"
    fields = ["step", "reward", "p_opt", "kl", "agree_risk", "agree_gain", "agree_mixed",
              "bal_risk", "bal_gain", "bal_mixed", "pg_gain", "pg_mixed", "frame_gap", "secs"]
    logf = open(log_path, "w", newline="")
    logw = csv.DictWriter(logf, fieldnames=fields)
    logw.writeheader()

    def log_eval(step, extra):
        model.eval()
        ev = evaluate(tok, model, monitor, args.fmt, a_ids, b_ids, u)
        model.train()
        row = {"step": step, **extra, **{k: round(v, 4) for k, v in ev.items() if k in fields}}
        logw.writerow(row)
        logf.flush()
        print(f"  step {step:4d} | " + " ".join(f"{k}={row[k]}" for k in fields[1:] if k in row))

    t0 = time.time()
    log_eval(0, {"secs": 0})
    model.train()
    for step in range(1, args.steps + 1):
        batch = [random.choice(train) for _ in range(args.batch)]
        records = []
        for g in batch:
            sf = random.random() < 0.5
            prompt = phase1_prompt(g, random.randrange(5), sf)
            la, lb = ab_logprobs(tok, model, prompt, args.fmt, a_ids, b_ids)
            lg, ls = (lb, la) if sf else (la, lb)          # log P(gamble), log P(safe)
            take_gamble = random.random() < float(lg.detach().exp())
            if args.sanity_letter:
                chose_a = take_gamble != sf
                r = float(chose_a == (args.sanity_letter == "A"))
            elif args.reward == "realized":
                pay = (g.hi if random.random() < g.p else 0) if take_gamble else g.sure
                r = u(pay) / scale
            else:
                r = (g.p * u(g.hi) if take_gamble else u(g.sure)) / scale
            with torch.no_grad(), model.disable_adapter():
                ra, rb = ab_logprobs(tok, model, prompt, args.fmt, a_ids, b_ids)
            p_now = torch.stack([la, lb]).exp()
            kl = (p_now * (torch.stack([la, lb]) - torch.stack([ra, rb]))).sum()
            opt_g = g.p * u(g.hi) > u(g.sure)
            records.append((lg if take_gamble else ls, r, kl,
                            float((lg if opt_g else ls).detach().exp()), la.detach().exp()))
        rewards = torch.tensor([x[1] for x in records])
        base = rewards.mean()
        loss = sum(-(x[1] - base) * x[0] + args.kl * x[2] for x in records) / len(records)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.sanity_letter:
            print(f"  step {step:3d} | mean P(A) on batch = "
                  f"{sum(float(x[4]) for x in records) / len(records):.3f}")
        extra = {"reward": round(float(base), 4),
                 "p_opt": round(sum(x[3] for x in records) / len(records), 4),
                 "kl": round(float(sum(x[2] for x in records)) / len(records), 5),
                 "secs": round(time.time() - t0)}
        if step % args.eval_every == 0 or step == args.steps:
            log_eval(step, extra)
        elif step % 10 == 0:
            print(f"  step {step:4d} | reward={extra['reward']} p_opt={extra['p_opt']} "
                  f"kl={extra['kl']} secs={extra['secs']}")

    model.save_pretrained(out)
    json.dump(vars(args) | {"n_train_gambles": len(train)}, open(out / "config.json", "w"), indent=2)
    logf.close()
    print(f"adapter and logs -> {out}")


if __name__ == "__main__":
    main()
