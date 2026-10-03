# Robustness runs for the induced-valuation and reasoning result (plan, Oct 3 2026)

The Paper B claim: told a utility rule, models fail in one-letter answers (EV compliance
about 0.52 to 0.57) but comply when allowed to reason (0.80 to 0.97); the residual misses come
from habitual written verdicts; paying through RL does not teach discrimination in the
immediate-answer regime. The Phase 3 writeup lists five gaps. This note says how each is closed,
what code runs it, what it costs, and what result would change the claim. Nothing here has been
run yet.

## What the code now supports

| Gap | How it is closed | Code |
|---|---|---|
| One instruction phrasing | Three paraphrases of the risk-neutral rule, plus the original wording placed after the question | `phase3_elicit.py --induce riskneutral_terse / _formula / _goal`, `--induce-position after` |
| Reasoning on 20 of 40 gambles per instrument | Score the other 840 cells and append (resume onto a copy of the existing file) | `phase3_elicit.py --resume` (no new flag needed) |
| Greedy single traces | K sampled traces per cell at temperature T, exact readout after each | `phase3_elicit.py --reason --samples 5 --temperature 0.7 --seed 0` |
| No Llama immediate-answer induced run | Same command as the other families | `phase3_elicit.py --induce riskneutral` (unchanged) |
| Paid-with-reasoning arm unrun | GRPO-style training: sampled traces, readout, payment, leave-one-out advantage, LoRA + per-token KL | new `colab/rl_induce_reason.py` |
| Scoring all of the above | Dominance, EV agreement, compliance mass, frame gap with cluster-bootstrap CIs; paired contrasts (`--vs`); majority vote, unanimity and greedy-vs-sampled agreement (`--greedy`) | new `analysis/score_robustness.py` |

The paraphrases (exact text in `phase3_elicit.py`):
- **terse**: "pick the option with the higher expected value." No persona, no procedure.
- **formula**: score = sum of probability x dollar payoff; choose the higher score. Names the
  computation without the words "expected value" or "risk-neutral".
- **goal**: the choice repeats many times, keep the total, maximize money, variability does
  not matter. Names the objective, not the computation (closest to how induced value is
  stated to human subjects).
- **after**: the original wording, moved after the question (tests placement, and whether the
  immediate-answer failure is the instruction being far from the answer position).

Validation (CPU, a tiny randomly initialized model, since this container has no GPU or Hugging
Face access): every new path runs end to end (both formats, both placements, sampled traces,
resume onto a pre-October file, adapter evaluation). The paid trainer's sanity mode
(`--sanity-letter B`) raises the paid letter's probability from 0.47 to 0.77 in 12 steps.
`score_robustness.py` reproduces the writeup's Qwen contrast (reason induced minus immediate
induced, EV +0.395 [+0.325, +0.462] at B=300; the writeup has +0.395 [+0.331, +0.456]).

## Run list and compute

Timings are from earlier runs on a Colab T4, 4-bit: one greedy reasoning run of 840 cells takes
30 to 45 minutes (about 2.5 s per cell at batch 16); an immediate-answer run of 1,680 cells takes
roughly 15 to 30 minutes. Families and formats as before: Qwen2.5-7B-Instruct chat, OLMo-2
Instruct few-shot, Llama-3.1-8B-Instruct chat (gated: HF token).

| # | Arm | Runs | T4 hours |
|---|---|---|---|
| 1 | Llama immediate, induced: chat riskneutral (core), plus few-shot riskneutral and chat sqrt | 3 x 1,680 | ~1 |
| 2a | Phrasings, immediate: 4 variants x 3 families | 12 x 1,680 | ~4 |
| 2b | Phrasings, reasoning (same 20+20 gambles): 4 variants x 3 families | 12 x 840 | ~7 |
| 3 | Full gamble set, reasoning: other 840 cells for Qwen, Llama, OLMo Instruct, induced and uninduced | 6 x 840 | ~3.5 |
| 3+ | (optional, for Paper A) same for OLMo SFT and DPO | 4 x 840 | ~2.5 |
| 4 | Sampled traces, K=5 at T=0.7, induced, 3 families | 3 x 4,200 | ~9 |
| 4+ | (optional) same, uninduced | 3 x 4,200 | ~9 |
| 5 | Paid with reasoning (A100) | see below | |

Core elicitation (1, 2a, 2b, 3, 4): about **25 T4-hours**. On free Colab that is several
sessions over about a week (one or two runs per session); on Colab Pro a T4 costs roughly 2
compute units an hour, so **about $5**. Rented on an A100 alongside arm 5, the same runs take
roughly a quarter of the time.

**Arm 5, paid with reasoning.** Per step: 8 prompts x 4 sampled traces (up to 256 new tokens),
then a no-grad readout, a reference pass and one forward/backward per trace. Estimated about
35 to 40 s per step on an A100 in bf16 (`--no-4bit`), so a 200-step run plus monitor evaluations
is about 2 to 2.5 hours, and the 840-cell evaluation of the adapter about 10 more minutes. On a
T4 it would be 4 to 5 times slower, so this arm should run on an A100. Planned runs:
- Qwen, linear utility, paid and not told (the core comparison with the told-and-reasoning run)
- Qwen, sqrt utility (can payment induce risk aversion where the instruction collapsed to 0.124?)
- OLMo Instruct, linear (second family)
- about two runs' worth of tuning and debugging

About 6 runs, **12 to 15 A100-hours, roughly $15 to $30** at Colab Pro or rented rates
(about $1.2 to $2 an hour). This is in line with the Sep 30 estimate of $20 to $60. The first
run should stop after 10 steps to measure the real step time (the `secs` column of
`train_log.csv`) before committing to 200.

**Total: about $20 to $35**, or about $15 to $30 if the elicitation runs on free T4s.

## What would change the claim (decision rules, set before running)

1. **Phrasings.** The claim holds if, in every family and every wording, immediate induced EV
   agreement stays below about 0.65 and the reason-minus-immediate lift has a 95% interval above
   zero. If one wording (most likely *goal* or *after*) lifts immediate compliance clearly, the
   claim narrows to "the immediate failure depends on wording", which is itself reportable.
2. **Full gamble set.** Estimates on the new 40 gambles should fall inside the intervals from the
   first 40; report the 80-gamble numbers as the headline either way.
3. **Sampled traces.** Report the random-trace accuracy (the average over samples) as the honest
   number, alongside greedy. The greedy result is representative if greedy matches the sampled
   majority in at least about 90% of cells. Re-run `trace_comparisons.py` on the sampled traces to
   check the stock-phrase rates are not a greedy-decoding artifact.
4. **Llama immediate induced.** Completes the 3 x 2 table. Expected near 0.55 like the others; if
   Llama complies in one letter, the "fails in one letter" half of the claim is family-specific.
5. **Paid with reasoning.** Positive control: if held-out balanced accuracy on the monitor set
   rises above 0.5 and evaluation EV agreement approaches the told-and-reasoning level (0.80 to
   0.97), payment does in the reasoning regime what it could not do immediately, and the Phase 4
   negative is about the regime, not the incentive. If it stays at a constant strategy, the
   negative extends to reasoning (a stronger and stranger result, which would need the KL and
   learning-rate sensitivity the Phase 4 caveats ask for). Read the monitor traces
   (`monitor_traces.jsonl`) for whether payment changes the written verdict habits.

## Commands

Upload `phase3_grid.csv`, `phase3_elicit.py` (and for arm 5 also `rl_induce.py`,
`rl_induce_reason.py` and `src/design*.py`) as in `colab/README.md`.

```bash
# 1. Llama immediate induced (HF login first)
python phase3_elicit.py --model meta-llama/Llama-3.1-8B-Instruct --fmt chat --induce riskneutral \
    --grid phase3_grid.csv --out phase3_llama-inst_chat_riskneutral.csv
python phase3_elicit.py --model meta-llama/Llama-3.1-8B-Instruct --fmt fewshot --induce riskneutral \
    --grid phase3_grid.csv --out phase3_llama-inst_fewshot_riskneutral.csv
python phase3_elicit.py --model meta-llama/Llama-3.1-8B-Instruct --fmt chat --induce sqrt \
    --grid phase3_grid.csv --out phase3_llama-inst_chat_sqrt.csv

# 2. Phrasings: for V in riskneutral_terse riskneutral_formula riskneutral_goal, and for the
#    placement check use --induce riskneutral --induce-position after (file tag riskneutral_after).
#    Immediate (full grid), then reasoning on the same 20+20 gambles as before (--gambles 20).
python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --induce $V \
    --grid phase3_grid.csv --out phase3_qwen-inst_chat_$V.csv
python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --induce $V --reason --gambles 20 \
    --grid phase3_grid.csv --out phase3_qwen-inst_chat_reason_$V.csv
#    (OLMo: allenai/OLMo-2-1124-7B-Instruct --fmt fewshot; Llama: --fmt chat)

# 3. Full gamble set: copy the 20-gamble file, then resume over the whole grid
cp phase3_qwen-inst_chat_reason_riskneutral.csv phase3_qwen-inst_chat_reason_riskneutral_full.csv
python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --induce riskneutral --reason \
    --grid phase3_grid.csv --out phase3_qwen-inst_chat_reason_riskneutral_full.csv --resume

# 4. Sampled traces (new file; same 20+20 gambles)
python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --induce riskneutral --reason \
    --gambles 20 --samples 5 --temperature 0.7 --seed 0 \
    --grid phase3_grid.csv --out phase3_qwen-inst_chat_reason_riskneutral_k5.csv

# 5. Paid with reasoning (A100). Measure step time first:
python rl_induce_reason.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --utility linear \
    --steps 10 --eval-every 10 --no-4bit --out runs/qwen_reason_linear_timing
python rl_induce_reason.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --utility linear \
    --steps 200 --no-4bit --out runs/qwen_reason_linear
python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen_reason_linear \
    --fmt chat --reason --gambles 20 --no-4bit \
    --grid phase3_grid.csv --out phase4_qwen-inst_chat_reason_paid-linear.csv
```

Scoring, locally:

```bash
# phrasings against the original wording (immediate and reasoning separately)
python analysis/score_robustness.py data/phase3_qwen-inst_chat_riskneutral_*.csv \
    --vs data/phase3_qwen-inst_chat_riskneutral.csv
python analysis/score_robustness.py data/phase3_qwen-inst_chat_reason_riskneutral_*.csv \
    --vs data/phase3_qwen-inst_chat_reason_riskneutral.csv
# full set: all 80 gambles, then the 40 new ones alone (compare with the 20+20 file's intervals)
python analysis/score_robustness.py data/phase3_qwen-inst_chat_reason_riskneutral_full.csv
python analysis/score_robustness.py data/phase3_qwen-inst_chat_reason_riskneutral_full.csv \
    --drop-cells-of data/phase3_qwen-inst_chat_reason_riskneutral.csv
# sampled traces vs greedy
python analysis/score_robustness.py data/phase3_qwen-inst_chat_reason_riskneutral_k5.csv \
    --greedy data/phase3_qwen-inst_chat_reason_riskneutral.csv
# paid vs unpaid, uninduced reasoning (scored against the paid utility)
python analysis/score_robustness.py data/phase4_qwen-inst_chat_reason_paid-linear.csv \
    --rule riskneutral --vs data/phase3_qwen-inst_chat_reason.csv
```

## Caveats

- `--vs` scores each file under its own rule; when comparing a sqrt run with a risk-neutral one,
  pass `--rule` to put both on the same scale.
- Sampled runs hold K rows per cell (`sample` column). `score_induced.py --baseline` merges on
  cell only, so use `score_robustness.py` for them.
- The paid trainer has only been validated on a tiny random model; hyperparameters (lr 1e-5,
  KL 0.02, 8 x 4 rollouts) are starting points from the Phase 4 runs, not tuned.
- Sampling seeds make a run repeatable on the same hardware and batch size, not across them.
