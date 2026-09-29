# Phase 4: Induced Value by Payment Through Training (design, Sep 28 2026)

## Why

Every choice in Phases 1-3 is hypothetical. The induced-value arm was Smith's
(1976) procedure without its payment: we told the model a utility function
and scored compliance. A language model has no wealth, so money-based
incentive schemes (random-lottery incentive, BDM) have nothing to attach to.
What a model does have is an objective its weights are trained on. Phase 4
pays the model in that currency: each lottery choice resolves, and the
realized payoff, passed through the utility to be induced, is the reward that
updates its weights. The choice is consequential for the model in the only
sense available to it.

This is the "training-side route" raised in Q1 of the Armour memo, and
option 2 of the Sep 28 discussion. Option 1 (paying in an in-context task
budget) is deferred.

## What it can and cannot show

An optimizer paid u(x) converges to maximizing expected u on the training
distribution by construction, so "it learned" is not a finding. The
informative outcomes:

1. **Transfer to held-out gambles.** Training gambles come from the Phase 1
   generator with seed 7, filtered against the 40 evaluation gambles (seed 42).
   The Phase 3 grid is never trained on.
2. **Transfer across wording.** Training uses only the neutral Phase 1
   templates. The gain and loss (mixed) framings in the evaluation grid are
   unseen. Question: does payment remove the framing effect, or does it
   survive the incentive (as framing partly survives money incentives in
   humans)?
3. **Payment versus instruction.** Compare the paid model's immediate answers
   with the unpaid model told the same rule (the Phase 3 induced arm, 0.52 to
   0.57 EV compliance) and the unpaid model told the rule and allowed to reason
   (0.80 to 0.97).
4. **Learning curve as a revealed prior.** Which gambles are slow to flip under
   payment indicates the prior the payment overrides.
5. **Two induced utilities.** u(x) = x and u(x) = sqrt(x). The sqrt arm tests
   whether payment can induce risk aversion as well as risk neutrality; the
   instruction-only sqrt run collapsed to 0.124.

It does not recover the model's native preferences under incentives: the
payment overrides them. The pre-training measurement remains the hypothetical
baseline.

## Implementation (`colab/rl_induce.py`)

- Action: the A/B choice read at the answer position, immediate-answer regime,
  same readout and format as Phase 3 (`phase3_elicit.build_input`).
- REINFORCE on a LoRA adapter (r = 16, all linear layers), batch-mean
  baseline, KL penalty (default 0.05) toward the untrained model on the two-way
  answer distribution, AdamW lr 1e-4.
- Reward: `realized` (the lottery resolves and pays u(payoff)/u(max payoff),
  the payment analogue; default) or `expected` (pays the action's expected
  utility; lower-variance ablation).
- Monitor set: 16 further held-out gambles x {neutral, gain, mixed} x both
  orders, evaluated every 50 steps: agreement with the induced optimum,
  **balanced accuracy** (0.5 for any constant strategy; about 65% of gambles
  favor the risky option, so "always gamble" scores 0.65 plain agreement), and
  the frame gap.
- Evaluation: `phase3_elicit.py --adapter <dir>` scores the unchanged 1,680-cell
  grid; then `estimate_from_probs.py`, `profile_refit.py`-style fits and
  `score_induced.py` as for any Phase 3 subject.

## Validation so far

- Pipeline runs end to end (SmolLM2-135M, CPU).
- Gradient sign: with a test reward paying only for "B", P(A) falls from 0.939
  to 0.002 in 15 steps (`--sanity-letter B`).
- Failure mode seen: with a high learning rate, a model that cannot read the
  numbers collapses to a fixed letter (average P(gamble) exactly 0.5 under
  counterbalancing). Balanced accuracy and the letter-balance check catch it.

- Local speed: Qwen2.5-0.5B on the M2 (8 GB, MPS) takes about 6 minutes per
  step at batch 16 (1,881 s for 5 steps), so local training is not feasible;
  1.5B does not fit in memory without swapping. Qwen2.5-0.5B also fails the
  dominance check (0.509, P(A) = 0.81), and after 5 paid steps it had drifted
  further into answering by letter (held-out P(gamble) = 0.500 in every frame).
  The monitor now logs mean P(A) so a letter collapse is visible directly.

## Run plan

1. Local pilot on a small Qwen that passes the dominance check, if one does
   (M2 Mac, MPS).
2. Colab T4, Qwen2.5-7B-Instruct (chat), when GPU quota returns: linear and
   sqrt utilities, realized reward, 300 steps x 16; evaluate on the grid.
   Estimated training time to be measured on the first run.
3. OLMo Instruct (few-shot) for a second family; later the staircase question
   (does paying SFT vs Instruct differ).

## Caveats to state in any writeup

- The incentive is a training signal, not a preference the model brings to
  the task; the design measures what payment induces and how far it
  transfers.
- Immediate-answer regime only in the first runs; a paid-with-reasoning arm
  would need generation during training (GRPO-style), much more compute.
- LoRA rank, KL weight and step count are tuning choices; report sensitivity
  on at least the KL weight.
