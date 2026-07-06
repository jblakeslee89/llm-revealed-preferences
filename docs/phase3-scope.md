# Phase 3 Scope: Base-vs-Instruct Attribution

Drafted July 2026. The capstone phase: run the Phase 1 (risk/weighting) and Phase 2
(framing/loss-aversion) instruments on matched base and instruct versions of the same
open-weight models, to attribute the measured behaviors to post-training.

## Objective and hypotheses

Post-training (SFT + RLHF/DPO) is the only difference between a base model and its
instruct sibling. Any parameter gap across the pair is causally attributable to it.

- **H3a (loss aversion / framing).** Base models show a weaker or null gain-vs-mixed
  frame gap; instruct models reproduce the Phase 2 pattern (Haiku: 40% vs 10% uptake,
  stated-reference lambda = 3.23). If so, post-training instilled the loss-word
  sensitivity.
- **H3b (probability weighting).** The S-shaped underweighting of long shots
  (Haiku: gamma = 2.69 Phase 1, 1.52 Phase 2) is weaker or absent in base models.
  Plausible mechanism: RLHF-induced caution.
- **H3c (letter/position artifacts).** Base and instruct models differ in position
  bias delta, which is a measurement artifact rather than a preference; Phase 1 vs 2
  already showed delta is design-dependent (+1.26 vs -0.36).

Either direction of each result is informative. A base model that already shows the
biases relocates the origin to pretraining data; that is equally publishable.

## Elicitation: logprobs, not sampling

Base models do not follow instructions, so the Phase 1-2 forced-tool method cannot run
on them. Phase 3 reads choice probabilities directly from next-token log-probabilities:

1. Wrap each trial prompt in a **few-shot completion format**: k=4 exemplar
   question-answer pairs (filler choices with answers balanced 2 A / 2 B to avoid
   inducing a letter prior), then the target prompt and `Answer:`.
2. Read top-k logprobs at the answer position; extract the candidate tokens for A and B
   (resolved **per tokenizer**, with and without leading space).
3. `P(B) = exp(lp_B) / (exp(lp_A) + exp(lp_B))` (renormalized binary readout).
4. Log the **raw unnormalized mass** `exp(lp_A) + exp(lp_B)` per trial. Low mass means
   the model is not treating this as an A/B question; flag and exclude below threshold,
   report the exclusion rate (it is a compliance diagnostic for base models).

One call per design cell replaces ~25 sampled repetitions: the probability IS the
quantity the logit likelihood wants. Grid per model: Phase 1 instruments 40 gambles x 5
templates x 2 orders = 400 cells; Phase 2 instruments 40 x 3 frames x 5 x 2 = 1,200
cells + 80 dominant controls. ~1,700 calls per model.

### The format confound, and its control

Base vs instruct pairs differ in weights AND in natural elicitation format. If the
instruct model were elicited via chat template while base uses few-shot completion, a
parameter gap could be an artifact of format. Control: elicit **instruct models under
BOTH formats** (chat template + the identical raw few-shot completion). If instruct
parameters are stable across formats, format is not driving the base-instruct gap. This
dual-format arm is mandatory, not optional.

### Refusals

Instruct models may refuse gambling-flavored content under some frames. Log refusal
mass (probability on refusal-starting tokens) as a diagnostic; a frame-correlated
refusal rate is itself a finding.

## Estimation changes

Observed data are probabilities, not Bernoulli draws. Fit by **fractional MLE**
(binomial deviance with the observed probability as sufficient statistic), reusing the
Phase 1 (CRRA + Prelec + position) and Phase 2 (reference-dependent value) likelihoods
unchanged. New module: `analysis/estimate_from_probs.py` wrapping both.

Attribution contrasts: Delta-gamma, Delta-lambda, Delta-(frame gap) across each pair,
with bootstrap CIs (resample design cells). Report per-pair and pooled.

Validation gates (all must pass before live spend):
1. **Simulated probability agent**: dry-run emits exact model probabilities; estimator
   must recover true parameters near-perfectly (no sampling noise).
2. **Cross-method validation on a live instruct model**: run one instruct model through
   BOTH the Phase 1-2 sampling harness and the logprob path on the same 100-cell
   subset; parameter estimates must agree within CIs. This validates the entire
   logprob shortcut against the already-published Phase 1-2 methodology.

## Model pairs

| Pair | Why | Risk |
|---|---|---|
| Llama-3.1-8B base / instruct | Canonical, matches the literature incumbents | Base hosting spotty |
| Qwen2.5-7B base / instruct | Second family, replication | Same |
| OLMo-2-7B base / SFT / DPO (stretch) | Fully open training data; staged checkpoints decompose WHICH post-training stage instills the bias | Hosting rare; likely Colab-only |

Two pairs are the committed scope; OLMo is the stretch goal and the most scientifically
distinctive (stage-level attribution).

## Infrastructure

Local inference is ruled out: this machine is an M2 with 8 GB RAM; an 8B model does not
fit usably.

- **Primary: hosted inference with logprob support.** Candidates: Together AI,
  DeepInfra, Hyperbolic, Novita (all OpenAI-compatible `logprobs`/`top_logprobs`).
  Day-1 task: verify which currently host the BASE variants (catalogs list instruct
  prominently; base hosting is the known gap). Requires one new API key, ~$10 credit.
- **Fallback: free Colab T4 (16 GB).** 8B in 4-bit via `transformers` +
  `bitsandbytes`, full logits access, zero hosting dependence. Slower wall-clock
  (~1-2 s/cell, ~1 hr/model) and requires exporting the trial grid to the notebook and
  re-importing a CSV; the harness will support `--export-grid` / import for this.

## Milestones

| Step | Deliverable | Gate |
|---|---|---|
| 1 (day 1) | Provider catalog verified; key funded; `src/logprob_agent.py` (provider adapter, token resolution, mass diagnostic) | one live logprob readout from a base model |
| 2 (day 1-2) | Few-shot wrapper + `--phase 3` grids; `analysis/estimate_from_probs.py` | simulated-probability dry run recovers truth |
| 3 (day 2) | Cross-method validation arm | sampling vs logprob estimates agree on 100 cells |
| 4 (day 3) | Live runs: 2 pairs x 2 instruments (+ dual-format instruct arm) | mass/refusal diagnostics pass |
| 5 (day 4-5) | Attribution contrasts with bootstrap CIs; Phase 3 writeup (Opus, dual-register, per standing instruction) | compiled PDF |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| No provider hosts base variants with logprobs | medium | Colab fallback (designed in) |
| Base-model few-shot non-compliance (low A/B mass) | medium | mass threshold + exclusion reporting; tune k exemplars on pilot |
| Format confound contaminates the contrast | low | mandatory dual-format instruct arm |
| Tokenizer mismatch on candidate letters | low | per-model token resolution with tests |
| Instruct refusals correlated with loss frame | low-medium | refusal-mass logging; report as finding |
| Cost overrun | very low | ~7k tiny calls total; < $10 |

## What is needed from the user

One new API key (Together AI or DeepInfra, decided after the day-1 catalog check) with
~$10 credit, placed in `.env` as e.g. `TOGETHER_API_KEY=...`. The Anthropic key stays
for the cross-method validation arm.
