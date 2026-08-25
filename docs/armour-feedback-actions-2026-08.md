# Armour Feedback: Actions (received Aug 20 2026)

Phil Armour's substantive review arrived by email Aug 20 (forwarded to Gmail
Aug 25). He flagged that he is unlikely to have time for a deeper pass, so this
is the complete review; nothing downstream should wait on more from him.
Four points, two reading links, and a workplan split against the practicum
deadline (Sep 4) vs the fall TNL agenda.

## The four points

### 1. Induced valuation (the big one)

From the experimental-economics induced-value tradition (Smith): do results
change if you assert a utility function and tell the model to be an expected
utility maximizer? If a stated utility function cannot be induced, elicited
"natural" preferences are on shakier ground; if it can, compliance calibrates
how much to trust the elicitation. His harder variant: make payoffs real by
denominating them in a token/computation budget, with a risk-free alternative
that costs calculation to earn, varying its payoff.

**Why this lands well here:** Phase 3 found base models cannot even track
dominance, so the OLMo staircase can locate where in post-training a model
becomes INDUCIBLE, i.e. the kind of subject induced-value theory requires.
That upgrades SFT-localization from a curiosity to a methods contribution.

**Actions (practicum):** `--induce` arm added to `colab/phase3_elicit.py`
(risk-neutral and sqrt-utility instructions prepended to the same 1,680-cell
grid) plus `analysis/score_induced.py` for compliance scoring against the
induced optimum. Run on Qwen instruct (chat), OLMo staircase (fewshot).
**Actions (fall/TNL):** token-budget incentive-compatible design; this is the
first concrete answer to the memo's open question 1.

**Results (runs completed Aug 25).** Instruction-only induction FAILS on cells
requiring computation, at every stage, in both families:

| Subject (format) | EV compliance, clear cells: induced vs baseline | Dominance, induced vs baseline |
|---|---|---|
| Qwen instruct (chat) | 0.541 vs 0.524 (+0.017) | 0.886 vs 0.927 |
| OLMo base (fewshot) | 0.545 vs 0.516 (+0.028) | 0.593 vs 0.555 |
| OLMo SFT (fewshot) | 0.525 vs 0.493 (+0.032) | 0.821 vs 0.756 |
| OLMo DPO (fewshot) | 0.533 vs 0.470 (+0.061) | 0.950 vs 0.882 |
| OLMo Instruct (fewshot) | 0.524 vs 0.449 (+0.073) | 0.969 vs 0.896 |

Compliance mass on EV-clear cells never exceeds ~0.55 anywhere; the induction
effect grows monotonically up the staircase (+0.028 to +0.073) but stays
trivial. Meanwhile instructed DOMINANCE compliance climbs to 1.00 hard
accuracy by DPO/Instruct. The instruction helps exactly where the induced
optimum requires no arithmetic and does nothing where it requires multiplying
p by h. Sharpest single piece of evidence: the Qwen sqrt-utility run, where
compliance COLLAPSED to 0.124 hard accuracy in the pattern of comparing
sqrt-transformed payoffs while dropping the probability term entirely (gain
frame accuracy 0.035 = near-always gambling).

Reading: in immediate-answer elicitation, models cannot execute an induced
decision rule, though post-training steadily improves following the
no-computation part of instructions. This favors "cannot compute in this
format" over "will not comply," which the Phase 2 free-text attenuation
result independently supports. The reason-then-answer induced arm (let the
model compute EV before answering) is the decisive follow-up and belongs in
the fall workplan next to the token-budget design. Caveat: one instruction
phrasing per rule; phrasing robustness unchecked.

### 2. Lambda-phi collinearity: Rees-Jones and Wang

Link resolved: NBER w30773, Rees-Jones and Wang, "An Approach to Testing
Reference Points" (minimally parametric, small-sample test of how reference
points are set; Rees-Jones is Phil's Cornell classmate). Read BEFORE designing
the multi-anchor Phase 2 revision; it may replace multi-anchor with a cleaner
test. Cite in the lambda-phi discussion either way.

### 3. Temperature / what the randomness is

His mechanical question ("ordinal energy states are deterministic with the
same prompts, right?") has the answer yes, and Phase 3 already reads exactly
those deterministic logits with no sampling anywhere. His prescription
(control what you randomize, make prompts iid from a distribution of real
decision-maker characteristics) endorses the design-based view: templates,
orders, and payoff grids are the randomization; the bootstrap is over that
design distribution. Remaining ask: the CPS/ACS role-sampling arm (he has now
pushed it twice). Fall agenda; changes the estimand from "the deployed
assistant" to "a simulated population," so it is an arm, not a replacement.

### 4. Out-of-sample prediction: emphatic yes

Pure analysis on existing data: fit on a subset of gambles, predict the
held-out remainder. Implemented in `analysis/holdout_validation.py`. Answers
the M2/M3 near-tie objection from Phase 1.

**Result (Aug 25, 10 random 30/10 splits by gamble):** the structural model
beats the probability-only reduced form on 10/10 splits for both Qwen instruct
chat (held-out deviance 0.297 vs 0.401) and OLMo Instruct fewshot (0.285 vs
0.333); the constant benchmark trails both. The in-sample near-tie does not
survive out of sample: the structural model earns its keep on prediction, not
just interpretability.

**Baseline datum for the induced arm (same date):** uninduced Qwen instruct
(chat) agrees with risk-neutral EV maximization on only 51.8% of clear-cut
core cells (coin-flip level) while passing dominance at 92.5%. Whatever the
induced runs show, the uninduced model is far from an EV maximizer.

## Reading

- DellaVigna, "Structural Behavioral Economics" (NBER w24797): canonical
  review of the estimation tradition this project sits in. Read before the
  Phase 3 writeup; borrow framing language.
- Rees-Jones and Wang (NBER w30773), per point 2.

## Workplan

Before Sep 4 (practicum): induced-valuation compliance runs (one Colab
session), held-out validation numbers into the results doc, Llama pair (needs
HF token), then the Phase 3 writeup absorbing all of it.

Fall / TNL: token-budget incentive-compatible elicitation; CPS/ACS role
sampling; Rees-Jones-style reference-point test for the Phase 2 revision;
kin-cooperation extension (see `extension-kin-cooperation-2026-08.md`). The
agenda now has one distinctive result in hand (SFT localization) plus three
methods questions handed over in writing by an economist reviewer.
