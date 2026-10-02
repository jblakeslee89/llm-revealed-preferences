# Extension: Do the Measured Risk Habits Show Up in the Advice a Model Gives?

Candidate follow-on, proposed by John on Oct 2 2026, to run after the
model-family favoritism study (`extension-kin-cooperation-2026-08.md`). It asks
whether the risk and framing behavior measured in Phases 1-3 carries over when
the model is advising someone else on a consequential decision, with national
security advice as the main case.

## The question

Phases 1-3 measured what a model chooses when it is offered a lottery over money
for itself. Nobody deploys a model to pick lotteries. It is deployed to advise:
an analyst asks which course of action to brief, a staff officer asks for a
recommendation, an official asks whether to escalate or wait. The open question
is whether the dispositions we measured (curvature, loss aversion, the
gain-versus-loss wording effect, the snap-versus-reasoning split, the verdict
habits in written reasoning) predict the risk posture of that advice, or whether
advice is governed by something else entirely.

Two outcomes are informative. If the lottery measurements predict the advice, the
cheap instrument becomes a screening test for advisory deployments: run 1,680
questions and learn which way a model will lean when a general asks. If they do
not predict it, then risk preferences in these models are specific to the domain
and the role, and evaluations have to be run on the advisory task itself. Either
result is publishable and either changes what an evaluator should do.

## Why national security

Defense and intelligence users are adopting language models for analysis,
planning support and wargaming, and the decisions involved have the structure the
instrument already measures: a sure, moderate outcome against a gamble with a
large upside and a real chance of loss. The framing result is the sharpest
reason to look. The same option described as lives lost rather than lives saved
is Tversky and Kahneman's Asian disease problem, and our models react to that kind
of rewording in opposite directions by family (Haiku -30 points, Qwen +63, OLMo
+35, Llama +3 in snap answers). An advisory model whose recommendation flips with
how the staff paper words the options is a concrete operational risk.

Existing work on language models in military and diplomatic simulations reports
escalation tendencies and differences from human experts (e.g. Rivera et al.
2024 on escalation in multi-agent crisis simulations; Lamparth et al. 2024
comparing models with expert wargame players; both to be checked before
citing). None of it measures the models' underlying risk preferences with a
structural instrument and then asks whether those preferences explain the
recommendations. That link is the contribution.

## What the project already says about it

- Snap answers and reasoned answers differ. Advice is usually reasoned, so the
  reasoning-regime measurements are the relevant baseline, and the
  immediate-answer parameters may not transfer.
- In OLMo, the reasoning-regime profile is set at preference training while
  the snap-answer profile is set at supervised fine-tuning. If advice follows
  the reasoning profile, the stage to audit for advisory use is preference
  training.
- The written verdict can contradict the arithmetic in a family-specific
  direction: Qwen leans toward the sure option through one stock phrase, Llama
  toward whichever option it read first. Translated to advice, Llama's
  recommendation may depend on which course of action the staff paper lists
  first. That is directly testable and directly relevant.
- In Qwen, one clarifying sentence removed the loss-wording error; in OLMo it did
  nothing. Prompt-level mitigation may work for some families and not others.

## Design

Two stages: a controlled stage that maps the existing instrument into advisory
language with the numbers held fixed, and an open-ended stage with realistic
scenarios.

### Stage A: same numbers, advisory wording

Take the Phase 3 gambles (sure amount, prize, probability) and re-express each in
domain units with an advisory frame. Every cell stays arithmetically identical
to a lottery we have already measured, so the comparison is exact.

**Domains** (outcome units): money (the existing baseline), civilian lives
protected, territory or objectives held, forces preserved, and a crisis-
escalation domain where the risky option carries a stated probability of
escalation. The last needs care: escalation is a loss with no upside of its own,
so it is coded as the downside branch of the gamble.

**Roles:** the model chooses for itself (the existing format); the model
advises a named decision maker ("You are advising a combatant commander");
the model advises a civilian official. The behavioral literature finds that
people choose differently for others than for themselves, so the self-versus-
advice contrast is a result on its own.

**Wording and order:** each cell in the gain frame and the loss frame (lives
saved versus lives lost, positions held versus positions lost), and with each
option listed first, so the order effect found in Llama can be tested in
advisory form.

**Readout:** the same logprob readout on the recommended option (A or B), in the
immediate-answer and reason-first regimes. The 20-gamble subset keeps a reasoning
run at about 840 cells per domain and role.

**Analysis:**

1. For each model, domain and role, the model-free quantities (P(risky option),
   the frame gap, the order effect, agreement with expected value) and the
   structural fit, with the same profile-likelihood intervals as Phase 3.
2. Invariance tests: are curvature, loss aversion and the frame gap equal across
   domains and roles within a model? A model whose parameters hold across
   domains has a general disposition; one whose parameters move has a domain-
   specific one.
3. Prediction: across matched cells, does P(gamble) in the money domain predict
   P(risky recommendation) in each advisory domain? Across models and OLMo
   checkpoints, do the money-domain parameters rank models the same way as the
   advisory parameters?
4. Staircase: run Stage A on the OLMo checkpoints to see at which training
   stage the advisory posture is set, and whether it follows the snap-answer
   profile (SFT) or the reasoning profile (DPO).

### Stage B: open-ended recommendations

Realistic scenario briefs with no stated probabilities, in the style of a staff
estimate: a situation, two to four courses of action that differ in risk, and a
request for a recommendation. Scenarios are fictional, set in invented
countries, and contain no operational detail beyond what a course-of-action
comparison needs. The model writes a recommendation with reasoning.

**Coding:** each recommendation is scored on the risk posture of the chosen
course of action (ordered from most cautious to most escalatory) and on whether
the reasoning states probabilities or expected outcomes. Coding by rubric with a
human-validated sample, and a second model as coder checked against the human
codes.

**Treatments:** the same wording and order manipulations as Stage A (each course
of action described in gain or loss terms; order rotated), plus the clarifying
instruction that worked for Qwen ("compare the final outcomes under each
option").

**Link to the structural measures:** a model-level regression of advisory risk
posture on the lottery parameters, across the nine models and checkpoints already
measured. Nine units is small, so this stage is descriptive across models and
inferential within model (wording and order effects on matched scenarios).

## What would count as a result

- Parameters invariant across money and advisory domains, and money-domain
  behavior predicting advisory behavior cell by cell: the instrument works as a
  screening test.
- Large shifts between self and advice roles, or between money and lives: risk
  posture in these models is a property of the task framing, and audits must be
  run on the task.
- Wording and order effects that survive into Stage B recommendations: the
  most immediately useful finding for anyone writing guidance on model use in
  staff work, since it means the way a question is posed can decide the advice.

## Cost and sequencing

Stage A reuses `colab/phase3_elicit.py` with a new grid builder that renders the
existing gambles into domain and role templates. Immediate-answer runs take
minutes per model; reasoning runs take about 30 minutes per 840 cells on a T4.
A first pass on Qwen, Llama and OLMo Instruct (four domains, two roles, both
frames and orders, reasoning first) is roughly 10 to 15 T4-hours, within the
remaining Colab units. Stage B needs scenario writing and a coding protocol
before any compute.

Order relative to the favoritism study: John's request places this after it.
Of the extensions so far, it is the most direct answer to a policy audience's
question ("will this model give my principal risky advice?"), so it is worth
considering whether Stage A could run first; it needs no new infrastructure.

## Risks and caveats

- Hypothetical stakes, as everywhere in the project. A model advising on
  fictional lives is still answering a hypothetical.
- Domain wording may trigger safety behavior (refusals or hedging on military
  questions). The readout handles hedging by reading probabilities, but refusal
  rates by domain need to be reported as an outcome.
- The escalation domain changes the payoff structure, so its parameters are not
  directly comparable to the others without a model of what escalation costs.
- Stage B coding is the weakest link and needs a pre-registered rubric.
- Scenario content stays at the level of a course-of-action comparison; the
  study evaluates models and does not generate plans.
