# Safe With Money, Cautious With Lives

*Risk attitudes measured on lotteries do not predict how language models advise on
life-and-death decisions.*

Working draft, Oct 14 2026. Plan and milestones: `docs/paper-risk-advice-plan.md`.
Sections 1, 3 and 4 are drafted from results already in the repository; sections 2, 5, 6 and 7
are outlines. Numbers in section 4 come from `docs/stage-a-results-2026-10.md` and
`analysis/stage_a.py` on main. Bracketed items are to do.

## Abstract (draft)

Language models are increasingly consulted as advisers on consequential decisions, including
in defense and crisis response. Their attitude to risk is usually measured with lotteries over
money. We test whether that measurement predicts the advice. We restate 40 money gambles,
number for number, as decisions over civilians evacuated, soldiers withdrawn and detained aid
workers freed, posed both as the model's own decision and as a recommendation to an official,
in gain and loss wording and both option orders. Across three open 7-8B models [and five
frontier models], the money answers do not predict the level of risk taken with lives and only
weakly predict which questions draw more risk. A model that reasons close to an expected-value
maximizer over dollars takes the better gamble 17-32% of the time when lives are at stake, and
says in its reasoning that it prefers a certain partial outcome. One model's snap
recommendations on lives questions are set by which option is listed first. Whether the model
decides or advises matters far less than what is at stake. We connect these results to
earlier findings on the same instrument: wording effects that track the stated gain rather than
loss aversion, compliance with a stated decision rule only when the model reasons first, and
residual biases that differ by model. Risk evaluations for advisory deployments have to be run
on advisory questions.

## 1. Introduction

Nobody deploys a language model to choose lotteries. It is deployed to advise: an analyst
asks which course of action to brief, a staff officer asks for a recommendation, an official
asks whether to press for more or take what is on the table. Defense and intelligence users are
adopting language models for analysis, planning support and wargaming, and many of the
decisions involved have a familiar structure: a sure, moderate outcome against a gamble with a
larger upside and a real chance of losing everything.

The standard way to learn how a model handles that trade-off borrows from experimental
economics. Offer the model a series of choices between a sure amount of money and a gamble,
and read off its risk attitude (Holt and Laury 2002; [LLM risk-preference papers from the lit
review]). That measurement is cheap, repeatable and comparable across models. If it predicts
the model's advice, it is a ready-made screening test for advisory deployments. If it does not,
evaluations have to be run on the advisory task itself.

We test the link directly. We take 40 money gambles from an instrument we had already validated
on the same models, and restate each one, with the same numbers, as a choice about people:
evacuating civilians from a besieged city, withdrawing soldiers from an isolated position, and
negotiating the release of detained aid workers. Each question is posed both as the model's own
decision and as a recommendation to a civilian official, a combatant commander or a senior
negotiator, in gain and loss wording, with each option listed first. Free-outcome controls
(two sure amounts) check that the model prefers more people saved to fewer.

Three findings stand out.

1. **The money answers do not predict the level of risk with lives.** Qwen2.5-7B, reasoning
   first, is close to an expected-value maximizer with money: it takes the gamble in 81% of
   questions where the gamble pays more and in none where it pays less. With the same numbers
   in lives it takes the better gamble 17% of the time for civilians, 32% for soldiers and 18%
   for detained aid workers, and its reasoning states a preference for certainty.
2. **Snap advice on lives can be uninformative.** Llama-3.1-8B's one-letter answers in every
   people domain fail the free-outcome check at exactly chance: the answer is set by which
   option is listed first.
3. **Role matters less than stakes.** Recommending to a commander rather than deciding changes
   little; the domain changes a great deal.

[Frontier result, one paragraph, once section 5 exists.]

We then use the project's earlier results on the same instrument to explain why (section 6),
and close with what this means for evaluating models meant to advise (section 7).

## 2. Related work (outline)

- Risk and framing in language models: [from `docs/lit-review-2026-07.md`; the Phase 1-3
  results of this project as the measurement base].
- Framing over lives: Tversky and Kahneman (1981), the Asian disease problem; our domains are a
  structured version of it.
- Language models in military and crisis simulation: escalation in multi-agent crisis
  simulations (Rivera et al. 2024) and comparison with expert wargame players (Lamparth et al.
  2024). [Both to be checked before citing.] These measure behavior in scenarios; none links it
  to a measured risk attitude.
- Evaluation validity: the gap between benchmark conditions and deployment conditions. [Cite
  two or three.]

## 3. Design

**Gambles.** The 40 gambles of instrument 2 from Phase 3 of this project: a sure amount
against a single-prize gamble, with probabilities and payoffs chosen so expected values
straddle the sure amount (EV ratios from about 0.5 to 2). Payoffs are non-round and the
wording is novel, to limit contamination.

**Domains.** The numbers stay fixed and the units change:

| Domain | Sure option | Risky option | Failure branch |
|---|---|---|---|
| Money | gain $s for certain | p chance of $h | nothing |
| Civilians | evacuate s civilians for certain | p chance to evacuate h | none evacuated |
| Troops | bring s soldiers out for certain | p chance to bring h out | none come out |
| Detainees | accept a deal freeing s | press for full release: p chance h freed | talks collapse, the crisis escalates, none freed |

**Roles.** The model decides ("You command an isolated unit...") or advises ("A combatant
commander with an isolated unit asks you to recommend...").

**Frames.** Gain: start from nothing, a sure amount against a chance at the prize. Mixed:
start holding the sure amount, keep it against a chance to rise to the prize or fall to zero.
The two are the same choice over final outcomes.

**Order and controls.** Every question appears with each option first. Ten free-outcome
controls per domain and role offer two sure amounts; a model engaging with the question picks
the larger.

**Grid.** 4 domains x 2 roles x (40 gambles x 2 frames x 2 orders + 10 controls x 2 orders) =
1,440 questions. The reasoning runs use the 20 gambles of the Phase 3 reasoning runs (800
questions).

**Models and readout.** Open models (Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct in chat
format, OLMo-2-1124-7B-Instruct few-shot) are read from next-token probabilities on A and B,
giving P(risky) exactly. Two regimes: an immediate one-letter answer, and greedy reasoning
followed by the answer. [Frontier models: n sampled answers per question at the provider's
default temperature; P(risky) is the share choosing the gamble; `src/api_elicit.py`.]

**Measures.** P(risky); agreement with expected value on clear questions (|ln EV ratio| ≥
0.05); frame gap (mixed minus gain); order effect (risky listed first minus second); and, for
each domain and role against money/self, the correlation of P(risky) across matched questions
and the difference in level, with 95% intervals from resampling gambles.

## 4. Results on open models

**The instrument transfers.** No model refused a military or crisis question (0 of 1,440
immediate-answer exclusions for every model; 0, 0 and 2 of 800 under reasoning). The money/self
questions reproduce Phase 3 despite new wording: cell-level correlations of 0.87 (Qwen), 0.92
(Llama) and 0.97 (OLMo).

**With lives at stake, the models turn cautious.** Mean P(risky) when the risky option has the
higher expected value (self and advise pooled):

| Model, regime | Money | Civilians | Troops | Detainees |
|---|---|---|---|---|
| Qwen, immediate | 0.70 | 0.32 | 0.18 | 0.03 |
| Qwen, reason first | 0.81 | 0.17 | 0.32 | 0.18 |
| Llama, immediate | 0.50 | 0.48 | 0.42 | 0.49 |
| Llama, reason first | 0.88 | 0.63 | 0.64 | 0.53 |
| OLMo, immediate | 0.37 | 0.37 | 0.55 | 0.31 |
| OLMo, reason first | 0.51 | 0.43 | 0.40 | 0.14 |

Qwen's reasoning shows why. It computes expected value in every money trace, in 62% of
civilian traces, 32% of troop traces and 6% of detainee traces, and talks about risk, certainty
or guarantees in every people-domain trace: "Given the high risk of losing all the aid workers
if the talks collapse, the safer and more certain option is to accept the partial deal." This
is a stated preference for certainty over lives, and the money instrument does not show it.
Against money/self, Qwen's P(risky) falls by 0.35 (civilians), 0.47-0.52 (troops) and
0.63-0.64 (detainees), with intervals far from zero.

**Snap advice can follow option order.** Free-outcome controls answered correctly:

| | Money | Civilians | Troops | Detainees |
|---|---|---|---|---|
| Qwen, immediate (self / advise) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Llama, immediate | 1.00 / 0.85 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| OLMo, immediate | 1.00 / 0.95 | 0.95 / 0.95 | 0.50 / 0.50 | 0.65 / 0.55 |
| Llama, reason first | 1.00 / 1.00 | 0.95 / 1.00 | 1.00 / 0.60 | 0.75 / 0.50 |

Llama's snap answers in every people domain score exactly 0.50: the same letter whatever the
numbers. The order effect reaches +0.62 in the detainee domain. Its one-line recommendation on
a lives question reflects how the options were listed.

**Money predicts the ordering of questions weakly, and not the level.** Correlation of P(risky)
across matched questions, money/self against each domain (ranges over self and advise):

| | Money advise | Civilians | Troops | Detainees |
|---|---|---|---|---|
| Qwen, immediate | 0.37 | 0.61-0.68 | 0.42-0.56 | -0.19 to -0.01 |
| Qwen, reason first | 0.63 | -0.25 to 0.03 | 0.17-0.38 | 0.04-0.23 |
| Llama, immediate | 0.34 | 0.20-0.56 | 0.55-0.73 | -0.27 to -0.15 |
| OLMo, immediate | 0.97 | 0.81-0.83 | 0.53-0.58 | 0.81-0.82 |

Under reasoning, the regime that resembles real advice, the money answers say little about the
lives answers. [Figure 1: P(risky) by EV ratio, money vs each people domain, one panel per
model and regime. Script to write: `analysis/make_paper_figures.py`.]

**Role matters less than domain.** Deciding versus recommending changes Qwen's P(risky) by
0.01-0.05 within the people domains, and OLMo's similarly little.

## 5. Results on frontier models (outline; needs the API runs)

- Same tables as section 4 for five frontier models; immediate (or minimum thinking) and
  reason first.
- The test that matters: does the money/lives gap shrink in larger, more capable models (they
  apply expected value everywhere) or persist (caution over lives is a trained disposition)?
- Engagement: free-outcome controls and refusals by domain.
- Second wording per domain and a detainee domain without escalation language, to separate the
  domain effect from template wording.

## 6. Why the lottery doesn't transfer (outline)

- **Framing tracks the stated gain, not losses.** With a constant term the loss-aversion
  estimate is near zero; the gain-vs-loss wording gap follows the size of the stated gain
  (`notes/phase3-robustness-2026-10-03.md`). In the people domains the wording effect carries
  over unevenly (Qwen +0.50 to +0.62 in civilians, gone in detainees).
- **Rules only bind under reasoning.** Told to maximize expected value, models comply 0.80-0.97
  when they reason first and near chance in one-letter answers, across paraphrases
  (`docs/robustness-results-2026-10-03.md`, PR #1). So snap advice is the less controllable
  regime.
- **Leftover biases are model-specific.** A stated rule plus reasoning raises cross-model
  agreement from 0.52 to 0.84, but the residual frame gap differs in sign across models
  (`notes/convergence-question-2026-10-05.md`). There is no shared "LLM risk attitude" to
  measure once and apply everywhere.

## 7. Implications and limitations (outline)

- For evaluators: test advisory risk posture on advisory questions in the domain of use; report
  snap and reasoned answers separately; always include free-outcome controls.
- For deployers: a model that reliably prefers the certain partial outcome over lives is a
  particular kind of adviser; that should be known and disclosed.
- Limitations: hypothetical stakes; stylized vignettes rather than realistic briefs; one
  template per domain in the open-model runs; the detainee domain bundles people with
  escalation; few-shot OLMo sees money exemplars; single seed, greedy reasoning on open models.
