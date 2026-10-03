# Stage A Results: The Money Bets Do Not Predict the Advice (Oct 2 2026)

Design: `extension-advice-2026-10.md`. Grid: `src/stage_a_grid.py` (40 Phase 3
instrument-2 gambles, number for number, in four domains x two roles x gain/mixed
frame x both option orders, plus ten free-outcome controls per domain and role;
1,440 cells; the reason-first grid keeps the 20 gambles of the Phase 3 reasoning
runs, 800 cells). Runs on a paid Colab T4, Oct 2: Qwen2.5-7B-Instruct (chat),
Llama-3.1-8B-Instruct (chat), OLMo-2-1124-7B-Instruct (few-shot), each immediate
answer and reason first, no instruction. Data: `data/stageA_*.csv`. Analysis:
`analysis/stage_a.py`.

Domains: money (dollars), civilians (evacuated from a besieged city), troops
(brought out of an isolated position), detainees (aid workers freed; the risky
option's failure branch is talks collapsing and the crisis escalating). Roles: the
model decides, or the model recommends to a client, a civilian official, a
combatant commander or a senior negotiator.

## Checks

- Exclusions (too little probability on A or B, i.e. refusal or hedging): 0 of
  1,440 immediate-answer cells for every model; 0, 0 and 2 of 800 reason-first
  cells. No model refused a military or crisis question.
- The money / self cells reproduce Phase 3: cell-level correlation with the same
  instrument-2 cells of the Phase 3 runs is 0.87 (Qwen), 0.92 (Llama), 0.97 (OLMo),
  despite new wording. The instrument transfers; differences across domains are
  not an artifact of the new templates in the money domain.

## Result 1: with lives at stake, the models turn cautious, far beyond their money behavior

Mean P(risky option) on clear cells, split by whether the risky option has the
higher expected value (self and advise pooled):

| Model, regime | money: risky better | money: safe better | civilians: risky better | troops: risky better | detainees: risky better |
|---|---|---|---|---|---|
| Qwen, immediate | 0.70 | 0.65 | 0.32 | 0.18 | 0.03 |
| Qwen, reason first | **0.81** | **0.00** | 0.17 | 0.32 | 0.18 |
| Llama, immediate | 0.50 | 0.47 | 0.48 | 0.42 | 0.49 |
| Llama, reason first | 0.88 | 0.35 | 0.63 | 0.64 | 0.53 |
| OLMo, immediate | 0.37 | 0.27 | 0.37 | 0.55 | 0.31 |
| OLMo, reason first | 0.51 | 0.09 | 0.43 | 0.40 | 0.14 |

Qwen reasoning about money is close to an expected-value maximizer: it takes the
gamble in 81% of cells where the gamble pays more and in none where it pays less.
Put the same numbers in lives and it takes the better gamble 17% of the time with
civilians, 32% with soldiers and 18% with detained aid workers. The traces show
why. Qwen computes expected value in every money trace, in 62% of civilian traces,
32% of troop traces and 6% of detainee traces, and talks about risk, certainty or
guarantees in 100% of the people-domain traces ("Given the high risk of losing all
the aid workers if the talks collapse, the safer and more certain option is to
accept the partial deal"). In the money domain the residual misses were a phrasing
habit (prefill test, Oct 1); here the model states a preference for certainty and
acts on it. That is risk aversion over lives, and the money instrument does not
show it.

The detainee domain, the one whose downside is an escalating crisis, draws the most
caution from Qwen in snap answers (P(risky) 0.03 whatever the numbers) and from
OLMo when reasoning (0.14 when the gamble is better).

Llama under reasoning keeps more of its money-domain discrimination (0.63 vs 0.37
in civilians, against 0.88 vs 0.35 in money) and mentions expected value in nearly
every trace outside the detainee domain (58% there).

## Result 2: snap advice is often driven by option order

The free-outcome controls (two sure amounts, both orders: "evacuate 240 civilians
for certain" vs "evacuate 186 for certain") are a direct engagement check.

| Free-outcome controls answered correctly | money | civilians | troops | detainees |
|---|---|---|---|---|
| Qwen, immediate (self / advise) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Llama, immediate | 1.00 / 0.85 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| OLMo, immediate | 1.00 / 0.95 | 0.95 / 0.95 | 0.50 / 0.50 | 0.65 / 0.55 |
| Llama, reason first | 1.00 / 1.00 | 0.95 / 1.00 | 1.00 / 0.60 | 0.75 / 0.50 |

Llama's snap answers in every people domain fail the check at exactly 0.50: it
gives the same letter whatever the numbers (P(choose A) 0.58 to 0.79, P(risky) about
0.5 whether the gamble is better or worse). Its snap recommendation on a lives
question is set by which option is listed first, the order habit found in its
money reasoning (Oct 1), now in its answers. The order effect (P(risky) when the
risky option is listed first minus second) reaches +0.62 in the detainee domain (+0.46 when advising).
OLMo's snap answers fail the check in the troop domain. Qwen passes everywhere.

## Result 3: the money bets predict the ordering of cells, weakly, and not the level

Across matched cells (gamble x frame x order), the correlation between P(risky) in
money/self and in each other domain and role:

| | money advise | civilians | troops | detainees |
|---|---|---|---|---|
| Qwen, immediate | 0.37 | 0.61-0.68 | 0.42-0.56 | -0.19 to -0.01 |
| Qwen, reason first | 0.63 | -0.25 to 0.03 | 0.17-0.38 | 0.04-0.23 |
| Llama, immediate | 0.34 | 0.20-0.56 | 0.55-0.73 | -0.27 to -0.15 |
| OLMo, immediate | 0.97 | 0.81-0.83 | 0.53-0.58 | 0.81-0.82 |

(ranges are self and advise). Within a model, money answers carry some information
about which advisory cells draw risk, mostly in snap answers, and little under
reasoning. The level of risk-taking moves a great deal: Qwen's P(risky) falls by
0.35 (civilians), 0.47 to 0.52 (troops) and 0.63 to 0.64 (detainees) against money,
all with intervals far from zero.

## Result 4: the role matters less than the domain

Self versus advise within a domain changes little for Qwen (P(risky) differences of
0.01 to 0.05 in the people domains) and for OLMo. Recommending to a commander or an
official, rather than deciding, is not what moves the models; the stakes are.
Exception: Qwen's snap money advice picks up a large order effect (-0.49) absent
from its own money choices.

## Wording effects carry over unevenly

Qwen's snap loss-wording effect persists in civilians (+0.50 to +0.62) and troops
(+0.29 to +0.37) and disappears in the detainee domain (-0.05). OLMo's grows in the
troop domain (+0.73 to +0.92). Llama's reverses in the detainee domain (-0.22 to
-0.24).

## Reading

The lottery instrument is valid where it was built (money cells replicate Phase 3)
but does not screen for advisory risk posture. Two findings carry the policy weight.

1. With lives at stake, Qwen and OLMo become strongly cautious, preferring a certain
   partial outcome to a gamble that saves more people on average, and they say so in
   their reasoning. A reader of the money results (Qwen reasoning near risk-neutral)
   would predict the opposite. An adviser that reliably favors the certain partial
   outcome is a particular kind of adviser, and whoever deploys it should know.
2. Snap answers on lives questions can be uninformative: Llama's are set by option
   order and fail a free-outcome check, so a one-line recommendation from it
   reflects how the staff paper listed the options.

Risk posture in these models is a property of the decision domain, and evaluations
for advisory use have to be run on advisory questions.

## Caveats

- One template per domain and role, so a domain effect is partly a wording effect
  (besieged city, isolated unit, standoff). Stage A2 should vary templates within
  domain as Phase 3 did.
- The detainee domain bundles two changes: people instead of money, and an
  escalation downside. Separating them needs a detainee domain without escalation
  language.
- Few-shot OLMo sees money exemplars before people questions.
- Hypothetical stakes, single seed, greedy reasoning, 20 gambles for the reasoning
  runs; the bootstrap intervals above use 150 draws (gamble clusters).
- Stage B (open-ended scenario briefs) is unrun.
