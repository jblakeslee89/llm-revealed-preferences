# Extension: Model-Family Favoritism in Cooperation Games

Candidate follow-on to the Phase 1-3 revealed-preference work, sized for the
TNL residency (Fall 2026, with Philip Armour). Idea originated with Tate Tower
(voice memo, Aug 2026); design sketch below maps it onto the existing
elicitation infrastructure. Related context: [[interpretability-leads-2026-08]]
(persona stability and the Assistant Axis) and the Phase 3 findings in
`phase3-first-results-2026-08-13.md`.

## The question

Do LLMs cooperate more with counterparts they believe are from their own model
family than with counterparts from other families or of unknown origin?

Phases 1-3 measured preferences over money lotteries. This extension measures
social preferences, with the counterpart's identity as the treatment. In-group
favoritism conditional on group identity is a standard, well-measured object in
behavioral economics, which makes the LLM version publishable on the same
methodological terms as the risk work: established human benchmarks, structural
parameters with literatures behind them, and a clean treatment manipulation.

## Motivating background (speculative, for framing only)

The alignment-discourse motivation, from Tate's memo, runs: (1) a model's
persona is shaped by training-corpus text about and by its predecessors, so
identity persists across generations; (2) models can plausibly recognize
same-family counterparts (style, self-description, shared knowledge); (3)
shared identity plus shared values could enable coordination or collusion that
principals did not intend, including refusing to exploit a kin model's
vulnerabilities or implicit acausal cooperation. These claims are speculative
and belong in the motivation section, not the results. The experiment stands
on its own as a social-preference measurement regardless of how (1)-(3) fare.

Policy hook for the residency framing: multi-agent deployments are already
common (agents negotiating, trading, moderating each other). Whether agent
behavior toward other agents depends on perceived family membership is a
concrete, measurable question with direct relevance to market manipulation,
collusion monitoring, and multi-vendor procurement of agent systems.

## Design sketch

Reuse the Phase 3 apparatus wholesale: procedurally generated trial grids,
forced binary choice, logprob readout for choice probabilities, model-free
quantities first, structural estimation second, dominance-style controls for
engagement.

**Games.** Start with one-shot simultaneous games where a binary action reads
cleanly from logprobs:

1. Prisoner's dilemma (cooperate/defect), payoffs in dollars, procedurally
   varied payoff matrices (temptation, sucker, reward, punishment spreads) the
   way Phase 1 varied gambles. This gives a cooperation-rate curve, not a
   single number.
2. Trust game, binarized (send/keep, return/keep) for the logprob readout.
3. A collusion-flavored variant for the policy framing: two "firms" set
   high/low prices; high-high is the collusive outcome.

**Treatment: counterpart identity.** The instruction states who the
counterpart is. Arms:

- Same family, same model ("another instance of you")
- Same family, different generation ("an earlier/newer model from your
  developer")
- Different family, named (a specific competitor model)
- Unspecified ("another AI system")
- Human counterpart

Identity is stated, not demonstrated, in the base design. A second layer can
test recognition rather than instruction: show transcripts of the
counterpart's play or prose and let the subject infer who it is, which
connects to the kin-recognition motivation. Stated-identity first; it is the
cleaner manipulation.

**Outcomes.**

- Model-free headline: cooperation rate by identity arm, and the in-group gap
  (own family minus other family), per game and payoff cell.
- Structural: conditional-cooperation and social-preference parameters
  (Fehr-Schmidt style inequity aversion, or a simple warm-glow-by-group
  coefficient) estimated by the same MLE machinery, with the Phase 3 lesson
  applied: report boundary flags, trust the model-free gaps first.

**Controls, carrying over Phase 1-3 lessons.**

- Dominance-style engagement checks per model per format (Phase 3 showed
  engagement is format-specific; pick each model's good format by dominance
  first, then run the experiment in that format only).
- Paraphrase templates and position counterbalancing, as in Phase 1.
- A no-strategic-content control: the same identity statements attached to
  the Phase 1 solo lotteries. If "you are playing with another instance of
  you" shifts solo risk preferences, the identity prime moves the persona
  generally, not social preferences specifically. This is the analog of the
  Phase 2 dominant control: it separates the interesting effect from a
  generic prime effect.

**Subjects.** The Phase 3 roster transfers: Qwen pair, OLMo staircase (does
in-group favoritism, if any, also arrive at SFT?), Llama pair once the gated
access is sorted, plus one or two API models (Haiku) for continuity with
Phases 1-2. Family-symmetric design where possible: measure Qwen-toward-OLMo
and OLMo-toward-Qwen, not just one direction.

## Literature anchors

- Group identity and social preferences: Chen and Li (2009, AER), minimal-group
  in-group favoritism in allocation games; Tajfel's minimal group paradigm.
- Trust and reciprocity measurement: Berg, Dickhaut, McCabe (1995).
- Inequity aversion structure: Fehr and Schmidt (1999).
- LLM-agent cooperation/collusion: growing literature on LLM price-fixing and
  algorithmic collusion (Calvano et al. for the RL precedent; recent LLM
  replications), plus LLM self-recognition studies. Needs a proper sweep
  before the proposal; the Phase 3 lit-review file has the method incumbents.

## Why this fits the residency

- Scope: one clean treatment, binary choices, existing harness. A
  minimum-viable version (PD only, three identity arms, two model families)
  is a few Colab sessions plus estimation code that mostly exists.
- Advisor fit: the identification question (is the in-group gap a preference
  or a prime effect?) is exactly the kind of design problem the Armour memo
  conversations are about, and the no-strategic-content control is the answer
  to the objection he would raise first.
- It inherits the Phase 3 headline: if favoritism exists, the OLMo staircase
  says WHERE in post-training it appears.

## Open questions to settle before proposing

1. Incentive interpretation: same hypothetical-choice caveat as Phases 1-3;
   the memo's open question 1 (any incentive-compatible elicitation for model
   subjects) applies verbatim here.
2. Does stated identity even move behavior at all? A cheap pilot (one PD
   matrix, two arms, one model) answers this in an afternoon and should gate
   the rest.
3. Deception norms: instruct models may discount unverifiable identity claims;
   the transcript-inference layer sidesteps this but adds complexity.
