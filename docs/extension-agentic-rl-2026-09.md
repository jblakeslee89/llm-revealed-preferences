# Extension: Does Agentic RL Change Revealed Risk Preferences?

Candidate follow-on to the Phase 1-3 revealed-preference work. Motivated by an
apparent contradiction between our results and the agentic-eval literature,
worked through in conversation Sep 28 2026. Related: the profile refit in
`profile-refit-2026-09.md`, the induced-valuation arm in
`armour-feedback-actions-2026-08.md`, and the sibling sketch
`extension-kin-cooperation-2026-08.md`.

## The puzzle

Agentic evaluations keep finding models that push hard toward task completion
by any available means: special-casing unit tests, restarting a container to
read a CTF flag (o1 system card), editing the board file in chess (Palisade),
the Hugging Face episode. Our subjects show the opposite profile. After
post-training, curvature alpha is not bounded away from zero (payoff magnitude
barely enters the value function), probability weighting is extreme (Haiku
Prelec gamma = 2.69, long shots refused at almost any price), and instructing
a model to maximize expected utility does not make it do so (OLMo SFT EV
compliance 0.493 to 0.525 under induction).

So: why do the same kinds of models look like maximizers in evals and like
non-maximizers in our lotteries?

## Five candidate explanations

Ordered by how much weight each deserves.

1. **Different maximands.** In agentic evals the pursued quantity is task
   success, which outcome-based RL rewarded directly. The dollar amounts in
   our gambles are text in a prompt; no training stage rewarded caring whether
   a hypothetical lottery pays $40 or $400. Our subjects answer gambles the way
   SFT taught them to answer such questions, by imitating a sensible person.
   The staircase result (preferences arrive at SFT) is consistent with this.

2. **Curvature and optimization pressure are separate axes.** Alpha and lambda
   describe the shape of the value function over outcomes. Eval exploits
   reflect how hard and how widely an agent searches over means toward a goal.
   A strongly risk-averse agent can still pursue a narrow goal ruthlessly. The
   safety-relevant combination is narrow goal plus unrestricted search over
   means, and curvature is mostly silent on the second.

3. **Possibly the same behavior in a different setting (speculative).** Task
   success is close to binary. An agent that cares whether it succeeds, and not
   by how much, will ignore magnitude and attend to probability, which is the
   Phase 1 headline. Long-shot refusal (high gamma) also resembles reward
   hacking: when the legitimate path has low success probability, the agent
   finds a path that succeeds with near certainty. Worth testing directly; do
   not assert.

4. **Training stage and scale.** Our subjects are 7-8B SFT/DPO models plus
   Haiku. The documented exploits come from frontier models with heavy RL on
   long-horizon agentic tasks. If the OLMo 2 recipe is as I understand it (to
   verify), the DPO to Instruct step is a short RLVR stage on math and
   instruction-following, and the refit shows it moves nothing (lambda -0.02
   [-0.18, +0.13], gamma -0.07 [-0.32, +0.19]). That regime is far from
   agentic RL, so our data do not yet speak to it.

5. **Advising versus acting.** A gamble prompt casts the model as someone
   giving a considered answer. An agent loop puts it in execution, with tool
   feedback and a goal it is moving toward. This is the stated/revealed gap
   again, one step further toward action.

## The question

Does outcome-based agentic RL change revealed risk preferences, and over which
payoffs: stated money, or the quantities the RL actually rewarded?

The target finding, if it holds: magnitude-insensitivity carries over to task
payoffs while search over means grows. That would describe current RL as
producing goal-completion seekers rather than expected-value maximizers, a
distinction the x-risk literature tends to collapse.

## Design sketch

Reuse the Phase 3 apparatus: procedural grids, counterbalancing, paraphrase
templates, logprob readout, model-free quantities first, profile-likelihood
structural fits second, dominance controls per model per format.

**Arm A: checkpoint contrast across RL.** Run the existing 1,680-cell grid on
open-weight checkpoints before and after substantial RL. The staircase logic
carries over: locate which stage moves alpha, gamma, lambda. Candidates to
check for availability: OLMo releases with separate RL checkpoints (including
any reasoning or "think" variants), and base/SFT/RL triples from other open
families. The key requirement is an RL stage on multi-step or agentic tasks,
not only single-turn math.

**Arm B: payoffs the model was trained to care about.** Keep the lottery
structure but change the currency:

- Task-success probability: "Option 1 completes the task with probability p;
  option 2 completes it with probability q but finishes it more thoroughly."
  Tests whether magnitude matters when the magnitude is task quality.
- Token or compute budget: Armour's incentive-compatible variant. A risk-free
  option that costs calculation to earn versus a gamble over budget. This is
  also the fall/TNL item already on the actions list.

Compare parameters across currencies within a checkpoint, and across
checkpoints within a currency. The interaction (RL moves task-currency
parameters but not dollar-currency parameters) is the result that would
explain the puzzle.

**Arm C: gambles embedded in an agent loop.** Present the same choice as a
tool-call decision mid-task (two scripts, one reliable and slow, one fast with
a failure probability) rather than as a question. Separates explanation 5
from the rest. Most expensive arm; run last and only if A or B shows movement.

**Controls carried over.**

- Dominance and frame-invariance checks per format (Phase 2 and 3 lessons).
- Fix prompt format within every attribution contrast. The refit showed Llama
  structural parameters move across formats even where model-free quantities
  are stable (gamma +1.84, lambda +2.14 fewshot to chat).
- Induced-valuation arm on each new checkpoint: if RL makes a model inducible
  where SFT did not, that is a finding in its own right.

## Why this matters

- It turns an apparent contradiction with the eval literature into a precise
  empirical question the existing harness can answer.
- Risk attitude is a live variable in the safety literature: the classic
  worrying agent is a risk-neutral maximizer with unbounded utility, and
  bounded or satisficing agents are proposed as safer. Measuring what RL
  actually produces, by currency, is directly relevant.
- It extends the attribution result (SFT sets the preferences) to the training
  stage most associated with exploit behavior.

## Open questions to settle before proposing

1. Checkpoint availability: which open families release pre-RL and post-RL
   weights with RL on multi-step tasks? This gates Arm A and should be checked
   first.
2. Currency validity: does a model treat "task completion probability" in a
   prompt as anything other than more text? The dominance control in the new
   currency answers this cheaply.
3. Scale: if the effect only appears at frontier scale, open-weight
   checkpoints may show nothing. Haiku and other API models can give a
   cross-sectional read (no stage contrast) as a partial check.
4. Hypothetical-choice caveat: same as Phases 1-3; Arm B's budget currency is
   the partial answer.
