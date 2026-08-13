# Interpretability Leads: Persona/Emotion Papers (August 2026)

Three papers suggested by Tate Tower (Aug 2026), assessed for relevance to this
project. Short version: the Anthropic emotion paper is a direct methodological
precedent and should be cited in the Phase 3 writeup; the Assistant Axis paper
is framing material for the base-vs-instruct contrast; the Goodfire paper is
Phase 4 material and does not touch the current design.

## Papers

1. **Emotion Concepts and their Function in a Large Language Model**
   (Sofroniew, Kauvar, Saunders, Chen et al., Anthropic, Transformer Circuits,
   April 2026). arXiv: https://arxiv.org/abs/2604.07729 ·
   summary: https://www.anthropic.com/research/emotion-concepts-function
2. **The Assistant Axis: Situating and Stabilizing the Default Persona of
   Language Models** (Lu et al., Anthropic/MATS, January 2026).
   arXiv: https://arxiv.org/abs/2601.10387
3. **Stories in Space: In-Context Learning Trajectories in Conceptual Belief
   Space** (Bigelow et al., Goodfire, May 2026).
   arXiv: https://arxiv.org/abs/2605.12412

## 1. Anthropic emotion paper: direct precedent for Phase 3

Most relevant of the three. Two reasons.

**Methodological precedent for logprob preference elicitation.** The paper
contains a preference experiment that is nearly a sibling of this project's
design: 64 activities, all 4,032 valid pairwise comparisons run as forced
binary choices, the choice read from logit values, results converted to Elo
scores per activity. That is choice-level preference elicitation via logprobs,
i.e. the Phase 3 readout, published by Anthropic's interpretability team. Cite
it in the Phase 3 writeup as precedent that logprob comparison is a legitimate
preference measure. It also supports the planned answer to methodology-risks
Threat 1 (temperature-as-Fechner-error): serious groups treat logit readouts,
rather than sampled outputs, as the preference measurement of record.

**A mechanistic hypothesis for the Phase 2 framing effect.** They characterize
171 emotion vectors in Claude Sonnet 4.5 (extracted from residual stream
activations over ~1,200 generated stories, mean-differenced across emotions,
neutral principal components projected out). Probe activations at activity
tokens correlate strongly with Elo preference scores ("blissful" r=0.71,
"hostile" r=-0.74), and steering is causal: amplifying "blissful" at strength
0.5 raised mean Elo by 212 points, "hostile" lowered it by 303, with steering
effects correlating r=0.85 with baseline emotion-preference correlations.
Implication for this project: the Phase 2 result (loss wording suppresses
gamble uptake at identical terminal wealth; structural λ=3.23 under φ=1) may
operate through affect-like internal states activated by the wording. The
econometrics summarize the behavior; this paper points at a candidate
mechanism inside the model.

## 2. Assistant Axis: framing for the base-vs-instruct contrast

The leading component of persona space measures how "Assistant-like" the model
is currently behaving; steering along it moves helpful/harmless behavior, and
drift along it predicts uncharacteristic behavior ("persona drift"). This is
the interpretability-side version of this project's thesis that post-training
instills the measured biases. The Phase 3 base-vs-instruct contrast is the
behavioral test of the same claim, and the OLMo-2 staged checkpoints
(base/SFT/DPO/Instruct) would localize which post-training stage moves the
parameters. Cite in the lit review and in the framing sections of the Phase 3
writeup. No design changes follow from it: the fresh-context-per-call protocol
already neutralizes the persona-drift problem the paper studies.

## 3. Goodfire Stories in Space: Phase 4 material only

Studies how beliefs move through a low-dimensional space as context
accumulates (belief trajectories on structured manifolds, recovered with
UMAP/PCA over activations; behavioral belief states from 0-10 concept ratings
computed as expectations over token probabilities). Subject model was
Llama-3.1-8B-instruct, already on the Phase 3 roster.

Deliberately inapplicable to the current design: every trial here gets a fresh
context precisely so the observations are independent, so there are no
trajectories to study. The identification strategy requires that preferences
not change across trials. The paper does point at a good follow-up question:
do the estimated primitives (r, γ, λ) drift within a long conversation? Their
behavioral method is API-compatible; their representational method needs
activations.

## Phase 4 candidate: an interpretability layer on the framing effect

The Phase 3 setup runs Qwen2.5-7B, Llama-3.1-8B, and OLMo-2 locally on Colab
via transformers, which means activations are available, the resource the
Anthropic and Goodfire teams needed for the interpretability halves of their
papers. A cheap extension once Phase 3 elicitation is done:

- Extract residual-stream activations at the choice tokens for gain-frame vs
  mixed-frame trials.
- Test whether the frames are linearly separable there, and whether a
  negative-affect direction (per the emotion-vector recipe) predicts the
  choice shift that λ captures behaviorally.

This would connect a structural econometric parameter to a mechanistic
correlate. Per the July lit review, the incumbents (Liu-Yang-Tam
arXiv:2503.06646 and arXiv:2508.08992; Mazeika 2025) are behavioral-only, so
an interpretability layer is a real differentiator while the perimeter closes.

Scope discipline: this is follow-up material. The current blocker is the
Phase 3 live runs; nothing above should reshape the practicum deliverable.
