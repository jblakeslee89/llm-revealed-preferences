# Methodology-Risk Survey: LLMs as Experimental Subjects for Discrete-Choice Risk-Preference Elicitation

Prepared July 2026 (web-search sweep). Eight threats, each with sources and a design implication, plus a prioritized top-5.

## Threat 1: Does temperature-sampled variation behave like random-utility logit error?

The weakest-supported assumption in the design. "The Illusion of Stochasticity in LLMs" (arXiv:2604.06543, 2026): token sampling does not produce well-calibrated probabilistic choices; observed A/B spread is driven by training-data token biases, not a structural utility-plus-noise process. Answer distributions depend jointly on next-token logits and sampling method (arXiv:2406.03009; arXiv:2511.04869). Calibration literature (Xiong et al., ICLR 2024, arXiv:2306.13063; arXiv:2410.06707): mixed results on sample-frequency vs. raw logprobs; both frequently miscalibrated.

Logprob vs. repeated sampling: logprobs give P(A) vs P(B) in one call (cheap, low variance, directly the quantity the logit likelihood wants) but only on open/exposing providers, are distorted by A/B token-ID priors, and capture only the first token. Repeated sampling works black-box but is expensive and (per the Illusion paper) frequencies are not a faithful readout of an underlying utility model. The Fechner/logit error is a statistical convenience, not a validated behavioral model.

**Implication:** read logprob choice probabilities directly where possible (open models); treat repeated-sampling frequencies as descriptive; report results under at least two temperatures.

## Threat 2: Prompt-wording / paraphrase sensitivity

Large and dominant source of instability. "Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses" (arXiv:2507.07188, 2025): typos, paraphrase, synonyms, priming, reordering all shift responses. EMNLP 2024 Findings (2024.findings-emnlp.108): accuracy fluctuates across semantically equivalent paraphrases. Liu-Yang-Tam (arXiv:2503.06646): estimated risk preferences "unstable across different temperature settings and prompt variations." QSTN (arXiv:2512.08646): framework built to average over templates and randomize order.

**Implication:** treat the prompt template as an experimental factor: 5+ equivalent templates per lottery, template as a random effect in the discrete-choice model, report between-template variance components.

## Threat 3: Position/label bias in A/B forced choice; tool-use vs. plain text

First-option/"A" bias is among the largest LLM artifacts: 13-85% swings across orderings (Pezeshkpour & Hruschka 2024). Zheng et al. (ICLR 2024): selection bias stems from token-level priors on the ID symbols themselves; proposes debiasing. LLM-as-judge (arXiv:2406.07791): systematic primacy. Mitigation: bidirectional/permutation evaluation; order-flip = indifference. Tool-use: "Let Me Speak Freely?" (Tam et al., EMNLP 2024, arXiv:2408.02442): strict structured-output constraints degrade reasoning ~10-30%; forcing the answer field before reasoning commits the model early. Function-calling surveys (arXiv:2605.00737): tool interfaces change decision behavior. Direct evidence on tool-use changing risk preferences specifically is thin.

**Implication:** present each lottery in both orders with safe/risky roles swapped; estimate residual position effect as a nuisance parameter; validate tool-use forced choice against plain-text answers on a calibration subset.

## Threat 4: Training-data contamination

First-order validity threat for canonical instruments. Machine Psychology (Hagendorff et al., arXiv:2303.13988): classic vignettes verbatim cause models to reproduce known token patterns; prescribes new wordings/names/orderings/actions. "This human study did not involve human subjects" (arXiv:2602.15785, 2026): no-training-leakage is a formal necessary condition. Detection: LogProber (arXiv:2408.14352). Holt-Laury ten-row menu, Ellsberg urns, Asian-disease framing are canonical and almost certainly in-corpus.

**Implication:** never use canonical instruments verbatim; randomized payoffs/probabilities in novel non-textbook cover stories; run an explicit contamination probe (canonical-vignette completion or LogProber-style).

## Threat 5: Immediate answer (no CoT) vs. reasoning

CoT materially changes risky/economic choice; direction is model- and task-dependent. Tam et al. (arXiv:2408.02442): suppressing reasoning changes outputs. Machine-psychology reviews (arXiv:2505.08245): deliberative processing reduces System-1 biases including framing. "Sparks of Rationality" (arXiv:2601.22329, 2026) and "Prospect Theory Fails for LLMs" (arXiv:2508.08992): reasoning-mode models are more internally consistent but less human-like in loss aversion. Immediate-answer vs. CoT-then-answer can recover different parameters from the same lotteries.

**Implication:** pre-register the reasoning regime as a design factor; run both arms, or disclose that recovered parameters are conditional on the immediate-answer regime.

## Threat 6: Incentive-compatibility / "do LLMs have preferences at all"

LLM choices are hypothetical, no stakes; standard IC guarantees do not transfer. Horton (NBER w31122): behavior is suggestive, not a substitute for incentivized data. "Alignment Revisited" (arXiv:2506.00751); "Can Revealed Preferences Clarify LLM Alignment" (arXiv:2605.08556): stated and revealed preferences of LLMs are often inconsistent. Hullman et al. (arXiv:2602.15785): measurement error must be uncorrelated with covariates or parameter estimates are biased. EconEvals (arXiv:2503.18825): rationality as emergent, context-dependent.

**Implication:** frame recovered parameters as properties of the model's conditional text distribution under the protocol, not incentivized preferences; validate against an internal-consistency check (transitivity/GARP).

## Threat 7: Model-version drift and API reproducibility

Chen, Zaharia & Zou (arXiv:2307.09009; HDSR 2024): GPT-3.5/4 outputs changed materially across quarterly snapshots. "Test Before You Deploy" (arXiv:2604.27789, 2026); output-drift studies (arXiv:2511.07585): double-digit variance across provider updates.

**Implication:** pin dated model snapshots; record snapshot IDs in the released dataset; re-elicit a calibration battery on a second snapshot to bound drift.

## Threat 8: Best-practice guides

- **Hullman et al. (arXiv:2602.15785, 2026)**: validate-then-simulate vs. statistical calibration (prediction-powered inference) vs. simulate-then-validate; formalizes no-training-leakage and moment-condition preservation; warns of compressed variance and token-driven fragility.
- **Machine Psychology (arXiv:2303.13988)**: methodological charter, contamination avoidance.
- **LLM Psychometrics review (arXiv:2505.08245, 2025)**: validation standards.
- **GPT-ology (arXiv:2406.09464)**; silicon-sampling cautions (arXiv:2507.02919; arXiv:2512.22725): variance compression biases elicited distributions toward the mean.

**Implication:** position the study within the Hullman framework; report against contamination and psychometric-validation checklists.

## Cross-cutting: the gain/loss framing manipulation may show a weak or null effect

Binz & Schulz (2023) found GPT-3 reproduced framing effects; newer models attenuate them. GPT-4 shows lower framing susceptibility; 2025 papers (arXiv:2508.08992; Kyoto DP e-25-006; arXiv:2503.06646) report weak/inconsistent loss aversion in current LLMs. A null gain/loss effect is a plausible TRUE result and must be distinguishable from design failure.

**Implication:** include a positive control (a manipulation with a known large effect) so a weak framing result is interpretable.

# Prioritized Top-5 Design Decisions

1. **De-contaminate the instruments (highest priority).** Randomized payoffs/probabilities in novel cover stories; explicit contamination probe. (2303.13988; 2408.14352; 2602.15785)
2. **Do not treat temperature sampling as validated random-utility noise.** Logprobs on open models (debiased for token priors); two temperatures on Claude; frequencies as descriptive. (2604.06543; 2306.13063)
3. **Counterbalance position/label bias and marginalize it out.** Both orders, roles swapped, nuisance parameter, tool-use vs. plain-text validation subset. (Zheng ICLR 2024; Pezeshkpour & Hruschka 2024; 2408.02442)
4. **Prompt template and reasoning regime as experimental factors.** 5+ templates with template random effects; immediate-answer and reason-then-answer arms; report variance components. (2507.07188; 2503.06646; 2408.02442)
5. **Pin snapshots; frame inference honestly; include a positive control.** Dated snapshots logged in the dataset; second-snapshot calibration battery; Hullman framing; positive-control manipulation so a null framing result is interpretable. (2307.09009; 2602.15785; 2508.08992)
