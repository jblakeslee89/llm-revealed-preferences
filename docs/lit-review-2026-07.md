# Literature Survey: Structural Estimation of Economic Preferences and Behavioral Biases in LLMs

Prepared July 2026 (web-search sweep). Updated September 2026: two post-training papers missed by the July sweep added to Category 3 (Wang, Li & Chen 2025; Itzhak, Belinkov & Stanovsky 2025), with a revised niche assessment at the end. Coverage through 2026. For each paper: method, finding, whether it does STRUCTURAL (likelihood-based parameter recovery) vs. DESCRIPTIVE work, and whether it addresses RLHF/post-training.

## Category 1: Foundational "LLM-as-subject" papers (descriptive)

**Horton, Filippas & Manning (2023/2025), "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?"** NBER w31122 / arXiv 2301.07543.
- Method: endows GPT-3 with preferences/information, replicates classic experiments (Charness-Rabin, Kahneman et al. 1986, Samuelson-Zeckhauser endowment/status-quo).
- Finding: qualitatively reproduces canonical behavioral results.
- STRUCTURAL: No. Pure qualitative replication. RLHF: not addressed.

**Binz & Schulz (2023), "Using cognitive psychology to understand GPT-3," PNAS 120(6).**
- Method: battery of canonical cognitive-psych vignettes on GPT-3.
- Finding: reproduces certainty effect, overweighting, framing, anchoring, loss-aversion indicators.
- STRUCTURAL: No, vignette-level bias demonstration. RLHF: No. Note: relevant contamination target — these are the memorized Kahneman-Tversky items.

**Mei, Xie, Yuan & Jackson (2024), "A Turing test of whether AI chatbots are behaviorally similar to humans," PNAS 121(9).**
- Method: dictator/trust/ultimatum/public-goods/finitely-repeated PD plus a risk task; compares GPT-3.5/GPT-4 distributions to a large human sample.
- Finding: largely human-indistinguishable but more altruistic/cooperative; on risk, ChatGPT is essentially risk-neutral vs. human risk-aversion.
- STRUCTURAL: Mostly descriptive/distributional; risk handled qualitatively, no CRRA MLE. RLHF: No.

## Category 2: Structural / revealed-preference estimation on LLMs (closest methodologically)

**Chen, Zeng, et al. (2023), "The Emergence of Economic Rationality of GPT," PNAS 120(51) / arXiv 2305.12763.**
- Method: 25 budget-allocation decisions over two goods at varying prices, in risk/time/social/food domains; tests GARP, computes CCEI (Afriat).
- Finding: GPT-4's choices are highly GARP-consistent (near utility-maximizing), often more rational and less heterogeneous than humans.
- STRUCTURAL: Yes, but revealed-preference/rationality-index style (CCEI), NOT parametric MLE of a risk-aversion coefficient. RLHF: No.

**Mazeika et al. (2025), "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs," arXiv 2502.08640.**
- Method: randomized forced-choice preference elicitation; fits **Thurstonian random utility models** to LLM choices; coherence metrics (transitivity, completeness); alignment via SFT/citizen-assembly.
- Finding: coherent, scale-increasing utility functions emerge; some anti-aligned values.
- STRUCTURAL: Yes, genuine likelihood-based random-utility estimation over LLM choices — the most direct methodological precedent for "fit a structural choice model to thousands of LLM binary choices." But the utilities are over general outcomes, NOT risk/lottery CRRA or loss-aversion λ, and not framing-invariance. RLHF: partially (scale and alignment interventions, not a clean base-vs-instruct econ-preference contrast).

**Liu, Yang & Tam (2025), "Evaluating and Aligning Human Economic Risk Preferences in LLMs," arXiv 2503.06646 (EMNLP 2025 main).**
- Method: three tasks (risk-category classification; S&P-vs-Treasury allocation; gambling with seven sequential sure-outcome options to derive certainty equivalents); fits **prospect-theory value function with loss-aversion λ and curvature α, β by least-squares** on certainty equivalents; then aligns to personas via DPO.
- Finding: Llama3-8B-Instruct and OLMo-2-7B-Instruct are strongly risk-seeking, deviating from human models; DPO can shift them toward target preferences.
- STRUCTURAL: Yes for λ/α/β, but via curve-fitting to certainty equivalents, NOT maximum-likelihood over binary Holt-Laury choices, and NO CRRA. Frame-invariance: not tested. RLHF: instruct-only, NO base-vs-instruct comparison. **The single closest published competitor; cite as the incumbent.**

**"Prospect Theory Fails for LLMs / Rethinking Prospect Theory for LLMs" (2025), arXiv 2508.08992.**
- Method: three-series lottery-choice experiment (Tversky-Kahneman style); estimates CPT parameters σ (curvature), λ (loss aversion), γ (probability weighting) for five instruct models; then swaps numeric probabilities for linguistic uncertainty markers.
- Finding: human-like risk curvature but **significantly weaker/unstable loss aversion** (λ: Llama-3.1-8B ≈0.01, Qwen2.5-14B ≈1.91, Qwen2.5-32B ≈1.21 vs. human ≈2.63); decisions unstable under epistemic-uncertainty phrasing.
- STRUCTURAL: Yes, parametric CPT estimation. Frame-invariance: tests linguistic-probability framing (not gain/loss framing). RLHF: instruct-only, NO base-vs-instruct.

**"Can Large Language Models Capture Human Risk Preferences? A Cross-Cultural Study" (2025), arXiv 2506.23107.**
- Method: lottery-choice tasks from transportation stated-preference surveys across four cities; **CRRA framework** applied to ChatGPT-4o and o1-mini vs. humans; multilingual prompts.
- Finding: both models systematically MORE risk-averse than humans; one-size-fits-all risk attitude ignoring cultural variation.
- STRUCTURAL: Uses CRRA but not clearly full-MLE structural recovery. Loss aversion λ: No. Gain/loss frame-invariance: No. RLHF: No.

## Category 3: RLHF / base-vs-instruct effect on biases (the post-training question)

**Itzhak, Stanovsky, Rosenfeld & Belinkov (2024), "Instructed to Bias: Instruction-Tuned Language Models Exhibit Emergent Cognitive Bias," TACL / arXiv 2308.00225.**
- Method: compares pretrained base vs. instruction-tuned/RLHF versions (Flan-T5, GPT-3 davinci variants, Mistral) on decoy, certainty, and belief-bias tasks.
- Finding: IT/RLHF **introduces or amplifies** these cognitive biases relative to base models.
- STRUCTURAL: No, qualitative bias-rate demonstration. RLHF: Yes — the paper that most directly establishes "post-training instills/amplifies the bias," but for certainty/decoy/belief bias, not structurally-estimated CRRA or λ. Strong precedent for hypothesis (c); the contribution is to make it structural and about risk/loss preferences.

**Bini, Cong, Huang & Jin (2026), "Behavioral Economics of AI: LLM Biases and Corrections," NBER w34745 / arXiv 2602.09362.**
- Method: broad audit of risk preferences, loss aversion, framing/reference dependence, base-rate neglect, ambiguity aversion (Ellsberg) in LLMs, plus prompt-based corrections.
- Finding: LLMs display human-like biases that are malleable to interventions.
- STRUCTURAL: No — explicitly qualitative/experimental demonstration, NOT MLE of CRRA or λ. Framing: Yes. RLHF: touches configuration effects but not a clean base-vs-instruct structural contrast. **The flagship econ-venue survey of the space; must be positioned against.**

**Wang, Li & Chen (2025), "Risk Profiling and Modulation for LLMs," arXiv 2509.23058 (v3, Oct 2025).** *Added Sept 2026.*
- Method: six models, including the exact Phase 3 pairs Llama-3.1-8B base/Instruct and Qwen2.5-7B base/Instruct, plus Qwen2.5-Math-7B and DeepSeek-R1-Distill-Llama-8B. Elicitation via Grable-Lytton and DOSPERT questionnaires and generated lottery choices; responses sampled (temperature 0.7, top-p 0.9) and parsed by regex, with no logit readout. Bayesian (MCMC) fits across a wide menu of utility classes: linear, power, CRRA, CARA, HARA, prospect theory, Epstein-Zin, Friedman-Savage. Three prompt tones (neutral, cautious, aggressive) with five phrasings each. Then modulates risk profiles by prompting, SFT, and DPO.
- Finding: pretrained models fit poorly under any utility class (best-model accuracy 69.6% Llama base, 74.9% Qwen base); instruction-tuned models fit standard formulations well (93.9% Llama Instruct, 82.7% Qwen Instruct); the reasoning-distilled DeepSeek-R1-Distill-Llama-8B fits worst (60.1%). SFT and DPO on choices generated by a target utility (e.g. CRRA) both shift risk preferences toward it, DPO more effectively. Base models get a plain-text few-shot prompt and instruct models a chat prompt, so stage and format are confounded. (Checked against arXiv 2509.23058, Oct 3 2026.)
- STRUCTURAL: Yes (Bayesian utility fits). Frame-invariance: No. Their "tone" manipulation instructs the model to be cautious or aggressive; it never relabels terminal-wealth-identical lotteries as gains or losses. RLHF: Yes, base vs instruct vs aligned, but no staged checkpoints along a single training path, no dominance/engagement control, no format control.
- **Closest competitor to Phase 3's base-vs-instruct result.** Their "pretrained models fit no utility model" anticipates Phase 3's "base models barely track value" on the same two families. Position Phase 3 as (i) a replication of that finding under a deterministic logit readout with a model-free dominance check, which says *why* the fit is poor (the base model is not engaging with value), and (ii) an extension to the OLMo-2 staircase, gain/loss framing, format control, and induced valuation, none of which they run.

**Itzhak, Belinkov & Stanovsky (2025), "Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs," COLM 2025 / arXiv 2507.07186.** *Added Sept 2026.*
- Method: 32 cognitive biases (including the framing effect, certainty effect, loss aversion and belief bias) measured as behavioral bias vectors. Two causal steps: (1) repeat instruction tuning with different random seeds to measure training randomness; (2) "cross-tuning," swapping instruction datasets (Tulu-2, Flan) between pretrained backbones (OLMo-7B, T5-11B; secondary checks on Llama2-7B, Mistral-7B). PCA over bias vectors.
- Finding: models cluster by pretraining backbone; instruction data is a secondary axis. Seed variation is moderate. Conclusion: pretraining plants a latent bias profile; finetuning brings it out and adjusts it.
- STRUCTURAL: No, bias scores and PCA, no preference parameters. Risk/λ: not estimated. RLHF: instruction tuning only, no preference-optimization stage separated.
- **The direct rival interpretation of Phase 3's headline.** The same authors' 2024 paper is cited in the Phase 3 writeup as evidence that post-training introduces bias; this paper is their updated position and points the other way. Phase 3 finds the measurable profile appears at SFT, but the OLMo base model fails the dominance check (0.555), so Phase 3 cannot distinguish "SFT installs the profile" from "SFT installs engagement and exposes a profile planted in pretraining." That second reading is this paper's thesis. Must be engaged directly; the natural test is a base-model elicitation that passes dominance, or a probe for a framing direction in base-model activations.

**"Where Do LLM Values Come From?" (LessWrong, 2025).** *Added Sept 2026; blog post, low weight.* Tracks value-laden behavior across OLMo training stages; reports the largest shifts at SFT and DPO, with some safety-related shifts from SFT partially reversed by DPO. Not peer reviewed and not about economic preferences, but it is further evidence that OLMo's staged checkpoints are becoming a standard testbed, which narrows the novelty of the staircase design itself (the novelty is the instruments run on it).

## Category 4: Other risk-behavior papers (mostly descriptive)

- **"AI as Decision-Maker: Ethics and Risk Preferences of LLMs" (2024), arXiv 2406.01168** — descriptive risk-attitude profiling across many models.
- **"How Personality Traits Shape LLM Risk-Taking Behaviour" (2025), arXiv 2503.04735** — persona manipulation; descriptive.
- **"LLM Agents Do Not Replicate Human Market Traders" (2025), arXiv 2502.15800** — Holt-Laury-style lotteries across six commercial models as a manipulation check, not structural estimation.
- **"Mind the (DH) Gap: A Contrast in Risky Choices Between Reasoning and Conversational LLMs" (2026), arXiv 2602.15173** — reasoning vs. chat models on risky choice; descriptive.
- **"LLM-driven Imitation of Subrational Behavior" (2024), arXiv 2402.08755** — descriptive.
- **"Are LLMs Rational Investors?" (2024), arXiv 2402.12713** — pronounced loss-aversion and framing bias (esp. GPT-4) in financial prompts; descriptive, no structural λ.
- **"Learning to be Homo Economicus: Can an LLM Learn Preferences from Choice Data?" (2024), arXiv 2401.07345** — adjacent, revealed-preference framing.

## Category 5: Methodological critiques of silicon sampling (must be engaged)

- **"GPT-ology, Computational Models, Silicon Sampling" (2024), arXiv 2406.09464** — cautions on treating LLM outputs as subject data.
- **"Do LLMs exhibit human-like response biases? A case study in survey design" (2023), arXiv 2311.04076** — order/label/wording sensitivity.
- **"Position: Stop Acting Like Language Model Agents Are Normal Agents" (2025), arXiv 2502.10420** — LLM "choices" are not stable preferences.
- **"The Illusion of Stochasticity in LLMs" (2026), arXiv 2604.06543** — sampling/temperature does not yield well-behaved choice randomness.
- **"Mitigating Social Desirability Bias in Random Silicon Sampling" (2025), arXiv 2512.22725**.
- **Yee & Koh (2026), "Large Language Models as Calibrated Measurement Instruments for Behavioral Parameters," arXiv 2602.01022** — VERIFIED July 2026: finance/asset-pricing orientation; 4 models, 24,000 agent-scenario pairs; documents systematic rationality bias (attenuated loss aversion, weak herding, near-zero disposition effects) in baseline LLMs; profile-based calibration shifts parameters toward human benchmarks; validates in an agent-based asset-pricing model. NOT a scoop: no Holt-Laury choice-level MLE, no Kőszegi-Rabin reference-point estimation, no gain/loss frame-invariance test, no base-vs-instruct contrast. Confirms "attenuated λ at baseline" as an expected finding.

## Assessment: What niche remains open (mid-2026)

The exact project has NOT been done, but the perimeter is closing fast. Pieces already exist: CRRA applied to LLM lottery choices (2506.23107); prospect-theory λ structurally estimated via certainty-equivalent curve-fitting (Liu-Yang-Tam) and CPT estimation (2508.08992, which already reports LLM λ far below human); Thurstonian random-utility MLE on LLM choices (Mazeika 2025); RLHF-instills-bias established qualitatively (Itzhak 2024; Bini et al. 2026). No single paper combines: (1) thousands of Holt-Laury binary choices with genuine choice-level maximum-likelihood CRRA recovery, (2) Kőszegi-Rabin/KT structural loss-aversion λ and endogenous reference point identified from gain/loss framing manipulations, (3) an explicit frame-invariance test, AND (4) a clean base-vs-instruct open-model contrast to attribute biases to post-training. The defensible niche is the integration plus (2)+(3)+(4): structural, choice-level, frame-manipulated λ with a base-vs-instruct causal design. Cite Liu-Yang-Tam and 2508.08992 as the incumbents extended; preempt reviewers with the stochasticity/contamination critiques (2604.06543, 2406.09464).

## Revised assessment (September 2026)

The two added papers change the niche in different ways.

Wang, Li & Chen (2025) take the plain base-vs-instruct contrast. On the same Llama and Qwen pairs, they already report that pretrained models fit no utility model and instruct models fit well. Phase 3's base-vs-instruct result is therefore a replication with a better readout (deterministic logits, dominance control, format control). It still counts, but it can no longer be the contribution.

Itzhak, Belinkov & Stanovsky (2025) take the other side of the attribution question. Their cross-tuning design says latent bias profiles come from pretraining. Phase 3's staircase says the measurable profile appears at SFT. Both can be true if SFT is where a pretrained disposition first becomes measurable, and the Phase 3 base model's failed dominance check is consistent with exactly that. The attribution claim has to be stated so that it survives this reading, or tested against it.

What remains open and defensible:

1. **Stage-level attribution of economic preference parameters along one training path** (OLMo-2 base, SFT, DPO, Instruct) with fixed instruments and fixed format. No paper runs structural risk and framing instruments on staged checkpoints.
2. **Engagement as a precondition for measurement.** A model-free dominance check that separates "odd preferences" from "no measurable preferences," applied per model and per format. Wang et al. observe poor base-model fit; nobody reports the check that explains it.
3. **Format as a measurement factor whose direction varies by family.** The reversal between Qwen and OLMo is new.
4. **Gain/loss frame-invariance tests with sign heterogeneity across families.** Still unclaimed; the incumbents estimate λ on instruct models without a terminal-wealth-identical relabeling.
5. **Induced valuation (Smith 1976) applied to LLMs.** Unclaimed in this literature.

Positioning: lead as a measurement paper (items 2 through 5 are the least exposed), with the staircase as the main application. Cite Wang et al. as the incumbent for base-vs-instruct and Itzhak et al. 2025 as the rival account of where the profile originates.
