# Literature Survey: Structural Estimation of Economic Preferences and Behavioral Biases in LLMs

Prepared July 2026 (web-search sweep). Coverage through 2026. For each paper: method, finding, whether it does STRUCTURAL (likelihood-based parameter recovery) vs. DESCRIPTIVE work, and whether it addresses RLHF/post-training.

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
