# Profile-Likelihood Refit of the Phase 3 Structural Fits (Sep 28 2026)

Working note for the Phase 3 writeup revision. Script: `analysis/profile_refit.py`.
Outputs: `results/profile_refit/{fits,intervals,contrasts,profiles}.csv`, figure
`writeups/figures/phase3_profiles.pdf`.

## What changed

- **Wider box.** Lambda was hard-clipped at log lambda in [-4, 4] ([0.018, 54.6]); the
  refit uses [-9, 9] ([1e-4, 8103]), gamma [0.018, 54.6], alpha (0, 1) with a_raw in
  [-12, 12]. Same objective, model, and core frames as `estimate_from_probs.py`, phi = 1.
  `estimate_reference.py` gained an optional `lam_clip` argument; its default is
  unchanged, so Phases 1-2 are unaffected.
- **Profile intervals replace the cell bootstrap.** Each of lambda, gamma, alpha is fixed
  on a grid and the other four parameters re-optimized. Interval = where the scaled
  profile crosses 3.84.
- **Calibration.** Primary: gamble-clustered sandwich (80 clusters; templates, orders and
  frames of one gamble are not independent), scaling 2*Delta(dev) by the robust/naive
  variance ratio for that parameter. Fallback where any parameter sits at the wide bound
  or the Hessian is unusable (condition > 1e10): Pearson dispersion (quasi). The quasi
  intervals ignore within-gamble correlation and are probably too narrow; they apply to
  qwen-inst, qwen-inst-chat, llama-base, llama-inst-chat, olmo-base.
- **Contrasts** combine the two subjects' profiles exactly (no shared nuisances). A side
  is reported open (inf) when either constituent interval is open toward it; on the
  level scale, the lower box edges of lambda and gamma and both alpha edges are natural
  limits and do not count as open.
- **Checks.** Dry-run recovery exact (alpha 0.8, lambda 2.5, gamma 1.6, mu 0.15, delta 0;
  deviance 6e-8). No profile found a better optimum than the full fit except qwen-inst
  (0.056 deviance; a flat ridge in the fully refusing fewshot fit).

## Results

**The staircase claim survives, now with finite intervals everywhere it matters.**

| Contrast (level) | Delta lambda | Delta gamma | Delta alpha |
|---|---|---|---|
| OLMo base -> SFT | +1.14 [+1.02, +1.26] | -1.46 [-1.79, -1.18] | -0.94 [-1.00, -0.78] |
| OLMo SFT -> DPO | -0.06 [-0.23, +0.11] | +0.04 [-0.24, +0.34] | -0.02 [-0.22, +0.20] |
| OLMo DPO -> Instruct | -0.02 [-0.18, +0.13] | -0.07 [-0.32, +0.19] | -0.01 [-0.20, +0.18] |
| Llama base -> instruct | +1.20 [+1.04, +1.27] | -1.00 [-1.35, -0.72] | -0.10 [-0.29, +0.07] |
| Qwen base -> instruct (fewshot) | +19.2 [+2.7, open] | +0.99 [+0.61, +1.41] | +0.10 [-0.12, +0.26] |

- The overflowed base-to-SFT gamma interval (lower bound ~ -3.8e48 in the writeup) is now
  [-1.79, -1.18]. OLMo base gamma is identified once lambda is allowed below 0.018
  (1.84 [1.65, 2.07]); it was the lambda clip, not gamma, that broke the bootstrap.
- The writeup's SFT-to-DPO lambda interval [-0.18, +1.06] had an upper tail produced by
  bootstrap draws at the bound; the profile interval is [-0.23, +0.11]. DPO and the final
  stage move nothing, more tightly than reported.
- **Base models: lambda goes to zero.** With the clip removed, every base model (and Qwen
  instruct in chat) puts lambda at the new floor, 1e-4, with an upper limit of 0.03 to
  0.14. The writeup's "0.018 at bound" was the clip. Substantively: base models put no
  weight on the loss side of mixed-frame gambles, which is the structural reading of
  their weak value-tracking. The level contrast is well defined because lambda cannot go
  below 0; the log-ratio contrast is open and should not be reported.
- **Alpha after post-training is not bounded away from zero.** Every instruct and SFT
  subject has an alpha interval that is open below (e.g. OLMo SFT [0, 0.23]). Near-zero
  curvature means payoff magnitude barely enters the value function: the structural
  version of the Phase 1 finding that the model prices probability, not payoffs.
- **Qwen instruct fewshot lambda is not identified above** (lower limit 3.6, open to the
  box). Report only that it is large; the old CI [+3.6, +53.7] had its upper end set by
  the clip. That subject is already excluded from attribution claims.

**One new finding: Llama is format-stable only on model-free quantities.** The writeup
calls Llama instruct "roughly format-stable" (dominance and frame gap similar across
formats). The structural parameters are not: fewshot to chat moves gamma +1.84
[+1.17, +3.47], lambda +2.14 [+0.89, +4.43], alpha +0.88 [+0.72, 1.00] (chat alpha sits at
its upper bound, quasi calibration). OLMo instruct also moves lambda (+0.31) and gamma
(-0.19) across formats. The writeup's format section should say that model-free stability
does not imply parameter stability, which strengthens the case for fixing format within
any attribution contrast.

## Edits the Phase 3 writeup needs (applied Sep 28 2026)

1. Staircase table: replace starred values (lambda 0.018*, alpha 1.000*) with the refit
   values and profile intervals; drop the boundary-flag caveat for gamma.
2. SFT->DPO and DPO->Instruct CIs: replace with the profile intervals above.
3. Llama and Qwen attribution sentences: Llama numbers barely change; Qwen lambda becomes
   "at least +2.7, upper end not identified".
4. Format section: add the Llama structural instability.
5. Limitations: replace the boundary-pinned paragraph with the quasi-calibration caveat
   for the five boundary subjects.
