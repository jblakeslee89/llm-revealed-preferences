# Phase 3 robustness: clustered bootstrap, template/gamble effects, and the intercept problem (Oct 3 2026)

Working note for the training-stage paper (Paper A). Scripts: `analysis/bootstrap_refit.py`,
`analysis/template_effects.py`. Outputs: `results/bootstrap_refit/`, `results/bootstrap_refit_intercept/`,
`results/template_effects/`.

## 1. Wild cluster bootstrap of the current specification (B = 400, all 11 subjects)

Response-scale wild bootstrap around the fitted model, Rademacher weights per gamble (80 clusters),
the same weight vector applied to every subject in a replicate so stage contrasts are paired. Clipped
cells per replicate: 0-8% for most subjects, 21% for Qwen Instruct few-shot (near-total refuser).

**Interior subjects: the clustered profile intervals hold.** Bootstrap intervals are 64-103% of the
profile-interval width for every OLMo post-SFT stage and Llama Instruct few-shot. The profile intervals
are, if anything, slightly conservative.

**Boundary subjects: the dispersion fallback was too narrow for OLMo base, as suspected.**

| OLMo base | profile (quasi) | bootstrap |
|---|---|---|
| lambda | [0, 0.03] | [0, 0.52] |
| gamma | [1.65, 2.07] | [1.32, 2.58] |
| alpha | [0.99, 1] | [0.87, 1] |

The base -> SFT contrasts still exclude zero with the wider intervals: Delta lambda +1.14 [+0.60, +1.25],
Delta gamma -1.46 [-2.14, -0.85], Delta alpha -0.94 [-1.00, -0.73]. Llama base is roughly right
(bootstrap 1.3-2.2x wider); Llama base -> instruct still excludes zero for lambda and gamma.

**Post-SFT contrasts, paired.** Gamble-level residuals correlate 0.96-0.97 between adjacent OLMo stages,
so pairing replicates is justified and sharply narrows the contrasts. Paired, some post-SFT moves are
detectably nonzero but an order of magnitude smaller than at SFT:

| | Delta lambda | Delta gamma | Delta alpha |
|---|---|---|---|
| SFT -> DPO | -0.06 [-0.10, -0.03] | +0.04 [+0.04, +0.13] | -0.02 [-0.05, +0.03] |
| DPO -> Instruct | -0.02 [-0.04, +0.00] | -0.07 [-0.10, -0.03] | -0.01 [-0.04, +0.00] |
| base -> SFT (for scale) | +1.14 [+0.60, +1.25] | -1.46 [-2.14, -0.85] | -0.94 [-1.00, -0.73] |

Unpaired, every post-SFT interval covers zero, matching the profile contrasts (which assume
independence). The writeup's "the parameters do not move after SFT" should become "after SFT the
parameters move by less than a tenth of the SFT step".

## 2. The intercept problem

Adding one constant c to the choice index (a fixed lean toward or away from the gamble, independent
of value) improves the fit significantly for nine of eleven subjects (quasi-LR 31-35 at the OLMo
post-SFT stages, 360 at OLMo base, 93 at Llama Instruct few-shot; not significant for Qwen Instruct
few-shot or Llama Instruct chat). It sends lambda to its floor (1e-4) for every subject except Qwen
Instruct few-shot (39), Llama Instruct chat (4.2) and OLMo Instruct chat (0.07).

**Correction (later the same day, see section 4):** an earlier version of this note said lambda is
not separately identified from c in this design. Simulation says otherwise: the current grid does
separate the two, because c shifts every frame and lambda only the mixed one. The intercept fit is the
data's answer, not an identification failure: once a constant is allowed, the post-trained models put
no weight on the loss side of a mixed gamble. A positive frame gap (more gambling when the same lottery
is described as a possible loss) is the opposite of what loss aversion predicts, and lambda about 1.1
in the base specification was the model using lambda as a partial intercept.

Under the intercept specification (bootstrap B = 200, paired):

| OLMo | c | lambda | gamma | alpha |
|---|---|---|---|---|
| base | +0.77 | 0.00 | 1.59 | 0.32 |
| SFT | -0.80 | 0.00 | 2.19 | 0.34 |
| DPO | -1.02 | 0.00 | 2.28 | 0.27 |
| Instruct | -1.30 | 0.00 | 2.03 | 0.26 |

- base -> SFT: only c moves detectably (-1.57). Lambda, gamma, alpha intervals all cover zero (gamma's
  is very wide, [-0.26, +5.09]).
- SFT -> DPO: c -0.22, DPO -> Instruct: c -0.28, both exclude zero paired. The structural parameters do
  not move detectably except a marginal gamma at DPO -> Instruct.
- Llama base -> instruct: c moves -1.06 [-1.13, -0.94]; lambda stays at zero; gamma interval barely
  excludes zero.
- Caveat: this bootstrap is visibly biased for c at the later OLMo stages (MLE -1.02 and -1.30 sit just
  outside their percentile intervals; replicate medians -0.81, -1.03). The ridge between c, gamma and
  alpha makes refits drift. Read the c contrasts' signs, not their exact bounds.

**What survives both specifications:** dominance and frame-gap results (model-free); SFT as the
largest single step; and after SFT, the structural parameters stay within a small neighborhood. **What
depends on the specification:** that lambda, gamma and alpha themselves move at SFT, and the numerical
values of lambda (about 1.1 without c, 0 with c). Under the intercept spec, what training does at every
stage is to move a constant lean toward the safe option, and that lean keeps growing through DPO and the
final stage, which is part of what the base spec was failing to capture after SFT.

## 3. Template and gamble effects (all 11 subjects)

- Template effects are significant for every subject (template RE vs intercept, LR 15-320), and at the
  OLMo post-SFT stages absorb about a fifth of the remaining deviance (20% SFT, 21% DPO, 17% Instruct;
  7% at base).
  The share does not grow at DPO, so template heterogeneity is not what DPO adds. One template
  (template 1) carries most of it, with a strong lean to the safe option that deepens across stages
  (-0.98 SFT, -1.46 DPO, -1.41 Instruct, relative to the stage mean). Model-free, the spread of uptake
  across templates triples from base (sd 0.05) to post-trained stages (0.12-0.16).
- Gamble random effects, with templates as fixed effects, are not significant for any OLMo stage
  (sigma at most 0.12, p >= 0.18) or Qwen subject. They are for Llama base and Llama Instruct
  few-shot (LR 11 and 38). Gamble-level misfit is not the missing channel on the staircase.
- Lambda, gamma and alpha are essentially unchanged by adding template or gamble effects on top of the
  intercept model. The parameter movement comes from c, not from heterogeneity.

## Implications for Paper A

1. State the attribution claim model-free first: dominance and frame gap, SFT the largest step.
2. Report the structural results under both specifications, with lambda flagged as not identified
   from a constant lean. The one structural result robust to specification is that post-trained models
   lean increasingly toward the safe option, a shift that starts at SFT and continues through DPO.
3. A design fix for identification: add mixed-frame gambles whose loss side varies independently of the
   gain side (or loss-only gambles), so lambda is pinned by a contrast a constant cannot mimic.

## 4. Identification check and the crossed gain x loss block (added Oct 3 2026)

`src/design_lossid.py` builds a 36-item block (`data/phase3_lossid_grid.csv`, 792 cells, instrument
`p3`) in the existing mixed-frame wording, with loss size L (14 to 271) and gain size G (0.45 to 5.5 x L)
crossed, p jittered near one half, gain-frame twins and dominance controls. `analysis/lossid_power.py`
simulates the SFT fit under two truths, A (lambda 1.14, c = 0) and B (lambda 0, c = -0.80), with noise
shaped like the real residuals, and refits with lambda and c both free (R = 40 each):

| truth | design | lambda-hat median [10%, 90%] |
|---|---|---|
| A | current grid | 0.97 [0.45, 1.55] |
| A | + block | 1.05 [0.44, 1.55] |
| B | current grid | 0.10 [0.00, 0.40] |
| B | + block | 0.02 [0.00, 0.31] |

The current grid already separates the truths; the block narrows lambda-hat only modestly. The reason:
with alpha near zero, loss aversion acts as a fixed penalty on any option with a loss side, which only
the frame contrast can identify, and the grid has that.

Model-free, the mixed-minus-gain logit gap (paired cells, gamble-clustered SEs) rises with the stated
gain and barely responds to the stated loss:

| subject | per log G | per log L |
|---|---|---|
| OLMo base | +0.13 (0.01) | -0.11 (0.03) |
| OLMo SFT | +0.26 (0.03) | -0.10 (0.08) |
| OLMo DPO | +0.43 (0.05) | -0.23 (0.12) |
| OLMo Instruct | +0.49 (0.06) | -0.23 (0.15) |
| Llama Instruct | +0.06 (0.03) | +0.10 (0.10) |

The block's real value is this model-free test, made sharp: in the current grid L spans 29-186 and is
correlated with G (r = 0.50); in the block L spans 14-271 and is orthogonal to G. It costs about half a
Phase 3 grid per model (immediate answers, T4).
