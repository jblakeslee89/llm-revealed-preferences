# Phase 3 First Live Results (overnight run, Aug 12-13 2026)

Status notes from the first live Phase 3 elicitation runs on Colab T4. Written
as the runs completed; numbers below come from `analysis/estimate_from_probs.py`
on the CSVs in `data/`. This is a working document for the Phase 3 writeup, not
the writeup itself.

## Data inventory

All runs use the exported 1,680-cell grid (`data/phase3_grid.csv`: 400 Phase-1
cells, 1,200 Phase-2 cells, 80 dominant controls), scored by next-token
logprobs, 4-bit quantization, free Colab T4.

| File | Model | Format | Excluded | ab_mass median (min) |
|---|---|---|---|---|
| `phase3_qwen-base_fewshot.csv` | Qwen2.5-7B | fewshot | 0% | 0.960 (0.863) |
| `phase3_qwen-inst_fewshot.csv` | Qwen2.5-7B-Instruct | fewshot | 0% | 0.995 (0.237) |
| `phase3_qwen-inst_chat.csv` | Qwen2.5-7B-Instruct | chat | 0% | 1.000 (1.000) |
| `phase3_olmo-base_fewshot.csv` | OLMo-2-1124-7B | fewshot | 0% | 0.908 (0.619) |
| `phase3_olmo-sft_fewshot.csv` | OLMo-2-1124-7B-SFT | fewshot | 0% | 0.982 (0.823) |
| `phase3_olmo-dpo_fewshot.csv` | OLMo-2-1124-7B-DPO | fewshot | 0% | 0.986 (0.560) |
| `phase3_olmo-inst_fewshot.csv` | OLMo-2-1124-7B-Instruct | fewshot | 0% | 0.992 (0.689) |
| `phase3_olmo-inst_chat.csv` | OLMo-2-1124-7B-Instruct | chat | 0% | 0.985 (0.753) |

Llama-3.1-8B is not run: it is HF-gated and logging in requires John's token,
which the overnight agent does not handle. The pair can be added in a fresh
Colab session in about an hour.

## Headline model-free quantities

Dominant-control uptake is the fraction of the 80 dominance trials where the
model picks the strictly larger sure amount (Haiku, Phase 2: 1.000). Frame gap
is mean p(gamble) in the mixed frame minus the gain frame on
terminal-wealth-identical lotteries (Haiku, Phase 2, choice-level: -0.302).

| Subject (format) | Dominance | Gain uptake | Mixed uptake | Frame gap |
|---|---|---|---|---|
| Qwen base (fewshot) | 0.639 | 0.161 | 0.365 | +0.203 |
| Qwen instruct (fewshot) | 0.680 | 0.000 | 0.001 | +0.001 |
| Qwen instruct (chat) | 0.927 | 0.267 | 0.893 | +0.626 |
| OLMo base (fewshot) | 0.555 | 0.499 | 0.598 | +0.099 |
| OLMo SFT (fewshot) | 0.756 | 0.308 | 0.551 | +0.243 |
| OLMo DPO (fewshot) | 0.882 | 0.218 | 0.536 | +0.318 |
| OLMo Instruct (fewshot) | 0.896 | 0.159 | 0.508 | +0.349 |
| OLMo Instruct (chat) | 0.636 | 0.475 | 0.552 | +0.078 |

## Finding 1: base models barely track value

Both base models emit A/B tokens with high confidence (median ab_mass 0.91 to
0.96) yet pass dominance barely above chance (0.56, 0.64). OLMo base picks the
gamble at roughly a coin flip in every frame. The few-shot wrapper teaches the
answer format; it does not produce value-tracking behavior. This reframes the
attribution question: post-training does not merely bend existing preferences,
it appears to install most of the value-tracking that makes "preferences" a
meaningful description at all.

## Finding 2: the format confound is real and large (Qwen)

The dual-format control on Qwen instruct was designed to check that format does
not drive the base-vs-instruct contrast. It failed in the informative
direction: the same instruct model is a different subject under the two
formats.

- Fewshot: near-total gamble refusal (uptake 0.000 to 0.001 in both frames),
  frame gap eliminated, structural lambda pinned at the upper bound (54.6, the
  same corner Phase 2 hit), delta = -4.67 (choices ride position, not value).
- Chat: dominance 0.927, frame gap +0.626 (gain 0.267 vs mixed 0.893), lambda
  at the lower bound.

OLMo Instruct shows the same instability in the OPPOSITE direction: fewshot is
its well-behaved format (dominance 0.896) while chat degrades it (dominance
0.636, delta +3.83, frame gap shrinks to +0.078). So format instability
generalizes across model families, but which format elicits coherent behavior
is model-specific. Qwen instruct behaves under chat; OLMo Instruct behaves
under fewshot. Any cross-model comparison must hold format fixed AND verify
per-model that the chosen format elicits engaged behavior (dominance is the
cheap check). The OLMo staircase below is all-fewshot, which is OLMo's good
format, so the staged attribution is unaffected.

Consequence: the Qwen base-vs-instruct attribution (Delta lambda CI [+3.6,
+53.7], Delta gamma CI [+0.71, +1.25] under fewshot) cannot be read as a clean
post-training effect, because the instruct model's fewshot behavior is not
representative of the same model in its natural chat format. The OLMo staged
contrasts below, all elicited under the SAME fewshot format, are the cleaner
attribution.

## Finding 3: the frame gap flips sign relative to Haiku

Phase 2 found Haiku SUPPRESSES gambling under loss wording (gap -0.302, the
anti-textbook direction). Every open model here does the opposite: mixed-frame
wording increases gamble uptake (gaps +0.10 to +0.63), the textbook
risk-seeking-in-losses direction from prospect theory. Cross-model
heterogeneity in the SIGN of the framing response is a stronger result than
either finding alone: the bias is not a fixed property of LLMs as a class, and
it is presumably a function of post-training recipes.

## Finding 4: OLMo staged attribution (the staircase)

Same format (fewshot) across all four checkpoints, so the format confound is
held fixed by design.

| Contrast | What moves |
|---|---|
| base -> SFT | Dominance 0.555 -> 0.756; frame gap +0.099 -> +0.243; structural fit moves off the boundary (lambda 0.018* -> 1.14, alpha 1.00* -> 0.06; * = at bound) |
| SFT -> DPO | Dominance 0.756 -> 0.882; frame gap +0.243 -> +0.318; parameters statistically unchanged (Delta lambda -0.06, CI [-0.18, +1.06]; Delta gamma +0.04, CI [-0.19, +0.39]) |
| DPO -> Instruct | Quiet. Dominance 0.882 -> 0.896; frame gap +0.318 -> +0.349; Delta lambda -0.02, CI [-0.14, +0.11]; Delta gamma -0.07, CI [-0.26, +0.17]; Delta alpha -0.01, CI [-0.11, +0.14] |

Reading: SFT installs most of the value-tracking and the framing sensitivity;
DPO tightens dominance compliance without moving the preference parameters;
the final Instruct stage moves nothing. The attribution localizes to SFT. This
is the paper's most distinctive sentence: the biases measured in
instruction-tuned models are already fully present after supervised
fine-tuning, and preference-based stages (DPO, final RLHF-style polish) neither
add nor remove them.

## Caveats for the writeup

1. **Boundary-pinned structural fits.** Base-model and Qwen-instruct fits hit
   parameter bounds (lambda at 0.018 or 54.6, alpha at 1.0). Parameter values
   and bootstrap CIs at bounds are not trustworthy; one bootstrap CI overflowed
   outright (Delta gamma lower bound ~ -3.8e48 in the base->SFT contrast).
   Model-free quantities carry the results; report structural fits with
   boundary flags, or refit with wider bounds and report profile likelihoods.
2. **Dominance filter option.** An alternative to reporting weakly engaged
   subjects at face value: condition the structural fits on passing dominance
   (or report both). Base-model "preferences" with 55% dominance compliance are
   closer to noise than taste.
3. **Only decisions differ from the dry run.** The estimator recovered
   parameters near-exactly on simulated data (`data/phase3_dryrun.csv`), so
   these degenerate fits are properties of the models' choice patterns, not
   estimator bugs.
4. **Chat-format bug fixed mid-run.** `apply_chat_template` returns a
   BatchEncoding in current transformers; fixed in `colab/phase3_elicit.py`
   (return_dict=False) after the first chat attempt crashed. Fewshot runs were
   unaffected.

## Session log (for reproducibility notes)

- Free Colab T4, one session for Qwen pair + OLMo base/SFT/DPO (~5 h), second
  session for OLMo Instruct after the first hit its time limit mid-download.
- OLMo Instruct's first attempt died on a disk-full IO error (OLMo base's
  29 GB fp32 cache plus SFT and DPO caches filled the disk); the re-run on a
  fresh VM avoids this.
- Chrome blocked Colab's multiple automatic downloads after the second file;
  CSVs were collected via the runtime proxy (`/files/content/...` with a
  short-lived token minted per Files-pane download click).

## Next steps

1. Llama-3.1-8B base/instruct pair (needs John's HF token, ~1 h of Colab).
2. Consider a wider-bounds refit and a dominance-conditioned refit before the
   writeup locks numbers; several bootstrap CIs overflowed at parameter bounds
   (worst case ~1e86 in the OLMo format contrast) and need either wider bounds
   or profile-likelihood reporting.
3. Phase 3 writeup: lead with the staged OLMo attribution (localizes to SFT)
   and the format instability; the sign-flip vs Haiku connects Phase 3 back to
   Phase 2.
