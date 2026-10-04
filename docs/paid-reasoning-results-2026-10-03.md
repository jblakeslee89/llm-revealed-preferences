# Arm 5 Results: Paid With Reasoning (Oct 3 2026)

Plan: `robustness-plan-2026-10.md`, arm 5. Trainer: `colab/rl_induce_reason.py` (GRPO-style
REINFORCE on a LoRA adapter; 8 prompts x 4 sampled traces per step; leave-one-out advantage;
per-token KL 0.02; lr 1e-5; expected-utility reward; balanced training gambles). The model is
paid and **not told** the rule (`--induce none`). Three 200-step runs on a Colab A100 in bf16,
Oct 3, back to back, about 7.5 hours and about 40 compute units. Each adapter was then scored on
the 840 reasoning-arm cells (20 gambles per instrument) with `phase3_elicit.py --reason`.

Files: `data/phase4_*_reason_paid-*.csv` (evaluations); `results/paid_reasoning/<run>/`
(training log, config, held-out monitor traces); `results/paid_reasoning/scores.txt`
(`score_robustness.py`, 1,000 cluster-bootstrap draws). The adapters (about 160 MB each) are
kept outside the repo, in `~/Code/llm-revealed-preferences-adapters/`.

Run notes. `peft` refuses to load next to the `torchao` 0.10 preinstalled on Colab's A100
image; `pip uninstall -y torchao` before training fixes it. The sqrt run used `--n-train 4000`:
balancing under sqrt keeps only about 114 of 800 generated gambles, so the larger pool gave 650
balanced training gambles (linear: 558).

## Results on the 840 evaluation cells

| Run | Scored against | Unpaid, untold, reasoning | **Paid, untold, reasoning** | Told the rule, reasoning |
|---|---|---|---|---|
| Qwen, linear | EV | 0.829 | **0.927** (+0.098 [+0.043, +0.163]) | 0.965 |
| Qwen, sqrt | sqrt rule | 0.606 | **0.739** (+0.133 [+0.069, +0.208]) | (immediate, told: 0.124) |
| OLMo Instruct, linear | EV | 0.741 | **0.776** (+0.035 [-0.017, +0.086]) | 0.803 |

(Agreement with the paid rule's optimum on clear core cells; brackets are paired 95%
intervals against the unpaid run on the same cells.)

**Qwen, linear: payment works in the reasoning regime.** Paid and never told the rule, Qwen's
agreement with expected value rises from 0.829 to 0.927, closing about 70% of the distance to
the told-and-reasoning level (it stays 0.037 [0.018, 0.059] short of it). Dominance stays at
1.000. Held-out balanced accuracy on the training monitor rose from 0.85 / 0.95 / 0.70 (risk /
gain / loss wording) to 1.00 / 1.00 / 0.95 by step 50. The loss-wording gap on the evaluation
cells is unchanged (-0.127 against -0.135 unpaid), although the monitor's gap shrank during
training.

**Qwen, sqrt: payment induces some risk aversion.** Agreement with the sqrt rule rises from 0.606
to 0.739, and agreement with risk-neutral EV falls by 0.139 [0.079, 0.210]: the paid model moved
away from expected value and toward the risk-averse rule it was paid under, which instruction
alone could not do (0.124 compliance told in one letter).

**OLMo: no detectable effect.** Agreement changes by +0.035 (interval includes zero), dominance
falls from 1.000 to 0.900 (-0.077 [-0.167, 0.000] on the 839 shared cells), and the loss-wording gap moves from about +0.06 to -0.195 (difference
-0.250 [-0.350, -0.146]). On the monitor, balanced accuracy drifted from 0.90 / 0.77 / 0.68 to
0.85 / 0.78 / 0.72 with no trend. This matches OLMo's non-response to the clarifying sentence
(`reason-pilot-2026-09-28.md`) and its noisy sampled traces (robustness arm 4).

## Against the pre-set decision rule

The plan's rule 5: if held-out balanced accuracy rises above 0.5 and evaluation EV agreement
approaches the told-and-reasoning level, payment does in the reasoning regime what it could not
do in one-letter answers, and the Phase 4 negative is about the regime, not the incentive. For
Qwen this holds: balanced accuracy reached 1.00 on two wordings, and evaluation agreement came
within 0.04 of the told level. For OLMo it does not hold. The result is family-specific.

## Caveats

- Peak, then slide. In both Qwen runs the monitor peaked mid-training (linear: loss-wording
  balanced accuracy 0.95 at steps 50 to 100, 0.78 at step 200; sqrt: gain wording 0.75 at steps
  100 to 125, 0.56 at step 200), and the evaluation used the step-200 adapter. A shorter run, or
  checkpoint selection on the monitor, may do better; the reported numbers are conservative on
  this point. The monitor has 16 prompts per wording, so single-step moves of 0.06 are one prompt.
- One seed per configuration, one learning rate and KL weight; the sensitivity analysis the
  Phase 4 caveats ask for is unrun.
- Training used neutral Phase 1 wording only; the gain and loss wordings in the evaluation are
  held out, and the 40 evaluation gambles are excluded from training.
- The sqrt run had a larger training pool than the plan's default (650 balanced gambles).
- `monitor_traces.jsonl` holds the written reasoning at each monitor evaluation; whether payment
  changed the verdict habits (Qwen's stock phrase) is not yet checked.
