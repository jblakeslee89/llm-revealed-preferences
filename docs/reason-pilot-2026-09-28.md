# Reason-then-answer pilot (Sep 28 2026)

First live run of `colab/phase3_elicit.py --reason`. Colab T4, 4-bit.

    --model Qwen/Qwen2.5-7B-Instruct --fmt chat --induce riskneutral --reason
    --gambles 2 --batch-size 16 --max-new-tokens 320

84 cells (2 gambles per instrument, all frames, templates, orders), 0 excluded,
every cell ended with the model's own "Answer:" line. The CSV
(`pilot_qwen_reason_rn.csv`) was not downloaded; numbers below were scored in
the notebook, matched on (instrument, trial_id).

| Qwen instruct, chat, same 40 clear cells | EV compliance mass | hard accuracy |
|---|---|---|
| immediate answer, uninduced | 0.532 | 0.550 |
| immediate answer, induced (risk-neutral) | 0.523 | 0.525 |
| reason first, induced (risk-neutral) | 1.000 | 1.000 |

Dominance controls 1.000 in all three. Reasoning samples compute
probability x payoff correctly and choose the larger EV, including in the
mixed (loss-worded) frame, e.g. "(0.60 * $234) + (0.40 * $0) = $140.40 ...
$140.40 > $139".

Reading: on this pilot, the induced-arm failure is a computation ceiling of
immediate-answer elicitation, not refusal to adopt the stated rule. Four gambles
only; the full 840-cell runs decide it.

Timing: ~3 min model download, ~1 min weight load, ~4 min for 84 cells
(~3 s/cell at batch 16). Full 840-cell run: roughly 40-45 min per
condition, so one model (induced + uninduced) per Colab session.

Bug found: `score_induced.py --baseline` matched on trial_id alone, which
repeats across instruments (1,280 unique ids for 1,680 cells), so the baseline
picked up extra cells (130 core instead of 80). Fixed to match on
(instrument, trial_id).

## Full run: Qwen instruct, chat, induced risk-neutral, reasoning (840 cells)

`data/phase3_qwen-inst_chat_reason_riskneutral.csv`: `--gambles 20` (20 of 40
gambles per instrument, seed 0), 840 cells, 0 excluded, every cell ended with
the model's own "Answer:" line. Run time about 39 min on a free T4.

Scored on matched cells (instrument, trial_id):

| Same 620 clear core cells | EV compliance mass | hard accuracy |
|---|---|---|
| immediate answer, uninduced | 0.536 | 0.532 |
| immediate answer, induced | 0.562 | 0.569 |
| reason first, induced | 0.965 | 0.965 |

Dominance controls (40): 1.000 with reasoning, vs 0.868 immediate induced and
0.925 uninduced.

Hard accuracy by frame with reasoning: gain 1.000, risk 1.000, neutral 0.994,
**mixed 0.869**. The pilot's reading holds: the immediate-answer failure was a
computation ceiling.

### The residual framing effect is a reference-point accounting error

21 of the 22 misses are in the mixed (loss-worded) frame, and all 21 are cells
where the gamble has the higher EV and the model chose the sure amount.
In the mixed frame, accuracy is 1.000 when EV favors the sure amount and 0.767
when it favors the gamble.

The reasoning shows how. The model computes the gamble as a change from current
wealth and compares it with the sure option as a level, e.g. sure $98,
gamble 47% of $227: "E_A = (0.47 x 227) + (0.53 x 0) - 98 = 8.69 ... E_B = 98 ...
choose B". Another variant nets the gain and loss correctly
("0.68 x 120 - 0.32 x 81 = 55.68") and then compares that net change with the
$81 level. Every error of this kind favors the safe option. The three misses read
closely all follow the pattern; a rough text check finds a subtracted or
negative loss term in all 21.

So with reasoning allowed, the loss wording no longer shifts a snap judgment; it
induces an inconsistent reference point in explicit arithmetic, which is the
Phase 2 framing effect reappearing as mental-accounting error. Worth a targeted
follow-up: score each reasoning trace for which reference point it uses, and
test whether a one-line clarification ("compare final dollar amounts") removes it.

## Full run: Qwen instruct, chat, uninduced, reasoning (840 cells)

`data/phase3_qwen-inst_chat_reason.csv`: same 840 cells, 0 excluded, every cell
ended with "Answer:". Both full runs took 1 h 12 min in one Colab session.

All four conditions on the same 840 cells (EV agreement on the 620 clear core
cells; uptake = mean P(gamble)):

| Condition | Dominance | EV agreement | Gain uptake | Mixed uptake | Frame gap |
|---|---|---|---|---|---|
| immediate, uninduced | 0.925 | 0.532 | 0.267 | 0.893 | +0.626 |
| immediate, induced | 0.868 | 0.569 | 0.441 | 0.843 | +0.401 |
| reason, uninduced | 1.000 | 0.829 | 0.535 | 0.400 | -0.135 |
| reason, induced | 1.000 | 0.965 | 0.600 | 0.455 | -0.145 |

Three readings.

1. **Reasoning alone moves the model most of the way to EV.** Uninduced, every
   trace (100%) frames the choice as expected value without being asked, and EV
   agreement rises from 0.532 to 0.829. The instruction adds the rest (0.965).
2. **The framing effect reverses sign under reasoning.** Immediate answer: loss
   wording raises gambling (+0.626, the textbook direction). With reasoning: it
   lowers it (-0.135), the direction Phase 2 found for Haiku (-0.302). The
   mechanism is visible in the traces: the same reference-point slip as in the
   induced run (gamble EV as net change, e.g. "0.68 x 120 + 0.32 x (-81) =
   55.68", compared with the sure amount as a level, $81), which always favors
   the safe option. 34 uninduced mixed-frame misses where EV favored the gamble.
3. **So "the framing effect" of a model depends on the elicitation regime, and
   under reasoning it is an arithmetic error with a fixed direction.** Hypothesis
   worth testing: Haiku's anti-textbook Phase 2 gap reflects the same deliberate
   but mis-referenced computation.

Open: the non-mixed EV misses in the uninduced reasoning run (risk and neutral
frames at ~0.79) are not yet classified (risk aversion stated in the trace vs
arithmetic error). A trace-coding pass would split them.

Next: same two runs on OLMo Instruct and SFT (fewshot), then Llama instruct
(chat), one model per Colab session.

## OLMo-2 Instruct, fewshot, reasoning (induced and uninduced, 840 cells each)

`data/phase3_olmo-inst_fewshot_reason_riskneutral.csv`,
`data/phase3_olmo-inst_fewshot_reason.csv`. Same 20-gamble subset, fewshot
format with the reasoning exemplars (`FEWSHOT_REASON`), greedy, 320-token cap.
Both runs took 1 h 05 min in one session. 1 cell excluded (uninduced).

End states: induced 829 answer / 7 eos / 4 blank; uninduced 518 answer / 234
eos / 88 blank. The eos traces are complete (median ~395 characters, ending in a
stated choice), not truncated at the token cap; the readout appends "Answer:"
after them as designed.

| OLMo Instruct, same cells | Dominance | EV agreement (620 clear) | Frame gap |
|---|---|---|---|
| immediate, uninduced | 0.897 | 0.497 | +0.346 |
| immediate, induced | 0.976 | 0.542 | +0.003 |
| reason, uninduced | 0.999 | 0.740 | +0.055 |
| reason, induced | 0.974 | 0.803 | -0.024 |

Induced, by frame (clear cells): gain 0.88, risk 0.94, neutral 0.83,
**mixed 0.59**. 91% of uninduced traces invoke expected value unprompted
(the fewshot exemplars demonstrate the arithmetic, so this is partly primed).

Same qualitative result as Qwen at a lower level: reasoning lifts EV agreement
substantially (0.50 to 0.74 uninduced; 0.54 to 0.80 induced) and removes the
immediate-answer framing gap (+0.346 to about zero). Two differences:

1. The gap goes to zero rather than reversing (Qwen: -0.135).
2. OLMo's mixed-frame misses run in both directions (accuracy 0.52 when EV
   favors the gamble, 0.67 when it favors the sure amount). The traces show the
   same reference-point slip as Qwen (using the $245 gain where the $362 final
   amount belongs) plus plain comparison errors ("Since $151.84 is less than
   $117"), which Qwen did not make.

Session note: a mis-targeted keystroke re-ran the finished Instruct cell; both
commands were interrupted during model download, before the script opens its
output file, and the CSVs were verified byte-identical against the copies taken
earlier (induced) or taken from the zip written at 16:31 (uninduced).

## OLMo-2 SFT, fewshot, reasoning (induced and uninduced, 840 cells each)

`data/phase3_olmo-sft_fewshot_reason_riskneutral.csv`,
`data/phase3_olmo-sft_fewshot_reason.csv`. The induced run hit CUDA OOM after
560 cells (the readout computed full-sequence logits; fixed in 9f0a926 with
`logits_to_keep=1`) and was completed with `--resume`; the uninduced run
resumed from 16 cells. Both files have 840 unique cells, no missing values,
0 excluded; the zip copy of the induced file is byte-identical to the resumed
download.

The SFT-to-Instruct comparison on the same cells:

| Condition | SFT EV | SFT gap | Instruct EV | Instruct gap |
|---|---|---|---|---|
| immediate, uninduced | 0.519 | +0.241 | 0.497 | +0.346 |
| immediate, induced | 0.545 | -0.008 | 0.542 | +0.003 |
| reason, uninduced | 0.681 | **+0.375** | 0.740 | **+0.055** |
| reason, induced | 0.713 | +0.358 | 0.803 | -0.024 |

(EV = hard EV agreement on the 620 clear core cells; gap = mean P(gamble) in
the mixed frame minus the gain frame. Dominance with reasoning: SFT 0.897 /
0.958, Instruct 0.999 / 0.975.)

**This qualifies the Phase 3 staircase claim.** Under immediate answer, the
structural parameters were fixed at SFT and the later stages moved nothing
detectable. Under reasoning, the later stages matter: the frame gap is large at
SFT (+0.375, larger than SFT's immediate-answer gap) and nearly gone by Instruct
(+0.055), and EV agreement rises (0.68 to 0.74 uninduced, 0.71 to 0.80 induced).
So preference optimization (DPO and the final stage) changes how the model
reasons about these choices, even though it leaves the immediate-answer
parameters where SFT put them. The attribution claim should be stated per
elicitation regime.

Caveats: one family; DPO checkpoint not yet run in the reasoning arm, so the
SFT-to-Instruct change cannot yet be split between DPO and the final stage;
fewshot reasoning exemplars show the EV arithmetic. Next: OLMo DPO with
reasoning (same two runs), then Llama instruct (chat).
