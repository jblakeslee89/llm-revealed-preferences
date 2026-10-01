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

## Bootstrap intervals (added with the Phase 3 writeup update)

95% intervals, resampling the 40 gamble clusters (20 per instrument), 2,000 draws:

| Contrast | Estimate | 95% CI |
|---|---|---|
| Qwen EV, reason induced minus immediate induced | +0.395 | [+0.331, +0.456] |
| Qwen EV, reason uninduced minus immediate uninduced | +0.297 | [+0.213, +0.379] |
| Qwen frame gap, reason uninduced | -0.135 | [-0.221, -0.053] |
| OLMo Instruct EV, reason induced minus immediate induced | +0.262 | [+0.199, +0.325] |
| OLMo SFT frame gap, reason uninduced | +0.375 | [+0.286, +0.466] |
| OLMo Instruct frame gap, reason uninduced | +0.055 | [-0.075, +0.183] |
| Instruct minus SFT, frame gap, reason uninduced | -0.320 | [-0.449, -0.201] |
| Instruct minus SFT, EV, reason induced | +0.090 | [+0.030, +0.160] |
| Instruct minus SFT, EV, reason uninduced | +0.060 | [-0.025, +0.148] |

Correction to the SFT section above: the uninduced SFT-to-Instruct EV rise
(0.68 to 0.74) is not distinguishable from zero; the induced rise and the
frame-gap drop are.

## Clarification check (Sep 30 2026): the loss-frame error is a fixable habit

One line appended to the instruction: "When you compare the options, compare
the final dollar amounts you would end up with under each, not the changes
from what you hold now." (`--induce riskneutral_clarify` / `clarify`,
`colab/phase3_elicit.py`.) Qwen2.5-7B-Instruct, chat, reasoning first, the same
20 gambles, gain and mixed frames only (400 cells), matched against the
earlier run on the same cells.

| Reasoning first, told to maximize EV | Mixed-frame accuracy (clear) | when gamble optimal | Frame gap |
|---|---|---|---|
| without the line | 0.869 (21 errors) | 0.767 | -0.145 |
| with the line | **1.000 (0 errors)** | **1.000** | **0.000** |

Gain-frame accuracy is 1.000 in both. The residual framing effect under
reasoning is removed completely by naming the right comparison, so it is a
bookkeeping habit, not a preference that survives correction. Consequence for
Phase 4: paying the model in the reasoning regime to remove this error would
buy, expensively, what one line of instruction already buys. The paid-reasoning
budget should target the sqrt (risk-averse) utility and transfer to snap
answers instead.

### Clarification without the EV instruction: partial

The same line without the expected-value instruction (`--induce clarify`), same
400 cells:

| Reasoning first | Gain EV agreement | Mixed EV agreement | when gamble optimal | Frame gap |
|---|---|---|---|---|
| uninduced | 0.938 | 0.787 | 0.622 | -0.135 |
| clarify only | 1.000 | 0.881 | 0.789 | -0.105 |
| induced (EV rule) | 1.000 | 0.869 | 0.767 | -0.145 |
| induced + clarify | 1.000 | 1.000 | 1.000 | 0.000 |

On its own the clarification cuts the mixed-frame misses where the gamble is
better from 34 to 19 and barely moves the frame gap. The remaining 19 are the
same bookkeeping error (e.g. "EV = 60.48 - 52 = 8.48 ... EV_B = 52"), and none of
the 19 traces mentions risk, safety or certainty, so they are not risk-averse
choices. The error disappears only when the instruction states both the
decision rule and the comparison. Reading: the loss-frame effect under
reasoning is an arithmetic habit, not a preference, and the model heeds a
correction to that habit reliably only when it is also told what to compute.
Every trace in the clarify-only run frames the choice in expected-value terms
(100%).

## Trace coding (Oct 1 2026): the stated comparison bends toward the safe answer

Open item from the Qwen section: what are the non-mixed misses in the uninduced
reasoning run (risk and neutral frames at about 0.79)? `analysis/trace_comparisons.py`
extracts every explicit numeric comparison in a trace and checks it against the
numbers; an EV-vs-sure comparison is one whose two numbers are within 2% of p x hi
and the sure amount.

The 72 Qwen misses outside the mixed frame, read by hand and by regex: 65 compute the
gamble's expected value correctly and then misstate the comparison ("The expected
value of Option A is $152.04, which is less than the guaranteed $117"); 6
acknowledge the higher expected value and choose the sure amount for its certainty;
1 is unclassified. So about 90% of these misses are false comparisons, not risk
aversion.

The false comparisons run one way. Numeric EV-vs-sure statements, reversed / total,
by which option the true comparison favors:

| Run | truth favors gamble | truth favors safe | Fisher p |
|---|---|---|---|
| Qwen, reason, uninduced | **40 / 62** | **0 / 153** | 7e-28 |
| Qwen, reason, EV rule | 0 / 23 | 0 / 50 | 1 |
| Qwen, reason, clarify only | 2 / 125 | 0 / 133 | 0.23 |
| Qwen, reason, EV rule + clarify | 0 / 73 | 0 / 62 | 1 |
| OLMo SFT, reason, uninduced | 2 / 19 | 5 / 10 | 0.03 |
| OLMo SFT, reason, EV rule | 4 / 13 | 0 / 8 | 0.13 |
| OLMo Instruct, reason, uninduced | 24 / 90 | 5 / 86 | 0.0002 |
| OLMo Instruct, reason, EV rule | 8 / 31 | 1 / 38 | 0.009 |

Random slips would reverse both directions at similar rates. Qwen never misstates a
comparison that favors the safe option and misstates two in three of those that
favor the gamble. In the wider regex set (any worded comparison against the sure
amount, cells where the gamble is better, n = 204), the false "less than" rate is
0.145 in the gain wording, 0.370 neutral and 0.452 in the risk wording. A clustered
logit (40 gamble clusters) finds the false statement predicted by the model's
immediate-answer P(gamble) on the same cell (coefficient -1.22, p = 0.006) and by
the wording (neutral +1.58, risk +1.71, both p < 0.001), and not by how far apart
the numbers are (log EV ratio, p = 0.61).

Reading (revised after the prefill test below). The arithmetic is right and the
verdict on it is not, and the errors favor the safe option only. The prefill test
locates the mechanism in a stock phrase: once a trace writes "$E, which is ___ than
the guaranteed $S", the word is "less" almost regardless of the numbers. Wording and
the immediate-answer lean predict whether the trace uses that phrase, not the verdict
given the phrase. So the safe lean enters through the choice of sentence template, a
weaker and more mechanical claim than "the model rationalizes its preferred answer".
Telling the model what to compute (the EV rule) removes the false statements (0 in
73), and the clarification line alone nearly does (2 in 258).

OLMo Instruct shows the same asymmetry more weakly and also makes two-way
comparison errors (14 of 51 other comparisons false, e.g. "$52 is greater than
$60.48"). OLMo SFT has few parsable statements and, if anything, errs the other
way (5 of 10 reversed toward the gamble); the sample is too small to say more.

Caveats: regex extraction covers only comparisons written with both numbers (about a
quarter of Qwen traces); one coder (automated, spot-checked by hand on 15 OLMo and
72 Qwen statements); Qwen is the only model with a large sample. The DPO check and
the prefill test are reported below.

## OLMo-2 DPO, fewshot, reasoning (Oct 1 2026): the change happens at DPO

`data/phase3_olmo-dpo_fewshot_reason_riskneutral.csv`,
`data/phase3_olmo-dpo_fewshot_reason.csv`. Same 840 cells, run on a paid Colab T4
(about 30 min per run). Excluded: 1 (induced), 3 (uninduced). End states:
induced 810 answer / 29 eos / 1 blank; uninduced 326 answer / 488 eos / 26 blank.
The eos traces are complete (median 373 characters, 6 over 1,100), ending in a
stated choice, as for Instruct. 91% of uninduced DPO traces mention expected value
(SFT 74%, Instruct 89%).

`analysis/score_reason_staircase.py` (dominance = hard accuracy on the 40 dominant
cells; EV = hard EV agreement on the 620 clear core cells; gap = mean P(gamble),
mixed minus gain):

| Condition | SFT EV | SFT gap | DPO EV | DPO gap | Instruct EV | Instruct gap |
|---|---|---|---|---|---|---|
| immediate, uninduced | 0.519 | +0.241 | 0.502 | +0.319 | 0.497 | +0.346 |
| immediate, induced | 0.545 | -0.008 | 0.549 | +0.014 | 0.542 | +0.003 |
| reason, uninduced | 0.681 | +0.375 | 0.741 | **+0.015** | 0.740 | +0.055 |
| reason, induced | 0.713 | +0.358 | **0.832** | -0.071 | 0.803 | -0.024 |

Stage contrasts under reasoning, 95% cluster bootstrap (40 gamble clusters, 2,000
draws):

| Contrast | DPO minus SFT | Instruct minus DPO |
|---|---|---|
| frame gap, uninduced | **-0.360** [-0.429, -0.287] | +0.040 [-0.076, +0.156] |
| frame gap, induced | **-0.429** [-0.547, -0.300] | +0.047 [-0.034, +0.127] |
| EV, induced | **+0.119** [+0.057, +0.182] | -0.029 [-0.054, -0.005] |
| EV, uninduced | +0.060 [-0.031, +0.148] | -0.001 [-0.045, +0.045] |

The SFT-to-Instruct change under reasoning, flagged in the SFT section as not yet
attributable, is entirely a DPO effect. Preference optimization removes the
reasoning-regime framing gap (+0.375 to +0.015) and raises EV compliance when the
rule is stated (0.713 to 0.832); the final RLVR stage moves nothing detectable and,
if anything, slightly lowers induced EV agreement. Under immediate answer, as
before, no stage after SFT moves anything.

So the staircase result is now regime-specific and stage-specific: the structural
parameters of the immediate answer are fixed at SFT; how the model reasons about the
same choices is set at DPO. The comparison-bias check (`trace_comparisons.py`) puts
DPO between SFT and Instruct (reversed toward safe 23/92, toward gamble 11/95,
p = 0.02), and DPO traces show the mixed-frame reference slip seen in Qwen
("the expected value of choosing Term B is $71.71 - $94 = -$22.29. Since -$22.29 is
less than $0, choosing Term A is the better option").

Caveats: one family, one seed of each checkpoint, fewshot reasoning exemplars that
demonstrate the EV arithmetic. The DPO and Instruct runs share the same 20 gambles
per instrument as SFT, so the contrasts are paired.

## OLMo clarification check (Oct 1 2026): the line does nothing for OLMo

`data/phase3_olmo-inst_fewshot_reason_clarify.csv`,
`data/phase3_olmo-inst_fewshot_reason_riskneutral_clarify.csv`: OLMo Instruct,
fewshot, reasoning first, gain and mixed frames of the same 20 gambles (400 cells
each), 0 excluded. `analysis/score_clarify.py olmo-inst_fewshot`:

| Reasoning first | Gain EV | Mixed EV | mixed misses | when gamble optimal | Frame gap |
|---|---|---|---|---|---|
| uninduced | 0.831 | 0.619 | 61 | 0.544 | +0.055 |
| clarify only | 0.781 | 0.588 | 66 | 0.378 | -0.044 |
| EV rule | 0.875 | 0.588 | 66 | 0.522 | -0.024 |
| EV rule + clarify | 0.887 | 0.581 | 67 | 0.489 | -0.055 |

Qwen for comparison: mixed EV 0.869 with the rule, 1.000 with rule + clarify.

The line that removed Qwen's loss-frame error leaves OLMo's untouched. The traces
show why: OLMo keeps computing the gamble as a change from current holdings after
being told to compare final amounts ("you have a 58% chance to gain $167 for a total
of $278 ... The expected value of Arrangement B is (0.58 * $167) + (0.42 * $0) =
$96.06. Since $96.06 is less than $111"). It even states the final total and then
multiplies the gain. Share of clear mixed-frame traces containing the
change-from-holdings number (p x gain, or the net change, within 2%):

| | uninduced | clarify | EV rule | EV rule + clarify |
|---|---|---|---|---|
| Qwen | 0.469 | | 0.350 | 0.244 (0 errors) |
| OLMo Instruct | 0.481 | 0.462 | 0.575 | 0.588 |

About 60% of OLMo's mixed-frame misses contain that slip; the rest are comparison
errors in both directions ("$59.38 > $64", "$161.64 is less than $111"). So the
"bookkeeping habit one sentence removes" is a Qwen result. For OLMo the same habit
is present and the sentence does not reach it, consistent with OLMo's weaker
instruction-following throughout (immediate-answer induced compliance 0.54, the
dominance and format results of Phase 3).

## Prefill test (Oct 1 2026): the false verdict comes from a stock phrase

`colab/prefill_comparison.py` on Qwen Instruct's uninduced reasoning traces: for the
492 of 800 traces with a comparison against the sure amount, cut the trace just
before the comparison word and read P(less group) / P(less + greater group).
Conditions: own (the model's own trace), rule (same trace, EV rule prepended to the
prompt), minimal (prompt, then "The expected value of Option X is $E, which is"
with the correct E). `data/prefill_qwen-inst_chat.csv`;
`analysis/prefill_analysis.py`. 318 non-mixed clear cells; the own-trace argmax
reproduces the generated word in 318 of 318.

Mean P(less):

| Truth favors | own | rule | minimal |
|---|---|---|---|
| gamble (n = 213) | 0.298 | 0.223 | **0.815** |
| safe (n = 105) | 0.723 | 0.698 | 0.964 |

Given only "The expected value of Option A is $152.04, which is", Qwen continues
with "less" 82% of the time even when $152.04 is above the sure amount (89% of cells
past 0.5; gain wording 0.69, neutral 0.93, risk 0.82). The verdict is close to
insensitive to the numbers at that position.

Generated word by construction (own traces):

| Truth | reached via "..., which is ___ than" | generated "less" | generated "greater" |
|---|---|---|---|
| gamble | yes | **63** | 1 |
| gamble | no | 2 | 147 |
| safe | yes | 53 | 0 |
| safe | no | 23 | 29 |

So 63 of the 65 false "less" statements are the phrase "$E, which is less than the
guaranteed $S"; reached any other way ("Since the expected value of Payout B
($162.84) is greater than ..."), the verdict is wrong 2 times in 149. What predicts
the error is whether the trace uses that construction (risk wording 55% of traces,
neutral 37%, gain 20%; clustered logit on cells where the gamble is better: risk
+1.68, neutral +1.27, immediate-answer P(gamble) -1.28 with p = 0.004, log EV ratio
p = 0.55).

On the false traces, prepending the EV rule with the identical trace text lowers
P(less) from 0.976 to 0.727: the instruction weakens the stock continuation at the
verdict but does not overturn it once the phrase is written. The rule's full effect
in the generated runs (0 false statements) comes mostly from steering the trace away
from the phrase.

Reading: the residual EV misses of uninduced Qwen reasoning outside the loss frame
are mostly a phrase-level default, "which is less than the guaranteed ...", that
the model reaches for more often when the wording mentions risk and when its snap
answer leans safe. The written trace states the right arithmetic and a verdict
supplied by the phrase. For trace-based auditing the lesson stands, with a narrower
mechanism than motivated reasoning.

Follow-up, same session: a fourth condition states the same correct number without
the construction, "Since the expected value of Option X, $E, is" (`--conditions
since`, `data/prefill_qwen-inst_chat_since.csv`):

| Truth favors | which-is prefix: mean P(less) | share > 0.5 | since prefix: mean P(less) | share > 0.5 |
|---|---|---|---|---|
| gamble | 0.815 | 0.892 | **0.168** | **0.113** |
| safe | 0.964 | 0.981 | 0.600 | 0.600 |

Same number, same prompt, different sentence frame: after "which is" the model says
"less" 89% of the time when the gamble is better; after "Since ... is" it says so 11%
of the time. The "since" frame discriminates (0.60 vs 0.17) where "which is" barely
does (0.96 vs 0.82), with some bias the other way (it calls the safe option's EV
"greater" 40% of the time when it is smaller). The false verdicts belong to the
phrase.

Caveat: one model, one trace per cell (greedy); the phrase result is specific to Qwen's
habitual wording and may not transfer to other families.
