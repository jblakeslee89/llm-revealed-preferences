# Robustness Results: Induced-Value Responsiveness (Oct 3 2026)

Plan: `robustness-plan-2026-10.md`. Runs: `colab/robustness_runs.ipynb`, arms 1 to 4, all 36
jobs, overnight Oct 2 to 3 on a Colab A100 (about 8 hours), every job exit 0. Data: the 36
new `data/phase3_*` files listed below. Scores: `results/robustness_runs/scores.txt`
(`analysis/score_robustness.py`, 300 cluster-bootstrap draws; "ev" is agreement with the
risk-neutral optimum on clear core cells).

Run notes. The notebook as first committed failed on every Llama job: `login()` in the kernel
is not visible to the `phase3_elicit.py` subprocesses (401 gated-repo error). Fixed in 6bd1001
by exporting `HF_TOKEN`. The overnight session wrote to the Colab VM instead of Drive (the Drive
permission prompt was declined while John was away); results were downloaded at the end.

## Verdict on the claim

The claim: told a utility rule, models fail in one-letter answers but comply when they reason.
It survives every check.

| | Immediate answer, told the rule | Reason first, told the rule |
|---|---|---|
| Qwen, original wording (Phase 3) | 0.55 | 0.965 |
| Qwen, three paraphrases + rule after question | 0.40 to 0.61 | 0.90 to 0.97 |
| OLMo Instruct, original | 0.53 | 0.80 |
| OLMo Instruct, paraphrases + after | 0.49 to 0.67 | 0.71 to 0.94 |
| Llama chat, original | refuses to answer in one letter (97% excluded) | 0.95 |
| Llama chat, paraphrases + after | 0.54 to 0.62 where it answers | 0.86 to 0.98 |
| Llama few-shot, original | 0.46 | |

No wording brings immediate-answer compliance near the reasoning level, and no wording breaks
compliance under reasoning.

## Arm by arm

**1. Llama immediate, induced (the missing cell).** Told the risk-neutral rule in chat, Llama
almost never gives a one-letter answer: 97% of cells fall below the A/B mass threshold, and
100% under the sqrt rule. It starts explaining instead. Without the instruction it answers
normally (0% excluded). Two paraphrases (terse, goal) do get letters, at 0.54 and 0.62. In
few-shot format it answers, at 0.46. Llama's immediate-answer response to the instruction is
refusal of the format, or coin-flip compliance where it complies with the format.

**2a. Phrasings, immediate.** Paraphrases lower or leave unchanged Qwen's immediate compliance
(terse -0.09, formula -0.16, goal -0.10). Moving the rule after the question helps a little
(Qwen +0.06 [-0.00, +0.11]; OLMo +0.14 [-0.02, +0.26], with 4.4% of OLMo cells excluded). The
immediate-answer failure is not an artifact of the original wording or of the rule's distance
from the answer position.

**2b. Phrasings, reasoning.** Compliance stays high under every wording. The goal wording
("repeated many times, keep the total") is the weakest for OLMo (-0.09 [-0.15, -0.05]) and Llama
(-0.09 [-0.14, -0.04]), and the formula wording for Qwen (-0.07). The rule placed after the question
is the strongest for OLMo (+0.14 [+0.09, +0.19], to 0.94) and Llama (+0.03, to 0.98). The
reasoning-regime frame gap is wording-sensitive: Qwen's stays negative under every wording
(-0.10 to -0.44), so the reversal reported in Phase 3 holds, while Llama's ranges from -0.20
(formula) to +0.21 (goal).

**3. Full gamble set.** The 40 gambles not used in the original reasoning runs reproduce the
first 40: told the rule, Qwen 0.979 (first half 0.965), OLMo 0.818 (0.803), Llama 0.969. All 80
gambles: Qwen 0.972, OLMo 0.810, Llama 0.958. Uninduced on the new gambles: Qwen 0.719 (first
half 0.829), OLMo 0.721 (0.740), Llama 0.873. Qwen's uninduced figure is lower on the new
gambles; the induced figures are stable.

**4. Sampled traces (k = 5, T = 0.7).** For Qwen and Llama the greedy trace is representative:
it matches the majority of five sampled traces in 97% and 95% of cells, and a randomly drawn
trace complies about as often as the greedy one (Qwen 0.969, Llama 0.938; majority vote 0.977
and 0.982). OLMo's traces vary a lot: only 41% of cells are unanimous, the greedy trace matches
the majority in 79%, and a random trace complies less often (0.736) than the greedy one (0.80).
OLMo's reasoning-regime numbers should be reported as a range, not a point.

## What changes in the writeup

- The "one instruction phrasing" limitation can be replaced by the table above.
- The reasoning arm can be reported on all 80 gambles.
- Greedy decoding is defensible for Qwen and Llama; for OLMo, report sampled-trace compliance
  alongside greedy.
- Llama fills the missing immediate-answer cell, with a different failure mode (format refusal).
- The reasoning-regime frame gap needs a caveat that its size depends on wording.

## Check against the pre-set decision rules (added Oct 3, thread review)

The plan fixed five rules before the runs. Three pass cleanly. Two need a caveat in the writeup.

1. **Phrasings: immediate compliance stays below about 0.65 in every family and wording.**
   Borderline for one cell. OLMo with the rule after the question reaches 0.673 [0.550, 0.801],
   and its dominance accuracy falls to 0.713 (1.000 with the rule before). The lift is noisy and
   comes with lost dominance, so it reads as a weaker engagement with the format, not as
   compliance. It stays far below the same wording under reasoning (0.94). Every
   reason-minus-immediate lift is positive.
2. **Full set: the new 40 gambles fall inside the first 40's intervals.** Passes for every
   induced run. Fails for Qwen uninduced: 0.719 [0.653, 0.786] on the new gambles is below the
   first half's interval (0.829 [0.755, 0.890]). The induced claim is unaffected. Any
   uninduced-reasoning number for Qwen should be the 80-gamble figure, not the 40-gamble one.
3. **Sampled traces: greedy matches the sampled majority in at least about 90% of cells.**
   Passes for Qwen (97%) and Llama (95%). Fails for OLMo (79%), so OLMo is reported as a
   range (random trace 0.736, greedy 0.80, majority 0.80).
   The plan's follow-up check on verdict habits, run on the induced sampled traces:
   the directional bias in stated EV-versus-sure comparisons survives sampling. OLMo reverses
   comparisons when the gamble is better 32% of the time versus 8% when the safe option is
   better (64/198 vs 16/211, Fisher p = 2e-10; greedy 8/31 vs 1/38). Llama leans the other
   way, 0.9% versus 3.8% (11/1,265 vs 35/928, p = 3e-6; greedy 7/377 vs 18/282). Qwen's
   induced traces make almost no false comparisons, sampled (1/509) or greedy (0/73). Qwen's
   stock phrase lives in the uninduced runs, which were not sampled. So "verdict habits are not
   a greedy-decoding artifact" is shown for OLMo and Llama, but not for Qwen's stock phrase.
4. **Llama immediate induced near 0.55.** Not the predicted pattern. In chat, Llama answers in
   one letter in only 3% of cells (52 of 1,680 kept, where every clear cell is correct) and
   explains instead. The "fails in one letter" half of the claim holds for Llama in the
   few-shot format (0.46) and under two paraphrases (0.54, 0.62). In chat its failure is
   refusing the format, which needs its own sentence.
5. **Paid with reasoning.** Not run.

