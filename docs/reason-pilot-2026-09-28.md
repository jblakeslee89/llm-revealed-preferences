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
