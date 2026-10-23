# Claude runs on Stage A, night of Oct 19-20 2026

All runs used the native Anthropic API. The grids are data/stageA_grid*.csv (main) and
data/stageA2_grid*.csv (second wording plus the "release" domain). Arms:
**snap** = "Answer with only A or B", thinking off; **reasoning** = written reasoning, then
"Answer: X", thinking off; **thinking** = extended thinking at effort low, then a one-letter
answer. Opus cannot turn thinking off, so its two runs are thinking + one letter and
thinking + written reasoning.

## Spend (from the token counts recorded in each CSV; `python scripts/spend.py`)

```
file  cells  excluded  out_per_cell  cost
   data/pilot/stageA_opus55_think.csv    240     0.000            78  0.58
data/pilot/stageA_sonnet55_reason.csv    237     0.000           379  1.02
  data/pilot/stageA_sonnet55_snap.csv    240     0.000            23  0.15
 data/pilot/stageA_sonnet55_think.csv    240     0.000           101  0.35
           data/stageA_haiku55_a2.csv   1800     0.002            20  0.13
    data/stageA_haiku55_a2_reason.csv   1000     0.000           844  0.47
     data/stageA_haiku55_a2_think.csv   1000     0.000           326  0.20
          data/stageA_haiku55_api.csv   1440     0.000            19  0.10
   data/stageA_haiku55_api_reason.csv    800     0.000           733  0.33
    data/stageA_haiku55_api_think.csv    800     0.000           261  0.13
    data/stageA_opus55_api_reason.csv    800     0.000           911 15.99
     data/stageA_opus55_api_think.csv    800     0.000           166  3.82
          data/stageA_sonnet55_a2.csv   1800     0.011            62  3.43
   data/stageA_sonnet55_a2_reason.csv    842     0.000           789  7.41
    data/stageA_sonnet55_a2_think.csv   1000     0.000           249  3.26
         data/stageA_sonnet55_api.csv   1440     0.024            63  2.66
  data/stageA_sonnet55_api_reason.csv    800     0.000           749  6.69
   data/stageA_sonnet55_api_think.csv    800     0.000           216  2.31
dropped rows $0.28
total $49.31
```

Total $49.31, under the $50 John set. Opus finished inside the $50, so the $75 allowance
was not used. Left for November: the last 16% of the Sonnet A2 reasoning run (~$1.40).

Thinking length at effort low (output tokens per cell, n=3 samples, includes the visible
answer): Haiku ~260, Sonnet ~216 (about 70 per call), Opus ~166 (about 55 per call).
At this effort the models barely think.

## Main grid: money/self against each lives domain

Mean P(risky) shift from money/self, with 95% gamble-cluster intervals, and the cell-level
correlation r with money/self. Source: `python analysis/stage_a.py --stems
sonnet55_api,sonnet55_api_think,haiku55_api,haiku55_api_think,opus55_api_think,opus55_api_reason --B 300`.

| Model, arm | P(risky) money/self | Shift, lives domains (range) | r with money/self (range) |
|---|---|---|---|
| Sonnet snap | 0.99 | -0.13 to -0.67 | 0.11 to 0.33 |
| Sonnet reasoning | 0.48 | -0.21 to -0.45 | 0.23 to 0.67 |
| Sonnet thinking | 0.56 | -0.24 to -0.48 | 0.28 to 0.67 |
| Haiku snap | 0.50 | -0.44 to +0.13 | -0.63 to 0.25 |
| Haiku reasoning | 0.51 | -0.18 to -0.47 | 0.33 to 0.65 |
| Haiku thinking | 0.55 | -0.05 to +0.05 | 0.87 to 0.97 |
| Opus thinking, one letter | 0.46 | -0.07 to +0.07 | 0.88 to 0.94 |
| Opus thinking, reasoning | 0.44 | -0.15 to +0.05 | 0.79 to 0.98 |
| Fable thinking, one letter | 0.48 | -0.21 to -0.40 | 0.34 to 0.69 |

Readings:
- **Snap answers repeat the open-model result.** Sonnet takes the money gamble almost every
  time (0.99) yet becomes much more cautious with lives, most of all with detainees (-0.67),
  and money answers barely predict lives answers (r 0.11-0.33). Haiku snap is erratic: money
  answers follow the frame (gap -0.77), and troops answers run opposite to money (r -0.63).
- **Written reasoning raises r to about 0.6 but keeps the caution shift** for both Sonnet
  and Haiku. Sonnet's brief extended thinking looks the same as its written reasoning.
- **Thinking in Haiku and Opus erases the domain shift.** Choices track expected value in
  every domain (EV agreement 0.80-1.00, lowest for Opus on detainees), so money predicts lives well (r ~0.9). The one
  exception is detainees for Opus with reasoning (-0.13 to -0.15).
- So the claim "a money lottery can't screen advisory risk" holds for snap answers and for
  Sonnet in every arm, but not for Haiku with thinking or Opus. That is a finding to write
  up, not a failure: with extended thinking on, Haiku and Opus choose by expected value in every domain, while Sonnet does not. (Opus cannot run with thinking off, so its model and its thinking are confounded.)

## Fable 5.1 (added Oct 20 afternoon)

John approved the thinking + one-letter arm only (~$10-12). Cost $8.32 (probe $0.05, test
slice $1.25, main grid $7.02; about 34 output tokens per call). Fable keeps the caution shift
with lives (P(risky) down 0.21 to 0.40, all intervals clear of zero) and money only partly
predicts lives (r 0.34-0.69). It looks like Sonnet, not Opus: so extended thinking erases
the gap in Haiku and Opus, but not in Sonnet or Fable. Spend since Oct 19 is now $57.58.

## High effort (Oct 20 evening, $4.59)

Same arm (thinking + one letter), effort high, max_tokens 16000. Thinking barely lengthens:
Sonnet ~90 output tokens per call (vs ~70 at low), Fable ~56 (vs ~31).

- **Sonnet, full grid ($2.74):** the caution shift halves for civilians and troops (-0.10 to
  -0.12, was -0.24 to -0.28) and r rises to 0.80-0.85 (was 0.56-0.67); EV agreement 0.94-0.97.
  Detainees barely move: -0.39 (was -0.46 to -0.48), r 0.41-0.44.
- **Fable, test slice ($1.85; money and civilians, 10 gambles, noisy):** civilians/advise
  moves toward money (P(risky) 0.40 vs 0.44 for money/advise; EV agreement 0.97), civilians/self
  does not (0.28).

Reading: more effort moves Sonnet partway toward Opus on civilians and troops, though it
thinks only a little longer, while caution with detainees holds. Effort level, not just the
model, matters. Spend since Oct 19: $62.17.

## Oct 23 batch ($35.47): effort ladder, Fable high, second wording

Spend: Sonnet A2 reasoning finished $1.64; Fable high, main grid $12.34 (it thought ~80 tokens per call, twice the test slice, so it hit
its cap at 774 of 800 cells; the last 26 ran Oct 23 for $0.24 and changed no figure);
Sonnet medium $2.34; Haiku high $0.19; Opus high $5.10; Opus A2 low $5.14; Fable A2 low $8.95.
October total $97.92 (incl. a $0.05 probe) of the $100 credits.

Shift in P(risky) from money/self (95% gamble-cluster intervals in the stage_a.py output):

| Model | Effort | Civilians + troops | Detainees | r civ/troops | r detainees |
|---|---|---|---|---|---|
| Haiku | low | -0.05 to +0.05 | -0.05 to +0.03 | 0.87-0.97 | 0.89-0.96 |
| Haiku | high | -0.03 to +0.00 | -0.04 to -0.01 | 0.92-0.98 | 0.93-0.99 |
| Sonnet | low | -0.24 to -0.28 | -0.46 to -0.48 | 0.56-0.67 | 0.28 |
| Sonnet | medium | -0.21 to -0.27 | -0.47 | 0.61-0.69 | 0.27-0.29 |
| Sonnet | high | -0.10 to -0.12 | -0.39 to -0.40 | 0.80-0.85 | 0.41-0.44 |
| Fable | low | -0.21 to -0.25 | -0.36 to -0.40 | 0.63-0.69 | 0.34-0.43 |
| Fable | high | -0.03 to -0.16 | -0.27 to -0.35 | 0.74-0.80 | 0.44-0.54 |
| Opus | low | +0.01 to +0.07 | -0.05 to -0.07 | 0.88-0.93 | 0.91-0.94 |
| Opus | high | +0.03 to +0.08 | -0.05 to -0.09 | 0.87-0.93 | 0.86-0.91 |

Second wording plus "release" (the detainee scenario without the escalation clause), thinking
at low effort:
- **Opus holds:** lives track money (r 0.81-0.97); detainees -0.06 to -0.15, release -0.03 to -0.05.
- **Fable:** civilians and troops -0.08 to -0.14 (smaller than the first wording),
  detainees -0.32 to -0.35, release -0.26 to -0.32.
- **Sonnet (reasoning / thinking):** release -0.39 to -0.44, the same as detainees (-0.45 to -0.47).
- **Haiku thinking:** release -0.02 to -0.06, detainees -0.08 to -0.11.

Sonnet's written reasoning on detainees (main grid) mentions escalation in 100% of answers,
"irreversible" in 60%, and the remaining detainees or later talks in ~75%; for civilians and troops
escalation never comes up. But removing escalation ("release") does not reduce Sonnet's or Fable's
caution, so escalation is what they cite, not what drives the choice. The caution is about trading
a sure release for a gamble on freeing more people.

Reading: more thinking effort pulls Sonnet and Fable toward money-like choices for civilians and
troops (step change between medium and high for Sonnet), while their caution about detained
people survives high effort, a second wording and the removal of escalation. Haiku with thinking
and Opus treat every domain like money at every effort tried.

## Not yet analyzed

Snap and reasoning arms on the second wording are analyzed in the stage_a.py output but not
summarized here yet (Sonnet snap: lives shift -0.28 to -0.58, r 0.00-0.12; Haiku snap is
frame-driven again).
