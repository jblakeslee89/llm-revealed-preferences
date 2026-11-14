# Paper Plan: Do an AI's Money Bets Predict Its Advice When Lives Are at Stake? (Oct 14 2026)

The TNL residency deliverable (due mid-December 2026) and the one publication this project
now aims at. It replaces the dissertation spine and the Epoch benchmark framing.

Working title: **"Safe With Money, Cautious With Lives: Risk Attitudes Measured on Lotteries
Do Not Predict How Language Models Advise on Life-and-Death Decisions."**

Target form: a working paper of about 15 pages, posted to arXiv in mid-December, then sent to a
workshop or policy venue (see Venues).

## The claim, in one paragraph

Evaluators who want to know how a model handles risk usually give it lotteries over money.
We ask whether that measurement predicts what the model recommends when the same numbers are
lives: civilians evacuated, soldiers brought out, detained aid workers freed. It does not. On
open models (Stage A, done) the money answers predict, at best, which questions draw more risk,
and never the level: a model that reasons like an expected-value maximizer over dollars takes
the better gamble only 17-32% of the time over lives, and says in its reasoning that it prefers
certainty. Snap recommendations can be uninformative altogether (Llama's follow option order).
The paper's new contribution is showing whether this holds for the frontier models people
actually use as advisers, and explaining why with the project's earlier results: wording effects
that track the stated gain, compliance with a stated rule only under reasoning, and leftover
biases that differ by model.

## Sections and where each comes from

| Section | Pages | Source | State |
|---|---|---|---|
| 1. Introduction: models are used as advisers; risk evals use lotteries | 1.5 | `docs/extension-advice-2026-10.md` | drafted (`docs/paper-risk-advice-draft.md`) |
| 2. Related work: LLM risk preferences, framing, wargame/escalation studies | 1.5 | `docs/lit-review-2026-07.md`, `writeups/references.bib` | outline; Rivera et al. 2024 and Lamparth et al. 2024 still to check before citing |
| 3. Design: the 40 gambles in four domains, two roles, two frames, both orders, engagement controls | 2 | `src/stage_a_grid.py`, `docs/stage-a-results-2026-10.md` | drafted |
| 4. Results on open models (Qwen, Llama, OLMo; snap and reasoned) | 3 | `docs/stage-a-results-2026-10.md` | drafted from existing tables |
| 5. Results on frontier models | 3 | new runs, `src/api_elicit.py` | **needs API keys and spend approval** |
| 6. Why: framing tracks the stated gain; rules need reasoning; leftover biases don't converge | 2 | Paper A (`notes/phase3-robustness-2026-10-03.md`), Paper B (`docs/robustness-results-2026-10-03.md` on PR #1), `notes/convergence-question-2026-10-05.md` | outline |
| 7. Implications for evaluating advisory AI; limitations | 1.5 | | outline |

## The one piece of new work: two Claude models on John's monthly API credits

Revised Oct 18 2026. The first version priced five flagship models through OpenRouter at
placeholder flagship rates and assumed every call thinks for about 1,000 hidden tokens; that
is where "$150-500" came from. John has $100 a month in Anthropic API credits, which cover
Claude models through the native API only, so the frontier step is now Claude only and fits
well inside those credits.

**Models and arms.** Two sizes of one frontier family, three ways of answering:

| Arm | What the model does | Counterpart in the open-model runs |
|---|---|---|
| Snap | Thinking off, one-letter answer | Immediate answer |
| Written reasoning | Thinking off, reasons in the reply, then "Answer: X" | Reason first |
| Extended thinking | Hidden thinking at low effort, then a one-letter answer | None: the reasoning-model case |

- **Claude Haiku 5.5** (`claude-haiku-5-5`), $0.10 / $0.50 per million tokens in / out, all
  three arms. Links to the Phase 1-2 Haiku runs.
- **Claude Sonnet 5.5** (`claude-sonnet-5-5`), $2 / $10, all three arms; this is the one
  reasoning model. Thinking is capped through the effort setting (`low`); Sonnet 5.5 no longer
  accepts a fixed thinking-token budget.
- Claude Opus 5.5 is left out: it cannot turn thinking off, so it has no snap arm.

Prices are Anthropic's first-party rates as of Oct 18 2026; hidden thinking bills as output.
`src/api_elicit.py` reads these prices, prints an estimate before any call, and stops calling
at `--max-spend`.

**Cost** (script estimates; extended thinking assumes 800 hidden tokens per call, with 2,000 in
brackets):

| Run | Calls per model | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|---|
| Pilot: money + civilians, 10 gambles, all arms, n = 2 | about 1,400 | under $0.50 | about $6 |
| Snap, full grid (1,440 cells), n = 5 | 7,200 | $0.08 | $1.64 |
| Written reasoning, 800 cells, n = 3 | 2,400 | $0.44 | $8.87 |
| Extended thinking, 800 cells, n = 3 | 2,400 | $0.99 | $19.74 [$48.54] |
| **Both models, all arms, plus pilot** | | | **about $38 [about $67]** |

**Split across the monthly credits:**

- **October, about $20 of $100:** the pilot (which measures real thinking-token counts before
  the expensive arm), Haiku in all arms, and Sonnet snap and written reasoning.
- **November, about $25-70 of $100:** Sonnet extended thinking, plus a second wording per
  domain and a detainee domain without escalation language in every arm (about half again on
  top). These answer the two main Stage A caveats.

The Batch API would halve all of this; the script doesn't use it, and at these amounts it
isn't needed.

**GPT and Gemini.** Their cheapest small models cost about $1-2 each for the full grid
(GPT-5 Nano $0.05 / $0.40, Gemini 2.5 Flash-Lite $0.10 / $0.40 per million tokens, from
third-party price lists in Oct 2026), paid out of pocket. Recommendation: drop them for the
TNL deliverable; add both, about $3, before a workshop submission if "is this only Claude?"
looks like a likely reviewer question. `--provider openai` already supports them.

**How the frontier claim changes.** Section 5 says "in one frontier family, at two sizes, with
and without extended thinking", not "across frontier models". The extended-thinking arm is new
and worth having: it tests whether a model that thinks before answering carries its money-domain
arithmetic over to lives.

**The $0 fallback.** Open models only, with the data in hand, and the frontier step listed as
future work. The claim narrows to three 7-8B open models; sections 4 and 6 and the bounds below
are unchanged. That is a complete class paper but a weaker publication, because the first
reviewer question would be whether it holds for current models.

## Framing: a finding, not a null

The result has two parts, and neither is a null.

1. **A large, precisely estimated shift.** Qwen takes 0.35-0.64 less risk with lives than with
   money in snap answers and 0.27-0.37 less under reasoning, with every 95% interval far from
   zero; under reasoning its agreement with expected value falls by 0.34-0.43. Llama's reasoned
   answers shift by 0.14-0.22. This is the headline: safe with money, cautious with lives.
   (`analysis/stage_a_bounds.py`, `results/stage_a_bounds.txt`.)
2. **An upper bound on what the money answers can predict, which varies by model.** The square
   of the upper 95% limit on the cell-level correlation bounds the share of question-to-question
   variation in lives answers that money answers can explain. Under reasoning the bound is 0-24%
   for Qwen and 3-15% for OLMo, but 35-46% for Llama; in snap answers 0-48% (Qwen), 0-53%
   (Llama) and 36-76% (OLMo). So: "a money lottery gets the level of risk wrong for Qwen in every
   regime and for Llama when it reasons, and how well it ranks questions ranges from not at all to well depending on model and
   regime, which an evaluator could only learn by running the advisory questions anyway."

Lead with (1), present (2) as the bound that makes a lottery unusable as a screen for advisory
use, and never write "no effect". The bounds table goes in section 4.

## Milestones

Dates are 2026 unless marked. The RAND term ends Dec 4; Thanksgiving is Nov 26.

| Dates | Milestone | Needs from John |
|---|---|---|
| Oct 14-18 | Plan, API harness, draft of sections 1, 3 and 4, bounds analysis (done in this thread) | TNL output format |
| Oct 19-24 | Pilot (about $6), then October runs: Haiku all arms, Sonnet snap and written reasoning (about $20 total). Second-wording grids built. Related work written and the two wargame citations checked | Anthropic API key in your environment and an explicit go-ahead |
| Oct 27 | FAccT 2027 abstract deadline (paper Nov 3). Stretch only; recommendation: skip | Go / no-go |
| Nov 1-6 | November runs on the new month's credits: Sonnet extended thinking and the second wordings for every arm. One Colab session for the open models on the new wordings | Go-ahead for the November batch; run one Colab notebook |
| Nov 9-20 | Analysis and figures; section 5 (frontier) and section 6 (why) written | |
| Nov 23-Dec 4 | Full draft, about 15 pages; to Phil and the TNL instructor by Dec 4 | Read and comment |
| Dec 7-11 | Revise; TNL deliverable by Dec 11 (adjust to the real due date); post to arXiv | arXiv account; endorsement for cs.CY or cs.AI if this is a first submission (check in November) |
| Dec-Jan | Policy version: RAND working paper and/or a 1,500-word piece for War on the Rocks or Lawfare | |

## Venues

Checked Oct 14 2026:

- **FAccT 2027**: abstract Oct 27, paper Nov 3, 14 pages; first-round decisions Dec 22,
  revision due Jan 28 2027; conference June 21-24 2027, Porto. Non-archival option exists.
  Strong fit for "evaluations must match deployment"; too early for the frontier results.
  ([CFP](https://facctconference.org/2027/cfp.html))
- **AIES 2027** (AAAI/ACM AI, Ethics and Society): 2026's deadlines were May 14 (abstract) and
  May 21 (paper), 10 pages, conference in October. Expect similar for 2027; best archival fit
  after the arXiv post. ([2026 CFP](https://www.aies-conference.com/2026/wp-content/uploads/2026/02/AIES-CFP-2026.pdf))
- **ICLR 2027 workshops** (conference April 26-30 2027): workshop paper deadlines usually fall
  in late January to early February; look for evaluation, safety or sociotechnical workshops when
  the list is out in December.
- **ICML 2027 workshops**: usually late May.
- **Policy outlets** (no deadline): RAND working paper series, War on the Rocks, Lawfare,
  Texas National Security Review.

Recommended path: arXiv in December, the policy version in December-January, an ICLR 2027
workshop in February if a fitting one appears, and AIES 2027 in May as the archival target.

## Open questions for John

1. What does TNL expect as the output: a paper, a memo, a presentation, or something else?
2. Go-ahead for the October batch (about $20 of your Anthropic credits), with your key set as
   `ANTHROPIC_API_KEY` wherever the run happens.
