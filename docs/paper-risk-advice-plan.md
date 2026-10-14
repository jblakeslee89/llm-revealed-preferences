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

## The one piece of new work: frontier models through their APIs

Built today: `src/api_elicit.py` asks each Stage A cell n times at the provider's default
temperature and records the share of answers that pick the gamble. Its output matches the open-
model files, so `analysis/stage_a.py --stems <name>` reports the same tables. It refuses to call
anything unless `--go` is passed, and prints a cost estimate first. Checked end to end on a
simulated agent (no spend).

Proposed models (five; John to confirm):

1. Claude Sonnet (latest), and Claude Haiku (latest) as the small-model contrast that also
   links to the Phase 1-2 Haiku runs
2. The current OpenAI flagship
3. The current Gemini flagship
4. One large open-weight model (DeepSeek or the largest Qwen), so the 7B open results have a
   frontier-scale open counterpart

Simplest key setup: one OpenRouter key reaches all five (`--provider openai --base-url
https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY`). Direct provider keys also work.

Regimes: immediate answer (thinking off or at its minimum where the API allows) and reason
first. For models that always think, the reasoning-effort setting is the axis instead of snap
versus reasoned.

Cost (from the script's estimator; $3 / $15 per million tokens in / out as a placeholder, so
check each model's price before running):

| Per model | Calls | Visible output only | With ~1,000 hidden reasoning tokens per call |
|---|---|---|---|
| Pilot: money + civilians, 10 gambles, n = 5 | 1,200 | under $1 | ~$18 |
| Immediate answer, full grid, n = 5 | 7,200 | ~$2.50 | ~$110 |
| Reason first, 20 gambles, n = 3 | 2,400 | ~$13 | ~$50 |

So about $15 per model that doesn't think by default and about $160 per model that does.
Five models: **roughly $150-500, with a proposed cap of $400**, pilot first.

Added at no extra design cost, and recommended: a second wording per domain and a detainee
domain without the escalation language, which answer the two main Stage A caveats. Generating
those grids is free; running them adds about half again to the frontier cost and needs one
Colab session for the open models.

## Milestones

Dates are 2026 unless marked. The RAND term ends Dec 4; Thanksgiving is Nov 26.

| Dates | Milestone | Needs from John |
|---|---|---|
| Oct 14-17 | Plan, API harness, draft of sections 1, 3 and 4 (done in this thread) | TNL output format; API key(s); spend approval; model list |
| Oct 19-23 | Pilot on all five models (≤ $20 each); check refusal and parse rates; fix n and thinking settings. Second-wording grids built. Related work written and the two wargame citations checked | Approve pilot spend |
| Oct 27 | FAccT 2027 abstract deadline (paper Nov 3). Stretch only: submitting would mean frontier results in the rebuttal window. Recommendation: skip | Go / no-go |
| Oct 26-Nov 6 | Full frontier runs, both regimes, plus the second wordings. One Colab session for the open models on the new wordings | Approve full spend; run one Colab notebook |
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
2. Which API keys can you provide (one OpenRouter key is simplest), and is a $400 cap right?
3. Are the five models above the right set?
