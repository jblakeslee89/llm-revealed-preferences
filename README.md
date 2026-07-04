# LLM Revealed Preferences

Structural estimation of economic preference primitives (risk aversion, loss aversion,
reference dependence) from elicited LLM choices, with a base-vs-instruct contrast to test
whether post-training instills the biases.

Summer OJT Practicum (Course 1012), Pardee RAND, Jun 29 - Sep 4 2026.

## Layout

- `proposal/` - LaTeX research proposal (revised Jul 2026 after literature sweep)
- `docs/` - literature review and methodology-risk survey (Jul 2026)
- `src/` - elicitation harness
  - `design.py` - procedural gamble generation, templates, counterbalanced trial construction
  - `harness.py` - runs trials against a model, writes tidy CSV
- `data/` - elicited choice datasets (CSV, one row per choice, self-documenting)
- `analysis/` - estimation
  - `estimate_crra.py` - CRRA + Fechner-logit MLE with sanity checks

## Quick start

```bash
pip install -r requirements.txt

# validate the whole pipeline against a simulated CRRA agent (no API key needed)
python src/harness.py --dry-run --reps 3
python analysis/estimate_crra.py data/phase1_dryrun.csv
# -> should recover r ~= 0.5, mu ~= 0.15

# live pilot against Claude (needs ANTHROPIC_API_KEY exported)
python src/harness.py --pilot 40
python analysis/estimate_crra.py data/phase1_claude-haiku-4-5-20251001.csv

# full Phase 1 run (~1,200 calls at default settings)
python src/harness.py --reps 3
```

## Design commitments (implemented in code, see proposal §3.4)

- **Contamination-safe**: procedural gambles, non-round payoffs, jittered probabilities,
  novel wording (no Holt-Laury rows or textbook vignettes)
- **Counterbalanced**: every cell presented in both orders; position bias estimable
- **Template as factor**: 5 equivalent paraphrases, logged per trial
- **Validation arm**: ~10% of trials use plain-text elicitation instead of forced tool
  output, so output-forcing effects are measurable
- **Self-documenting data**: model snapshot, temperature, seed, raw response, discard
  flag, and timestamp on every row

## Status

- [x] Proposal (revised for mid-2026 literature)
- [x] Phase 1 harness + estimator, validated end-to-end on simulated agent
- [x] Phase 1 live run: 1,200 choices, claude-haiku-4-5-20251001
- [x] Extended estimator (`analysis/estimate_cpt.py`): Prelec weighting + position nuisance
- [ ] Phase 2: gain/loss framing, loss aversion, reference point
- [ ] Phase 3: base-vs-instruct logprob elicitation on open models

## Phase 1 headline (July 2026)

Haiku prices probability, not payoffs. Extreme S-shaped probability weighting
(Prelec gamma = 2.69 vs. human ~0.7: long shots refused at almost any price),
human-plausible curvature once weighting is controlled (r = 0.59), and a ~3.5x
odds shift toward whichever option is listed second. A probability-only heuristic
nearly ties the structural model on fit. Full writeup in `writeups/`.
