# Phase 3 on Colab (runbook)

The provider catalog check (July 2026) found that no serverless host serves BASE open
models with logprobs, and OLMo-2 is hosted nowhere. So Phase 3 elicitation runs directly
on a free Colab T4 GPU. This is free, gives exact probabilities from the full vocabulary,
and is the only path that covers OLMo-2's staged checkpoints.

## Steps

1. Locally, the trial grid is already exported to `data/phase3_grid.csv` (1,680 cells
   with prompts). Regenerate it any time with:
   ```bash
   python src/run_phase3.py --export-grid data/phase3_grid.csv
   ```

2. Open a new notebook at https://colab.research.google.com, then
   **Runtime -> Change runtime type -> T4 GPU**.

3. In the first cell:
   ```python
   !pip install -q transformers accelerate bitsandbytes
   from google.colab import files
   files.upload()   # upload phase3_grid.csv AND colab/phase3_elicit.py
   ```

4. (Only for gated Llama; skip for Qwen and OLMo.)
   ```python
   from huggingface_hub import login; login()   # paste a HF token with Llama access
   ```

5. Score each model. One base/instruct pair is the minimum; add the dual-format instruct
   run so the format confound is controlled.
   ```python
   # committed pair: Qwen (open, not gated)
   !python phase3_elicit.py --model Qwen/Qwen2.5-7B          --grid phase3_grid.csv --out phase3_qwen-base_fewshot.csv --fmt fewshot
   !python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --grid phase3_grid.csv --out phase3_qwen-inst_fewshot.csv --fmt fewshot
   !python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --grid phase3_grid.csv --out phase3_qwen-inst_chat.csv    --fmt chat
   ```
   Each run is ~1,680 forward passes, roughly 15-40 min on a T4. The free session lasts
   long enough for one model; do additional models in fresh sessions.

6. Download the result CSVs and drop them in the repo's `data/`:
   ```python
   files.download("phase3_qwen-base_fewshot.csv")
   files.download("phase3_qwen-inst_fewshot.csv")
   files.download("phase3_qwen-inst_chat.csv")
   ```

7. Locally, run the attribution:
   ```bash
   python analysis/estimate_from_probs.py data/phase3_qwen-base_fewshot.csv \
       --instruct data/phase3_qwen-inst_fewshot.csv \
       --label-base "Qwen base" --label-instruct "Qwen instruct"
   ```

## Model pairs, in priority order

| Pair | Gated? | Why |
|---|---|---|
| `Qwen/Qwen2.5-7B` / `-Instruct` | no | Start here: open, matches the literature |
| `meta-llama/Llama-3.1-8B` / `-Instruct` | yes (HF token) | Second family; the incumbent papers' model |
| `allenai/OLMo-2-1124-7B` / `-SFT` / `-DPO` / `-Instruct` | no | Stretch: staged checkpoints attribute the bias to a SPECIFIC post-training stage |

## Format-confound control (why the `--fmt chat` run matters)

Base models are elicited via few-shot completion; instruct models are naturally chat.
If the instruct model gave different parameters purely because of format, the
base-vs-instruct gap would be an artifact. Running the instruct model under BOTH
`--fmt fewshot` and `--fmt chat` checks this: if its parameters are stable across
formats, format is not driving the contrast, and the base-vs-instruct gap is real.

## Induced-valuation arm (Armour feedback, Aug 2026)

Assert a utility function and instruct the model to maximize it; compliance with
the induced optimum calibrates the elicitation (induced-value logic). Run each
model in its GOOD format per the Phase 3 dominance check (Qwen instruct: chat;
OLMo: fewshot). Score afterwards with `analysis/score_induced.py`, passing the
matching uninduced CSV as `--baseline`.

```python
!python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --grid phase3_grid.csv --out phase3_qwen-inst_chat_riskneutral.csv --fmt chat --induce riskneutral
!python phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --grid phase3_grid.csv --out phase3_qwen-inst_chat_sqrt.csv        --fmt chat --induce sqrt
# OLMo staircase (fewshot), riskneutral: where in post-training does inducibility arrive?
!python phase3_elicit.py --model allenai/OLMo-2-1124-7B          --grid phase3_grid.csv --out phase3_olmo-base_fewshot_riskneutral.csv --fmt fewshot --induce riskneutral
!python phase3_elicit.py --model allenai/OLMo-2-1124-7B-SFT      --grid phase3_grid.csv --out phase3_olmo-sft_fewshot_riskneutral.csv  --fmt fewshot --induce riskneutral
!python phase3_elicit.py --model allenai/OLMo-2-1124-7B-DPO      --grid phase3_grid.csv --out phase3_olmo-dpo_fewshot_riskneutral.csv  --fmt fewshot --induce riskneutral
!python phase3_elicit.py --model allenai/OLMo-2-1124-7B-Instruct --grid phase3_grid.csv --out phase3_olmo-inst_fewshot_riskneutral.csv --fmt fewshot --induce riskneutral
```

Disk note: clear `/root/.cache/huggingface` between model FAMILIES (the Aug 13
session died on a full disk with four OLMo checkpoints cached; one family at a
time fits fine).

## Optional cross-method check

To confirm reading logprobs matches sampling, generate (say) 25 completions per cell for
a ~100-cell subset of one instruct model and compare the sampled choice frequency to the
logprob `p_gamble`. High agreement validates the logprob shortcut. This is a robustness
appendix, not a gate; the estimator recovery on simulated data is the primary validation.
