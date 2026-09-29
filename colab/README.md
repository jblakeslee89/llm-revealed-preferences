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

## Reason-then-answer arm (Sep 2026)

The induced arm failed on every cell that needs arithmetic (compliance never above ~0.55),
while instructed dominance reached 1.00. In immediate-answer elicitation the model must
emit a letter at the next token, so that result cannot separate "cannot compute in this
format" from "will not comply." This arm lets the model reason first: greedy generation
until it writes `Answer:`, then the same A/B logprob readout at that position. Output
CSVs match the immediate-answer ones, plus `reason`, `reason_ended` and the `reasoning`
text for auditing.

Reading the result: if reasoning lifts induced EV compliance toward 1, the immediate-answer
ceiling was computation and the elicited "natural" parameters describe an intuitive,
no-computation regime. If compliance stays near 0.5 with reasoning allowed, the model will
not adopt the stated utility, which is the worse outcome for induced-value logic.

Reasoning runs are roughly 10-20x slower than immediate answer, so they use a gamble
subset (`--gambles 20` keeps 20 of 40 gambles per instrument with all their frames,
templates and orders: 840 cells). Time a `--gambles 2` pilot first and scale.

```python
# ---- Colab cell: reason-then-answer arm (T4 GPU runtime) ----
!pip install -q -U transformers accelerate bitsandbytes
from google.colab import files
files.upload()   # phase3_grid.csv and colab/phase3_elicit.py

import shlex, shutil, subprocess
G = "--grid phase3_grid.csv --reason --gambles 20 --batch-size 16 --max-new-tokens 320"
RUNS = [
    # (model, format, tag): each subject in the format where it passed dominance
    ("Qwen/Qwen2.5-7B-Instruct",         "chat",    "qwen-inst_chat"),
    ("allenai/OLMo-2-1124-7B-SFT",       "fewshot", "olmo-sft_fewshot"),
    ("allenai/OLMo-2-1124-7B-Instruct",  "fewshot", "olmo-inst_fewshot"),
]
for model, fmt, tag in RUNS:
    for induce in ["riskneutral", "none"]:   # decisive run first, uninduced baseline second
        out = f"phase3_{tag}_reason" + ("" if induce == "none" else f"_{induce}") + ".csv"
        cmd = f"python phase3_elicit.py --model {model} --fmt {fmt} --induce {induce} --out {out} {G}"
        print(">>", cmd)
        subprocess.run(shlex.split(cmd), check=True)
        files.download(out)   # download as each finishes, in case the session dies
    shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)   # disk: clear between models
```

Llama (`meta-llama/Llama-3.1-8B-Instruct`, chat) is the fourth subject; it needs the
`login()` cell first. Run the base OLMo checkpoint only if time allows: it barely passes
dominance, so reasoning there mostly tests whether it produces coherent reasoning at all.

Scoring, locally:

```bash
# decisive comparison: induced with reasoning vs induced immediate answer, same cells
python analysis/score_induced.py data/phase3_qwen-inst_chat_reason_riskneutral.csv \
    --induce riskneutral --baseline data/phase3_qwen-inst_chat_riskneutral.csv
# does reasoning alone move the uninduced model toward EV?
python analysis/score_induced.py data/phase3_qwen-inst_chat_reason.csv \
    --induce riskneutral --baseline data/phase3_qwen-inst_chat.csv
```

Caveat for fewshot runs: the reasoning exemplars show the EV arithmetic (without stating
a rule), so an uninduced fewshot reasoning run is partly primed toward EV. Compare
induced vs uninduced within the reasoning arm; chat runs have no exemplars.

## Phase 4: induced value by payment through training

Design and caveats: `docs/phase4-paid-induction.md`. `colab/rl_induce.py` trains a
LoRA adapter with REINFORCE, paying the model the realized lottery payoff (through the
induced utility) for each A/B choice on fresh training gambles; `phase3_elicit.py
--adapter` then scores the unchanged 1,680-cell grid with the paid model.

```python
# ---- Colab cell: Phase 4, Qwen instruct (chat), linear and sqrt utilities ----
!test -d /content/repo || (git clone -q https://github.com/jblakeslee89/llm-revealed-preferences.git /content/repo && pip install -q -U transformers accelerate bitsandbytes peft)
%cd /content/repo
!git pull -q && git log --oneline -1
!python colab/rl_induce.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --utility linear --steps 300 --batch 16 --out runs/qwen_paid_linear
!python colab/phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen_paid_linear --fmt chat --grid data/phase3_grid.csv --out phase4_qwen-inst_chat_paid-linear.csv
!python colab/rl_induce.py --model Qwen/Qwen2.5-7B-Instruct --fmt chat --utility sqrt --steps 300 --batch 16 --out runs/qwen_paid_sqrt
!python colab/phase3_elicit.py --model Qwen/Qwen2.5-7B-Instruct --adapter runs/qwen_paid_sqrt --fmt chat --grid data/phase3_grid.csv --out phase4_qwen-inst_chat_paid-sqrt.csv
!zip -qr phase4_qwen.zip phase4_qwen-inst_chat_paid-*.csv runs/*/train_log.csv runs/*/config.json runs/*/adapter_config.json
```

Download `phase4_qwen.zip` from the Files pane. Score locally:

```bash
python analysis/score_induced.py data/phase4_qwen-inst_chat_paid-linear.csv --induce riskneutral --baseline data/phase3_qwen-inst_chat.csv
python analysis/score_induced.py data/phase4_qwen-inst_chat_paid-sqrt.csv --induce sqrt --baseline data/phase3_qwen-inst_chat.csv
```

Check the first run's training speed from the `secs` column in `train_log.csv` before
queueing the second; if 300 steps will not fit in the session, lower `--steps`.

## Optional cross-method check

To confirm reading logprobs matches sampling, generate (say) 25 completions per cell for
a ~100-cell subset of one instruct model and compare the sampled choice frequency to the
logprob `p_gamble`. High agreement validates the logprob shortcut. This is a robustness
appendix, not a gate; the estimator recovery on simulated data is the primary validation.
