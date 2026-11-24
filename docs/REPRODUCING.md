# Reproducing the study

There are two routes: reproduce the reported analysis from the archived predictions on a CPU, or run fresh model inference on a CUDA GPU. The original notebook is preserved at [`notebooks/more_thinking_less_reading_experiment.ipynb`](../notebooks/more_thinking_less_reading_experiment.ipynb). The Python package is an extracted and organized implementation of its code, with safer checkpoint handling and extra run metadata. The archived notebook and saved results remain the historical source.

## 1. Recompute tables, statistical tests, and plots without a model

Run from the repository root using Python 3.10 or newer:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/analyze_results.py --results-dir results --output-dir reproduction/analysis
```

Once the four analysis dependencies are installed, this command reads local CSV/JSONL files only. It does not import PyTorch or Transformers, contact Hugging Face, load weights, or require a GPU. It validates the matched prediction grid and the separation between screening and headline evaluation before writing outputs. It reads the stored headline `correct` field, which was produced using the complete PopQA alias set. Comparing answers against only the `gold` string in that CSV would incorrectly mark alias answers as wrong.

Expected independent evaluation: 150 facts, 900 predictions; Think − NoThink accuracy difference **−3.5556 percentage points**, fact-clustered 95% bootstrap CI **[−7.7778, 0.0000]**, paired fact-level Wilcoxon **p = 0.0781164**, and **39.3974×** generated tokens. Explicit aligned and conflict records have 100% adherence in both modes, with zero prior-reversion flips.

Generated files include:

| Output | Purpose |
|---|---|
| `analysis_summary.json` | Paired effect, CI, p values, token multiplier, cap diagnostic |
| `paper_results_table.csv` | Headline independent factual-recall table |
| `abstract_combined_table.csv` | Headline result plus boundary controls |
| `condition_summary.csv`, `control_table.csv` | All three conditions on the screened boundary set |
| `conflict_results_table.csv`, `relation_summary.csv` | Conflict-only and relation-wise summaries |
| `transition_counts.csv`, `grounding_flip_examples.csv` | Paired answer-source transitions and any flips |
| `manual_audit.csv` | Rows prepared for human inspection; **not proof that a human audit was completed** |
| Three PNG files | Independent accuracy, conflict comparison, and transitions |

The original notebook writes its conflict table to `paper_results_table.csv`, then overwrites it with the independent table in cell 40. The CLI preserves both under separate filenames. Plot pixels can vary with Matplotlib/fonts; the checks compare numerical tables and reported statistics.

## 2. Run fresh inference

The historical inference used Qwen3-4B in 4-bit NF4 on a single NVIDIA A10. Use a CUDA-capable environment with a compatible PyTorch/driver setup and enough free GPU memory for the quantized model and generation state. The source does not establish a minimum VRAM guarantee or execution time. The first run needs network access and disk space for model weights and PopQA. No API key is embedded in this repository.

```bash
python -m pip install -r requirements-reproduce.txt
python scripts/run_experiment.py --output-dir reproduction/full
```

`requirements-reproduce.txt` preserves the notebook's three explicit install pins: `transformers==5.15.0`, `accelerate==1.14.0`, and `bitsandbytes==0.50.0`. These are **the recorded install command**, not a verified package lock or evidence that those exact versions were installed when the archived outputs were generated. PyTorch, datasets, and scientific libraries were not pinned in the notebook. If a recorded version is unavailable in your package index, inspect the notebook and choose a compatible environment explicitly; do not silently treat that change as an exact replication. New runs record installed package versions in `environment.json`.

A smaller diagnostic run can explicitly change the sample sizes:

```bash
python scripts/run_experiment.py --smoke --candidate-pool 100 --target-examples 5 --independent-examples 10 --output-dir reproduction/smoke
```

This is a changed-size smoke check, not the reported experiment; it is not guaranteed to find five known facts. The original smoke defaults screen only 60 candidates while requesting 20 known facts, and can fail because too few facts are recalled. The original independent evaluation is hardcoded to 150 facts, including in smoke mode; the CLI keeps that default unless `--independent-examples` is specified. A failed known-fact screen raises an error with the observed count. Choose a new output directory if changing the pool, target, or any other configuration.

Optional `--model-revision` and `--dataset-revision` arguments pin future runs. The original notebook does not pin model or dataset revisions. See [data provenance](DATA_AND_PROVENANCE.md) for any recovered cache evidence; a cache observation is not a guaranteed historical runtime lock. Row indices depend on the cleaned dataset order.

The CLI checkpoints predictions every 25 rows and screening every 10 rows. Rerunning the identical command resumes completed keys. It writes `run_config.json`, `environment.json`, and `dataset_sha256.txt` and refuses to mix old checkpoints with a different config or cleaned dataset. Fresh independent facts, including their aliases, are saved in `independent_examples.jsonl`; this additional manifest was absent from the archived run. Atomic checkpoint replacement prevents a partial write from truncating a completed checkpoint. Use a separate reproduction directory; never overwrite the historical `results/` directory. The default `reproduction/` output tree is ignored by Git so regenerated files do not modify the archived release.

## 3. Exact experimental protocol

| Component | Recorded setting |
|---|---|
| Dataset | `akariasai/PopQA`, test split |
| Cleaning | Keep `(subj, prop)` pairs occurring once; retain relations with at least two distinct objects |
| Boundary screening | Up to 2,500 candidates sampled without replacement with seed `20260811`; NoThink greedy generation |
| Boundary selection | Sample 150 recalled facts with Pandas `random_state=20260811` |
| Counterfactual | Different object from the same relation, excluding normalized gold aliases; RNG seed `20260812` |
| Boundary conditions | No context, aligned record, conflicting fictional record; both modes; three seeds → 2,700 predictions |
| Independent selection | 150 cleaned rows not in **any** screened row, sampled without replacement with seed `20260812` |
| Independent evaluation | Both modes × seeds `(17, 29, 41)` → 900 predictions |
| Matched generation seed | Boundary: `decode_seed + 100003 * example_id`; independent: `decode_seed + 200003 * example_id` |
| Sampling in evaluation | Both modes: temperature `0.6`, top-p `0.95`, top-k `20` |
| Token caps | NoThink: 64 new tokens; Think: 256 new tokens |
| Weights and attention | Same model; NF4 + double quantization; BF16 compute when supported; eager attention; GPU 0 |
| Thinking switch | Official chat template `enable_thinking=False/True` |
| Answer extraction | Text after the last `</think>` token if present; otherwise all generated tokens |
| Scoring | Lowercase, first line, strip common answer prefixes, remove punctuation/articles; exact match to normalized aliases |
| Headline inference | Average correctness over three seeds per fact; paired Wilcoxon on 150 fact differences; 5,000 fact bootstrap draws, RNG seed `20260813` |
| Boundary inference | Original paired exact binomial test on prior/non-prior discordances; 5,000 decode-pair bootstrap draws, seed `20260811` |

The headline bootstrap treats facts as the independent sampling unit. The original boundary bootstrap resamples matched **decode pairs**, not facts; it is retained for fidelity and labeled accordingly. Here every conflict pair is context→context, so both boundary CI endpoints are zero. Three decoding seeds are repeated samples from one model, not independent training runs.

## 4. Interpretation and limitations

The CI touches zero and p is about 0.078; this does not establish statistically significant accuracy harm. The strongest observation is the extra generated-token cost without an observed accuracy gain. Token count is not wall-clock time, energy, FLOPs, or a measured bill.

In the archived independent evaluation, **368 of 450 Think outputs (81.8%) reach the 256-token cap**. The independent CSV does not contain reasoning text or finish reasons, so it cannot establish exactly which answers were truncated or whether every generation completed its reasoning. This is a material limitation, and results may change with a different cap or extraction policy. The CLI intentionally preserves the original policy, including treating empty answers as unmatched and using the full output when no closing thinking token is found.

Results cover one model/size, NF4 quantization, one entity-centric dataset, and these prompts. The low independent accuracy limits claims about high-performing systems. The boundary set is intentionally conditioned on NoThink knowledge, so its no-context accuracy gap is selection-biased and is **not** the headline comparison. Explicit fictional records and instructions can make adherence unusually easy. Lexical matching misses paraphrases; the manual audit CSV alone does not prove a completed human audit. Downloaded dataset/model contents and package versions can also change in the absence of historical locks.

## 5. Offline validation

```bash
python -m pip install pytest
python -m pytest -q
```

The tests check all 2,700 boundary labels against the archived aliases and all 900 headline labels against the recovered alias metadata in `data/independent_facts.jsonl` (including 16 alias-only successes). They also check the 150-fact independent design and screening separation, the reported paired statistics, the 368 at-cap outputs, reproduction of the saved numerical tables, source-preserving output paths, cleaning logic, CSV booleans, and resume metadata safeguards. They do not download a model or run inference. Full GPU inference has not been rerun as part of repository preparation.
