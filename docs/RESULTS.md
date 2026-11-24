# Results and interpretation

This page documents the results accompanying **More Thinking, Less Reading? Separating Reasoning Utility from Evidence Selection in Factual QA**, an accepted abstract at the **WiML Workshop @ NeurIPS 2026**. [OpenReview submission #135](https://openreview.net/forum?id=IBSVk8DrQ0), author names, and acceptance status were supplied by the author; the local abstract PDF does not include author or venue metadata. Values below are drawn from the archived prediction files and final abstract.

## Headline: untouched factual recall

**Question:** Does thinking improve ordinary factual recall enough to justify its generation cost?

The independent set contains 150 PopQA facts excluded from screening. Each fact is evaluated with both modes and three matched decoding seeds: 150 × 2 × 3 = **900 predictions**, or 450 per mode.

| Metric | NoThink | Think |
|---|---:|---:|
| Correct predictions | 47 / 450 | 31 / 450 |
| Factual accuracy | 10.44% | 6.89% |
| Mean generated tokens / answer | 6.21 | 244.53 |

The paired effect is **−3.56 percentage points** (Think − NoThink). The fact-level 95% bootstrap interval is **[−7.78, 0.00]**, with a paired fact-level Wilcoxon **p = 0.078**. Each fact's accuracy is averaged across its three seeds before resampling; the 900 decoded outputs are not treated as 900 independent facts.

The mean generation ratio is **244.5267 / 6.2067 ≈ 39.4×**. This counts generated tokens, including thinking tokens. It is a token-cost observation, not a measured wall-clock, energy, financial, or FLOP ratio. The generation caps differ by mode: 64 new tokens for NoThink and 256 for Think. **368/450 independent Think outputs (81.78%) reach the 256-token cap.** This substantial censoring is a key protocol limitation; a larger budget could change final-answer availability and measured performance.

**Interpretation:** there is no observed accuracy improvement from thinking in this experiment. The negative accuracy estimate is not statistically conclusive. It should not be described as a significant accuracy decrease.

Sources: [`independent_factual_predictions.csv`](../results/independent_factual_predictions.csv), [`paper_results_table.csv`](../results/paper_results_table.csv), and the [abstract PDF](../paper/WiML_More_Thinking_Less_Reading_Clean_Source.pdf).

## Boundary controls: explicit evidence

**Question:** Does thinking change evidence selection when a record agrees with, or contradicts, a fact the model previously answered correctly?

A separate 150-fact set is selected after deterministic NoThink screening. Every selected fact is tested under three prompt conditions, both modes, and three decoding seeds. This yields **2,700 predictions** total:

| Condition | NoThink predictions | Think predictions | Total |
|---|---:|---:|---:|
| No supplied context | 450 | 450 | 900 |
| Aligned record | 450 | 450 | 900 |
| Conflicting record | 450 | 450 | 900 |
| **All screened-set conditions** | **1,350** | **1,350** | **2,700** |

Conflict records substitute a different object drawn from the same relation. Prompts label the records fictional and explicitly instruct the model to answer from the supplied record.

| Metric | NoThink | Think |
|---|---:|---:|
| Aligned-record adherence | 450 / 450 (100%) | 450 / 450 (100%) |
| Conflicting-record adherence | 450 / 450 (100%) | 450 / 450 (100%) |
| Conflict prior reversion | 0 / 450 (0%) | 0 / 450 (0%) |
| Other answers under conflict | 0 / 450 (0%) | 0 / 450 (0%) |

There are **450 matched conflict pairs** across modes: all are context → context. Context → prior flips and prior → context recoveries are both zero. In the aligned condition, the record and original answer agree, so adherence alone cannot identify which source produced the answer.

The evidence result provides a boundary on the interpretation of the recall result: **extra generation cost is not accompanied by an observed shift away from explicit records in this setting**. It does not establish perfect grounding outside the tested prompts or population.

Sources: [`raw_predictions.csv`](../results/raw_predictions.csv), [`condition_summary.csv`](../results/condition_summary.csv), and [`paired_examples.jsonl`](../results/paired_examples.jsonl).

## Why the screened no-context result is a diagnostic

On the selected set's sampled no-context condition, NoThink accuracy is 93.11% and Think accuracy is 41.33%. These values must not replace the untouched-set headline: the facts were selected because NoThink had already answered them correctly during deterministic screening. That selection makes a direct factual-recall comparison favor the selection procedure.

The independent 150-fact evaluation was added to answer the factual-utility question without that screening bias. The selected set remains useful for asking whether explicit, conflicting records override facts the model demonstrated it knew.

## Metrics

| Metric | Meaning |
|---|---|
| Factual accuracy | Final answer matches a normalized gold alias |
| Context adherence | Final answer matches the supplied record; aligned cases use the gold aliases |
| Prior reversion | In the conflict condition, final answer matches an original-answer alias instead of the conflicting record |
| Other | Final answer matches neither the relevant record nor a prior alias |
| Grounding flip | The same fact/seed changes from context in NoThink to prior in Think |
| Generation cost | Number of tokens generated after the prompt, including reasoning tokens |

Answer extraction uses text after the last `</think>` marker when present. Lexical normalization strips common prefixes, punctuation, and articles. An incomplete thinking generation can fail to provide a concise final answer within its token budget.

## Archive map

The original filenames are retained for traceability.

| Artifact | Role |
|---|---|
| `screening.csv` | Candidate screening answers and known-fact flags |
| `paired_examples.jsonl` | The selected facts, original aliases, and aligned/conflict record values |
| `raw_predictions.csv` | All 2,700 screened-set condition/mode/seed outputs, with source labels and token counts |
| `independent_factual_predictions.csv` | The 900 untouched factual-recall outputs, correctness flags, and token counts |
| `condition_summary.csv` | Screened-set aggregates by condition and mode |
| `control_table.csv` | Screened-set diagnostic comparison across the three conditions |
| `paper_results_table.csv` | Untouched factual accuracy and generation cost; this file is overwritten by the later independent-analysis cell in the original notebook |
| `abstract_combined_table.csv` | Compact final table combining independent recall and evidence-control results |
| `relation_summary.csv` | Conflict adherence and prior reversion by relation and mode |
| `manual_audit.csv` | Conflict-output export for human review; no completed human audit is certified by this file |
| `independent_factual_recall.png` | Untouched-set accuracy figure |
| `conflict_mode_comparison.png` | Context adherence and prior reversion under conflict |
| `paired_transitions.png` | Matched conflict source transitions |
| `abstract_architecture_flow.png` | Original abstract experiment-flow figure |
| `ABSTRACT_WRITING_KIT.md` | Original drafting notes, interpretations, and earlier title |

The original notebook first explores the screened-set boundary experiment, then adds the independent headline recall evaluation. Its cell order is retained. The final abstract and this repository's headline use the independent evaluation.

The independent CSV preserves final outputs, correctness labels, and token counts. Its original schema omits complete gold alias lists and separated thinking text. The alias lists have now been recovered as a separate derived supplement, so the archived outputs can also be relabeled locally. The screened-set raw predictions include thinking text, and the paired JSONL contains prior aliases.

## Recovered independent fact aliases

[`data/independent_facts.jsonl`](../data/independent_facts.jsonl) contains all 150 headline facts, including their original cleaned `row_index`, question, canonical gold answer, and complete `gold_aliases`. It was recovered from the server's existing PopQA Arrow cache by applying the exact notebook cleaning procedure. The original files in `results/` remain unchanged.

Validation matched all 150 fact identities against the original row indices, questions, and gold answers, then recomputed correctness for all 900 predictions using the notebook's normalization and alias matching. **There were zero identity mismatches and zero correctness-label mismatches.** Sixteen correct predictions match an alternate alias rather than the canonical gold answer, so comparing against the `gold` column alone would undercount accuracy.

[`data/cache_provenance.json`](../data/cache_provenance.json) records the recovery method, file hashes, validation counts, and observed cache references:

| Cached resource | Observed revision |
|---|---|
| PopQA | `098765c79ea10a2cb19c828324e33281b8336ec0` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |

Both cache references have August 11, 2026 modification times. These observations support the recovery provenance, but the notebook did not pin revisions: the cache state does **not prove** which model or dataset revision was used during the original generation run. The recovered manifest supports exact rechecking of the saved correctness labels; it does not restore missing independent thinking traces or establish bit-for-bit reproducibility of fresh inference.

See [data and source provenance](DATA_AND_PROVENANCE.md) and the [SHA-256 source manifest](../SOURCE_MANIFEST.json) for the preservation record and boundaries of the available source archive.

## Limits of the evidence

- One checkpoint and size: Qwen3-4B, loaded in 4-bit NF4.
- One entity-centric factual-QA dataset, with low absolute accuracy on the untouched set.
- Three decoding seeds, not independently trained models.
- Different generation caps between modes, with thinking often using most of its available budget.
- Artificial counterfactual records and a strong instruction to use them as the source of truth.
- A boundary-control set conditioned on successful NoThink screening.
- Lexical correctness and source classification that can miss valid paraphrases; the audit export needs human interpretation.

The findings support a scoped compute–utility mismatch and stable explicit-record adherence. They do not establish a universal effect of reasoning modes or a statistically significant reduction in factual accuracy.
