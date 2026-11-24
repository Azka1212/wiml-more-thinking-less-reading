# WiML One-Page Abstract Writing Kit

## Recommended title

**More Thinking, Less Reading? The Cost and Boundary Conditions of Reasoning Mode for Factual QA**

## One-sentence problem

Reasoning modes greatly expand inference-time computation, but it is unclear whether this extra computation improves concise factual recall or weakens reliance on explicit textual evidence.

## Research question

Does enabling reasoning mode improve factual recall enough to justify its token cost, and does it weaken adherence to explicit textual evidence?

## Experimental design

- Model: Qwen3-4B, loaded in 4-bit NF4 for single-A10 inference.
- Headline evaluation: 150 untouched PopQA facts, never used for screening or selection.
- Pairing: Think and NoThink use the same facts and three matched decoding seeds (900 predictions).
- Inference: factual differences and bootstrap confidence intervals are computed at the fact level, so repeated decoding seeds are not treated as independent examples.
- Boundary controls: a separate set of 150 facts known to the model is paired with aligned and counterfactual conflicting textual records (2,700 predictions).
- Automatic matching: normalized aliases classify factual correctness, record adherence, prior reversion, and other answers.

## Two clean contributions

1. **Independent efficiency evaluation.** We provide a matched Think/NoThink evaluation on an untouched factual QA subset, avoiding the selection bias created by choosing facts that NoThink already answers correctly.
2. **Empirical finding with a boundary condition.** Thinking uses approximately 39× more generated tokens without an observed factual-accuracy improvement, while explicit aligned or conflicting evidence preserves 100% context adherence in both modes.

## Headline results

| Mode | Independent factual accuracy | Tokens / answer | Aligned adherence | Conflict adherence | Conflict prior reversion |
|---|---:|---:|---:|---:|---:|
| NoThink | **10.44%** | **6.21** | 100% | 100% | 0% |
| Think | 6.89% | 244.53 | 100% | 100% | 0% |

- Think − NoThink factual accuracy: **−3.56 percentage points**.
- Fact-clustered 95% bootstrap CI: **[−7.78, 0.00]**.
- Paired fact-level Wilcoxon test: **p = 0.078**.
- Token multiplier: **244.53 / 6.21 ≈ 39.4×**.
- Context→prior grounding flips: **0**.
- Reasoning recoveries under conflict: **0**.

## Correct interpretation

Thinking did not provide a reliable factual-accuracy improvement and consumed roughly 39× more generated tokens. The accuracy estimate is directionally worse but not conventionally statistically significant because its confidence interval touches zero. Perfect aligned and conflict-record adherence shows that this inefficiency is not evidence of weaker textual grounding under the tested prompts.

## Claims to avoid

- Do not say thinking significantly harms factual accuracy.
- Do not say reasoning causes grounding drift; the experiment found zero drift.
- Do not claim universal behavior across models, datasets, precisions, or prompts.
- Do not treat the three decoding seeds as three independent model-training runs.
- Do not use the screened-set no-context recall gap as the headline accuracy result; that set was selected using NoThink.

## Limitations

- One model family and size: Qwen3-4B.
- Model inference used 4-bit NF4 quantization.
- PopQA is short, entity-centric factual QA; results may not transfer to multi-hop or generative knowledge tasks.
- Absolute accuracy on the untouched subset is low, limiting conclusions about high-performing regimes.
- The independent accuracy difference is suggestive rather than statistically conclusive.
- Counterfactual records are artificial interventions with explicit instructions to use the record.
- The boundary-control set is intentionally conditioned on facts demonstrated during NoThink screening.
- Lexical alias matching can miss semantically correct paraphrases; manual audit should be reported.
- Thinking generations were capped at 256 new tokens and occasionally failed to produce a concise final answer.

## Recommended figures

1. **Architecture/flow:** `abstract_architecture_flow.png`.
   Caption: *We compare Think and NoThink on an untouched PopQA subset, then use aligned and conflicting records on separately screened facts to test whether explicit evidence bounds any reasoning-mode degradation.*
2. **Headline result:** `independent_factual_recall.png`.
   Caption: *Thinking mode does not improve factual accuracy on the untouched subset (6.9% vs. 10.4%) while increasing mean generation from 6.2 to 244.5 tokens.*

Use only one of these if space is tight. The architecture communicates design; the result graph communicates the main observation. The compact combined table already contains both performance and boundary-control results.

## Suggested abstract text

Reasoning modes increase inference-time computation, but their value for concise factual recall and their interaction with explicit evidence remain unclear. We compare Qwen3-4B with thinking enabled and disabled on 150 untouched PopQA facts using three matched decoding seeds, and separately test aligned and counterfactual records on 150 facts demonstrated to be known by the model. On the untouched subset, thinking does not improve factual accuracy (6.9% vs. 10.4%; paired difference −3.56 points, fact-clustered 95% CI [−7.78, 0.00]) while increasing generation from 6.2 to 244.5 tokens per answer. In contrast, both modes follow aligned and conflicting records in 100% of cases, with no prior-reversion flips. These results identify a bounded inefficiency: reasoning mode adds approximately 39× generation cost without an observed factual-recall benefit, yet explicit textual evidence remains fully controlling in this setting.

## One-page layout

- Title and motivation: 15%.
- Method and experimental flow: 25%.
- Combined two-row table: 25%.
- One small figure: 20%.
- Conclusion and limitations: 15%.

## Recommended final conclusion

For Qwen3-4B on PopQA, reasoning mode substantially increases token cost without an observed factual-recall benefit. This inefficiency does not imply reduced grounding: explicit records remain perfectly controlling under both aligned and conflicting evidence.
