# Data and source provenance

## Preserved research archive

The `notebooks/` and `results/` directories preserve every available file from the original `wiml-abstract` project as copied on November 24, 2026: one notebook and fifteen result/figure/documentation files. `paper/` contains the author-supplied one-page abstract PDF. No original source files were deleted as part of preparing this repository.

[`SOURCE_MANIFEST.json`](../SOURCE_MANIFEST.json) records the original relative paths, repository paths, byte sizes, and SHA-256 hashes. The notebook and result files are preserved byte for byte, including notebook execution outputs. The maintained `src/`, command-line scripts, tests, README, and supporting documentation are new repository additions derived from that research archive.

Run `python tools/verify_archive.py` from the repository root to verify all seventeen original artifacts (sixteen project files plus the supplied PDF) using only the Python standard library.

The supplied PDF title supersedes the notebook's earlier subtitle. Both refer to the same Qwen3-4B/PopQA experiment and the same saved headline results.

## External model and dataset

- Model: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), evaluated with 4-bit NF4 loading. Model weights are downloaded separately; none are committed here.
- Dataset: [`akariasai/PopQA`](https://huggingface.co/datasets/akariasai/PopQA), `test` split. The complete dataset is downloaded separately. Saved sampled records and generated predictions are included in the research archive.
- The historical notebook did not pin model or dataset revision hashes. Existing server cache metadata identifies PopQA revision `098765c79ea10a2cb19c828324e33281b8336ec0` and Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, with cache timestamps on August 11, 2026. The recovered dataset matches all 150 independent facts and all 900 correctness labels. This supports reproduction but does not establish which revision the historical runtime selected; see [`data/cache_provenance.json`](../data/cache_provenance.json).
- The notebook records Python 3.13.5 and pins Transformers 5.15.0, Accelerate 1.14.0, and bitsandbytes 0.50.0. It does not provide a complete dependency/CUDA environment lock. New validation environment details are distinct from that historical execution environment.

Upstream model, dataset, and third-party package terms continue to apply. No new open-source license is granted at this private review stage.

## What the saved records establish

- `screening.csv`: 2,500 screening records; 244 marked known.
- `paired_examples.jsonl`: 150 separately selected boundary-control facts, with aliases and aligned/conflicting values.
- `raw_predictions.csv`: 2,700 predictions over those 150 facts, three seeds, three evidence conditions, and two reasoning modes.
- `independent_factual_predictions.csv`: 900 predictions over 150 facts disjoint from all screened items, three seeds, and two modes.
- The independent CSV preserves correctness labels, final answers, and generated-token counts. It omits gold aliases, full thinking traces, token IDs, and stopping reasons. The derived supplement [`data/independent_facts.jsonl`](../data/independent_facts.jsonl) recovers aliases for all 150 facts from the existing PopQA cache using the notebook's exact cleaning sequence. All row indices, questions, gold answers, and 900 correctness labels were checked against the saved predictions with zero mismatches. Sixteen correct predictions require noncanonical aliases, so comparing against canonical `gold` alone is insufficient.
- `manual_audit.csv` is an audit worksheet. Its presence does not establish that every row received human review; the archived file has no separate human-annotation or reviewer-notes columns.

## Original artifacts without editable generators

The original project includes `abstract_architecture_flow.png`, `abstract_combined_table.csv`, and `ABSTRACT_WRITING_KIT.md`, but their original creation scripts were not present. These artifacts are retained unchanged. Any maintained code that regenerates a related table or figure is a documented reconstruction rather than a recovered historical generator.

The supplied PDF filename contains `Clean_Source`, but no editable LaTeX or document source was supplied or found in the original WiML project. The PDF is archived as received.

## Venue and authorship

The authors identified by the submission owner are **Azka Ikramullah** and **Seong Oun Hwang**. The owner reports acceptance at **WiML @ NeurIPS 2026** and supplied [OpenReview submission 135](https://openreview.net/forum?id=IBSVk8DrQ0). Workshop acceptance is distinct from a NeurIPS main-conference paper. Affiliations are not inferred.

## Preservation and review

This repository is a private review copy. Creating or verifying it does not authorize removal of the server project. Before any later source cleanup, review the repository and preserve the original artifacts and the Git commit that contains them.
