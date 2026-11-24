# Repository validation

Validated during repository preparation on November 24, 2026. These checks concern the preserved evidence and maintained CPU analysis; the Qwen GPU inference experiment was not rerun.

## Source integrity

- All sixteen original project files match the source server's SHA-256 checksums.
- The supplied PDF is retained unchanged.
- The original notebook passes the Jupyter notebook schema, including its saved execution outputs.
- `CITATION.cff` passes the official Citation File Format 1.2.0 schema.
- The README hero and regenerated plots were rendered and visually inspected.

Run `python tools/verify_archive.py` to verify the seventeen original artifacts locally after cloning. The derived alias/provenance files are identified separately in `SOURCE_MANIFEST.json`.

## Saved-result audit

- Complete prediction grids: 2,700 boundary predictions and 900 independent predictions, with no duplicate prediction keys.
- The 150 independent facts are disjoint from all 2,500 screened facts.
- Every boundary label was rechecked using the archived paired-example aliases: zero mismatches across 2,700 outputs.
- Every independent correctness label was rechecked using recovered PopQA aliases: zero mismatches across 900 outputs. Sixteen correct outputs require a noncanonical alias.
- Regenerated condition, control, headline, combined, relation, and audit tables agree with the original numerical data.
- Recomputed headline difference: -3.5555556 percentage points; fact-clustered 95% CI [-7.7777778, 0]; Wilcoxon p = 0.0781164; token multiplier = 39.3974221.
- Ten offline tests pass, covering label audits, sampling design, statistics, tables, protocol helpers, checkpoint consistency, and the fresh-run analysis path.

The CPU validation environment used Python 3.12, NumPy 2.5.5, Pandas 3.1.1, SciPy 1.18.1, Matplotlib 3.11.2, and Pytest 9.2.0 on Windows. This is a validation environment, not a reconstructed historical GPU environment.

## Automated checks

The GitHub Actions workflow verifies original checksums, runs the offline tests, and regenerates analysis on Linux with Python 3.12. It does not download model weights, require API credentials, or run GPU inference. Future numerical differences should be investigated against the preserved original CSVs and recorded method, rather than updating the archive to match a new environment.
