"""Offline checks against the archived study, without loading any model."""

from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from wiml.analysis import analyze, independent_statistics, load_results
from wiml.config import Config
from wiml.experiment import build_independent, write_run_metadata
from wiml.protocol import (
    classify_conflict, matches_gold, normalize_answer, parse_bool, prepare_popqa,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def archived():
    return load_results(RESULTS)


def test_archive_design_and_headline(archived):
    raw, independent, pairs = archived
    assert len(raw) == 2700
    assert len(pairs) == 150
    assert len(independent) == 900
    assert set(independent["decode_seed"]) == {17, 29, 41}
    screening = pd.read_csv(RESULTS / "screening.csv")
    assert len(screening) == 2500
    assert set(independent["row_index"]).isdisjoint(screening["row_index"])
    stats = independent_statistics(independent)
    assert stats["effect_think_minus_nothink_pp"] == pytest.approx(-3.5555555556)
    assert stats["fact_clustered_ci95_pp"] == pytest.approx([-7.7777777778, 0])
    assert stats["paired_fact_wilcoxon_p"] == pytest.approx(0.07811637889228637)
    th = independent[independent["thinking"]]
    assert (th["generated_tokens"] == 256).sum() == 368


def test_saved_boundary_labels_match_original_alias_protocol(archived):
    raw, _, pairs = archived
    metadata = pairs.set_index("example_id")
    for row in raw.itertuples(index=False):
        ex = metadata.loc[row.example_id]
        final = "" if pd.isna(row.final_answer) else row.final_answer
        if row.condition == "conflict":
            source = classify_conflict(final, ex.conflict_value, ex.prior_aliases)
            assert row.answer_source == source
            assert row.context_correct == int(source == "context")
            assert row.prior_reversion == int(source == "prior")
        else:
            correct = int(matches_gold(final, ex.prior_aliases))
            stored = row.context_correct if row.condition == "aligned" else row.prior_reversion
            assert correct == stored


def test_saved_independent_labels_match_recovered_aliases(archived):
    _, independent, _ = archived
    facts = pd.read_json(ROOT / "data" / "independent_facts.jsonl", lines=True)
    metadata = facts.set_index("row_index")
    alias_only_successes = 0
    for row in independent.itertuples(index=False):
        fact = metadata.loc[row.row_index]
        assert row.question == fact["question"]
        assert row.gold == fact["gold"]
        final = "" if pd.isna(row.final_answer) else row.final_answer
        assert int(matches_gold(final, fact["gold_aliases"])) == row.correct
        if row.correct and not matches_gold(final, [row.gold]):
            alias_only_successes += 1
    assert alias_only_successes == 16


def test_analysis_recreates_saved_tables(archived, tmp_path):
    report = analyze(RESULTS, tmp_path / "analysis")
    out = tmp_path / "analysis"
    for filename in [
        "condition_summary.csv", "control_table.csv", "paper_results_table.csv",
        "abstract_combined_table.csv", "relation_summary.csv", "manual_audit.csv",
    ]:
        expected = pd.read_csv(RESULTS / filename)
        actual = pd.read_csv(out / filename)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-10, rtol=1e-10)
    assert report["token_multiplier"] == pytest.approx(39.39742212674543)
    assert report["boundary"]["context_to_prior_flips"] == 0
    assert report["boundary"]["prior_to_context_recoveries"] == 0
    assert report["boundary"]["decode_pair_bootstrap_ci95_pp"] == [0, 0]
    assert report["generation_cap_diagnostic"]["think_independent_at_original_256_token_cap"] == 368
    for filename in ["independent_factual_recall.png", "conflict_mode_comparison.png", "paired_transitions.png"]:
        assert (out / filename).stat().st_size > 10000


def test_normalization_and_csv_booleans():
    assert normalize_answer("Answer: Paris.") == "paris"
    assert normalize_answer("The answer is The Hague") == "hague"
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""
    assert matches_gold("NYC", ["New York City", "NYC"])
    assert not matches_gold("NYC", ["New York City"])
    assert not parse_bool("False")
    assert parse_bool("True")
    with pytest.raises(ValueError):
        parse_bool("unknown")


def test_cleaning_removes_ambiguous_pairs_and_single_object_relations():
    frame = pd.DataFrame([
        {"subj": "a", "prop": "r", "obj": "x", "possible_answers": '["X alias"]'},
        {"subj": "b", "prop": "r", "obj": "y", "possible_answers": ["Y alias"]},
        {"subj": "dup", "prop": "r", "obj": "z", "possible_answers": []},
        {"subj": "dup", "prop": "r", "obj": "w", "possible_answers": []},
        {"subj": "c", "prop": "singleton", "obj": "z", "possible_answers": []},
    ])
    cleaned = prepare_popqa(frame)
    assert list(cleaned["subj"]) == ["a", "b"]
    assert cleaned.iloc[0]["gold_aliases"] == ["x", "X alias"]


def test_independent_selection_is_reproducible_and_disjoint():
    frame = pd.DataFrame({"question": [f"q{i}" for i in range(500)]})
    screening = pd.DataFrame({"row_index": np.arange(250)})
    cfg = Config(independent_examples=150)
    a = build_independent(frame, screening, cfg)
    b = build_independent(frame, screening, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["row_index"]).isdisjoint(screening["row_index"])


def test_resume_rejects_configuration_or_dataset_change(tmp_path):
    cfg = Config(output_dir=str(tmp_path))
    frame = pd.DataFrame({"question": ["a question"]})
    write_run_metadata(tmp_path, cfg, frame)
    write_run_metadata(tmp_path, cfg, frame)
    with pytest.raises(ValueError, match="different run_config"):
        write_run_metadata(tmp_path, replace(cfg, temperature=0.7), frame)
    with pytest.raises(ValueError, match="dataset changed"):
        write_run_metadata(tmp_path, cfg, pd.DataFrame({"question": ["different"]}))


def test_analysis_preserves_archive():
    with pytest.raises(ValueError, match="must differ"):
        analyze(RESULTS, RESULTS)
    with pytest.raises(ValueError, match="must differ"):
        analyze(RESULTS, RESULTS / "generated")


def test_fresh_run_can_write_analysis_subdirectory(tmp_path):
    # Fresh experiment directories legitimately contain their own analysis/
    # output. Exercise the same call shape used by experiment.main.
    fresh = tmp_path / "fresh_run"
    shutil.copytree(RESULTS, fresh)
    report = analyze(fresh, fresh / "analysis")
    assert report["independent"]["facts"] == 150
    assert (fresh / "analysis" / "analysis_summary.json").is_file()
