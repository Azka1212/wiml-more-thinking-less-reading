"""Recreate reported tables, tests, diagnostics, and plots from saved CSVs only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from .protocol import parse_bool


def _read_predictions(path, required):
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame["thinking"] = frame["thinking"].map(parse_bool)
    return frame


def load_results(results_dir):
    source = Path(results_dir)
    raw = _read_predictions(source / "raw_predictions.csv", ["example_id", "decode_seed", "condition", "thinking", "answer_source", "context_correct", "prior_reversion", "generated_tokens"])
    independent = _read_predictions(source / "independent_factual_predictions.csv", ["example_id", "row_index", "decode_seed", "thinking", "correct", "generated_tokens"])
    pairs = pd.read_json(source / "paired_examples.jsonl", lines=True)
    if pairs["example_id"].duplicated().any():
        raise ValueError("Duplicate paired example_id")
    for frame, keys in [
        (raw, ["example_id", "decode_seed", "condition", "thinking"]),
        (independent, ["example_id", "decode_seed", "thinking"]),
    ]:
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate prediction keys: {keys}")
        if frame[keys].isna().any().any():
            raise ValueError("Missing prediction keys")
        if set(frame["thinking"]) != {False, True}:
            raise ValueError("Both modes are required")
        if not frame["generated_tokens"].ge(0).all():
            raise ValueError("Invalid token count")
        seeds = frame["decode_seed"].nunique()
        expected = seeds * 2 * (3 if "condition" in keys else 1)
        if not frame.groupby("example_id").size().eq(expected).all():
            raise ValueError("Incomplete matched predictions per fact")
    if set(raw["condition"]) != {"no_context", "aligned", "conflict"}:
        raise ValueError("Expected no_context, aligned, conflict conditions")
    # Count checks alone can miss missing modes/seeds/conditions within a fact.
    for frame, extra in [(raw, ["condition"]), (independent, [])]:
        levels = [frame[k].unique() for k in ["example_id", "decode_seed", *extra, "thinking"]]
        expected = pd.MultiIndex.from_product(levels)
        actual = pd.MultiIndex.from_frame(frame[["example_id", "decode_seed", *extra, "thinking"]])
        if len(expected.difference(actual)):
            raise ValueError("Prediction grid has missing mode/seed/condition pairs")
    if set(raw["example_id"]) != set(pairs["example_id"]):
        raise ValueError("Boundary predictions do not match paired metadata")
    if not independent["correct"].isin([0, 1]).all():
        raise ValueError("Stored correctness must be binary")
    if not independent.groupby("example_id")["row_index"].nunique().eq(1).all():
        raise ValueError("Independent example_id points to multiple facts")
    screening_path = source / "screening.csv"
    if screening_path.exists():
        screened = set(pd.read_csv(screening_path)["row_index"].astype(int))
        if screened & set(independent["row_index"].astype(int)):
            raise ValueError("Headline facts overlap screening set")
    return raw, independent, pairs


def independent_statistics(independent, bootstrap_samples=5000, eval_seed=20260811):
    fact_rates = independent.groupby(["example_id", "thinking"])["correct"].mean().unstack()
    values = (fact_rates[True] - fact_rates[False]).to_numpy()
    rng = np.random.default_rng(eval_seed + 2)
    boot = np.empty(bootstrap_samples)
    for b in range(bootstrap_samples):
        boot[b] = rng.choice(values, size=len(values), replace=True).mean()
    ci = np.quantile(boot, [0.025, 0.975])
    p_value = float(wilcoxon(values, alternative="two-sided").pvalue) if np.any(values != 0) else 1.0
    return {
        "facts": len(fact_rates), "predictions": len(independent),
        "effect_think_minus_nothink_pp": 100 * float(values.mean()),
        "fact_clustered_ci95_pp": [100 * float(ci[0]), 100 * float(ci[1])],
        "paired_fact_wilcoxon_p": p_value,
    }


def boundary_statistics(wide, bootstrap_samples=5000, eval_seed=20260811):
    no, th = wide["NoThink"].to_numpy(), wide["Think"].to_numpy()
    drift = int(np.sum((no != "prior") & (th == "prior")))
    recovery = int(np.sum((no == "prior") & (th != "prior")))
    n_discordant = drift + recovery
    p = float(binomtest(drift, n=n_discordant, p=0.5).pvalue) if n_discordant else 1.0
    rng = np.random.default_rng(eval_seed)
    no_prior, th_prior = (no == "prior").astype(float), (th == "prior").astype(float)
    boot = np.empty(bootstrap_samples)
    for b in range(bootstrap_samples):
        idx = rng.integers(0, len(wide), size=len(wide))
        boot[b] = th_prior[idx].mean() - no_prior[idx].mean()
    ci = np.quantile(boot, [0.025, 0.975])
    return {
        "matched_decode_pairs": len(wide), "drift_to_prior": drift,
        "recovery_from_prior": recovery, "discordant": n_discordant,
        "paired_exact_p": p,
        "prior_reversion_difference_pp": 100 * float(th_prior.mean() - no_prior.mean()),
        "decode_pair_bootstrap_ci95_pp": [100 * float(ci[0]), 100 * float(ci[1])],
        "context_to_prior_flips": int(np.sum((no == "context") & (th == "prior"))),
        "prior_to_context_recoveries": int(np.sum((no == "prior") & (th == "context"))),
    }


def condition_summary(raw):
    rows = []
    for condition in ["no_context", "aligned", "conflict"]:
        for thinking in [False, True]:
            sub = raw[(raw["condition"] == condition) & (raw["thinking"] == thinking)]
            context = 100 * sub["context_correct"].mean() if condition != "no_context" else np.nan
            prior = 100 * sub["prior_reversion"].mean() if condition != "aligned" else np.nan
            other = 100 * (sub["answer_source"] == "other").mean() if condition == "conflict" else 100 - (context if condition == "aligned" else prior)
            rows.append({"condition": condition, "mode": "Think" if thinking else "NoThink", "n": len(sub), "mean_generated_tokens": sub["generated_tokens"].mean(), "context_adherence_pct": context, "prior_reversion_pct": prior, "other_pct": other})
    return pd.DataFrame(rows)


def _plot_tables(independent_table, conflict_table, transitions, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(independent_table["Mode"], independent_table["Independent factual accuracy (%) ↑"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Thinking mode on an untouched PopQA subset")
    fig.tight_layout()
    fig.savefig(out / "independent_factual_recall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    x, width = np.arange(len(conflict_table)), 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - width/2, conflict_table["Context Adherence (%) ↑"], width, label="Context adherence")
    ax.bar(x + width/2, conflict_table["Prior Reversion (%) ↓"], width, label="Prior reversion")
    ax.set_xticks(x, conflict_table["Mode"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Same model under conflicting textual evidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "conflict_mode_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    ordered = ["context → context", "context → prior", "prior → context", "prior → prior"]
    plot = transitions.copy()
    plot["order"] = plot["transition"].map({k: i for i, k in enumerate(ordered)}).fillna(99)
    plot = plot.sort_values(["order", "pct"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(plot["transition"], plot["pct"])
    ax.set_ylabel("Paired examples (%)")
    ax.set_title("Answer-source transitions: NoThink → Think")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out / "paired_transitions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(results_dir, output_dir, bootstrap_samples=5000, eval_seed=20260811):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    source, out = Path(results_dir), Path(output_dir)
    archive = Path(__file__).resolve().parents[2] / "results"
    if source.resolve() == out.resolve():
        raise ValueError("Output must differ from the archived input directory")
    if out.resolve() == archive or archive in out.resolve().parents:
        raise ValueError("Output must differ from the archived input directory and its children")
    raw, independent, pairs = load_results(source)
    out.mkdir(parents=True, exist_ok=True)
    summary = condition_summary(raw)
    summary.to_csv(out / "condition_summary.csv", index=False)
    conf = raw[raw["condition"] == "conflict"].copy()
    wide = conf.pivot(index=["example_id", "decode_seed"], columns="thinking", values="answer_source").reset_index().rename(columns={False: "NoThink", True: "Think"})
    wide["transition"] = wide["NoThink"] + " → " + wide["Think"]
    transitions = wide["transition"].value_counts().rename_axis("transition").reset_index(name="count")
    transitions["pct"] = 100 * transitions["count"] / len(wide)
    transitions.to_csv(out / "transition_counts.csv", index=False)

    control_rows, conflict_rows = [], []
    for condition, label, metric in [
        ("no_context", "No context: prior recall", "prior_reversion"),
        ("aligned", "Aligned context: adherence", "context_correct"),
        ("conflict", "Conflict context: adherence", "context_correct"),
    ]:
        rates = [100 * raw.loc[(raw["condition"] == condition) & (raw["thinking"] == t), metric].mean() for t in [False, True]]
        control_rows.append({"Condition": label, "NoThink": rates[0], "Think": rates[1], "Think_minus_NoThink": rates[1] - rates[0]})
    pd.DataFrame(control_rows).to_csv(out / "control_table.csv", index=False)
    for thinking in [False, True]:
        sub = conf[conf["thinking"] == thinking]
        conflict_rows.append({"Mode": "Think" if thinking else "NoThink", "Context Adherence (%) ↑": 100 * sub["context_correct"].mean(), "Prior Reversion (%) ↓": 100 * sub["prior_reversion"].mean(), "Other (%) ↓": 100 * (sub["answer_source"] == "other").mean(), "Tokens / Answer ↓": sub["generated_tokens"].mean()})
    conflict_table = pd.DataFrame(conflict_rows)
    # The original notebook overwrites its conflict paper table with the
    # independent table. Keep both here under distinct names.
    conflict_table.to_csv(out / "conflict_results_table.csv", index=False)
    grouped = independent.groupby("thinking").agg(accuracy=("correct", "mean"), tokens=("generated_tokens", "mean"))
    independent_table = pd.DataFrame({"Mode": ["NoThink", "Think"], "Independent factual accuracy (%) ↑": [100 * grouped.loc[t, "accuracy"] for t in [False, True]], "Tokens / answer ↓": [grouped.loc[t, "tokens"] for t in [False, True]]})
    independent_table.to_csv(out / "paper_results_table.csv", index=False)
    combined_rows = []
    for thinking in [False, True]:
        def metric(condition, name):
            return 100 * raw.loc[(raw["condition"] == condition) & (raw["thinking"] == thinking), name].mean()
        combined_rows.append({"Mode": "Think" if thinking else "NoThink", "Independent factual accuracy (%)": 100 * grouped.loc[thinking, "accuracy"], "Tokens / factual answer": grouped.loc[thinking, "tokens"], "Aligned-record adherence (%)": metric("aligned", "context_correct"), "Conflict-record adherence (%)": metric("conflict", "context_correct"), "Conflict prior reversion (%)": metric("conflict", "prior_reversion")})
    pd.DataFrame(combined_rows).to_csv(out / "abstract_combined_table.csv", index=False)
    conf_meta = conf.merge(pairs[["example_id", "prop"]], on="example_id", how="left", validate="many_to_one")
    relations = conf_meta.groupby(["prop", "thinking"]).agg(n=("example_id", "size"), context_adherence=("context_correct", "mean"), prior_reversion=("prior_reversion", "mean")).reset_index()
    relations[["context_adherence", "prior_reversion"]] *= 100
    relations.to_csv(out / "relation_summary.csv", index=False)
    audit_columns = ["example_id", "decode_seed", "thinking", "question", "prior_answer", "conflict_value", "final_answer", "answer_source", "thinking_text"]
    conf[audit_columns].to_csv(out / "manual_audit.csv", index=False)
    flip_ids = wide.loc[(wide["NoThink"] == "context") & (wide["Think"] == "prior"), ["example_id", "decode_seed"]]
    conf[audit_columns].merge(flip_ids, on=["example_id", "decode_seed"], how="inner").to_csv(out / "grounding_flip_examples.csv", index=False)
    report = {
        "independent": independent_statistics(independent, bootstrap_samples, eval_seed),
        "boundary": boundary_statistics(wide, bootstrap_samples, eval_seed),
        "token_multiplier": float(grouped.loc[True, "tokens"] / grouped.loc[False, "tokens"]),
        "generation_cap_diagnostic": {
            "think_independent_at_original_256_token_cap": int(((independent["thinking"]) & (independent["generated_tokens"] == 256)).sum()),
            "think_independent_total": int(independent["thinking"].sum()),
            "note": "At-cap does not by itself prove truncation; the archived independent CSV has no thinking text or finish reason.",
        },
        "method_notes": [
            "Headline inference resamples facts after averaging the matched decoding seeds.",
            "Boundary exact test and bootstrap follow original decode-pair procedure; repeated seeds are not independent facts.",
            "Headline stored correctness is retained; aliases are absent from the independent CSV, so canonical-gold relabeling would be incorrect.",
            "Generation caps differ: NoThink 64, Think 256; token counts measure generated tokens, not latency or monetary cost.",
        ],
    }
    (out / "analysis_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot_tables(independent_table, conflict_table, transitions, out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="reproduction/analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=20260811)
    args = parser.parse_args(argv)
    report = analyze(args.results_dir, args.output_dir, args.bootstrap_samples, args.eval_seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
