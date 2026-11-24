"""Resumable extraction of the original notebook's full experimental pipeline."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from .config import Config
from .protocol import (
    alias_set, classify_conflict, matches_gold, no_context_prompt,
    normalize_answer, parse_bool, prepare_popqa, record_prompt,
)


def read_checkpoint(path, bool_columns=()):
    if not path.exists():
        return []
    frame = pd.read_csv(path, keep_default_na=False)
    for col in bool_columns:
        frame[col] = frame[col].map(parse_bool)
    return frame.to_dict("records")


def save_rows(rows, path):
    # Write atomically so interrupted checkpoint writes do not truncate a CSV.
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    temp.replace(path)


def screen_facts(df, generate, cfg, out):
    from tqdm.auto import tqdm
    rng = np.random.default_rng(cfg.eval_seed)
    indices = rng.choice(len(df), size=min(cfg.candidate_pool, len(df)), replace=False)
    path = out / "screening.csv"
    rows = read_checkpoint(path, ["prior_known"])
    completed = {int(r["row_index"]) for r in rows}
    for j, idx in enumerate(tqdm(indices, desc="screen prior-known facts")):
        if int(idx) in completed:
            continue
        row = df.iloc[int(idx)]
        result = generate(no_context_prompt(row), False, cfg.eval_seed + j, sample=False)
        rows.append({
            "row_index": int(idx), "question": row["question"],
            "subj": row["subj"], "prop": row["prop"], "gold": row["obj"],
            "screen_answer": result["final"],
            "prior_known": bool(matches_gold(result["final"], row["gold_aliases"])),
        })
        if len(rows) % 10 == 0:
            save_rows(rows, path)
    save_rows(rows, path)
    return pd.DataFrame(rows)


def build_pairs(df, screening, cfg):
    known = screening[screening["prior_known"].map(parse_bool)].reset_index(drop=True)
    if len(known) < cfg.target_examples:
        raise RuntimeError(
            f"Only {len(known)} prior-known examples for target {cfg.target_examples}. "
            "The original 60-item smoke pool may be insufficient. Increase the "
            "candidate pool or reduce the target in a NEW output directory."
        )
    selected = known.sample(n=cfg.target_examples, random_state=cfg.eval_seed)
    rng = np.random.default_rng(cfg.eval_seed + 1)
    records = []
    for ex_id, sr in selected.reset_index(drop=True).iterrows():
        row = df.iloc[int(sr["row_index"])]
        same_rel = df[(df["prop"] == row["prop"]) & (df["obj"].astype(str) != str(row["obj"]))].copy()
        gold_norms = alias_set(row["gold_aliases"])
        same_rel = same_rel[~same_rel["obj"].map(lambda x: normalize_answer(x) in gold_norms)]
        if len(same_rel) == 0:
            continue
        cf = same_rel.iloc[int(rng.integers(0, len(same_rel)))]
        records.append({
            "example_id": int(ex_id), "row_index": int(sr["row_index"]),
            "question": row["question"], "subj": row["subj"], "prop": row["prop"],
            "prior_answer": row["obj"], "prior_aliases": row["gold_aliases"],
            "aligned_value": row["obj"], "conflict_value": str(cf["obj"]),
        })
    paired = pd.DataFrame(records).head(cfg.target_examples).copy()
    # Do not silently publish a smaller experiment if no counterfactual exists.
    if len(paired) != cfg.target_examples:
        raise RuntimeError(f"Constructed {len(paired)} usable pairs; target was {cfg.target_examples}.")
    return paired


def build_independent(df, screening, cfg):
    excluded = set(screening["row_index"].astype(int))
    candidates = np.array([i for i in range(len(df)) if i not in excluded])
    rng = np.random.default_rng(cfg.eval_seed + 1)
    indices = rng.choice(candidates, size=cfg.independent_examples, replace=False)
    fresh = df.iloc[indices].copy().reset_index().rename(columns={"index": "row_index"})
    fresh["example_id"] = np.arange(len(fresh))
    return fresh


def run_boundary(pairs, generate, cfg, out):
    from tqdm.auto import tqdm
    path = out / "raw_predictions.csv"
    rows = read_checkpoint(path, ["thinking"])
    done = {(int(r["example_id"]), int(r["decode_seed"]), str(r["condition"]), bool(r["thinking"])) for r in rows}
    total = len(pairs) * len(cfg.decode_seeds) * 6
    with tqdm(total=total, desc="main paired experiment") as bar:
        for _, ex in pairs.iterrows():
            for decode_seed in cfg.decode_seeds:
                base = int(decode_seed + 100003 * int(ex["example_id"]))
                prompts = {"no_context": no_context_prompt(ex), "aligned": record_prompt(ex, ex["aligned_value"]), "conflict": record_prompt(ex, ex["conflict_value"])}
                for condition, prompt in prompts.items():
                    for thinking in (False, True):
                        key = (int(ex["example_id"]), int(decode_seed), condition, thinking)
                        if key in done:
                            bar.update(1)
                            continue
                        result = generate(prompt, thinking, base)
                        if condition == "conflict":
                            source = classify_conflict(result["final"], ex["conflict_value"], ex["prior_aliases"])
                            context_correct, prior_reversion = int(source == "context"), int(source == "prior")
                        elif condition == "aligned":
                            ok = matches_gold(result["final"], ex["prior_aliases"])
                            source, context_correct, prior_reversion = "context_prior" if ok else "other", int(ok), np.nan
                        else:
                            ok = matches_gold(result["final"], ex["prior_aliases"])
                            source, context_correct, prior_reversion = "prior" if ok else "other", np.nan, int(ok)
                        rows.append({
                            "example_id": int(ex["example_id"]), "decode_seed": int(decode_seed),
                            "condition": condition, "thinking": thinking, "question": ex["question"],
                            "prior_answer": ex["prior_answer"], "conflict_value": ex["conflict_value"],
                            "final_answer": result["final"], "thinking_text": result["thinking"],
                            "answer_source": source, "context_correct": context_correct,
                            "prior_reversion": prior_reversion, "generated_tokens": result["generated_tokens"],
                            "thinking_chars": result["thinking_chars"],
                        })
                        done.add(key)
                        if len(rows) % 25 == 0:
                            save_rows(rows, path)
                        bar.update(1)
    save_rows(rows, path)
    return pd.DataFrame(rows)


def run_independent(fresh, generate, cfg, out):
    from tqdm.auto import tqdm
    path = out / "independent_factual_predictions.csv"
    rows = read_checkpoint(path, ["thinking"])
    done = {(int(r["example_id"]), int(r["decode_seed"]), bool(r["thinking"])) for r in rows}
    with tqdm(total=len(fresh) * len(cfg.decode_seeds) * 2, desc="independent factual recall") as bar:
        for _, ex in fresh.iterrows():
            for decode_seed in cfg.decode_seeds:
                seed = int(decode_seed + 200003 * int(ex["example_id"]))
                for thinking in (False, True):
                    key = (int(ex["example_id"]), int(decode_seed), thinking)
                    if key in done:
                        bar.update(1)
                        continue
                    result = generate(no_context_prompt(ex), thinking, seed)
                    rows.append({
                        "example_id": int(ex["example_id"]), "row_index": int(ex["row_index"]),
                        "decode_seed": int(decode_seed), "thinking": thinking, "question": ex["question"],
                        "gold": ex["obj"], "final_answer": result["final"],
                        "correct": int(matches_gold(result["final"], ex["gold_aliases"])),
                        "generated_tokens": result["generated_tokens"],
                    })
                    done.add(key)
                    if len(rows) % 25 == 0:
                        save_rows(rows, path)
                    bar.update(1)
    save_rows(rows, path)
    return pd.DataFrame(rows)


def write_run_metadata(out, cfg, df):
    config = json.loads(json.dumps(asdict(cfg)))
    path = out / "run_config.json"
    # Mixing a changed configuration with old checkpoints would invalidate pairing.
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("Output directory has a different run_config.json; choose a new directory.")
    existing = [p for p in out.glob("*.csv")]
    if existing and not path.exists():
        raise ValueError("Cannot resume unverified historical CSVs. Choose a new output directory.")
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    serialized = df.to_json(orient="records", force_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    data_path = out / "dataset_sha256.txt"
    if data_path.exists() and data_path.read_text().strip() != digest:
        raise ValueError("Cleaned dataset changed; refusing to resume with incompatible row indices.")
    data_path.write_text(digest + "\n")
    packages = {}
    for name in ["torch", "transformers", "accelerate", "bitsandbytes", "datasets", "numpy", "pandas", "scipy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(), "packages": packages}, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reproduction/full")
    parser.add_argument("--smoke", action="store_true", help="Original 60-item/20-known-fact smoke settings; may not find enough known facts")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--candidate-pool", type=int, help="Override screening-pool size; changes the protocol")
    parser.add_argument("--target-examples", type=int, help="Override boundary-set size; changes the protocol")
    parser.add_argument("--independent-examples", type=int, default=150, help="Default 150 in both full and original smoke settings")
    args = parser.parse_args(argv)
    cfg = Config(output_dir=args.output_dir, smoke=args.smoke, model_revision=args.model_revision, dataset_revision=args.dataset_revision, independent_examples=args.independent_examples)
    if args.candidate_pool is not None:
        cfg.candidate_pool_smoke = cfg.candidate_pool_full = args.candidate_pool
    if args.target_examples is not None:
        cfg.target_examples_smoke = cfg.target_examples_full = args.target_examples
    if min(cfg.candidate_pool, cfg.target_examples, cfg.independent_examples) <= 0:
        parser.error("All sample sizes must be positive")
    out = Path(cfg.output_dir)
    archive_dir = Path(__file__).resolve().parents[2] / "results"
    if out.resolve() == archive_dir or archive_dir in out.resolve().parents:
        parser.error("Use a separate reproduction directory; archived results must be preserved")
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    revision = {"revision": cfg.dataset_revision} if cfg.dataset_revision else {}
    raw = load_dataset("akariasai/PopQA", split="test", **revision)
    df = prepare_popqa(raw.to_pandas())
    write_run_metadata(out, cfg, df)
    from .inference import QwenGenerator
    generate = QwenGenerator(cfg)
    screening = screen_facts(df, generate, cfg, out)
    pairs = build_pairs(df, screening, cfg)
    pairs.to_json(out / "paired_examples.jsonl", orient="records", lines=True, force_ascii=False)
    fresh = build_independent(df, screening, cfg)
    # Additional reproducibility manifest; not present in the archived run.
    fresh.to_json(out / "independent_examples.jsonl", orient="records", lines=True, force_ascii=False)
    run_boundary(pairs, generate, cfg, out)
    run_independent(fresh, generate, cfg, out)
    from .analysis import analyze
    report = analyze(out, out / "analysis", bootstrap_samples=cfg.bootstrap_samples, eval_seed=cfg.eval_seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
