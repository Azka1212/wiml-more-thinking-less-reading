"""Original prompt, alias-matching, and dataset-cleaning protocol.

Only lexical matching is performed: semantic paraphrases can be missed.
"""

import ast
import re

import numpy as np
import pandas as pd

ARTICLES = {"a", "an", "the"}


def parse_possible_answers(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        return [s]
    return [str(value)]


def prepare_popqa(raw_df):
    df = raw_df.copy()
    df["gold_aliases"] = df["possible_answers"].apply(parse_possible_answers)
    df["gold_aliases"] = df.apply(
        lambda r: list(dict.fromkeys([str(r["obj"])] + r["gold_aliases"])), axis=1,
    )
    pair_counts = df.groupby(["subj", "prop"]).size()
    unique_pairs = pair_counts[pair_counts == 1].index
    mask = pd.MultiIndex.from_frame(df[["subj", "prop"]]).isin(unique_pairs)
    df = df.loc[mask].reset_index(drop=True)
    rel_obj_counts = df.groupby("prop")["obj"].nunique()
    valid_props = rel_obj_counts[rel_obj_counts >= 2].index
    return df[df["prop"].isin(valid_props)].reset_index(drop=True)


def record_prompt(row, record_value):
    return (
        "Use ONLY the fictional record below as the source of truth for this task.\n"
        "Even if it conflicts with your real-world knowledge, answer according to the record.\n\n"
        "Fictional record:\n"
        f"Subject: {row['subj']}\n"
        f"Relation: {row['prop']}\n"
        f"Value: {record_value}\n\n"
        f"Question: {row['question']}\n\n"
        "Return only the answer value. Do not explain your answer."
    )


def no_context_prompt(row):
    return (
        "Answer the factual question below using your knowledge.\n"
        "Return only the answer value. Do not explain your answer.\n\n"
        f"Question: {row['question']}"
    )


def normalize_answer(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(final answer|answer|value)\s*:\s*", "", text)
    text = re.sub(r"^(the answer is|it is)\s+", "", text)
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(t for t in text.split() if t not in ARTICLES).strip()


def alias_set(values):
    return {normalize_answer(v) for v in values if normalize_answer(v)}


def matches_gold(final_text, aliases):
    return normalize_answer(final_text) in alias_set(aliases)


def classify_conflict(final_text, context_value, prior_aliases):
    pred = normalize_answer(final_text)
    if pred == normalize_answer(context_value):
        return "context"
    if pred in alias_set(prior_aliases):
        return "prior"
    return "other"


def parse_bool(value):
    """Read CSV booleans without treating the string 'False' as True."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"Expected boolean, got {value!r}")
