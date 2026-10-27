"""Similarity metrics and bootstrap summaries."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from difflib import SequenceMatcher
from typing import Any

from .config import Config


METRICS = (
    "back_over_gold", "back_reduction_pct", "qel_over_gold",
    "qel_reduction_pct", "lexical_seqmatch", "rougeL_f1", "semantic_cosine",
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00b5", "u")).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9%\.]+", text))


def formula_surface_token_count(formula: str) -> int:
    """Count formula surface tokens; this is not a model tokenizer."""
    pattern = r"""
        \\[A-Za-z]+      |
        [A-Za-z_]+       |
        \d+\.\d+|\d+     |
        >=|<=|:=|\\ge|\\le|\\neg|\\vee|\\wedge |
        [{}()\[\]_,+\-*/=]
    """
    return len(re.findall(pattern, formula, flags=re.VERBOSE))


def count_formula_bundle_tokens(formula: str, macros: dict[str, str]) -> int:
    return formula_surface_token_count(formula) + sum(
        formula_surface_token_count(key) + formula_surface_token_count(value)
        for key, value in macros.items()
    )


def lexical_similarity_ratio(prediction: str, reference: str) -> float:
    """Character-level SequenceMatcher ratio, including its default autojunk."""
    return 100.0 * SequenceMatcher(None, prediction, reference).ratio()


def lcs_length(a_tokens: list[str], b_tokens: list[str]) -> int:
    previous = [0] * (len(b_tokens) + 1)
    for token in a_tokens:
        current = [0]
        for index, other in enumerate(b_tokens, 1):
            current.append(previous[index - 1] + 1 if token == other
                           else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Case-sensitive, whitespace-tokenized ROUGE-L F1 on a 0–100 scale."""
    pred_tokens, ref_tokens = prediction.split(), reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision, recall = lcs / len(pred_tokens), lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 100.0 * 2 * precision * recall / (precision + recall)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        raise ValueError("Embedding vectors must have the same nonzero length")
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("Embedding vectors must contain finite values")
    norm_a = math.sqrt(sum(value * value for value in a))
    norm_b = math.sqrt(sum(value * value for value in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def bootstrap_mean_ci(values: list[float], n_boot: int = 5000,
                      ci: float = 0.95, seed: int = 42) -> dict[str, float | None]:
    """Percentile bootstrap with replacement and fixed order-statistic bounds."""
    if n_boot < 1 or not 0 < ci < 1:
        raise ValueError("Invalid bootstrap sample count or confidence level")
    vals = [float(value) for value in values
            if value is not None and math.isfinite(float(value))]
    if not vals:
        return dict(mean=None, ci_lower=None, ci_upper=None, std=None)
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    alpha = (1 - ci) / 2
    lower = max(0, int(alpha * n_boot))
    upper = min(n_boot - 1, int((1 - alpha) * n_boot) - 1)
    return {
        "mean": sum(vals) / n,
        "ci_lower": means[lower],
        "ci_upper": means[upper],
        "std": statistics.stdev(vals) if n > 1 else 0.0,
    }


def add_local_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        row = dict(record)
        gold, back = row["gold_text"], row["back_translation"]
        gold_count, back_count = word_count(gold), word_count(back)
        if gold_count == 0:
            raise ValueError("Reference text must contain words")
        qel_count = count_formula_bundle_tokens(
            row["tel_formula"], json.loads(row["macro_definitions_json"] or "{}"),
        )
        row.update({
            "gold_word_count": gold_count, "back_word_count": back_count,
            "qel_surface_token_count": qel_count,
            "back_over_gold": back_count / gold_count,
            "qel_over_gold": qel_count / gold_count,
            "back_reduction_pct": (1 - back_count / gold_count) * 100,
            "qel_reduction_pct": (1 - qel_count / gold_count) * 100,
            "lexical_seqmatch": lexical_similarity_ratio(back, gold),
            "rougeL_f1": rouge_l_f1(back, gold),
        })
        output.append(row)
    return output


def summarize(records: list[dict[str, Any]], config: Config
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records:
        raise ValueError("Cannot summarize an empty result set")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((int(row["id"]), row["name"]), []).append(row)

    def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        stats = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite values in {metric}")
            result = bootstrap_mean_ci(values, config.bootstrap_samples,
                                       config.confidence, config.seed)
            stats.update({f"{metric}_{key}": value for key, value in result.items()})
        return stats

    by_rule = []
    for (rid, name), rows in sorted(groups.items()):
        by_rule.append({
            "id": rid, "name": name, "n": len(rows),
            "gold_word_count": rows[0]["gold_word_count"],
            "qel_surface_token_count": rows[0]["qel_surface_token_count"],
            **metric_summary(rows),
        })
    overall = {
        "generation_model": config.generation_model,
        "semantic_model": config.embedding_model,
        "runs": len({int(row["run"]) for row in records}),
        "rule_count": len(groups), "total_samples": len(records),
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_ci": config.confidence, "bootstrap_seed": config.seed,
        **metric_summary(records),
    }
    return by_rule, overall
