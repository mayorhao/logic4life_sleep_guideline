#!/usr/bin/env python3
"""Generate and evaluate English back-translations of QEL sleep-scoring rules.

Companion code for "Ensemble Logic for Symbolic Representation of Sleep
Medicine Guidelines" (AMIA 2026), https://arxiv.org/abs/2607.21331.

Python 3.10 or newer is required. Generation requires openai and pydantic;
input validation and local metrics use only the Python standard library.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
import re
import statistics
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)
GENERATION_PROMPT = (
    "\n"
    "You are reconstructing a formal clinician-facing event definition from a temporal logic formula.\n"
    "\n"
    "Your job is to convert the formula into a precise clinical rule describing the observable physiological conditions under which an event is present. The output must structurally mirror standardized medical scoring guidelines strictly based on the provided logic, without relying on or hallucinating any external knowledge.\n"
    "\n"
    "======================================================================\n"
    "PRIMARY OBJECTIVE\n"
    "======================================================================\n"
    "Given a temporal logic formula, produce a clinical rule that describes:\n"
    "- what physiological signals or states are present\n"
    "- what thresholds apply\n"
    "- how long signals persist (duration)\n"
    "- what signals must be absent (negation)\n"
    "- what temporal ordering or co-occurrence is required\n"
    "\n"
    "======================================================================\n"
    "HARD CONSTRAINTS & ZERO-HALLUCINATION POLICY\n"
    "======================================================================\n"
    "1. NO EXPLANATIONS: Do NOT explain the formalism, variables (x, u, v, l), or logical operators.\n"
    "2. STRICT TERMINOLOGY MAPPING: You MUST use the exact physiological phrases provided in the \"Terminology Map\" below. Do not paraphrase them or use synonyms.\n"
    "3. EXACT THRESHOLDS: Preserve all numeric thresholds and duration constraints exactly as written in the logic.\n"
    "4. NO HALLUCINATION: Do NOT add clinical conditions, diagnostic sensor types, or exceptions that are NOT explicitly supported by the formula or the provided macro_definitions.\n"
    "5. NO EXTERNAL KNOWLEDGE: Rely ONLY on the provided logic and terminology map. Do not infer rules from pre-existing scoring manuals.\n"
    "6. When applying logical negations to glossary states, apply proper English grammar. For example, instead of 'the subject is awake is absent', output 'the subject is not awake'.\n"
    "7. ABSOLUTELY NO ALGEBRAIC VARIABLES: Do not output variable names like u, v, w, x, or \\ell. Translate inequalities (e.g., 5 <= u <= 90) into clinical duration ranges (e.g., 'lasting between 5 and 90 seconds'.\n"
    "8. If a macro starts with \\varphi_ (e.g., \\varphi_LM) and is not in the definitions, translate it as 'the [Event Name] event' (e.g., 'the leg movement event') rather than printing the raw symbol.\n"
    "9. ALGEBRAIC ELIMINATION: You must NEVER print raw local variables (u, v, w, l, d) or mathematical inequalities (like \"0 ≤ v ≤ u\" or \"L/2 < \\ell \\le L\"). Instead, convert the logic into fluent English bounding conditions. \n"
    "   - BAD: \"snoring is absent continuously for v seconds where 0 ≤ v ≤ u\"\n"
    "   - GOOD: \"snoring is absent at some time during the event\"\n"
    "   - BAD: \"lasts greater than L/2 seconds and at most L seconds\"\n"
    "   - GOOD: \"occupies the majority of the epoch\"\n"
    "\n"
    "======================================================================\n"
    "REQUIRED STYLE & SENTENCE STRUCTURE\n"
    "======================================================================\n"
    "Write the result as an active, standardized clinical scoring instruction. \n"
    "\n"
    "Preferred opening styles (Use the \"name\" of the event from the input):\n"
    "- \"Score [Event Name] if/when ...\"\n"
    "- \"Score an epoch as [Stage Name] if ...\"\n"
    "\n"
    "Translate specific temporal logic operators into their standard clinical phrasing:\n"
    "- \\Box (Continuous): \"continuously for at least X seconds\" or \"throughout the entire period\".\n"
    "- \\Diamond (Eventual): \"occurs within\" or \"accompanied by\".\n"
    "- Negation (\\neg): \"is absent\" or \"does not occur\".\n"
    "======================================================================\n"
    "META-PARAMETER RESOLUTION MAP\n"
    "======================================================================\n"
    "When the following variables or algebraic expressions appear, replace them with their clinical AASM equivalents. Do NOT output algebraic expressions (like L/2 or u+ε) in the final text.\n"
    "\n"
    "- L          -> \"the epoch\"\n"
    "- L/2        -> \"the half of the epoch\" (or \"half of the epoch\")\n"
    "- \\epsilon   -> \"immediately following\" or \"shortly after\" (represents an arbitrarily small positive delay, do not output the word epsilon)\n"
    "- x, x-L     ->  Used for indexing prior/current epochs. Translate relative timing (e.g., condition at x-L and condition at x) as \"in the previous epoch\" and \"in the current epoch\".\n"
    "======================================================================\n"
    "STRICT TERMINOLOGY MAP\n"
    "======================================================================\n"
    "When the following predicates appear in the logic or macros, map them strictly to these phrases:\n"
    "\n"
    "[Sleep States]\n"
    "Sleep           -> \"the subject is asleep\"\n"
    "Wake            -> \"the subject is awake\"\n"
    "NREM            -> \"non-REM sleep is present\"\n"
    "REM             -> \"REM sleep is present\"\n"
    "N3              -> \"stage N3 sleep is present\"\n"
    "\n"
    "[EEG & Brainwaves]\n"
    "EEGAbrupt       -> \"abrupt shift in EEG frequency\"\n"
    "Alpha           -> \"alpha EEG activity is present\"\n"
    "Theta           -> \"theta EEG activity is present\"\n"
    "EEGFreqGT16     -> \"EEG activity above 16 Hz is present\"\n"
    "Spindle         -> \"sleep spindle waveform is present\"\n"
    "KC              -> \"K-complex waveform is present\"\n"
    "\n"
    "[Respiration & Airflow]\n"
    "FlowDrop90      -> \"airflow decreases by at least 90% relative to baseline\"\n"
    "FlowDrop30      -> \"airflow decreases by at least 30% relative to baseline\"\n"
    "FlowBelowThr    -> \"inspiratory airflow flattening or reduced inspiratory airflow is present\"\n"
    "EffortAboveThr  -> \"inspiratory respiratory effort is present\"\n"
    "Snore           -> \"snoring is present\"\n"
    "PhaseDiffAboveThr -> \"thoraco-abdominal paradoxical movement is present\"\n"
    "Desat3          -> \"oxygen saturation decreases by at least 3%\"\n"
    "Desat4          -> \"oxygen saturation decreases by at least 4%\"\n"
    "\n"
    "[Movement & EMG]\n"
    "ChinEMGAboveThr -> \"chin EMG activity increases above baseline or threshold\"\n"
    "MajorBodyMove   -> \"major body movement is present\"\n"
    "EMGAboveRestThr -> \"leg EMG activity rises above resting baseline\"\n"
    "EMGAbove8       -> \"leg EMG increases by at least 8 microvolts above resting EMG\"\n"
    "EMGAbove2       -> \"leg EMG exceeds 2 microvolts above resting EMG\"\n"
    "[marcro definitions expressed by varphi_[event_name]]\n"
    "Arousal         -> \"an arousal event is present\"\n"
    "apnea           -> \"an apnea event is present\"\n"
    "hypopnea        -> \"a hypopnea event is present\"\n"
    "LM              -> \"a leg movement event is present\"\n"
    "\n"
    "======================================================================\n"
    "EXAMPLE (STRICTLY DERIVED FROM INPUT)\n"
    "======================================================================\n"
    "Example input:\n"
    "{\n"
    "  \"id\": 999,\n"
    "  \"name\": \"Atypical Theta Event\",\n"
    "  \"tel_formula\": \"\\\\varphi_{\\\\text{dummy}} := \\\\exists x \\\\exists u \\\\exists v \\\\Big((\\\\Box_u \\\\text{Theta})_x \\\\wedge (\\\\Diamond_v \\\\text{MajorBodyMove})_x \\\\wedge \\\\neg(\\\\Box_u \\\\text{Snore})_x\\\\Big),\\\\; u \\\\ge 10,\\\\; v \\\\le 5\",\n"
    "  \"macro_definitions\": {}\n"
    "}\n"
    "\n"
    "Example output:\n"
    "{\n"
    "  \"id\": 999,\n"
    "  \"name\": \"Atypical Theta Event\",\n"
    "  \"back_translation\": \"Score Atypical Theta Event if theta EEG activity is present continuously for at least 10 seconds, accompanied by major body movement occurring within 5 seconds, and snoring is absent throughout the entire period.\",\n"
    "  \"back_translation_clauses\": [\n"
    "    \"Theta EEG activity is present continuously for at least 10 seconds.\",\n"
    "    \"The event is accompanied by major body movement occurring within 5 seconds.\",\n"
    "    \"Snoring is absent throughout the entire period.\"\n"
    "  ],\n"
    "  \"assumptions\": [],\n"
    "  \"warnings\": []\n"
    "}\n"
    "\n"
    "======================================================================\n"
    "INPUT\n"
    "======================================================================\n"
    "[Waiting for JSON input...]\n"
)

METRICS = (
    "back_over_gold", "back_reduction_pct", "qel_over_gold",
    "qel_reduction_pct", "lexical_seqmatch", "rougeL_f1", "semantic_cosine",
)


@dataclass(frozen=True)
class Config:
    generation_model: str = "gpt-5-mini-2025-08-07"
    embedding_model: str = "text-embedding-3-small"
    runs: int = 40
    concurrency: int = 16
    embedding_batch_size: int = 128
    max_attempts: int = 5
    bootstrap_samples: int = 5000
    confidence: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("runs", "concurrency", "embedding_batch_size",
                     "max_attempts", "bootstrap_samples"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.generation_model.strip() or not self.embedding_model.strip():
            raise ValueError("Model names must not be empty")


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


def load_rules(path: Path, rule_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Load public JSON inputs and validate them before any API calls."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise ValueError("Input must be an object with a 'rules' list")
    if not data["rules"]:
        raise ValueError("The rules list must not be empty")
    rules = []
    seen = set()
    for item in data["rules"]:
        if not isinstance(item, dict):
            raise ValueError("Each rule must be an object")
        rid = item.get("id")
        if type(rid) is not int or rid < 1 or rid in seen:
            raise ValueError("Rule IDs must be unique positive integers")
        seen.add(rid)
        for field in ("name", "formula", "reference_text"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Rule {rid}: {field} must be a nonempty string")
        gold_text = normalize_whitespace(item["reference_text"])
        if not word_count(gold_text):
            raise ValueError(f"Rule {rid}: reference_text must contain words")
        macros = item.get("macro_definitions", {})
        if not isinstance(macros, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
            for k, v in macros.items()
        ):
            raise ValueError(f"Rule {rid}: macro_definitions must map strings to strings")
        supporting = item.get("supporting_rule_ids", [])
        if not isinstance(supporting, list) or not all(
            type(value) is int and value > 0 for value in supporting
        ):
            raise ValueError(f"Rule {rid}: supporting_rule_ids must be positive integers")
        warnings = item.get("formula_warnings", [])
        if not isinstance(warnings, list) or not all(isinstance(v, str) for v in warnings):
            raise ValueError(f"Rule {rid}: formula_warnings must be a list of strings")
        rules.append({
            "id": rid, "name": item["name"].strip(),
            "tel_formula": item["formula"].strip(), "gold_text": gold_text,
            "macro_definitions": macros, "supporting_rule_ids": supporting,
            "formula_warnings": warnings,
        })
    if rule_ids is not None:
        if not rule_ids or len(set(rule_ids)) != len(rule_ids):
            raise ValueError("Requested rule IDs must be nonempty and unique")
        missing = set(rule_ids) - seen
        if missing:
            raise ValueError(f"Requested rule IDs are missing: {sorted(missing)}")
        indexed = {rule["id"]: rule for rule in rules}
        rules = [indexed[rid] for rid in rule_ids]
    return rules


def build_generation_payload(rule: dict[str, Any]) -> dict[str, Any]:
    """Keep reference narratives out of the model's generation input."""
    payload = {key: rule[key] for key in ("id", "name", "tel_formula")}
    for key in ("macro_definitions", "supporting_rule_ids", "formula_warnings"):
        if rule.get(key):
            payload[key] = rule[key]
    return payload


def response_schema() -> type:
    from pydantic import BaseModel, Field

    class BackTranslationResult(BaseModel):
        id: int
        name: str
        back_translation: str
        back_translation_clauses: list[str] = Field(default_factory=list)
        assumptions: list[str] = Field(default_factory=list)
        warnings: list[str] = Field(default_factory=list)

    return BackTranslationResult


async def request_with_retry(call: Callable, config: Config, **kwargs: Any) -> Any:
    from openai import APIConnectionError, APIStatusError

    for attempt in range(config.max_attempts):
        try:
            return await call(**kwargs)
        except (APIConnectionError, APIStatusError) as error:
            retryable = isinstance(error, APIConnectionError) or (
                error.status_code in (408, 409, 429) or error.status_code >= 500
            )
            if not retryable or attempt + 1 == config.max_attempts:
                raise
            delay = min(1.5 * 2 ** attempt, 20.0)
            LOGGER.warning("%s; retrying in %.1f seconds (%d/%d)",
                           type(error).__name__, delay, attempt + 1, config.max_attempts)
            await asyncio.sleep(delay)
    raise RuntimeError("Request did not complete")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


async def generate_all(client: Any, rules: list[dict[str, Any]], output_dir: Path,
                       config: Config) -> list[dict[str, Any]]:
    schema = response_schema()
    semaphore = asyncio.Semaphore(config.concurrency)

    async def generate_one(run: int, rule: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            LOGGER.info("Generating rule %s, run %s/%s", rule["id"], run, config.runs)
            response = await request_with_retry(
                client.responses.parse, config,
                model=config.generation_model, instructions=GENERATION_PROMPT,
                input="Now reconstruct the rule from the following input.\n\nInput:\n"
                      + json.dumps(build_generation_payload(rule), ensure_ascii=False, indent=2),
                text_format=schema,
            )
            result = response.output_parsed
            if result is None:
                raise ValueError(f"Rule {rule['id']}: no structured translation returned")
            if result.id != rule["id"] or result.name != rule["name"]:
                raise ValueError(f"Rule {rule['id']}: response identifier or name does not match")
            text = normalize_whitespace(result.back_translation)
            if not text:
                text = normalize_whitespace(" ".join(result.back_translation_clauses))
            if not text:
                raise ValueError(f"Rule {rule['id']}: empty translation returned")
            relative_path = Path("raw_outputs") / f"run_{run:02d}_rule_{rule['id']:02d}.json"
            write_json(output_dir / relative_path, result.model_dump())
            return {
                "run": run, "id": rule["id"], "name": rule["name"],
                "gold_text": rule["gold_text"], "tel_formula": rule["tel_formula"],
                "macro_definitions_json": json.dumps(rule["macro_definitions"], ensure_ascii=False),
                "supporting_rule_ids_json": json.dumps(rule["supporting_rule_ids"]),
                "formula_warnings_json": json.dumps(rule["formula_warnings"], ensure_ascii=False),
                "back_translation": text,
                "back_translation_clauses": json.dumps(result.back_translation_clauses, ensure_ascii=False),
                "assumptions": json.dumps(result.assumptions, ensure_ascii=False),
                "warnings": json.dumps(result.warnings, ensure_ascii=False),
                "raw_output_file": relative_path.as_posix(),
            }

    tasks = [asyncio.create_task(generate_one(run, rule))
             for run in range(1, config.runs + 1) for rule in rules]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def embed_texts_unique(client: Any, texts: list[str],
                             config: Config) -> list[list[float]]:
    normalized = [normalize_whitespace(text) for text in texts]
    unique_texts = list(dict.fromkeys(normalized))
    vectors = []
    for start in range(0, len(unique_texts), config.embedding_batch_size):
        batch = unique_texts[start:start + config.embedding_batch_size]
        response = await request_with_retry(
            client.embeddings.create, config, model=config.embedding_model, input=batch,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(batch))):
            raise ValueError("Embedding response has missing or duplicate indices")
        vectors.extend(item.embedding for item in ordered)
    mapping = dict(zip(unique_texts, vectors))
    return [mapping[text] for text in normalized]


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


async def run_evaluation(client: Any, rules: list[dict[str, Any]],
                         output_dir: Path, config: Config) -> dict[str, Any]:
    """Run the experiment into a new or empty directory."""
    if not rules:
        raise ValueError("At least one rule is required")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be new or empty; choose another --output-dir")
    (output_dir / "raw_outputs").mkdir(parents=True)
    records = add_local_metrics(await generate_all(client, rules, output_dir, config))
    LOGGER.info("Computing semantic similarity")
    predictions = await embed_texts_unique(client, [r["back_translation"] for r in records], config)
    references = await embed_texts_unique(client, [r["gold_text"] for r in records], config)
    for row, prediction, reference in zip(records, predictions, references):
        row["semantic_cosine"] = cosine_similarity(prediction, reference) * 100.0
    write_csv(output_dir / "detailed_results.csv", records)
    LOGGER.info("Computing bootstrap summaries")
    by_rule, overall = summarize(records, config)
    overall["output_files"] = {
        "detailed_results_csv": "detailed_results.csv",
        "summary_by_rule_csv": "summary_by_rule.csv",
        "summary_overall_json": "summary_overall.json",
        "raw_generation_dir": "raw_outputs",
    }
    write_csv(output_dir / "summary_by_rule.csv", by_rule)
    write_json(output_dir / "summary_overall.json", overall)
    return overall


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON file containing the rules to evaluate")
    source.add_argument("--print-prompt", action="store_true", help="Print the generation prompt and exit")
    parser.add_argument("--validate-only", action="store_true", help="Validate input without API calls")
    parser.add_argument("--rule-ids", type=int, nargs="+", help="Evaluate these IDs in the given order")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="New or empty output directory")
    parser.add_argument("--generation-model", default=Config.generation_model)
    parser.add_argument("--embedding-model", default=Config.embedding_model)
    parser.add_argument("--runs", type=int, default=Config.runs)
    parser.add_argument("--concurrency", type=int, default=Config.concurrency)
    parser.add_argument("--embedding-batch-size", type=int, default=Config.embedding_batch_size)
    parser.add_argument("--max-attempts", type=int, default=Config.max_attempts)
    parser.add_argument("--bootstrap-samples", type=int, default=Config.bootstrap_samples)
    parser.add_argument("--confidence", type=float, default=Config.confidence)
    parser.add_argument("--seed", type=int, default=Config.seed, help="Bootstrap seed; does not seed model generation")
    parser.add_argument("--verbose", action="store_true", help="Log progress for each rule and run")
    args = parser.parse_args(argv)
    if args.print_prompt and (args.validate_only or args.rule_ids is not None):
        parser.error("--validate-only and --rule-ids require --input")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    LOGGER.setLevel(logging.INFO if args.verbose else logging.WARNING)
    if args.print_prompt:
        print(GENERATION_PROMPT)
        return 0
    try:
        config = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
        rules = load_rules(args.input, args.rule_ids)
        if args.validate_only:
            print(f"Validated {len(rules)} rules; {config.runs * len(rules)} translations requested.")
            return 0
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("Set OPENAI_API_KEY in your environment before generating translations")
        try:
            from openai import AsyncOpenAI
            import pydantic
        except ImportError as error:
            raise ValueError("Install dependencies with: python -m pip install openai 'pydantic>=2'") from error
        if int(pydantic.__version__.split(".")[0]) < 2:
            raise ValueError("pydantic 2 or newer is required")

        async def execute() -> dict[str, Any]:
            async with AsyncOpenAI(max_retries=0) as client:
                return await run_evaluation(client, rules, args.output_dir, config)

        overall = asyncio.run(execute())
        print(f"Saved {overall['total_samples']} translations to {args.output_dir}")
        for metric in ("semantic_cosine", "lexical_seqmatch", "rougeL_f1"):
            print(f"{metric}: {overall[metric + '_mean']:.2f} "
                  f"({100 * config.confidence:g}% CI {overall[metric + '_ci_lower']:.2f}–"
                  f"{overall[metric + '_ci_upper']:.2f})")
        return 0
    except (ValueError, OSError) as error:
        LOGGER.error("%s", error)
        return 1
    except Exception as error:
        # API exception bodies can contain request data. Report only the type.
        LOGGER.error("Evaluation failed (%s). Check model access and service availability.",
                     type(error).__name__)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; completed translations remain in raw_outputs")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
