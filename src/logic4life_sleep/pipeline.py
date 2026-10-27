"""Asynchronous generation, embeddings, and result files."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .config import Config
from .metrics import add_local_metrics, cosine_similarity, normalize_whitespace, summarize
from .prompt import GENERATION_PROMPT
from .rules import build_generation_payload

LOGGER = logging.getLogger(__name__)


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


async def run_evaluation(client: Any, rules: list[dict[str, Any]],
                         output_dir: Path, config: Config) -> dict[str, Any]:
    """Run the experiment into a new or empty directory."""
    if not rules:
        raise ValueError("At least one rule is required")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be new or empty; choose another --output-dir")
    (output_dir / "raw_outputs").mkdir(parents=True)
    write_json(output_dir / "config.json", asdict(config))
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
        "config_json": "config.json",
        "detailed_results_csv": "detailed_results.csv",
        "summary_by_rule_csv": "summary_by_rule.csv",
        "summary_overall_json": "summary_overall.json",
        "raw_generation_dir": "raw_outputs",
    }
    write_csv(output_dir / "summary_by_rule.csv", by_rule)
    write_json(output_dir / "summary_overall.json", overall)
    return overall
