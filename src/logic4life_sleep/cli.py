"""Command-line interface for QEL back-translation."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .pipeline import run_evaluation
from .prompt import GENERATION_PROMPT
from .rules import load_rules

LOGGER = logging.getLogger("logic4life_sleep")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON file containing the rules to evaluate")
    source.add_argument("--print-prompt", action="store_true", help="Print the generation prompt and exit")
    parser.add_argument("--config", type=Path, help="JSON experiment settings; explicit CLI options take precedence")
    parser.add_argument("--validate-only", action="store_true", help="Validate input without API calls")
    parser.add_argument("--rule-ids", type=int, nargs="+", help="Evaluate these IDs in the given order")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="New or empty output directory")
    parser.add_argument("--generation-model")
    parser.add_argument("--embedding-model")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--seed", type=int, help="Bootstrap seed; does not seed model generation")
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
        overrides = {key: getattr(args, key) for key in Config.__dataclass_fields__
                     if getattr(args, key) is not None}
        config = load_config(args.config, overrides)
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
            raise ValueError("Install the project with: python -m pip install -e .") from error
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
