"""End-to-end pipeline tests using mocked HTTP responses."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import AsyncOpenAI, AuthenticationError

from logic4life_sleep.config import Config
from logic4life_sleep import pipeline
from logic4life_sleep.prompt import GENERATION_PROMPT
from logic4life_sleep.rules import load_rules


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_outputs_and_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rules.json"
            path.write_text(json.dumps({"rules": [
                {"id": 1, "name": "First event", "formula": r"\Box Sleep",
                 "reference_text": "REFERENCE_ONLY first event"},
                {"id": 2, "name": "Second event", "formula": r"\Box Theta",
                 "reference_text": "REFERENCE_ONLY second event"},
            ]}))
            rules = load_rules(path)
            config = Config(runs=2, concurrency=2, embedding_batch_size=2, bootstrap_samples=20)
            generation_calls, embedding_calls = [], []

            async def handler(request):
                payload = json.loads(request.content)
                if request.url.path.endswith("/responses"):
                    generation_calls.append(payload)
                    self.assertEqual(payload["instructions"], GENERATION_PROMPT)
                    self.assertNotIn("REFERENCE_ONLY", payload["input"])
                    rule = json.loads(payload["input"].split("Input:\n", 1)[1])
                    self.assertNotIn("gold_text", rule)
                    result = {
                        "id": rule["id"], "name": rule["name"],
                        "back_translation": f"Score {rule['name']} if sleep persists.",
                        "back_translation_clauses": ["Sleep persists."],
                        "assumptions": [], "warnings": [],
                    }
                    return httpx.Response(200, json={
                        "id": "resp_test", "created_at": 0, "model": payload["model"],
                        "object": "response", "parallel_tool_calls": True,
                        "tool_choice": "auto", "tools": [],
                        "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                                    "status": "completed", "content": [{"type": "output_text",
                                    "text": json.dumps(result), "annotations": []}]}],
                    })
                self.assertTrue(request.url.path.endswith("/embeddings"))
                embedding_calls.append(payload)
                entries = [{"object": "embedding", "index": i,
                            "embedding": [float(len(text)), 1.0]}
                           for i, text in enumerate(payload["input"])]
                return httpx.Response(200, json={
                    "object": "list", "model": payload["model"], "data": entries[::-1],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                })

            async with AsyncOpenAI(
                api_key="test-key", base_url="https://example.invalid/v1", max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            ) as client:
                output = root / "results"
                summary = await pipeline.run_evaluation(client, rules, output, config)
                self.assertEqual(summary["total_samples"], 4)
                self.assertEqual(len(generation_calls), 4)
                self.assertEqual(len(embedding_calls), 2)
                self.assertTrue(all(len(call["input"]) == 2 for call in embedding_calls))
                self.assertEqual(json.loads((output / "config.json").read_text()), asdict(config))
                self.assertEqual(len(list((output / "raw_outputs").glob("*.json"))), 4)
                with (output / "detailed_results.csv").open() as handle:
                    records = list(csv.DictReader(handle))
                self.assertEqual([(r["run"], r["id"]) for r in records],
                                 [("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")])
                self.assertTrue(all(not Path(r["raw_output_file"]).is_absolute() for r in records))
                self.assertTrue((output / "summary_by_rule.csv").is_file())
                self.assertTrue((output / "summary_overall.json").is_file())
                with self.assertRaises(ValueError):
                    await pipeline.run_evaluation(client, rules, output, config)

    async def test_transient_errors_retry_but_authentication_errors_do_not(self):
        for status in (429, 401):
            with self.subTest(status=status):
                attempts = []

                async def handler(request):
                    attempts.append(request)
                    if len(attempts) == 1:
                        return httpx.Response(status, json={"error": {"message": "Test error", "type": "test"}})
                    return httpx.Response(200, json={"object": "list", "model": "test",
                        "data": [{"index": 0, "object": "embedding", "embedding": [1.0, 0.0]}]})

                async with AsyncOpenAI(
                    api_key="test-key", base_url="https://example.invalid/v1", max_retries=0,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                ) as client:
                    with patch.object(pipeline.asyncio, "sleep", new_callable=AsyncMock):
                        if status == 429:
                            result = await pipeline.embed_texts_unique(client, ["test"], Config())
                            self.assertEqual(result, [[1.0, 0.0]])
                            self.assertEqual(len(attempts), 2)
                        else:
                            with self.assertRaises(AuthenticationError):
                                await pipeline.embed_texts_unique(client, ["test"], Config())
                            self.assertEqual(len(attempts), 1)
