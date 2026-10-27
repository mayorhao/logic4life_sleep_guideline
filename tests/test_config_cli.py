"""Configuration and command-line behavior without API access."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from logic4life_sleep.cli import main
from logic4life_sleep.config import Config, load_config
from logic4life_sleep.prompt import GENERATION_PROMPT
from logic4life_sleep.rules import build_generation_payload, load_rules

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_file_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"runs": 3, "confidence": 0.9}))
            result = load_config(path, {"runs": 2})
        self.assertEqual(result.runs, 2)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.embedding_model, Config.embedding_model)

    def test_invalid_settings(self):
        for values in ({"runs": True}, {"runs": 0}, {"runs": "40"},
                       {"seed": 1.5}, {"seed": False}, {"confidence": True},
                       {"confidence": float("nan")}, {"confidence": 1},
                       {"generation_model": None}, {"embedding_model": " "}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values)

    def test_invalid_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for data in ([], {"run": 40}, {"runs": "bad"}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    load_config(path, {"runs": 1})
        with self.assertRaises(ValueError):
            load_config(overrides={"run": 1})

    def test_default_config_file(self):
        self.assertEqual(load_config(ROOT / "config/default.json"), Config())


class InputAndCliTests(unittest.TestCase):
    def test_payload_excludes_reference(self):
        rule = load_rules(ROOT / "examples/rules.json")[0]
        rule["gold_text"] = "REFERENCE_ONLY_TEXT"
        payload = build_generation_payload(rule)
        self.assertNotIn("gold_text", payload)
        self.assertNotIn("REFERENCE_ONLY_TEXT", json.dumps(payload))
        self.assertEqual(set(payload), {"id", "name", "tel_formula"})

    def test_reject_bad_rules(self):
        sample = json.loads((ROOT / "examples/rules.json").read_text())["rules"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            for data in ({"rules": []}, {"rules": [sample, sample]},
                         {"rules": [{**sample, "id": True}]},
                         {"rules": [{**sample, "reference_text": "---"}]},
                         {"rules": [{**sample, "macro_definitions": []}]}):
                path.write_text(json.dumps(data))
                with self.subTest(data=data), self.assertRaises(ValueError):
                    load_rules(path)
        with self.assertRaises(ValueError):
            load_rules(ROOT / "examples/rules.json", [999])

    def test_cli_validation_and_override_without_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = main(["--config", str(ROOT / "config/default.json"),
                               "--input", str(ROOT / "examples/rules.json"),
                               "--runs", "2", "--validate-only"])
        self.assertEqual(status, 0)
        self.assertIn("2 translations requested", output.getvalue())

    def test_print_prompt(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--print-prompt"]), 0)
        self.assertEqual(output.getvalue(), GENERATION_PROMPT + "\n")

    def test_missing_key_is_clear(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), self.assertLogs("logic4life_sleep", "ERROR") as logs:
            self.assertEqual(main(["--input", str(ROOT / "examples/rules.json")]), 1)
        self.assertIn("OPENAI_API_KEY", "\n".join(logs.output))

    def test_both_python_entry_points(self):
        for prefix in ([sys.executable, "-m", "logic4life_sleep"],
                       [sys.executable, str(ROOT / "back_translation.py")]):
            result = subprocess.run([*prefix, "--input", str(ROOT / "examples/rules.json"),
                                     "--validate-only"], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated 1 rules", result.stdout)
