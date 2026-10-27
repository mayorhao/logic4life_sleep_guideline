"""Small numerical checks for the published metric definitions."""

import json
import unittest

from logic4life_sleep.config import Config
from logic4life_sleep.metrics import (
    add_local_metrics, bootstrap_mean_ci, cosine_similarity,
    lexical_similarity_ratio, normalize_whitespace, rouge_l_f1, summarize,
)


class MetricTests(unittest.TestCase):
    def test_text_and_overlap(self):
        self.assertEqual(normalize_whitespace("  8 µV\n threshold  "), "8 uV threshold")
        self.assertEqual(lexical_similarity_ratio("sleep", "sleep"), 100)
        self.assertEqual(rouge_l_f1("sleep", "Sleep"), 0)
        self.assertAlmostEqual(rouge_l_f1("a b c", "a x c"), 200 / 3)
        self.assertEqual(rouge_l_f1("", "sleep"), 0)

    def test_cosine(self):
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0)
        self.assertAlmostEqual(cosine_similarity([1, 1], [2, 2]), 1)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan")], [1])

    def test_bootstrap(self):
        result = bootstrap_mean_ci([1, 2, 3], n_boot=100, seed=42)
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["std"], 1)
        self.assertLessEqual(result["ci_lower"], 2)
        self.assertGreaterEqual(result["ci_upper"], 2)
        self.assertEqual(result, bootstrap_mean_ci([1, 2, 3], n_boot=100, seed=42))
        self.assertEqual(bootstrap_mean_ci([7], n_boot=10),
                         {"mean": 7, "std": 0, "ci_lower": 7, "ci_upper": 7})

    def test_identical_translation_summary(self):
        records = [{"id": 1, "name": "Example", "run": run,
                    "gold_text": "sleep occurs", "back_translation": "sleep occurs",
                    "tel_formula": r"\Box Sleep", "macro_definitions_json": json.dumps({}),
                    "semantic_cosine": 100} for run in (1, 2)]
        by_rule, overall = summarize(add_local_metrics(records), Config(bootstrap_samples=10))
        self.assertEqual(overall["total_samples"], 2)
        self.assertEqual(overall["runs"], 2)
        self.assertEqual(by_rule[0]["n"], 2)
        self.assertEqual(overall["back_reduction_pct_mean"], 0)
        for metric in ("rougeL_f1", "lexical_seqmatch", "semantic_cosine"):
            self.assertEqual(overall[metric + "_mean"], 100)
            self.assertEqual(overall[metric + "_ci_lower"], 100)
            self.assertEqual(overall[metric + "_ci_upper"], 100)
