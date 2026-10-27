"""Validated rule inputs and generation payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import normalize_whitespace, word_count


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
