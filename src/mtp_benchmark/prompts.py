from __future__ import annotations

import json
from pathlib import Path

REQUIRED_CATEGORIES = {"structured_json", "code", "instruction", "creative"}


def load_prompts(path: Path) -> list[dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Prompt file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prompt file is not valid JSON: {path}: {exc}") from exc
    return validate_prompts(payload)


def validate_prompts(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Prompts JSON must be a non-empty list.")
    seen_ids: set[str] = set()
    categories: set[str] = set()
    prompts: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt entry {index} must be an object.")
        prompt_id = _read_field(item, "id", index)
        category = _read_field(item, "category", index)
        text = _read_field(item, "text", index)
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        if category not in REQUIRED_CATEGORIES:
            raise ValueError(
                f"Prompt {prompt_id} has unsupported category '{category}'. "
                f"Expected one of {sorted(REQUIRED_CATEGORIES)}."
            )
        seen_ids.add(prompt_id)
        categories.add(category)
        prompts.append({"id": prompt_id, "category": category, "text": text})
    missing = sorted(REQUIRED_CATEGORIES - categories)
    if missing:
        raise ValueError(f"Prompt suite is missing required categories: {missing}")
    return prompts


def _read_field(item: dict[str, object], field: str, index: int) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt entry {index} is missing non-empty '{field}'.")
    return value.strip()
