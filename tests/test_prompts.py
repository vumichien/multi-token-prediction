from pathlib import Path

import pytest

from mtp_benchmark.prompts import load_prompts, validate_prompts


def test_validate_prompts_requires_all_categories() -> None:
    with pytest.raises(ValueError, match="missing required categories"):
        validate_prompts([{"id": "only-one", "category": "instruction", "text": "hello"}])


def test_validate_prompts_rejects_duplicate_ids() -> None:
    payload = [
        {"id": "dup", "category": "instruction", "text": "a"},
        {"id": "dup", "category": "creative", "text": "b"},
        {"id": "json", "category": "structured_json", "text": "c"},
        {"id": "code", "category": "code", "text": "d"},
    ]
    with pytest.raises(ValueError, match="Duplicate prompt id"):
        validate_prompts(payload)


def test_load_prompts_from_fixture() -> None:
    prompts = load_prompts(Path("data/prompts.json"))
    assert {prompt["category"] for prompt in prompts} == {
        "structured_json",
        "code",
        "instruction",
        "creative",
    }
