import pytest

from mtp_benchmark.model_loader import default_assistant_model, execute_policy_attempts, policy_attempts


def test_policy_attempts_auto_order() -> None:
    assert policy_attempts("auto") == ["8bit", "4bit", "cpu_offload"]


def test_execute_policy_attempts_falls_back() -> None:
    result, history = execute_policy_attempts(
        ["8bit", "4bit"],
        lambda name: {"policy": name} if name == "4bit" else (_ for _ in ()).throw(RuntimeError("nope")),
    )
    assert result["policy"] == "4bit"
    assert [item["status"] for item in history] == ["failed", "loaded"]


def test_default_assistant_model_suffix() -> None:
    assert default_assistant_model("google/gemma-4-E2B-it") == "google/gemma-4-E2B-it-assistant"


def test_policy_attempts_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unsupported --policy"):
        policy_attempts("bad")
