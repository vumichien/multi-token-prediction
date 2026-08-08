"""Boundary behaviour of prompt/completion tokenisation.

`build_example` decides which positions carry a training signal. If the mask
boundary slipped by one, the heads would train on prompt tokens and every
accuracy number in the article would be measured against the wrong population,
with nothing failing loudly. These tests pin the boundary with a stub tokenizer
so they run on CPU in milliseconds.
"""

from __future__ import annotations

import pytest

from mtp_finetune.base_runtime import IGNORE_INDEX, build_example


class StubTokenizer:
    """Chat template emits [1, 2] around the prompt; encode maps chars to ids."""

    def apply_chat_template(self, conversation, **kwargs):
        del kwargs
        content = conversation[0]["content"]
        return [1, 2] + [ord(character) for character in content] + [3]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


def test_prompt_positions_are_masked_and_completion_positions_are_not(tokenizer) -> None:
    input_ids, labels = build_example(tokenizer, "ab", "xy", max_length=64)
    # prompt = [1, 2, 'a', 'b', 3] -> 5 masked positions, then the completion.
    assert input_ids == [1, 2, ord("a"), ord("b"), 3, ord("x"), ord("y")]
    assert labels == [IGNORE_INDEX] * 5 + [ord("x"), ord("y")]


def test_labels_and_input_ids_always_line_up(tokenizer) -> None:
    input_ids, labels = build_example(tokenizer, "hello", "world", max_length=64)
    assert len(input_ids) == len(labels)
    # Every unmasked label must equal the token at the same position: the loss
    # shift happens in offset_cross_entropy, not here.
    for index, label in enumerate(labels):
        if label != IGNORE_INDEX:
            assert label == input_ids[index]


def test_truncation_keeps_the_mask_aligned(tokenizer) -> None:
    input_ids, labels = build_example(tokenizer, "ab", "xyzw", max_length=6)
    assert len(input_ids) == 6 and len(labels) == 6
    assert labels == [IGNORE_INDEX] * 5 + [ord("x")]


def test_completion_truncated_away_entirely_is_dropped(tokenizer) -> None:
    # max_length lands inside the prompt, so there is no target left at all.
    assert build_example(tokenizer, "ab", "xyz", max_length=5) is None
    assert build_example(tokenizer, "ab", "xyz", max_length=3) is None


def test_empty_completion_is_dropped(tokenizer) -> None:
    assert build_example(tokenizer, "ab", "", max_length=64) is None
