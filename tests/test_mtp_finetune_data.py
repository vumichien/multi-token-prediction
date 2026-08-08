import json
import zlib
from collections import Counter

import pytest
import torch

from mtp_benchmark.prompts import REQUIRED_CATEGORIES
from mtp_finetune.base_runtime import IGNORE_INDEX, collate
from mtp_finetune.dataset import load_samples, split_samples
from mtp_finetune.generate_data import (
    batch_seed_offset,
    expand_completions,
    rewrite_in_bank_order,
)
from mtp_finetune.prompt_bank import bank_size, build_prompt_bank


def test_prompt_bank_ids_are_unique() -> None:
    prompts = build_prompt_bank()
    assert len(prompts) == bank_size()
    assert len({prompt["id"] for prompt in prompts}) == len(prompts)


def test_prompt_bank_uses_the_benchmark_categories() -> None:
    assert {prompt["category"] for prompt in build_prompt_bank()} == REQUIRED_CATEGORIES


def test_prompt_bank_is_deterministic_and_shuffled() -> None:
    assert build_prompt_bank() == build_prompt_bank()
    # A shuffled bank keeps every family in any prefix; an unshuffled one would
    # put whole families on one side of the train/held-out split.
    assert len(Counter(p["category"] for p in build_prompt_bank(limit=40))) == 4


def test_prompt_bank_limit_truncates() -> None:
    assert len(build_prompt_bank(limit=7)) == 7


def test_expand_completions_is_a_noop_for_one_completion() -> None:
    prompts = build_prompt_bank(limit=5)
    assert expand_completions(prompts, 1) == prompts


def test_expand_completions_keeps_a_prompts_completions_adjacent() -> None:
    # Adjacency is what stops the tail split from holding out one completion of
    # a prompt whose other completion was trained on.
    expanded = expand_completions(build_prompt_bank(limit=3), 2)
    assert [item["id"] for item in expanded] == [
        f"{prompt['id']}#c{index}"
        for prompt in build_prompt_bank(limit=3)
        for index in (0, 1)
    ]
    assert all(item["prompt_id"] in item["id"] for item in expanded)


def test_expand_completions_rejects_zero() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        expand_completions(build_prompt_bank(limit=2), 0)


def test_rewrite_in_bank_order_regroups_appended_completions(tmp_path) -> None:
    bank = expand_completions(build_prompt_bank(limit=2), 2)
    path = tmp_path / "samples.jsonl"
    # Written the way a resumed run writes it: every c0, then every c1.
    appended = [bank[0], bank[2], bank[1], bank[3]]
    path.write_text(
        "".join(json.dumps({"id": item["id"]}) + "\n" for item in appended),
        encoding="utf-8",
    )
    assert rewrite_in_bank_order(path, bank) == 0
    assert [row["id"] for row in load_samples(path)] == [item["id"] for item in bank]


def test_rewrite_in_bank_order_keeps_unknown_rows_at_the_end(tmp_path) -> None:
    bank = expand_completions(build_prompt_bank(limit=1), 2)
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"id": "from-an-older-bank"}) + "\n"
        + "".join(json.dumps({"id": item["id"]}) + "\n" for item in bank),
        encoding="utf-8",
    )
    assert rewrite_in_bank_order(path, bank) == 1
    assert [row["id"] for row in load_samples(path)][-1] == "from-an-older-bank"


def test_batch_seed_offset_is_stable_across_processes() -> None:
    # zlib.crc32, not the salted built-in hash(), so a resumed run reproduces
    # the completions it would have written.
    assert batch_seed_offset("code-01-03#c0") == zlib.crc32(b"code-01-03#c0") % 1_000_003


def test_split_samples_reserves_a_tail() -> None:
    samples = [{"id": str(index)} for index in range(20)]
    train, heldout = split_samples(samples, 0.1)
    assert len(train) == 18
    assert [item["id"] for item in heldout] == ["18", "19"]


def test_split_samples_moves_the_cut_to_a_prompt_boundary() -> None:
    # 7 prompts x 3 completions, 10% of 21 = 2 rows, which would land inside
    # prompt p6 and train on one of its completions while scoring another.
    samples = [
        {"id": f"p{prompt}#c{completion}", "prompt_id": f"p{prompt}"}
        for prompt in range(7)
        for completion in range(3)
    ]
    train, heldout = split_samples(samples, 0.1)
    assert len(heldout) == 3  # the whole p6 group, not the 2 rows requested
    assert {s["prompt_id"] for s in heldout} == {"p6"}
    assert not {s["prompt_id"] for s in heldout} & {s["prompt_id"] for s in train}


def test_split_samples_leaves_aligned_groups_alone() -> None:
    samples = [
        {"id": f"p{prompt}#c{completion}", "prompt_id": f"p{prompt}"}
        for prompt in range(10)
        for completion in range(2)
    ]
    train, heldout = split_samples(samples, 0.1)
    assert len(heldout) == 2 and len(train) == 18


def test_split_samples_rejects_a_file_that_is_not_grouped() -> None:
    # Completions scattered instead of adjacent: no single cut can separate the
    # prompts, so this must fail loudly rather than leak.
    samples = [
        {"id": f"p{prompt}#c{completion}", "prompt_id": f"p{prompt}"}
        for completion in range(2)
        for prompt in range(5)
    ]
    with pytest.raises(ValueError, match="appear in both splits"):
        split_samples(samples, 0.3)


def test_split_samples_keeps_at_least_one_heldout() -> None:
    _, heldout = split_samples([{"id": "a"}, {"id": "b"}, {"id": "c"}], 0.01)
    assert len(heldout) == 1


def test_split_samples_with_zero_fraction_holds_nothing_out() -> None:
    train, heldout = split_samples([{"id": "a"}, {"id": "b"}], 0.0)
    assert len(train) == 2 and heldout == []


def test_split_samples_rejects_out_of_range_fractions() -> None:
    with pytest.raises(ValueError, match="heldout_fraction"):
        split_samples([{"id": "a"}], 1.0)


def test_load_samples_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_data"):
        load_samples(tmp_path / "absent.jsonl")


def test_load_samples_rejects_an_empty_file(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_samples(path)


def test_load_samples_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"id": "a", "category": "code", "prompt": "p", "completion": "c"}) + "\n",
        encoding="utf-8",
    )
    assert load_samples(path)[0]["id"] == "a"


def test_collate_right_pads_and_masks_padding() -> None:
    examples = [([1, 2, 3], [IGNORE_INDEX, 2, 3]), ([4, 5], [IGNORE_INDEX, 5])]
    input_ids, attention_mask, labels = collate(examples, pad_token_id=0, device=torch.device("cpu"))
    assert input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    # Padded label positions must be ignored, not learned as token 0.
    assert labels.tolist() == [[IGNORE_INDEX, 2, 3], [IGNORE_INDEX, 5, IGNORE_INDEX]]
