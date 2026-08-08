"""Turning generated samples into training batches.

Split rule: the samples file is written in the prompt bank's shuffled order, so
taking the tail as a held-out set keeps all four prompt families on both sides
of the split without any extra stratification logic.

A prompt can appear several times with different sampled completions. Those
rows sit next to each other in the file, and the split is moved to the nearest
prompt boundary so a prompt is never trained on and scored on at once. Adjacency
alone would not guarantee that: an arbitrary cut can land mid-group.
"""

from __future__ import annotations

import json
from pathlib import Path

from mtp_finetune.base_runtime import build_example


def load_samples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No generated samples at {path}. Run "
            "`python -m mtp_finetune.generate_data` first."
        )
    with path.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    if not samples:
        raise ValueError(f"{path} is empty.")
    return samples


def sample_prompt_id(sample: dict[str, str]) -> str:
    """The prompt a sample came from, falling back to its own id."""
    return sample.get("prompt_id", sample["id"])


def split_samples(
    samples: list[dict[str, str]],
    heldout_fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split off a held-out tail, moving the cut to a prompt boundary."""
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must be in [0, 1), got {heldout_fraction}")
    heldout_size = int(len(samples) * heldout_fraction)
    if heldout_fraction > 0 and heldout_size == 0:
        heldout_size = 1
    cut = len(samples) - heldout_size

    # Walk the cut back to the start of whatever prompt group it landed inside,
    # so every completion of a prompt ends up on the same side.
    while 0 < cut < len(samples) and sample_prompt_id(samples[cut]) == sample_prompt_id(
        samples[cut - 1]
    ):
        cut -= 1

    train, heldout = samples[:cut], samples[cut:]
    overlap = {sample_prompt_id(s) for s in heldout} & {sample_prompt_id(s) for s in train}
    if overlap:
        raise ValueError(
            f"{len(overlap)} prompts appear in both splits (e.g. {sorted(overlap)[:3]}). "
            "The samples file is not grouped by prompt; regenerate it so each "
            "prompt's completions are adjacent."
        )
    return train, heldout


def encode_samples(
    tokenizer: object,
    samples: list[dict[str, str]],
    *,
    max_length: int,
) -> list[tuple[list[int], list[int]]]:
    """Tokenise samples, dropping any that are too short to supply head targets."""
    encoded = []
    for sample in samples:
        example = build_example(
            tokenizer,
            sample["prompt"],
            sample["completion"],
            max_length=max_length,
        )
        if example is not None:
            encoded.append(example)
    if not encoded:
        raise ValueError("Every sample was dropped during tokenisation.")
    return encoded


def encode_samples_by_category(
    tokenizer: object,
    samples: list[dict[str, str]],
    *,
    max_length: int,
) -> dict[str, list[tuple[list[int], list[int]]]]:
    """Same as `encode_samples`, grouped by prompt family for per-category metrics."""
    grouped: dict[str, list[tuple[list[int], list[int]]]] = {}
    for sample in samples:
        example = build_example(
            tokenizer,
            sample["prompt"],
            sample["completion"],
            max_length=max_length,
        )
        if example is not None:
            grouped.setdefault(sample["category"], []).append(example)
    return grouped
