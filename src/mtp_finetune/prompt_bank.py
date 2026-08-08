"""Deterministic prompt bank for self-distillation.

The inference experiment (`data/prompts.json`) uses four prompt families:
structured_json, code, instruction, creative. The heads are trained on the same
four families so the fine-tune section and the benchmark section of the article
talk about the same kinds of text.

One template crossed with one subject gives one prompt, so the bank is fully
reproducible from this file alone - no dataset download, no `datasets`
dependency, and the exact prompt list is recoverable from the article.
"""

from __future__ import annotations

import random

# Each family: templates with a single {subject} slot, crossed with subjects.
_FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "structured_json": (
        (
            "Return only JSON with keys name, category, and price_usd for {subject}.",
            "Produce a JSON object describing {subject} with at least four fields.",
            "Give me {subject} as JSON with a nested `details` object.",
            "Return a JSON array of three variants of {subject}, each with id and label.",
            "Output JSON for {subject} using snake_case keys and no prose.",
            "Describe {subject} as JSON including a boolean `available` field.",
            "Return only JSON: {subject}, with keys summary, tags (array), and score.",
            "Emit a JSON record for {subject} with an ISO 8601 `updated_at` timestamp.",
            "Serialize {subject} to JSON with a `metadata` object holding two keys.",
            "Return JSON for {subject} where every numeric field is a float.",
        ),
        (
            "a waterproof notebook",
            "a second-hand mountain bike",
            "an espresso machine for a small office",
            "a rescue cat listed for adoption",
            "a weekend train ticket between two cities",
            "a rooftop solar panel kit",
            "a paperback novel in a used bookshop",
            "a rented studio apartment",
            "a refurbished laptop with a one-year warranty",
            "a set of cast-iron pans",
            "a bicycle repair stand",
            "a monthly gym membership",
            "a noise-cancelling headset",
            "a sourdough starter kit",
            "a garden shed delivered flat-packed",
            "a vintage film camera",
            "a standing desk with a manual crank",
            "a wool blanket woven in a small mill",
            "a portable water filter for hiking",
            "a two-person tent rated for winter",
        ),
    ),
    "code": (
        (
            "Write a Python function that {subject}.",
            "Write a small Python function that {subject}, with a docstring.",
            "Show me Python code that {subject}, then explain it in two sentences.",
            "Implement a Python helper that {subject} and handles the empty input case.",
            "Write a Python function that {subject}; add type hints.",
            "Give a Python snippet that {subject} without using any third-party library.",
            "Write a Python generator that {subject}.",
            "Write a Python function that {subject}, plus one pytest test for it.",
            "Refactor this idea into clean Python: something that {subject}.",
            "Write a Python function that {subject} and raises ValueError on bad input.",
        ),
        (
            "removes duplicate dictionaries from a list while preserving order",
            "flattens an arbitrarily nested list of integers",
            "parses a log line into a timestamp and a message",
            "merges two sorted lists into one sorted list",
            "counts word frequencies in a string, ignoring case",
            "retries a callable with exponential backoff",
            "chunks an iterable into fixed-size batches",
            "converts a snake_case string to camelCase",
            "returns the longest common prefix of a list of strings",
            "validates that a string is a well-formed IPv4 address",
            "computes a rolling average over a list of numbers",
            "groups a list of records by one of their keys",
            "reads a CSV file and yields rows as dictionaries",
            "finds the first missing positive integer in a list",
            "deep-merges two nested dictionaries",
            "truncates a string on a word boundary",
            "expands a range expression like '1-3,7' into integers",
            "removes ANSI escape codes from a terminal string",
            "checks whether two strings are anagrams",
            "formats a byte count as a human-readable size",
        ),
    ),
    "instruction": (
        (
            "Explain in three bullet points how to {subject}.",
            "Give me a numbered checklist for how to {subject}.",
            "In under 120 words, explain how to {subject}.",
            "Explain how to {subject} to someone who has never done it before.",
            "List the three most common mistakes people make when they {subject}.",
            "Walk me through how to {subject}, one step per line.",
            "Explain how to {subject} and what to check afterwards.",
            "Summarize how to {subject} in two short paragraphs.",
            "What do I need to prepare before I {subject}?",
            "Explain how to {subject}, and when you should not do it at all.",
        ),
        (
            "back up a PostgreSQL database before a schema migration",
            "rotate an API key without downtime",
            "set up a Python virtual environment on Windows",
            "review a pull request that touches shared code",
            "profile a slow web endpoint",
            "restore a laptop from a system backup",
            "write a bug report that a developer can act on",
            "roll back a failed container deployment",
            "add an index to a busy production table",
            "hand over an on-call rotation",
            "read a stack trace from a language you do not know",
            "decide whether a flaky test should be quarantined",
            "migrate a repository to a new default branch name",
            "size a cache before you have traffic data",
            "audit which third-party packages a project actually uses",
            "recover a git commit you deleted by accident",
            "run a postmortem that people are willing to attend",
            "split one long function into readable pieces",
            "check whether a dependency upgrade is safe",
            "set up logging that is useful at three in the morning",
        ),
    ),
    "creative": (
        (
            "Write a short opening paragraph for a story about {subject}.",
            "Write two paragraphs of fiction featuring {subject}.",
            "Describe a quiet scene involving {subject}.",
            "Write the first three sentences of a novel about {subject}.",
            "Write a short vignette about {subject}, present tense.",
            "Describe {subject} from the point of view of someone leaving.",
            "Write a paragraph about {subject} with no dialogue.",
            "Write an atmospheric opening about {subject}.",
            "Write a short character sketch of {subject}.",
            "Write a paragraph about {subject} that ends on an unresolved note.",
        ),
        (
            "a botanist on Europa",
            "a night-shift lighthouse keeper",
            "a cartographer who has run out of blank paper",
            "a retired train driver on their last journey",
            "a translator working on a dead language",
            "a beekeeper during an unusually warm winter",
            "a radio operator in an empty research station",
            "a locksmith who has forgotten a combination",
            "a piano tuner in a concert hall before dawn",
            "a ferry pilot on a route nobody takes any more",
            "an archivist who has found an unlabelled reel",
            "a glassblower on the last night of a workshop",
            "a night nurse counting a quiet ward",
            "a seed collector in a burnt valley",
            "a clockmaker who no longer keeps time",
            "a border guard at an abandoned crossing",
            "a diver mapping a flooded quarry",
            "a stonemason repairing a wall she once built",
            "a courier carrying a package with no address",
            "an organ repairer in an empty church",
        ),
    ),
}

BANK_SEED = 20260725


def build_prompt_bank(limit: int | None = None, seed: int = BANK_SEED) -> list[dict[str, str]]:
    """Return prompts as `{id, category, text}`, shuffled deterministically.

    Shuffling matters: a train/held-out split taken as a prefix/suffix of this
    list must not end up with whole families on one side.
    """
    prompts: list[dict[str, str]] = []
    for category, (templates, subjects) in _FAMILIES.items():
        for template_index, template in enumerate(templates):
            for subject_index, subject in enumerate(subjects):
                prompts.append(
                    {
                        "id": f"{category}-{template_index:02d}-{subject_index:02d}",
                        "category": category,
                        "text": template.format(subject=subject),
                    }
                )
    random.Random(seed).shuffle(prompts)
    return prompts[:limit] if limit is not None else prompts


def bank_size() -> int:
    return sum(len(templates) * len(subjects) for templates, subjects in _FAMILIES.values())
