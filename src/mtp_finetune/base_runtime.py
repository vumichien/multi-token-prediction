"""Loading and tokenising helpers shared by every stage of the fine-tune run.

Data generation, training, evaluation, and the self-speculation loop all need
the same frozen base and the same prompt/completion tokenisation, so the rules
live here once. In particular `build_example` decides which positions carry a
training signal: only the tokens the base itself generated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mtp_finetune import BASE_MODEL

IGNORE_INDEX = -100


@dataclass(slots=True)
class BaseRuntime:
    tokenizer: object
    model: object
    device: torch.device

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def lm_head_weight(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight


def load_base(model_id: str = BASE_MODEL, *, device: str | None = None) -> BaseRuntime:
    """Load the base in bf16 and freeze it. Nothing here is ever trained."""
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(resolved)
    model.eval()
    model.requires_grad_(False)
    return BaseRuntime(tokenizer=tokenizer, model=model, device=resolved)


def prompt_token_ids(tokenizer: object, prompt: str) -> list[int]:
    """Tokenise a user turn through the chat template, ready for generation."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    # Transformers has returned both a flat list and a batch of one over the
    # 4.x/5.x line; normalise so callers only ever see list[int].
    if encoded and isinstance(encoded[0], list):
        return list(encoded[0])
    return list(encoded)


def build_example(
    tokenizer: object,
    prompt: str,
    completion: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int]] | None:
    """Return (input_ids, labels) with prompt positions masked out.

    Returns None when the completion has too few tokens left after truncation to
    supply a target for the furthest head.
    """
    prompt_ids = prompt_token_ids(tokenizer, prompt)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = (prompt_ids + completion_ids)[:max_length]
    if len(input_ids) <= len(prompt_ids):
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return input_ids, labels[: len(input_ids)]


def collate(
    examples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch into (input_ids, attention_mask, labels) on `device`."""
    width = max(len(input_ids) for input_ids, _ in examples)
    input_batch, mask_batch, label_batch = [], [], []
    for input_ids, labels in examples:
        padding = width - len(input_ids)
        input_batch.append(input_ids + [pad_token_id] * padding)
        mask_batch.append([1] * len(input_ids) + [0] * padding)
        label_batch.append(labels + [IGNORE_INDEX] * padding)
    return (
        torch.tensor(input_batch, dtype=torch.long, device=device),
        torch.tensor(mask_batch, dtype=torch.long, device=device),
        torch.tensor(label_batch, dtype=torch.long, device=device),
    )


@torch.no_grad()
def final_hidden_and_logits(
    model: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen base once, returning (final hidden state, base logits).

    The base runs under no_grad, which is what keeps the whole experiment inside
    10 GB: none of its activations have to be retained for backprop.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[-1], outputs.logits


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / (1024**2), 1)


def reserved_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_reserved() / (1024**2), 1)
