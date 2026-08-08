"""The draft/verify loop must reproduce greedy decoding exactly.

On the real GPU run the two paths can differ at bf16 near-ties, which is why
`self_speculate` measures the divergence instead of asserting equality. That
tolerance would also hide a genuine off-by-one in the KV-cache crop or in the
hidden-state index used for the next drafts.

Here the model is tiny and runs in fp32 on CPU, so no rounding excuse exists:
speculative output must equal plain greedy output token for token, whatever the
heads happen to draft. Random heads make that a real test - they force a mix of
accepted and rejected drafts, exercising every crop length.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from mtp_finetune.base_runtime import BaseRuntime
from mtp_finetune.heads import MTPHeads
from mtp_finetune.self_speculate import speculative_generate

VOCAB = 64
HIDDEN = 32
EOS_ID = 2


class StubTokenizer:
    eos_token_id = EOS_ID


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch) -> None:
    """The loop synchronises CUDA for timing; on CPU that is a no-op."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)


@pytest.fixture
def runtime() -> BaseRuntime:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        eos_token_id=EOS_ID,
    )
    model = LlamaForCausalLM(config).eval()
    model.requires_grad_(False)
    return BaseRuntime(
        tokenizer=StubTokenizer(), model=model, device=torch.device("cpu")
    )


@torch.no_grad()
def plain_greedy(runtime: BaseRuntime, prompt_ids: list[int], limit: int) -> list[int]:
    """One token per forward, no cache, no heads: the reference decoding."""
    tokens: list[int] = []
    while len(tokens) < limit:
        logits = runtime.model(
            input_ids=torch.tensor([prompt_ids + tokens])
        ).logits[0, -1]
        tokens.append(int(logits.argmax()))
        if tokens[-1] == EOS_ID:
            break
    return tokens


def build_heads(seed: int) -> MTPHeads:
    torch.manual_seed(seed)
    heads = MTPHeads(HIDDEN, VOCAB, 3)
    # Break the zero-init identity so the heads actually draft something.
    for block in heads.blocks:
        torch.nn.init.normal_(block.linear.weight, std=0.5)
    return heads.eval()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_speculative_output_equals_greedy_output(runtime, seed) -> None:
    heads = build_heads(seed)
    prompt = [5, 9, 13, 21]
    result = speculative_generate(runtime, heads, prompt, 24)
    assert result["tokens"] == plain_greedy(runtime, prompt, 24)


def test_heads_that_draft_perfectly_still_produce_greedy_output(runtime) -> None:
    # Copying the base LM head into every head makes the drafts wrong in a
    # different way (they all predict t+1), so acceptance is driven by real
    # agreement rather than by the heads being untrained noise.
    heads = MTPHeads(
        HIDDEN, VOCAB, 3, base_lm_head_weight=runtime.lm_head_weight()
    ).eval()
    prompt = [7, 11, 3, 19]
    result = speculative_generate(runtime, heads, prompt, 24)
    assert result["tokens"] == plain_greedy(runtime, prompt, 24)


def test_every_pass_emits_at_least_one_token(runtime) -> None:
    heads = build_heads(4)
    result = speculative_generate(runtime, heads, [5, 9, 13, 21], 24)
    # The priming pass emits one token; each later pass emits accepted+1.
    expected = 1 + sum(accepted + 1 for accepted in result["accepted_per_pass"])
    assert result["tokens_emitted"] == expected
    assert result["forward_passes"] == 1 + len(result["accepted_per_pass"])
    assert result["tokens_emitted"] >= result["forward_passes"]


def test_accepted_counts_never_exceed_the_number_of_heads(runtime) -> None:
    heads = build_heads(5)
    result = speculative_generate(runtime, heads, [5, 9, 13, 21], 24)
    assert all(0 <= accepted <= 3 for accepted in result["accepted_per_pass"])


def test_generation_stops_at_eos_even_when_a_draft_supplied_it(runtime) -> None:
    heads = build_heads(6)
    result = speculative_generate(runtime, heads, [5, 9, 13, 21], 200)
    tokens = result["tokens"]
    if EOS_ID in tokens:
        # EOS must be the final token, never buried mid-sequence.
        assert tokens.index(EOS_ID) == len(tokens) - 1
