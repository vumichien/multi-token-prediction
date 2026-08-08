"""Medusa-style future-token heads and the offset cross-entropy objective.

This module is the teaching core of the fine-tune experiment. The whole idea of
multi-token prediction fits in two pieces:

1. Take the final hidden state `h_t` the frozen base already computes for
   position `t`. The base's own LM head turns it into a guess for `x_{t+1}`.
2. Bolt on K extra heads. Head `k` reads the same `h_t` and is trained to guess
   `x_{t+k+1}` instead. Nothing else about the model changes.

The loss is therefore just cross-entropy with a shifted label window, one shift
per head. That is the Meta MTP objective; because the trunk stays frozen here it
is the Medusa retrofit of it rather than MTP pretraining.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# Head 0 predicts t+2, head 1 predicts t+3, and so on. The frozen base LM head
# already covers t+1, so extra heads start one step further out.
FIRST_EXTRA_OFFSET = 2


def head_offsets(num_heads: int) -> list[int]:
    """Token offsets the extra heads predict, e.g. [2, 3, 4] for K=3."""
    return [FIRST_EXTRA_OFFSET + index for index in range(num_heads)]


class ResBlock(nn.Module):
    """Residual SiLU block, as in the Medusa paper.

    The linear layer is zero-initialised so each head starts as the identity:
    at step 0 a head is exactly the base LM head, and training only has to learn
    the correction that shifts its prediction further into the future.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.act(self.linear(hidden))


class MTPHeads(nn.Module):
    """K independent (ResBlock -> LM head) stacks reading one shared hidden state."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int,
        *,
        base_lm_head_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.offsets = head_offsets(num_heads)
        self.blocks = nn.ModuleList(ResBlock(hidden_size) for _ in range(num_heads))
        self.lm_heads = nn.ModuleList(
            nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_heads)
        )
        if base_lm_head_weight is not None:
            self.copy_base_lm_head(base_lm_head_weight)

    def copy_base_lm_head(self, weight: torch.Tensor) -> None:
        """Warm-start every head's output projection from the base LM head."""
        if tuple(weight.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"Base LM head is {tuple(weight.shape)}, expected "
                f"{(self.vocab_size, self.hidden_size)}."
            )
        with torch.no_grad():
            for lm_head in self.lm_heads:
                lm_head.weight.copy_(weight.to(lm_head.weight.dtype))

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """hidden: (batch, seq, hidden) -> one (batch, seq, vocab) logit tensor per head."""
        return [
            lm_head(block(hidden))
            for block, lm_head in zip(self.blocks, self.lm_heads, strict=True)
        ]

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def offset_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    offset: int,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy between logits at position t and the label at position t+offset.

    Positions whose target falls off the end of the sequence are dropped, and
    label positions marked -100 (prompt tokens, padding) are ignored.

    `reduction="sum"` returns the summed loss instead of the per-token mean.
    Accumulating sums and dividing by the total token count at the end gives a
    token-weighted average across batches; averaging the per-batch means would
    instead weight a short completion the same as a long one.
    """
    if offset < 1:
        raise ValueError(f"offset must be >= 1, got {offset}")
    if offset >= labels.size(1):
        raise ValueError(
            f"offset {offset} needs a sequence longer than {labels.size(1)} tokens."
        )
    shifted_logits = logits[:, : -offset, :]
    shifted_labels = labels[:, offset:]
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )


def head_losses(
    heads: MTPHeads,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    base_logits: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Return `{offset: loss}` for every trained head, plus the frozen t+1 baseline.

    `base_logits` is the frozen base LM head's output on the same hidden state.
    Its t+1 loss is not trained on - it is logged as the floor the extra heads
    are measured against, which is the whole "predicting further is harder"
    point of the article's chart.
    """
    losses: dict[int, torch.Tensor] = {}
    if base_logits is not None:
        losses[1] = offset_cross_entropy(base_logits, labels, 1)
    for offset, logits in zip(heads.offsets, heads(hidden), strict=True):
        losses[offset] = offset_cross_entropy(logits, labels, offset)
    return losses


@torch.no_grad()
def head_top1_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    offset: int,
) -> tuple[int, int]:
    """Return (correct, counted) top-1 hits for one head at its own offset."""
    shifted_logits = logits[:, : -offset, :]
    shifted_labels = labels[:, offset:]
    mask = shifted_labels != -100
    if not bool(mask.any()):
        return 0, 0
    predictions = shifted_logits.argmax(dim=-1)
    correct = int((predictions[mask] == shifted_labels[mask]).sum().item())
    return correct, int(mask.sum().item())
