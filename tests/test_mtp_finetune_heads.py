import pytest
import torch

from mtp_finetune.heads import (
    MTPHeads,
    head_losses,
    head_offsets,
    head_top1_accuracy,
    offset_cross_entropy,
)

HIDDEN = 8
VOCAB = 11


def build_heads(num_heads: int = 3) -> MTPHeads:
    torch.manual_seed(0)
    return MTPHeads(HIDDEN, VOCAB, num_heads)


def test_head_offsets_start_at_two() -> None:
    assert head_offsets(3) == [2, 3, 4]


def test_heads_start_as_copies_of_the_base_head() -> None:
    base_weight = torch.randn(VOCAB, HIDDEN)
    heads = MTPHeads(HIDDEN, VOCAB, 3, base_lm_head_weight=base_weight)
    hidden = torch.randn(2, 6, HIDDEN)
    base_logits = torch.nn.functional.linear(hidden, base_weight)
    for logits in heads(hidden):
        torch.testing.assert_close(logits, base_logits)


def test_copy_base_lm_head_rejects_wrong_shape() -> None:
    heads = build_heads()
    with pytest.raises(ValueError, match="expected"):
        heads.copy_base_lm_head(torch.randn(VOCAB + 1, HIDDEN))


def test_offset_cross_entropy_aligns_logits_with_future_labels() -> None:
    # Position t is given a confident vote for the token that truly sits at t+2,
    # so a correctly aligned loss must be near zero.
    labels = torch.tensor([[3, 5, 7, 9, 2, 4]])
    logits = torch.zeros(1, 6, VOCAB)
    for position in range(4):
        logits[0, position, labels[0, position + 2]] = 30.0
    assert offset_cross_entropy(logits, labels, 2).item() == pytest.approx(0.0, abs=1e-4)


def test_offset_cross_entropy_ignores_masked_labels() -> None:
    labels = torch.tensor([[-100, -100, 4, 6]])
    logits = torch.zeros(1, 4, VOCAB)
    # Positions 0 and 1 predict labels 4 and 6; the masked leading labels never
    # become targets because the shift moves the window past them.
    logits[0, 0, 4] = 30.0
    logits[0, 1, 6] = 30.0
    assert offset_cross_entropy(logits, labels, 2).item() == pytest.approx(0.0, abs=1e-4)


def test_offset_cross_entropy_sum_reduction_is_the_mean_times_token_count() -> None:
    # Token-weighted averaging depends on this: sums accumulated across batches
    # and divided by the total scored-token count.
    torch.manual_seed(0)
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    logits = torch.randn(1, 6, VOCAB)
    scored = labels.size(1) - 2
    mean = offset_cross_entropy(logits, labels, 2)
    total = offset_cross_entropy(logits, labels, 2, reduction="sum")
    assert total.item() == pytest.approx(mean.item() * scored, rel=1e-5)


def test_offset_cross_entropy_sum_skips_masked_positions() -> None:
    labels = torch.tensor([[1, 2, -100, 4, 5, 6]])
    logits = torch.zeros(1, 6, VOCAB)
    total = offset_cross_entropy(logits, labels, 2, reduction="sum")
    mean = offset_cross_entropy(logits, labels, 2)
    # 4 shifted targets, one of which is masked -> 3 contribute.
    assert total.item() == pytest.approx(mean.item() * 3, rel=1e-5)


def test_offset_cross_entropy_rejects_impossible_offsets() -> None:
    labels = torch.tensor([[1, 2, 3]])
    logits = torch.zeros(1, 3, VOCAB)
    with pytest.raises(ValueError, match="offset must be"):
        offset_cross_entropy(logits, labels, 0)
    with pytest.raises(ValueError, match="longer than"):
        offset_cross_entropy(logits, labels, 3)


def test_head_losses_covers_base_reference_and_every_head() -> None:
    heads = build_heads()
    hidden = torch.randn(1, 12, HIDDEN)
    labels = torch.randint(0, VOCAB, (1, 12))
    base_logits = torch.randn(1, 12, VOCAB)
    losses = head_losses(heads, hidden, labels, base_logits=base_logits)
    assert sorted(losses) == [1, 2, 3, 4]


def test_head_losses_without_base_logits_skips_the_reference() -> None:
    heads = build_heads()
    hidden = torch.randn(1, 12, HIDDEN)
    labels = torch.randint(0, VOCAB, (1, 12))
    assert sorted(head_losses(heads, hidden, labels)) == [2, 3, 4]


def test_head_top1_accuracy_counts_only_unmasked_positions() -> None:
    labels = torch.tensor([[-100, -100, 4, 6, 8]])
    logits = torch.zeros(1, 5, VOCAB)
    logits[0, 0, 4] = 5.0  # matches label at offset 2
    logits[0, 1, 0] = 5.0  # misses label 6
    logits[0, 2, 8] = 5.0  # matches label 8
    assert head_top1_accuracy(logits, labels, 2) == (2, 3)


def test_head_top1_accuracy_handles_fully_masked_windows() -> None:
    labels = torch.full((1, 5), -100)
    logits = torch.zeros(1, 5, VOCAB)
    assert head_top1_accuracy(logits, labels, 2) == (0, 0)
