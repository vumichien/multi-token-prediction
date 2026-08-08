"""Held-out accuracy per prediction horizon.

This is the evidence for the article's central claim about MTP training: the
further ahead a head has to guess, the worse it does. The frozen base's own t+1
head is measured on the same batches as the reference point, so t+1 through t+4
are directly comparable numbers rather than a curve with no scale.

Usage:
    python -m mtp_finetune.eval_heads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from mtp_finetune import BASE_MODEL, NUM_HEADS
from mtp_finetune.base_runtime import (
    collate,
    final_hidden_and_logits,
    load_base,
    peak_vram_mb,
)
from mtp_finetune.dataset import (
    encode_samples_by_category,
    load_samples,
    split_samples,
)
from mtp_finetune.heads import MTPHeads, head_top1_accuracy, offset_cross_entropy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "finetune" / "samples.jsonl"
DEFAULT_RUN = "finetune-smollm2-mtp-heads"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--dev-examples", type=int, default=32,
                        help="Held-out samples reserved for checkpoint selection "
                             "during training (train_heads --eval-examples). They "
                             "are skipped here so the reported score is not "
                             "measured on the data the checkpoint was picked on.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to results/<run-name>/heldout-accuracy.json")
    return parser.parse_args(argv)


def default_checkpoint(run_name: str) -> Path:
    """Prefer the best held-out checkpoint over the last one written.

    The last checkpoint is whatever the run happened to stop on, which after an
    overfitting run is the worst one. Scoring that would understate the heads.
    """
    directory = REPO_ROOT / "checkpoints" / run_name
    best = directory / "heads-best.safetensors"
    return best if best.exists() else directory / "heads.safetensors"


def load_heads(
    checkpoint: Path,
    hidden_size: int,
    vocab_size: int,
    num_heads: int,
    device: torch.device,
) -> MTPHeads:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No trained heads at {checkpoint}. Run "
            "`python -m mtp_finetune.train_heads` first."
        )
    heads = MTPHeads(hidden_size, vocab_size, num_heads)
    heads.load_state_dict(load_file(str(checkpoint)))
    return heads.to(device, dtype=torch.float32).eval()


@torch.no_grad()
def score_examples(
    runtime: object,
    heads: MTPHeads,
    examples: list[tuple[list[int], list[int]]],
    *,
    batch_size: int,
) -> dict[int, dict[str, float]]:
    """Return `{offset: {top1_accuracy, cross_entropy, tokens}}` over the examples.

    Both metrics are token-weighted over the same scored positions, so the
    reported `cross_entropy` and `top1_accuracy` describe one population and
    `tokens_scored` is the size of it.
    """
    offsets = [1] + heads.offsets
    correct = dict.fromkeys(offsets, 0)
    counted = dict.fromkeys(offsets, 0)
    loss_totals = dict.fromkeys(offsets, 0.0)

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, attention_mask, labels = collate(
            batch, runtime.tokenizer.pad_token_id, runtime.device
        )
        hidden, base_logits = final_hidden_and_logits(
            runtime.model, input_ids, attention_mask
        )

        hit, total = head_top1_accuracy(base_logits, labels, 1)
        correct[1] += hit
        counted[1] += total
        loss_totals[1] += float(
            offset_cross_entropy(base_logits, labels, 1, reduction="sum").item()
        )
        del base_logits

        for index, offset in enumerate(heads.offsets):
            logits = heads.lm_heads[index](heads.blocks[index](hidden.float()))
            hit, total = head_top1_accuracy(logits, labels, offset)
            correct[offset] += hit
            counted[offset] += total
            loss_totals[offset] += float(
                offset_cross_entropy(logits, labels, offset, reduction="sum").item()
            )
            del logits
        del hidden, input_ids, attention_mask, labels

    return {
        offset: {
            "top1_accuracy": round(correct[offset] / counted[offset], 4) if counted[offset] else 0.0,
            "cross_entropy": round(loss_totals[offset] / counted[offset], 4) if counted[offset] else 0.0,
            "tokens_scored": counted[offset],
        }
        for offset in offsets
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = REPO_ROOT / "results" / args.run_name
    checkpoint = args.checkpoint or default_checkpoint(args.run_name)
    output = args.output or (results_dir / "heldout-accuracy.json")

    _, heldout = split_samples(load_samples(args.samples), args.heldout_fraction)
    # Training selects the best checkpoint on the first --eval-examples of this
    # same list. Scoring those again would report a number the checkpoint was
    # chosen to be good at, so they are dropped here.
    dev_samples, test_samples = heldout[: args.dev_examples], heldout[args.dev_examples :]
    if not test_samples:
        raise ValueError(
            f"No held-out samples left after reserving {args.dev_examples} for "
            "checkpoint selection; raise --heldout-fraction or lower --dev-examples."
        )
    heldout = test_samples

    runtime = load_base(args.model)
    runtime.tokenizer.padding_side = "right"
    heads = load_heads(
        checkpoint, runtime.hidden_size, runtime.vocab_size, args.num_heads, runtime.device
    )

    by_category = encode_samples_by_category(
        runtime.tokenizer, heldout, max_length=args.max_length
    )
    everything = [example for group in by_category.values() for example in group]
    print(f"Scoring {len(everything)} held-out examples across "
          f"{len(by_category)} prompt families")

    overall = score_examples(runtime, heads, everything, batch_size=args.batch_size)
    per_category = {
        category: score_examples(runtime, heads, group, batch_size=args.batch_size)
        for category, group in sorted(by_category.items())
    }

    payload = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "heldout_examples": len(everything),
        "heldout_prompts": len({sample["prompt_id"] for sample in heldout}),
        "dev_examples_excluded": len(dev_samples),
        "max_length": args.max_length,
        "head_offsets": heads.offsets,
        "note": (
            "t+1 is the frozen base LM head, not a trained head. It is the "
            "reference the trained heads are compared against. Scores are "
            "token-weighted over `tokens_scored` positions, and `dev_examples_"
            "excluded` samples were withheld because training picked the "
            "checkpoint on them. `heldout_examples` counts sampled completions; "
            "`heldout_prompts` is the number of distinct prompts behind them, "
            "which is the honest unit of independent evidence."
        ),
        "overall": {f"t+{offset}": scores for offset, scores in overall.items()},
        "by_category": {
            category: {f"t+{offset}": scores for offset, scores in group.items()}
            for category, group in per_category.items()
        },
        "peak_allocated_mb": peak_vram_mb(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nhorizon  top-1 accuracy  cross-entropy  tokens")
    for offset, scores in overall.items():
        marker = " (frozen base)" if offset == 1 else ""
        print(f"  t+{offset}      {scores['top1_accuracy']:>8.4f}  "
              f"{scores['cross_entropy']:>13.4f}  {scores['tokens_scored']:>6}{marker}")
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
