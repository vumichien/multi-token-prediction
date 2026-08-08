"""Train K future-token heads on a frozen SmolLM2 inside 10 GB.

What actually keeps this inside a consumer GPU:

* the base runs under `no_grad`, so none of its activations are kept for backward;
* each head is independent, so heads are backpropagated one at a time and the
  vocabulary-sized logits of head k are freed before head k+1 is computed;
* head parameters are fp32 masters with a bf16 autocast forward and an 8-bit
  optimiser, which is where most of the remaining budget would otherwise go.

The training log is appended and flushed every logging interval, so a crash or a
hard stop at the time budget still leaves a usable loss curve on disk.

Usage:
    python -m mtp_finetune.train_heads --max-steps 100 --run-name smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from mtp_finetune import BASE_MODEL, NUM_HEADS
from mtp_finetune.base_runtime import (
    collate,
    final_hidden_and_logits,
    load_base,
    peak_vram_mb,
    reserved_vram_mb,
)
from mtp_finetune.dataset import encode_samples, load_samples, split_samples
from mtp_finetune.eval_heads import score_examples
from mtp_finetune.heads import MTPHeads, offset_cross_entropy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "finetune" / "samples.jsonl"
DEFAULT_RUN = "finetune-smollm2-mtp-heads"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--run-name", default=DEFAULT_RUN,
                        help="Results directory name under results/.")
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Optimiser steps. Training also stops at --max-minutes.")
    parser.add_argument("--max-minutes", type=float, default=240.0,
                        help="Hard wall-clock budget; whichever limit hits first wins.")
    parser.add_argument("--log-every", type=int, default=5,
                        help="Optimiser steps per training-log.csv row.")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Optimiser steps between held-out evaluations "
                             "(0 disables). Held-out loss is what shows the run "
                             "tipping over into memorising the samples.")
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after this many evaluations with no held-out "
                             "improvement (0 disables). Requires --eval-every.")
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--fresh-log", action="store_true",
                        help="Truncate an existing training-log.csv instead of appending.")
    return parser.parse_args(argv)


def build_optimizer(parameters: list[torch.nn.Parameter], args: argparse.Namespace) -> object:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "8-bit AdamW needs bitsandbytes. Install it, or rerun with "
            "--optimizer adamw (about 1.8 GB more VRAM)."
        ) from exc
    return bnb.optim.AdamW8bit(parameters, lr=args.lr, weight_decay=args.weight_decay)


def learning_rate_at(step: int, total_steps: int, args: argparse.Namespace) -> float:
    """Linear warmup into cosine decay."""
    warmup = max(1, int(total_steps * args.warmup_fraction))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def verify_identity_init(
    heads: MTPHeads,
    hidden: torch.Tensor,
    base_logits: torch.Tensor,
) -> dict[str, float]:
    """Zero-init ResBlocks mean every head must start as a copy of the base head.

    Checking it costs one forward pass and catches a silently wrong warm start,
    which would otherwise only show up as heads that never converge. Only the
    first head is probed, so this does not allocate K vocabulary-sized tensors.

    The comparison is scale-relative on purpose: the base runs its head in bf16
    and the copy runs in fp32, so a raw gap of a tenth of a logit is ordinary
    rounding. Top-1 agreement is loosely bounded for the same reason - across
    49k classes a couple of percent of positions have their top two logits
    within bf16 rounding of each other, so they flip harmlessly. A head that had
    not actually been copied would agree with the base almost never, which is
    the failure this is here to catch.
    """
    with torch.no_grad():
        head_logits = heads.lm_heads[0](heads.blocks[0](hidden.float()))
        reference = base_logits.float()
        scale = float(reference.abs().max().item())
        gap = float((head_logits - reference).abs().max().item())
        agreement = float(
            (head_logits.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean().item()
        )
    return {
        "max_logit_gap": gap,
        "logit_scale": scale,
        "relative_gap": gap / max(scale, 1e-6),
        "top1_agreement": agreement,
    }


def last_logged_step(path: Path) -> int | None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return int(float(rows[-1]["step"])) if rows else None


def open_log(path: Path, offsets: list[int], fresh: bool) -> tuple[object, object]:
    """Open training-log.csv for append, writing the header only when new.

    A second run into the same directory would append rows whose step counter
    restarts from 1. Nothing downstream sorts by step - the GIF scene plots the
    file in order - so that would render as a sawtooth curve in a published
    article. Refuse instead, and say which flag resolves it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and path.exists():
        path.unlink()
    if path.exists() and last_logged_step(path) is not None:
        raise FileExistsError(
            f"{path} already holds a run ending at step {last_logged_step(path)}. "
            "Appending would restart the step counter mid-file. Pass --fresh-log "
            "to replace it, or --run-name to write somewhere else."
        )
    is_new = not path.exists()
    handle = path.open("a", encoding="utf-8", newline="")
    fieldnames = ["step"] + [f"head_{offset}_loss" for offset in offsets]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    if is_new:
        writer.writeheader()
        handle.flush()
    return handle, writer


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Head training needs CUDA: the loop relies on bf16 autocast and "
            "reports peak VRAM. On CPU it would take days."
        )
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    results_dir = REPO_ROOT / "results" / args.run_name
    checkpoint_dir = args.checkpoint_dir or (REPO_ROOT / "checkpoints" / args.run_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_samples, heldout_samples = split_samples(
        load_samples(args.samples), args.heldout_fraction
    )
    print(f"Samples: {len(train_samples)} train / {len(heldout_samples)} held out")

    runtime = load_base(args.model)
    runtime.tokenizer.padding_side = "right"  # training pads on the right
    examples = encode_samples(runtime.tokenizer, train_samples, max_length=args.max_length)
    print(f"Tokenised {len(examples)} training examples "
          f"(max_length {args.max_length})")
    eval_examples: list[tuple[list[int], list[int]]] = []
    if args.eval_every:
        eval_examples = encode_samples(
            runtime.tokenizer, heldout_samples, max_length=args.max_length
        )[: args.eval_examples]
        print(f"Held-out evaluation every {args.eval_every} steps on "
              f"{len(eval_examples)} examples")

    heads = MTPHeads(
        runtime.hidden_size,
        runtime.vocab_size,
        args.num_heads,
        base_lm_head_weight=runtime.lm_head_weight(),
    ).to(runtime.device, dtype=torch.float32)
    trainable = heads.trainable_parameter_count()
    print(f"Heads: K={args.num_heads} predicting t+{heads.offsets}, "
          f"{trainable / 1e6:.1f}M trainable parameters")

    # Sanity gate: at step 0 each head must reproduce the base head exactly.
    probe_ids, probe_mask, _ = collate(examples[:1], runtime.tokenizer.pad_token_id, runtime.device)
    probe_hidden, probe_logits = final_hidden_and_logits(runtime.model, probe_ids, probe_mask)
    identity = verify_identity_init(heads, probe_hidden, probe_logits)
    print(f"Identity check at init: max gap {identity['max_logit_gap']:.4f} on a "
          f"{identity['logit_scale']:.1f} logit scale "
          f"({identity['relative_gap'] * 100:.2f}%), "
          f"top-1 agreement {identity['top1_agreement'] * 100:.1f}%")
    if identity["relative_gap"] > 0.02 or identity["top1_agreement"] < 0.90:
        raise RuntimeError(
            f"Heads did not warm-start from the base LM head ({identity}). "
            "Refusing to train from a broken initialisation."
        )
    del probe_ids, probe_mask, probe_hidden, probe_logits
    torch.cuda.empty_cache()

    optimizer = build_optimizer(list(heads.parameters()), args)
    log_offsets = [1] + heads.offsets  # head_1 is the frozen base reference
    log_handle, log_writer = open_log(
        results_dir / "training-log.csv", log_offsets, args.fresh_log
    )
    heldout_handle = heldout_writer = None
    if eval_examples:
        heldout_handle, heldout_writer = open_log(
            results_dir / "heldout-log.csv", log_offsets, args.fresh_log
        )

    config = {
        "model": args.model,
        "num_heads": args.num_heads,
        "head_offsets": heads.offsets,
        "trainable_parameters": trainable,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "max_steps": args.max_steps,
        "max_minutes": args.max_minutes,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "train_samples": len(train_samples),
        "heldout_samples": len(heldout_samples),
        "train_examples": len(examples),
        "heldout_fraction": args.heldout_fraction,
        "samples_path": str(args.samples.relative_to(REPO_ROOT)),
    }
    (results_dir / "train-config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    torch.cuda.reset_peak_memory_stats()
    order = list(range(len(examples)))
    rng.shuffle(order)
    cursor = 0
    interval_totals = dict.fromkeys(log_offsets, 0.0)
    interval_batches = 0
    tokens_seen = 0
    stop_reason = "max_steps"
    best_heldout_loss = float("inf")
    best_step = 0
    evaluations_without_improvement = 0
    started = time.perf_counter()

    for step in range(args.max_steps):
        learning_rate = learning_rate_at(step, args.max_steps, args)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum):
            if cursor + args.batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            batch = [examples[index] for index in order[cursor : cursor + args.batch_size]]
            cursor += args.batch_size

            input_ids, attention_mask, labels = collate(
                batch, runtime.tokenizer.pad_token_id, runtime.device
            )
            tokens_seen += int(attention_mask.sum().item())
            hidden, base_logits = final_hidden_and_logits(
                runtime.model, input_ids, attention_mask
            )

            # The frozen base's own t+1 loss: the floor the extra heads sit above.
            interval_totals[1] += float(
                offset_cross_entropy(base_logits, labels, 1).item()
            )
            del base_logits

            # One head at a time: its vocab-sized logits are freed before the next.
            for index, offset in enumerate(heads.offsets):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = heads.lm_heads[index](heads.blocks[index](hidden))
                loss = offset_cross_entropy(logits, labels, offset)
                (loss / args.grad_accum).backward()
                interval_totals[offset] += float(loss.item())
                del logits, loss
            del hidden, input_ids, attention_mask, labels
            interval_batches += 1

        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        optimizer.step()

        completed = step + 1
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        if completed % args.log_every == 0 or completed == args.max_steps:
            row = {"step": completed}
            for offset in log_offsets:
                row[f"head_{offset}_loss"] = round(
                    interval_totals[offset] / max(interval_batches, 1), 5
                )
            log_writer.writerow(row)
            log_handle.flush()
            losses = "  ".join(
                f"t+{offset}={row[f'head_{offset}_loss']:.3f}" for offset in log_offsets
            )
            print(f"step {completed}/{args.max_steps}  {losses}  "
                  f"lr={learning_rate:.2e}  {elapsed_minutes:.1f}min  "
                  f"peak={peak_vram_mb()}MB", flush=True)
            interval_totals = dict.fromkeys(log_offsets, 0.0)
            interval_batches = 0

        if heldout_writer is not None and (
            completed % args.eval_every == 0 or completed == args.max_steps
        ):
            heads.eval()
            scores = score_examples(runtime, heads, eval_examples, batch_size=1)
            heads.train()
            heldout_writer.writerow({
                "step": completed,
                **{
                    f"head_{offset}_loss": scores[offset]["cross_entropy"]
                    for offset in log_offsets
                },
            })
            heldout_handle.flush()

            # Mean over the trained heads only: t+1 is the frozen base and never
            # moves, so including it would just damp the signal.
            heldout_loss = statistics.fmean(
                scores[offset]["cross_entropy"] for offset in heads.offsets
            )
            improved = heldout_loss < best_heldout_loss - 1e-4
            if improved:
                best_heldout_loss = heldout_loss
                best_step = completed
                evaluations_without_improvement = 0
                save_file(heads.state_dict(), str(checkpoint_dir / "heads-best.safetensors"))
            else:
                evaluations_without_improvement += 1
            print("  held out  " + "  ".join(
                f"t+{offset}={scores[offset]['cross_entropy']:.3f}"
                f"/acc {scores[offset]['top1_accuracy']:.3f}"
                for offset in log_offsets
            ) + f"  mean={heldout_loss:.4f}"
                + ("  <- best, saved" if improved
                   else f"  (no gain x{evaluations_without_improvement})"), flush=True)

            if (
                args.early_stop_patience
                and evaluations_without_improvement >= args.early_stop_patience
            ):
                stop_reason = "early_stop"
                print(f"Held-out loss has not improved in "
                      f"{evaluations_without_improvement} evaluations; stopping. "
                      f"Best was {best_heldout_loss:.4f} at step {best_step}.")
                break

        if args.save_every and completed % args.save_every == 0:
            save_file(heads.state_dict(), str(checkpoint_dir / "heads.safetensors"))

        if elapsed_minutes >= args.max_minutes:
            stop_reason = "max_minutes"
            print(f"Stopping at the {args.max_minutes} minute budget after "
                  f"{completed} steps.")
            break

    log_handle.close()
    if heldout_handle is not None:
        heldout_handle.close()
    save_file(heads.state_dict(), str(checkpoint_dir / "heads.safetensors"))

    wall_clock_minutes = (time.perf_counter() - started) / 60.0
    summary = {
        "status": "completed",
        "stop_reason": stop_reason,
        "steps_completed": completed,
        "wall_clock_minutes": round(wall_clock_minutes, 2),
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / max(wall_clock_minutes * 60, 1e-6), 1),
        "peak_allocated_mb": peak_vram_mb(),
        "peak_reserved_mb": reserved_vram_mb(),
        "identity_init_check": {key: round(value, 6) for key, value in identity.items()},
        "best_heldout_mean_loss": (
            round(best_heldout_loss, 5) if best_step else None
        ),
        "best_step": best_step or None,
        "best_checkpoint": (
            str(checkpoint_dir / "heads-best.safetensors") if best_step else None
        ),
        "checkpoint": str(checkpoint_dir / "heads.safetensors"),
        "config": config,
    }
    (results_dir / "train-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "config"}, indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
