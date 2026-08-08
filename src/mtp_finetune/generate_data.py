"""Self-distillation: let the frozen base write the text its own heads learn from.

Medusa-style heads are trained on the base model's own outputs, not on an
external corpus. That keeps the heads aligned with the distribution they will
have to draft for at inference time - a head is only useful if the base would
have accepted its guess anyway.

Usage:
    python -m mtp_finetune.generate_data --limit 320
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import torch

from mtp_finetune import BASE_MODEL
from mtp_finetune.base_runtime import load_base, peak_vram_mb
from mtp_finetune.prompt_bank import bank_size, build_prompt_bank

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "finetune" / "samples.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--limit", type=int, default=320,
                        help=f"Prompts to sample from the bank of {bank_size()}.")
    parser.add_argument("--completions-per-prompt", type=int, default=1,
                        help="Sampled completions per prompt. Temperature "
                             "sampling makes each one a different continuation, "
                             "which is more training signal per prompt written.")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate from scratch instead of resuming.")
    return parser.parse_args(argv)


def batch_seed_offset(sample_id: str) -> int:
    """Process-stable seed offset derived from a sample id."""
    return zlib.crc32(sample_id.encode("utf-8")) % 1_000_003


def already_generated(path: Path) -> set[str]:
    """Prompt ids already on disk, so an interrupted run can resume."""
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def expand_completions(
    prompts: list[dict[str, str]],
    completions_per_prompt: int,
) -> list[dict[str, str]]:
    """One work item per (prompt, completion) pair, each with its own sample id.

    Completions of the same prompt stay adjacent. The train/held-out split is a
    tail of this file, so keeping them together is what stops one completion of
    a prompt training the heads that are then scored on another completion of
    the same prompt.
    """
    if completions_per_prompt < 1:
        raise ValueError("--completions-per-prompt must be at least 1.")
    if completions_per_prompt == 1:
        return prompts
    return [
        {**prompt, "id": f"{prompt['id']}#c{index}", "prompt_id": prompt["id"]}
        for prompt in prompts
        for index in range(completions_per_prompt)
    ]


def rewrite_in_bank_order(path: Path, bank: list[dict[str, str]]) -> int:
    """Reorder the samples file to match the bank, keeping completions adjacent.

    Resuming a run with a higher --completions-per-prompt appends the new
    completions after every existing row, which would scatter a prompt's
    completions across the train/held-out boundary. Rewriting in bank order
    after every run makes the file's layout independent of how it was built.
    Returns the number of rows that were not in the bank and kept at the end.
    """
    with path.open(encoding="utf-8") as handle:
        rows = {
            record["id"]: line
            for line in handle
            if line.strip()
            for record in (json.loads(line),)
        }
    ordered = [rows.pop(item["id"]) for item in bank if item["id"] in rows]
    leftovers = list(rows.values())
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        sink.writelines(ordered + leftovers)
    temporary.replace(path)
    return len(leftovers)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = already_generated(args.output)

    bank = expand_completions(
        build_prompt_bank(limit=args.limit), args.completions_per_prompt
    )
    prompts = [p for p in bank if p["id"] not in done]
    if not prompts:
        print(f"Nothing to do: {len(done)} samples already in {args.output}")
        print(f"Reordered to bank order ({rewrite_in_bank_order(args.output, bank)} "
              "rows not in the bank kept at the end)")
        return 0
    print(f"Generating {len(prompts)} samples ({len(done)} already present) "
          f"with {args.model}")

    runtime = load_base(args.model)
    tokenizer = runtime.tokenizer
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left pad

    started = time.perf_counter()
    generated_tokens = 0
    with args.output.open("a", encoding="utf-8", newline="\n") as sink:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            # Seed per batch, not once per run, so resuming an interrupted run
            # reproduces the same completions it would have written. Python's
            # built-in hash() is salted per process and would not survive that.
            torch.manual_seed(args.seed + batch_seed_offset(batch[0]["id"]))
            encoded = tokenizer.apply_chat_template(
                [[{"role": "user", "content": item["text"]}] for item in batch],
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            ).to(runtime.device)

            with torch.no_grad():
                outputs = runtime.model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = outputs[:, encoded["input_ids"].shape[1] :]

            for item, row in zip(batch, new_tokens, strict=True):
                completion_ids = [
                    token for token in row.tolist() if token != tokenizer.pad_token_id
                ]
                generated_tokens += len(completion_ids)
                sink.write(json.dumps({
                    "id": item["id"],
                    "prompt_id": item.get("prompt_id", item["id"]),
                    "category": item["category"],
                    "prompt": item["text"],
                    "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                    "completion_tokens": len(completion_ids),
                }, ensure_ascii=False) + "\n")
            sink.flush()

            elapsed = time.perf_counter() - started
            finished = start + len(batch)
            print(f"  {finished}/{len(prompts)} prompts | "
                  f"{generated_tokens} tokens | {elapsed:.0f}s | "
                  f"{generated_tokens / max(elapsed, 1e-6):.1f} tok/s", flush=True)

    elapsed = time.perf_counter() - started
    stray = rewrite_in_bank_order(args.output, bank)
    print(f"Done: {len(prompts)} samples, {generated_tokens} tokens in {elapsed / 60:.1f} min "
          f"(peak VRAM {peak_vram_mb()} MB) -> {args.output}")
    print(f"Rewritten in bank order; {stray} rows were not in the bank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
