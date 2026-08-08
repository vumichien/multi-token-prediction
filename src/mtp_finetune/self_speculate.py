"""Self-speculative greedy decoding: the heads draft, the base verifies.

One pass of the loop:

1. Feed `[known_token, draft_2, draft_3, draft_4]` through the base in a single
   forward. The base's greedy prediction at each position tells us what it
   would have produced there anyway.
2. Accept the longest prefix of drafts the base agrees with. The first
   disagreement is replaced by the base's own token, so every token emitted is
   one the base's own argmax chose. The heads can cost speed, never content.

   That is not quite the same as being bit-identical to a token-at-a-time
   greedy run, and this module measures the difference rather than claiming it
   away. A verification pass scores four positions in one forward; plain greedy
   scores one. In bf16 those two paths round a 2048-wide matmul slightly
   differently, so wherever the base's top two logits sit within a couple of
   ulps of each other, the two runs can pick different tokens - both of them
   the argmax of a legitimate evaluation of the same prefix. When outputs
   diverge, the run records where and reports the base's own top-1/top-2 gap
   there, so a reader can see whether it was a near-tie or something worse.
3. The hidden state at the last accepted position supplies the next three
   drafts, so the whole pass costs exactly one forward.

`accepted + 1` tokens therefore come out of every forward pass. That ratio is
the algorithmic result and does not depend on how fast this Python loop is.
Wall-clock tok/s is reported next to it but is implementation-bound: an
unfused fp32 head stack in eager PyTorch is not a serving runtime.

Usage:
    python -m mtp_finetune.self_speculate
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import DynamicCache

from mtp_finetune import BASE_MODEL, NUM_HEADS
from mtp_finetune.base_runtime import load_base, peak_vram_mb, prompt_token_ids
from mtp_finetune.eval_heads import default_checkpoint, load_heads
from mtp_finetune.heads import MTPHeads

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = REPO_ROOT / "data" / "prompts.json"
DEFAULT_RUN = "finetune-smollm2-mtp-heads"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to results/<run-name>/self-speculation.json")
    return parser.parse_args(argv)


@torch.no_grad()
def draft_tokens(heads: MTPHeads, hidden: torch.Tensor) -> list[int]:
    """Greedy draft for t+2..t+K+1 from a single hidden state vector."""
    row = hidden.float().unsqueeze(0)
    return [
        int(lm_head(block(row)).argmax(dim=-1).item())
        for block, lm_head in zip(heads.blocks, heads.lm_heads, strict=True)
    ]


@torch.no_grad()
def greedy_baseline(
    runtime: object,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> tuple[list[int], float]:
    """Plain one-token-at-a-time greedy decoding, for both timing and ground truth."""
    input_ids = torch.tensor([prompt_ids], device=runtime.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = runtime.model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=runtime.tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return output[0, len(prompt_ids) :].tolist(), elapsed


@torch.no_grad()
def speculative_generate(
    runtime: object,
    heads: MTPHeads,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> dict[str, object]:
    """Greedy decode using the heads as a drafter. Returns tokens and pass statistics."""
    eos_id = runtime.tokenizer.eos_token_id
    cache = DynamicCache()

    torch.cuda.synchronize()
    started = time.perf_counter()

    primed = runtime.model(
        input_ids=torch.tensor([prompt_ids], device=runtime.device),
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
    )
    cache = primed.past_key_values
    generated = [int(primed.logits[0, -1].argmax().item())]
    drafts = draft_tokens(heads, primed.hidden_states[-1][0, -1])
    pending = [generated[0]] + drafts
    forward_passes = 1
    accepted_per_pass: list[int] = []

    # An accepted draft can itself be the end-of-sequence token, so the stop
    # check looks at everything emitted this pass, not just the last token.
    while len(generated) < max_new_tokens and eos_id not in generated:
        prefix_length = cache.get_seq_length()
        outputs = runtime.model(
            input_ids=torch.tensor([pending], device=runtime.device),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        cache = outputs.past_key_values
        forward_passes += 1
        predictions = outputs.logits[0].argmax(dim=-1).tolist()

        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == predictions[accepted]:
            accepted += 1
        accepted_per_pass.append(accepted)

        generated.extend(drafts[:accepted])
        correction = predictions[accepted]
        generated.append(correction)

        # Rejected drafts must leave no trace in the cache.
        cache.crop(prefix_length + 1 + accepted)
        drafts = draft_tokens(heads, outputs.hidden_states[-1][0, accepted])
        pending = [correction] + drafts

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    # A pass emits accepted+1 tokens at once, so the loop can run past the
    # requested length or past EOS. Those tokens are trimmed from the output but
    # the pass that produced them still happened: `tokens_emitted` keeps the
    # untrimmed count so tokens-per-pass is not credited against work that was
    # never charged for, or charged for work whose output was thrown away.
    tokens_emitted = len(generated)
    if eos_id in generated:
        generated = generated[: generated.index(eos_id) + 1]
    return {
        "tokens": generated[:max_new_tokens],
        "tokens_emitted": tokens_emitted,
        "seconds": elapsed,
        "forward_passes": forward_passes,
        "accepted_per_pass": accepted_per_pass,
    }


def bfloat16_ulp(value: float) -> float:
    """Spacing between representable bf16 values near `value` (8-bit mantissa)."""
    magnitude = abs(value)
    if magnitude == 0.0:
        return 0.0
    return float(2.0 ** (math.floor(math.log2(magnitude)) - 7))


@torch.no_grad()
def divergence_report(
    runtime: object,
    prompt_ids: list[int],
    reference: list[int],
    candidate: list[int],
) -> dict[str, object] | None:
    """Where two greedy runs first differ, and how close the base's own call was.

    A near-tie means the two runs disagreed about a coin flip. A wide gap would
    mean one of them actually took a token the base did not prefer, which would
    be a real defect rather than arithmetic.
    """
    limit = min(len(reference), len(candidate))
    index = next((i for i in range(limit) if reference[i] != candidate[i]), None)
    if index is None:
        return None
    context = torch.tensor([prompt_ids + reference[:index]], device=runtime.device)
    logits = runtime.model(input_ids=context).logits[0, -1].float()
    top = torch.topk(logits, 2)
    gap = float(top.values[0] - top.values[1])
    ulp = bfloat16_ulp(float(top.values[0]))
    return {
        "index": index,
        "reference_token": runtime.tokenizer.decode([reference[index]]),
        "speculative_token": runtime.tokenizer.decode([candidate[index]]),
        "base_top1_top2_gap": round(gap, 5),
        "bf16_ulp_at_that_scale": round(ulp, 5),
        "gap_in_ulps": round(gap / ulp, 2) if ulp else None,
        "explained_by_rounding": bool(ulp and gap <= 4 * ulp),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Self-speculation needs CUDA: the timings it reports are only "
            "meaningful against a GPU baseline."
        )
    results_dir = REPO_ROOT / "results" / args.run_name
    checkpoint = args.checkpoint or default_checkpoint(args.run_name)
    output = args.output or (results_dir / "self-speculation.json")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))

    runtime = load_base(args.model)
    heads = load_heads(
        checkpoint, runtime.hidden_size, runtime.vocab_size, args.num_heads, runtime.device
    )

    records: list[dict[str, object]] = []
    for prompt in prompts:
        ids = prompt_token_ids(runtime.tokenizer, prompt["text"])
        reference, _ = greedy_baseline(runtime, ids, args.max_new_tokens)
        for trial in range(args.trials):
            baseline_tokens, baseline_seconds = greedy_baseline(
                runtime, ids, args.max_new_tokens
            )
            speculative = speculative_generate(runtime, heads, ids, args.max_new_tokens)
            spec_tokens = speculative["tokens"]
            comparable = min(len(reference), len(spec_tokens))
            accepted = speculative["accepted_per_pass"]
            records.append({
                "prompt_id": prompt["id"],
                "prompt_category": prompt["category"],
                "trial": trial,
                "baseline_tokens": len(baseline_tokens),
                "baseline_seconds": round(baseline_seconds, 4),
                "baseline_tokens_per_second": round(
                    len(baseline_tokens) / max(baseline_seconds, 1e-6), 2
                ),
                "speculative_tokens": len(spec_tokens),
                "speculative_seconds": round(speculative["seconds"], 4),
                "speculative_tokens_per_second": round(
                    len(spec_tokens) / max(speculative["seconds"], 1e-6), 2
                ),
                "forward_passes": speculative["forward_passes"],
                "tokens_emitted": speculative["tokens_emitted"],
                "tokens_per_forward_pass": round(
                    speculative["tokens_emitted"] / max(speculative["forward_passes"], 1), 3
                ),
                "mean_accepted_drafts": round(
                    statistics.fmean(accepted) if accepted else 0.0, 3
                ),
                "output_matches_greedy": (
                    len(spec_tokens) == len(reference)
                    and spec_tokens == reference
                ),
                "output_prefix_matches_greedy": (
                    spec_tokens[:comparable] == reference[:comparable]
                ),
                "divergence": divergence_report(runtime, ids, reference, spec_tokens),
            })
            print(f"  {prompt['id']} trial {trial}: "
                  f"{records[-1]['tokens_per_forward_pass']} tok/pass, "
                  f"accept {records[-1]['mean_accepted_drafts']}/{args.num_heads}, "
                  f"match={records[-1]['output_matches_greedy']}", flush=True)

    payload = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "num_heads": args.num_heads,
        "max_new_tokens": args.max_new_tokens,
        "trials_per_prompt": args.trials,
        "note": (
            "tokens_per_forward_pass is the algorithmic speedup. Wall-clock "
            "tok/s is implementation-bound: eager fp32 heads in a Python loop, "
            "not a serving runtime. Every emitted token is the base model's own "
            "argmax; where output_matches_greedy is false, `divergence` records "
            "the base's top-1/top-2 logit gap at the first differing position, "
            "which separates bf16 near-ties from an actual decoding defect."
        ),
        "records": records,
        "overall": {
            "mean_tokens_per_forward_pass": round(
                statistics.fmean(r["tokens_per_forward_pass"] for r in records), 3
            ),
            "mean_accepted_drafts": round(
                statistics.fmean(r["mean_accepted_drafts"] for r in records), 3
            ),
            "median_baseline_tokens_per_second": round(
                statistics.median(r["baseline_tokens_per_second"] for r in records), 2
            ),
            "median_speculative_tokens_per_second": round(
                statistics.median(r["speculative_tokens_per_second"] for r in records), 2
            ),
            "all_outputs_match_greedy": all(r["output_matches_greedy"] for r in records),
            "all_divergences_explained_by_rounding": all(
                record["divergence"]["explained_by_rounding"]
                for record in records
                if record["divergence"] is not None
            ),
        },
        "peak_allocated_mb": peak_vram_mb(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n{json.dumps(payload['overall'], indent=2)}\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
