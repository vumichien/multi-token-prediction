"""Seal a fine-tune run into the same immutable evidence layout as the benchmark.

Training and evaluation each write their own artifacts as they go, because a
run that dies at hour two should still leave usable data. This step is what
turns that working directory into evidence: one `raw-results.json`, one
`summary.csv`, and a `manifest.json` of sha256 digests, written through the
benchmark's own writer so both experiments in the article seal the same way.

The manifest covers `raw-results.json` and `summary.csv`. That is enough to
pin every number, because the per-stage artifacts are copied wholesale into
`raw-results.json` as records - but the loose copies left beside it
(`training-log.csv`, `heldout-accuracy.json`, `self-speculation.json`) are not
themselves digested. Check a claim against the sealed pair, not against those.

Usage:
    python -m mtp_finetune.seal_results
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from mtp_benchmark.environment import collect_environment_metadata
from mtp_benchmark.results import build_raw_payload, write_results
from mtp_finetune import BASE_MODEL

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "finetune-smollm2-mtp-heads"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN)
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "data" / "prompts.json")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing run artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def horizon_rows(accuracy: dict[str, object]) -> list[dict[str, object]]:
    """One summary row per prediction horizon, overall and per prompt family."""
    rows: list[dict[str, object]] = []
    for horizon, scores in accuracy["overall"].items():
        rows.append({
            "scope": "overall",
            "mode": "frozen_base_head" if horizon == "t+1" else "trained_head",
            "horizon": horizon,
            "top1_accuracy": scores["top1_accuracy"],
            "cross_entropy": scores["cross_entropy"],
            "tokens_scored": scores["tokens_scored"],
        })
    for category, group in accuracy["by_category"].items():
        for horizon, scores in group.items():
            rows.append({
                "scope": "per_category",
                "mode": "frozen_base_head" if horizon == "t+1" else "trained_head",
                "prompt_category": category,
                "horizon": horizon,
                "top1_accuracy": scores["top1_accuracy"],
                "cross_entropy": scores["cross_entropy"],
                "tokens_scored": scores["tokens_scored"],
            })
    return rows


def speculation_rows(speculation: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{
        "scope": "overall",
        "mode": "self_speculative",
        "tokens_per_forward_pass": speculation["overall"]["mean_tokens_per_forward_pass"],
        "accepted_tokens_mean": speculation["overall"]["mean_accepted_drafts"],
        "tokens_per_second_median": speculation["overall"][
            "median_speculative_tokens_per_second"
        ],
        "all_outputs_match": speculation["overall"]["all_outputs_match_greedy"],
        "divergences_explained_by_rounding": speculation["overall"][
            "all_divergences_explained_by_rounding"
        ],
    }, {
        "scope": "overall",
        "mode": "greedy_baseline",
        "tokens_per_second_median": speculation["overall"][
            "median_baseline_tokens_per_second"
        ],
    }]
    # One row per prompt, not per trial. `trials` means a count everywhere else
    # in this repo's summary.csv (see mtp_benchmark.metrics), and the trials of
    # one prompt differ only in wall-clock, so a median is the honest summary.
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in speculation["records"]:
        grouped.setdefault(record["prompt_id"], []).append(record)
    for prompt_id, trials in grouped.items():
        first = trials[0]
        rows.append({
            "scope": "per_prompt",
            "mode": "self_speculative",
            "prompt_id": prompt_id,
            "prompt_category": first["prompt_category"],
            "trials": len(trials),
            "tokens_per_forward_pass": first["tokens_per_forward_pass"],
            "accepted_tokens_mean": first["mean_accepted_drafts"],
            "tokens_per_second_median": round(
                statistics.median(t["speculative_tokens_per_second"] for t in trials), 2
            ),
            "all_outputs_match": all(t["output_matches_greedy"] for t in trials),
            "output_match_rate": round(
                sum(bool(t["output_matches_greedy"]) for t in trials) / len(trials), 4
            ),
            "divergence_gap_in_ulps": (
                first["divergence"]["gap_in_ulps"] if first.get("divergence") else None
            ),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = REPO_ROOT / "results" / args.run_name

    train_summary = read_json(run_dir / "train-summary.json")
    accuracy = read_json(run_dir / "heldout-accuracy.json")
    speculation = read_json(run_dir / "self-speculation.json")

    summary_rows = horizon_rows(accuracy) + speculation_rows(speculation)
    raw_payload = build_raw_payload(
        config=train_summary["config"],
        environment=collect_environment_metadata(
            target_model=BASE_MODEL,
            assistant_model=f"mtp_finetune heads K={train_summary['config']['num_heads']}",
            prompts_path=args.prompts,
        ),
        prompts=json.loads(args.prompts.read_text(encoding="utf-8")),
        records=[
            {"artifact": "training_log", "rows": read_csv_rows(run_dir / "training-log.csv")},
            {"artifact": "heldout_log", "rows": read_csv_rows(run_dir / "heldout-log.csv")},
            {"artifact": "heldout_accuracy", "payload": accuracy},
            {"artifact": "self_speculation", "payload": speculation},
        ],
        summaries=[{"train": train_summary}],
        dry_run=False,
    )

    written = write_results(run_dir, raw_payload, summary_rows)
    print(json.dumps(written, indent=2))
    print(f"Sealed {args.run_name}: {len(summary_rows)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
