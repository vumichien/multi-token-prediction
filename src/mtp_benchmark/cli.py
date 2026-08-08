from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model_loader import default_assistant_model
from .runner import BenchmarkConfig, run_benchmark

DEFAULT_TARGET_MODEL = "google/gemma-4-E2B-it"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Gemma baseline versus assistant generation.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--assistant-model")
    parser.add_argument("--prompts-path", type=Path, default=Path("data/prompts.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/latest"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--policy", choices=["auto", "8bit", "4bit", "cpu_offload", "none"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 1 or args.warmup_runs < 0 or args.timed_runs < 1:
        print("error: max-new-tokens must be >= 1, warmup-runs >= 0, timed-runs >= 1", file=sys.stderr)
        return 2
    assistant_model = args.assistant_model or default_assistant_model(args.target_model)
    config = BenchmarkConfig(
        target_model=args.target_model,
        assistant_model=assistant_model,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        policy=args.policy,
    )
    try:
        artifacts = run_benchmark(config, dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.dry_run:
        import json

        payload = json.loads(Path(artifacts["raw_json"]).read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            print(
                f"error: benchmark finished with recorded trial failures; see {artifacts['raw_json']}",
                file=sys.stderr,
            )
            return 1
    mode = "dry run" if args.dry_run else "benchmark"
    print(
        f"{mode} complete | raw={artifacts['raw_json']} | summary={artifacts['summary_csv']} | "
        f"assistant={assistant_model}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
