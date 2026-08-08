from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

SCHEMA_VERSION = "1.0.0"


def build_raw_payload(
    config: dict[str, object],
    environment: dict[str, object],
    prompts: list[dict[str, str]],
    records: list[dict[str, object]],
    summaries: list[dict[str, object]],
    *,
    dry_run: bool,
    errors: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if errors else ("dry_run" if dry_run else "completed"),
        "config": config,
        "environment": environment,
        "prompts": prompts,
        "records": records,
        "summaries": summaries,
        "errors": errors or [],
    }


def write_results(
    output_dir: Path,
    raw_payload: dict[str, object],
    summary_rows: list[dict[str, object]],
) -> dict[str, str]:
    raw_path = output_dir / "raw-results.json"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.json"
    lock_path = output_dir / ".benchmark-artifacts.lock"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in (raw_path, summary_path, manifest_path)):
            raise FileExistsError(
                f"Refusing to overwrite benchmark evidence in {output_dir}; choose a new output directory."
            )
        try:
            with lock_path.open("x", encoding="utf-8") as lock:
                lock.write("immutable benchmark artifact directory\n")
        except FileExistsError as exc:
            raise FileExistsError(
                f"Benchmark artifact directory is locked: {output_dir}; choose a new output directory."
            ) from exc
        raw_text = json.dumps(raw_payload, indent=2)
        summary_text = _summary_csv_text(summary_rows)
        _atomic_write_text(raw_path, raw_text)
        _atomic_write_text(summary_path, summary_text)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifacts": {
                raw_path.name: _digest(raw_text),
                summary_path.name: _digest(summary_text),
            },
        }
        _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    except OSError as exc:
        raise RuntimeError(f"Unable to write result artifacts under {output_dir}: {exc}") from exc
    return {
        "raw_json": str(raw_path),
        "summary_csv": str(summary_path),
        "manifest_json": str(manifest_path),
    }


def _summary_csv_text(rows: list[dict[str, object]]) -> str:
    fieldnames = _fieldnames(rows)
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})
    return handle.getvalue()


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def _digest(content: str) -> dict[str, object]:
    encoded = content.encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    fields = [
        "scope",
        "mode",
        "prompt_id",
        "prompt_category",
        "trials",
        "successes",
        "failures",
        "latency_seconds_median",
        "latency_seconds_mean",
        "latency_seconds_stdev",
        "tokens_per_second_median",
        "tokens_per_second_mean",
        "tokens_per_second_stdev",
        "paired_speedup_median",
        "paired_speedup_mean",
        "paired_speedup_stdev",
        "generated_tokens_total",
        "peak_allocated_mb_max",
        "peak_reserved_mb_max",
        "accepted_tokens_mean",
        "acceptance_rate_mean",
        "output_match_rate",
        "all_outputs_match",
        "errors",
    ]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
