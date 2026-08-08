from __future__ import annotations

import statistics
import time


def measure_call(run_call: object, torch: object) -> tuple[object | None, dict[str, float | None], Exception | None]:
    cuda = bool(torch and torch.cuda.is_available())
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = run_call()
    except Exception as exc:  # pragma: no cover - exercised through caller behavior
        error = exc
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated() if cuda else None
    reserved = torch.cuda.max_memory_reserved() if cuda else None
    return result, {
        "latency_seconds": round(elapsed, 6),
        "peak_allocated_mb": _to_mb(allocated),
        "peak_reserved_mb": _to_mb(reserved),
    }, error


def extract_speculative_metrics(output: object) -> dict[str, float | int | None | str]:
    for accepted_name, rate_name in _SPECULATIVE_FIELDS:
        accepted = getattr(output, accepted_name, None)
        rate = getattr(output, rate_name, None)
        if accepted is not None or rate is not None:
            return {"accepted_tokens": accepted, "acceptance_rate": rate, "speculative_note": None}
    return {
        "accepted_tokens": None,
        "acceptance_rate": None,
        "speculative_note": "Transformers output did not expose assistant acceptance counters.",
    }


def summarize_records(records: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for record in records:
        if record.get("stage") != "timed":
            continue
        key = tuple(record.get(name) for name in keys)
        grouped.setdefault(key, []).append(record)
    summaries: list[dict[str, object]] = []
    for key, items in grouped.items():
        success = [item for item in items if not item.get("error")]
        equality = [item["output_match"] for item in success if item.get("output_match") is not None]
        summaries.append(
            {name: value for name, value in zip(keys, key)}
            | {
                "trials": len(items),
                "successes": len(success),
                "failures": len(items) - len(success),
                "latency_seconds_median": _median(success, "latency_seconds"),
                "latency_seconds_mean": _mean(success, "latency_seconds"),
                "latency_seconds_stdev": _stdev(success, "latency_seconds"),
                "tokens_per_second_median": _median(success, "tokens_per_second"),
                "tokens_per_second_mean": _mean(success, "tokens_per_second"),
                "tokens_per_second_stdev": _stdev(success, "tokens_per_second"),
                "generated_tokens_total": int(sum(item.get("generated_tokens") or 0 for item in success)),
                "peak_allocated_mb_max": _maximum(success, "peak_allocated_mb"),
                "peak_reserved_mb_max": _maximum(success, "peak_reserved_mb"),
                "accepted_tokens_mean": _mean(success, "accepted_tokens"),
                "acceptance_rate_mean": _mean(success, "acceptance_rate"),
                "output_match_rate": round(sum(bool(flag) for flag in equality) / len(equality), 6)
                if equality
                else None,
                "all_outputs_match": all(equality) if equality else None,
                "errors": " | ".join(sorted({str(item["error"]) for item in items if item.get("error")})),
            }
        )
    return summaries


def summarize_paired_speedups(records: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs: dict[tuple[object, object, object], dict[str, dict[str, object]]] = {}
    for record in records:
        if (
            record.get("stage") != "timed"
            or record.get("error")
            or record.get("output_match") is not True
        ):
            continue
        key = (record.get("prompt_id"), record.get("prompt_category"), record.get("pair_index"))
        pairs.setdefault(key, {})[str(record.get("mode"))] = record
    ratios: list[tuple[object, object, float]] = []
    for (prompt_id, category, _), pair in pairs.items():
        baseline = pair.get("baseline", {}).get("tokens_per_second")
        assistant = pair.get("assistant", {}).get("tokens_per_second")
        if baseline and assistant:
            ratios.append((prompt_id, category, float(assistant) / float(baseline)))
    rows: list[dict[str, object]] = []
    for prompt_id, category in sorted({(item[0], item[1]) for item in ratios}):
        values = [item[2] for item in ratios if item[:2] == (prompt_id, category)]
        rows.append(_speedup_row("paired_prompt", values, prompt_id, category))
    if ratios:
        rows.append(_speedup_row("paired_overall", [item[2] for item in ratios], None, None))
    return rows


def quality_gate_errors(records: list[dict[str, object]]) -> list[str]:
    """Return one publication-blocking error for every mismatched generation pair."""
    mismatches = {
        (
            str(record.get("stage")),
            str(record.get("prompt_id")),
            str(record.get("pair_index")),
        )
        for record in records
        if record.get("output_match") is False
    }
    return [
        f"Output mismatch in {stage} pair: prompt={prompt_id}, pair_index={pair_index}"
        for stage, prompt_id, pair_index in sorted(mismatches)
    ]


def _speedup_row(scope: str, values: list[float], prompt_id: object, category: object) -> dict[str, object]:
    return {
        "scope": scope,
        "mode": "assistant_vs_baseline",
        "prompt_id": prompt_id,
        "prompt_category": category,
        "trials": len(values),
        "paired_speedup_median": round(statistics.median(values), 6),
        "paired_speedup_mean": round(statistics.mean(values), 6),
        "paired_speedup_stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def _mean(records: list[dict[str, object]], field: str) -> float | None:
    values = [float(item[field]) for item in records if item.get(field) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _median(records: list[dict[str, object]], field: str) -> float | None:
    values = [float(item[field]) for item in records if item.get(field) is not None]
    return round(statistics.median(values), 6) if values else None


def _stdev(records: list[dict[str, object]], field: str) -> float | None:
    values = [float(item[field]) for item in records if item.get(field) is not None]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return round(statistics.stdev(values), 6)


def _maximum(records: list[dict[str, object]], field: str) -> float | None:
    values = [float(item[field]) for item in records if item.get(field) is not None]
    return round(max(values), 6) if values else None


def _to_mb(value: int | None) -> float | None:
    return round(value / (1024**2), 3) if value is not None else None


_SPECULATIVE_FIELDS = (
    ("accepted_tokens", "acceptance_rate"),
    ("assistant_accepted_tokens", "assistant_acceptance_rate"),
)
