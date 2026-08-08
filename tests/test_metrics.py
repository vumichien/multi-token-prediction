from mtp_benchmark.metrics import (
    extract_speculative_metrics,
    quality_gate_errors,
    summarize_paired_speedups,
    summarize_records,
)


def test_summarize_records_excludes_failures_from_numeric_aggregation() -> None:
    records = [
        {
            "stage": "timed",
            "mode": "baseline",
            "prompt_id": "p1",
            "prompt_category": "instruction",
            "latency_seconds": 2.0,
            "tokens_per_second": 10.0,
            "generated_tokens": 20,
            "peak_allocated_mb": 100.0,
            "peak_reserved_mb": 120.0,
            "accepted_tokens": None,
            "acceptance_rate": None,
            "output_match": True,
        },
        {
            "stage": "timed",
            "mode": "baseline",
            "prompt_id": "p1",
            "prompt_category": "instruction",
            "error": "boom",
        },
    ]
    row = summarize_records(records, ("mode",))[0]
    assert row["failures"] == 1
    assert row["generated_tokens_total"] == 20
    assert row["tokens_per_second_median"] == 10.0
    assert row["output_match_rate"] == 1.0


def test_extract_speculative_metrics_returns_note_when_fields_missing() -> None:
    metrics = extract_speculative_metrics(object())
    assert metrics["accepted_tokens"] is None
    assert "did not expose" in str(metrics["speculative_note"])


def test_summarize_paired_speedups_uses_matched_trials() -> None:
    records = [
        {"stage": "timed", "pair_index": 0, "mode": "baseline", "prompt_id": "p1", "prompt_category": "code", "tokens_per_second": 10.0, "output_match": True},
        {"stage": "timed", "pair_index": 0, "mode": "assistant", "prompt_id": "p1", "prompt_category": "code", "tokens_per_second": 15.0, "output_match": True},
    ]
    rows = summarize_paired_speedups(records)
    assert rows[-1]["paired_speedup_median"] == 1.5


def test_output_mismatch_fails_quality_gate_and_is_excluded_from_speedup() -> None:
    records = [
        {"stage": "timed", "pair_index": 0, "mode": "baseline", "prompt_id": "p1", "prompt_category": "code", "tokens_per_second": 10.0, "output_match": False},
        {"stage": "timed", "pair_index": 0, "mode": "assistant", "prompt_id": "p1", "prompt_category": "code", "tokens_per_second": 20.0, "output_match": False},
    ]
    assert summarize_paired_speedups(records) == []
    assert quality_gate_errors(records) == [
        "Output mismatch in timed pair: prompt=p1, pair_index=0"
    ]
