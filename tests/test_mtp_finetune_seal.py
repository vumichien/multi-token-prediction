import pytest

from mtp_finetune.seal_results import horizon_rows, read_json, speculation_rows

ACCURACY = {
    "overall": {
        "t+1": {"top1_accuracy": 0.81, "cross_entropy": 0.62, "tokens_scored": 100},
        "t+2": {"top1_accuracy": 0.21, "cross_entropy": 5.1, "tokens_scored": 99},
    },
    "by_category": {
        "code": {
            "t+1": {"top1_accuracy": 0.9, "cross_entropy": 0.4, "tokens_scored": 50},
            "t+2": {"top1_accuracy": 0.3, "cross_entropy": 4.2, "tokens_scored": 49},
        }
    },
}

SPECULATION = {
    "overall": {
        "mean_tokens_per_forward_pass": 1.6,
        "mean_accepted_drafts": 0.6,
        "median_baseline_tokens_per_second": 40.0,
        "median_speculative_tokens_per_second": 25.0,
        "all_outputs_match_greedy": False,
        "all_divergences_explained_by_rounding": True,
    },
    "records": [{
        "prompt_id": "code-refactor",
        "prompt_category": "code",
        "trial": 0,
        "tokens_per_forward_pass": 1.7,
        "mean_accepted_drafts": 0.7,
        "speculative_tokens_per_second": 26.0,
        "output_matches_greedy": True,
        "divergence": None,
    }, {
        "prompt_id": "creative-brief",
        "prompt_category": "creative",
        "trial": 0,
        "tokens_per_forward_pass": 1.3,
        "mean_accepted_drafts": 0.3,
        "speculative_tokens_per_second": 22.0,
        "output_matches_greedy": False,
        "divergence": {"gap_in_ulps": 1.0, "explained_by_rounding": True},
    }],
}


def test_horizon_rows_labels_the_frozen_base_head_separately() -> None:
    rows = horizon_rows(ACCURACY)
    modes = {(row["scope"], row["horizon"]): row["mode"] for row in rows}
    assert modes[("overall", "t+1")] == "frozen_base_head"
    assert modes[("overall", "t+2")] == "trained_head"
    assert modes[("per_category", "t+1")] == "frozen_base_head"


def test_horizon_rows_covers_overall_and_per_category() -> None:
    rows = horizon_rows(ACCURACY)
    assert sum(row["scope"] == "overall" for row in rows) == 2
    assert sum(row["scope"] == "per_category" for row in rows) == 2


def test_speculation_rows_include_both_modes_and_per_prompt_detail() -> None:
    rows = speculation_rows(SPECULATION)
    modes = [row["mode"] for row in rows if row["scope"] == "overall"]
    assert modes == ["self_speculative", "greedy_baseline"]
    per_prompt = [row for row in rows if row["scope"] == "per_prompt"]
    assert per_prompt[0]["prompt_id"] == "code-refactor"
    assert per_prompt[0]["all_outputs_match"] is True


def test_speculation_rows_collapse_trials_and_report_trials_as_a_count() -> None:
    # `trials` means a count everywhere else in this repo's summary.csv, so a
    # trial index in that column would read as "0 trials" to anything comparing
    # the two experiments.
    speculation = {
        "overall": dict(SPECULATION["overall"]),
        "records": [
            {**SPECULATION["records"][0], "trial": index,
             "speculative_tokens_per_second": speed}
            for index, speed in enumerate((20.0, 26.0, 24.0))
        ],
    }
    per_prompt = [
        row for row in speculation_rows(speculation) if row["scope"] == "per_prompt"
    ]
    assert len(per_prompt) == 1
    assert per_prompt[0]["trials"] == 3
    assert per_prompt[0]["tokens_per_second_median"] == 24.0
    assert per_prompt[0]["output_match_rate"] == 1.0


def test_speculation_rows_carry_the_rounding_caveat() -> None:
    rows = speculation_rows(SPECULATION)
    overall = next(row for row in rows if row["mode"] == "self_speculative")
    # A false all_outputs_match must not reach the summary without the
    # measurement that explains it.
    assert overall["all_outputs_match"] is False
    assert overall["divergences_explained_by_rounding"] is True
    diverged = next(row for row in rows if row.get("prompt_id") == "creative-brief")
    assert diverged["divergence_gap_in_ulps"] == 1.0
    matched = next(row for row in rows if row.get("prompt_id") == "code-refactor")
    assert matched["divergence_gap_in_ulps"] is None


def test_read_json_names_the_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="train-summary.json"):
        read_json(tmp_path / "train-summary.json")
