from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mtp_benchmark.model_loader import LoadedRuntime, RuntimeLoadError
from mtp_benchmark.runner import BenchmarkConfig, run_benchmark


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def to(self, _device: str) -> "FakeTensor":
        return self

    def __getitem__(self, item: slice | int) -> "FakeTensor | int":
        if isinstance(item, slice):
            return FakeTensor(self.values[item])
        return self.values[item]

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, _text: str, *, return_tensors: str) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        return {"input_ids": FakeTensor([10, 11])}

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return ",".join(str(item) for item in ids)


class FakeModel:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def parameters(self):
        yield SimpleNamespace(device="cuda:0")

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        target_model="google/gemma-4-E2B-it",
        assistant_model="google/gemma-4-E2B-it-assistant",
        prompts_path=tmp_path / "prompts.json",
        output_dir=tmp_path / "results",
        max_new_tokens=16,
        warmup_runs=1,
        timed_runs=2,
        policy="auto",
    )


def test_run_benchmark_writes_failed_status_and_loader_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    config = _config(tmp_path)
    monkeypatch.setattr(
        "mtp_benchmark.runner.load_prompts",
        lambda _path: [{"id": "p1", "category": "instruction", "text": "hello"}],
    )
    monkeypatch.setattr("mtp_benchmark.runner.collect_environment_metadata", lambda *_args: {"env": "ok"})
    monkeypatch.setattr(
        "mtp_benchmark.runner.load_runtime",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeLoadError(
                "no runtime",
                attempts=[{"policy": "8bit", "status": "failed", "error": "oom"}],
                support={"assistant_model_arg": True, "recognized_configs": ["gemma4"]},
            )
        ),
    )
    monkeypatch.setattr("mtp_benchmark.runner.unload_runtime", lambda _runtime: None)
    monkeypatch.setattr(
        "mtp_benchmark.runner.write_results",
        lambda _output_dir, raw_payload, summary_rows: captured.update(
            {"raw_payload": raw_payload, "summary_rows": summary_rows}
        )
        or {"raw_json": "raw.json", "summary_csv": "summary.csv"},
    )

    with pytest.raises(RuntimeLoadError, match="no runtime"):
        run_benchmark(config, dry_run=False)

    assert captured["raw_payload"]["status"] == "failed"
    assert captured["raw_payload"]["environment"]["loader"]["attempts"][0]["policy"] == "8bit"
    assert captured["raw_payload"]["errors"] == ["no runtime"]


def test_run_benchmark_pairs_baseline_and_assistant_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    config = _config(tmp_path)
    prompts = [{"id": "p1", "category": "instruction", "text": "hello"}]
    outputs = [
        SimpleNamespace(sequences=[FakeTensor([10, 11, 21, 22])]),
        SimpleNamespace(
            sequences=[FakeTensor([10, 11, 21, 22])],
            accepted_tokens=2,
            acceptance_rate=1.0,
        ),
        SimpleNamespace(sequences=[FakeTensor([10, 11, 21, 22])]),
        SimpleNamespace(
            sequences=[FakeTensor([10, 11, 21, 22])],
            accepted_tokens=2,
            acceptance_rate=1.0,
        ),
        SimpleNamespace(
            sequences=[FakeTensor([10, 11, 21, 22])],
            accepted_tokens=2,
            acceptance_rate=1.0,
        ),
        SimpleNamespace(sequences=[FakeTensor([10, 11, 21, 22])]),
    ]
    runtime = LoadedRuntime(
        torch=None,
        tokenizer=FakeTokenizer(),
        model=FakeModel(outputs),
        assistant_model=object(),
        metadata={"selected_policy": "8bit"},
    )
    monkeypatch.setattr("mtp_benchmark.runner.load_prompts", lambda _path: prompts)
    monkeypatch.setattr("mtp_benchmark.runner.collect_environment_metadata", lambda *_args: {"env": "ok"})
    monkeypatch.setattr("mtp_benchmark.runner.load_runtime", lambda *_args: runtime)
    monkeypatch.setattr("mtp_benchmark.runner.unload_runtime", lambda _runtime: None)
    monkeypatch.setattr(
        "mtp_benchmark.runner.measure_call",
        lambda run_call, _torch: (
            run_call(),
            {"latency_seconds": 0.5, "peak_allocated_mb": 10.0, "peak_reserved_mb": 12.0},
            None,
        ),
    )
    monkeypatch.setattr(
        "mtp_benchmark.runner.write_results",
        lambda _output_dir, raw_payload, summary_rows: captured.update(
            {"raw_payload": raw_payload, "summary_rows": summary_rows}
        )
        or {"raw_json": "raw.json", "summary_csv": "summary.csv"},
    )

    artifacts = run_benchmark(config, dry_run=False)

    assert artifacts["raw_json"] == "raw.json"
    assert captured["raw_payload"]["status"] == "completed"
    assert [("assistant_model" in call) for call in runtime.model.calls] == [
        False,
        True,
        False,
        True,
        True,
        False,
    ]
    timed_records = [record for record in captured["raw_payload"]["records"] if record["stage"] == "timed"]
    assert all(record["output_match"] is True for record in timed_records)
    assert all(record["max_new_tokens"] == 16 for record in timed_records)
    assert all(record["do_sample"] is False for record in timed_records)
    assert any(row["scope"] == "paired_overall" for row in captured["summary_rows"])
