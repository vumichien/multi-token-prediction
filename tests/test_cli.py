from __future__ import annotations

import pytest

from mtp_benchmark import cli


def test_build_parser_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "Benchmark Gemma baseline versus assistant generation." in capsys.readouterr().out


def test_main_dry_run_calls_runner(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "mtp_benchmark.cli.run_benchmark",
        lambda config, *, dry_run: {
            "raw_json": str(config.output_dir / "raw.json"),
            "summary_csv": str(config.output_dir / "summary.csv"),
        }
        if dry_run
        else (_ for _ in ()).throw(AssertionError("dry_run flag not passed")),
    )

    exit_code = cli.main(["--dry-run", "--timed-runs", "1"])

    assert exit_code == 0
    assert "dry run complete" in capsys.readouterr().out
