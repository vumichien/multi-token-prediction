import csv
import hashlib
import json
from pathlib import Path

import pytest

from mtp_benchmark.results import build_raw_payload, write_results


def test_write_results_serializes_json_and_csv(tmp_path: Path) -> None:
    payload = build_raw_payload({"policy": "auto"}, {"env": "ok"}, [], [], [{"scope": "overall", "mode": "baseline"}], dry_run=False)
    artifacts = write_results(tmp_path, payload, [{"scope": "overall", "mode": "baseline", "trials": 1}])
    raw = json.loads(Path(artifacts["raw_json"]).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(Path(artifacts["summary_csv"]).open(encoding="utf-8")))
    manifest = json.loads(Path(artifacts["manifest_json"]).read_text(encoding="utf-8"))
    assert raw["schema_version"] == "1.0.0"
    assert rows[0]["mode"] == "baseline"
    raw_bytes = Path(artifacts["raw_json"]).read_bytes()
    assert manifest["artifacts"]["raw-results.json"]["sha256"] == hashlib.sha256(raw_bytes).hexdigest()


def test_write_results_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    payload = build_raw_payload({}, {}, [], [], [], dry_run=True)
    write_results(tmp_path, payload, [])
    with pytest.raises(RuntimeError, match="Refusing to overwrite benchmark evidence"):
        write_results(tmp_path, payload, [])


def test_write_results_raises_actionable_error_for_bad_output_dir(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unable to write result artifacts"):
        write_results(blocker / "nested", {"ok": True}, [])


def test_payload_marks_errors_as_failed() -> None:
    payload = build_raw_payload({}, {}, [], [], [], dry_run=False, errors=["boom"])
    assert payload["status"] == "failed"
