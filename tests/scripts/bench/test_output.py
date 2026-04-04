from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.bench.output import _resolve_output_dir, _write_json, _write_jsonl


class TestWriteJson:
    def test_sorted_keys_and_trailing_newline(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        _write_json(p, {"b": 2, "a": 1})
        content = p.read_text()
        parsed = json.loads(content)
        assert parsed == {"a": 1, "b": 2}
        assert content.endswith("\n")


class TestWriteJsonl:
    def test_one_record_per_line(self, tmp_path: Path) -> None:
        p = tmp_path / "out.jsonl"
        _write_jsonl(p, [{"x": 1}, {"y": 2}])
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"x": 1}
        assert json.loads(lines[1]) == {"y": 2}


class TestResolveOutputDir:
    def test_closed_loop_includes_concurrency(self) -> None:
        args = Namespace(
            out="results",
            scenario="short_short",
            endpoint="stream_v2",
            mode="closed",
            concurrency=8,
        )
        path = str(_resolve_output_dir(args))
        assert "concurrency=8" in path
        assert "scenario=short_short" in path
        assert "mode=closed" in path

    def test_open_loop_includes_arrival_rate(self) -> None:
        args = Namespace(
            out="results",
            scenario="burst",
            endpoint="generate_v2",
            mode="open",
            arrival_rate=10.0,
        )
        path = str(_resolve_output_dir(args))
        assert "arrival_rate=10.0" in path
        assert "mode=open" in path
