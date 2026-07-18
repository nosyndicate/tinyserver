from __future__ import annotations

import pytest

pytest.importorskip("triton")

from scripts.bench_paged_attention import (  # noqa: E402
    BenchResult,
    _build_results_table,
    _build_row,
    _format_ratio,
)


def test_build_results_table_renders_expected_headers_and_values() -> None:
    patched = BenchResult(
        prefill_ms=10.25,
        prefill_tokens_per_s=512.0,
        decode_tokens_per_s=256.0,
        e2e_ms=20.5,
        e2e_tokens_per_s=0.0,
    )
    original = BenchResult(
        prefill_ms=20.5,
        prefill_tokens_per_s=256.0,
        decode_tokens_per_s=128.0,
        e2e_ms=41.0,
        e2e_tokens_per_s=0.0,
    )

    table = _build_results_table(phase="all")
    table.add_row(_build_row(4, [16, 64, 128, 512], patched, original, phase="all"))

    rendered = table.get_string()

    assert "prompt_len" in rendered
    assert "patched_pre_ms" in rendered
    assert "original_e2e_ms" in rendered
    assert "16-512" in rendered
    assert "10.25" in rendered
    assert "20.50" in rendered
    assert "  2.00x" in rendered


def test_build_results_table_prefill_phase_has_only_prefill_columns() -> None:
    rendered = _build_results_table(phase="prefill").get_string()

    for col in (
        "patched_pre_ms",
        "original_pre_ms",
        "pre_ratio",
        "patched_ptok/s",
        "original_ptok/s",
        "pre_tps_ratio",
    ):
        assert col in rendered
    for absent in (
        "patched_dtok/s",
        "original_dtok/s",
        "decode_tps_ratio",
        "patched_e2e_ms",
        "original_e2e_ms",
        "e2e_ratio",
    ):
        assert absent not in rendered


def test_build_results_table_decode_phase_has_only_decode_columns() -> None:
    rendered = _build_results_table(phase="decode").get_string()

    for col in ("patched_dtok/s", "original_dtok/s", "decode_tps_ratio"):
        assert col in rendered
    for absent in (
        "patched_pre_ms",
        "original_pre_ms",
        "pre_ratio",
        "patched_ptok/s",
        "original_ptok/s",
        "pre_tps_ratio",
        "patched_e2e_ms",
        "original_e2e_ms",
        "e2e_ratio",
    ):
        assert absent not in rendered


def test_build_row_prefill_phase_produces_correct_column_count() -> None:
    result = BenchResult(prefill_ms=5.0, prefill_tokens_per_s=200.0)
    row = _build_row(2, [32], result, result, phase="prefill")
    # batch + prompt_len + 6 prefill cols
    assert len(row) == 8


def test_build_row_decode_phase_produces_correct_column_count() -> None:
    result = BenchResult(decode_tokens_per_s=100.0)
    row = _build_row(2, [32], result, result, phase="decode")
    # batch + prompt_len + 3 decode cols
    assert len(row) == 5


def test_format_ratio_higher_is_better() -> None:
    assert _format_ratio(2.0, 1.0, True) == "  2.00x"


def test_format_ratio_lower_is_better() -> None:
    assert _format_ratio(1.0, 2.0, False) == "  2.00x"


def test_format_ratio_zero_original_returns_na() -> None:
    assert _format_ratio(1.0, 0.0, True) == "n/a"


def test_format_ratio_zero_patched_returns_na() -> None:
    assert _format_ratio(0.0, 1.0, False) == "n/a"
