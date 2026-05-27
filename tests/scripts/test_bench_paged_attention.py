from __future__ import annotations

from scripts.bench_paged_attention import BenchResult, _build_results_table, _build_row


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

    table = _build_results_table()
    table.add_row(_build_row(4, [16, 64, 128, 512], patched, original))

    rendered = table.get_string()

    assert "prompt_len" in rendered
    assert "patched_pre_ms" in rendered
    assert "original_e2e_ms" in rendered
    assert "16-512" in rendered
    assert "10.25" in rendered
    assert "20.50" in rendered
    assert "  2.00x" in rendered
