import itertools

import torch
from transformers import DynamicCache

from server.executor.executor import BatchExecutor
from server.executor.types import (
    DecodeResult,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
)
from server.model.batch_ops import DecodeBatchOutput, PrefillBatchOutput
from server.model.sampling import SamplingParams

VOCAB = 100
EOS = 2


class FakeTokenizer:
    def __init__(self, token_map: dict[int, str] | None = None) -> None:
        self._map = token_map or {}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._map.get(tid, "x") for tid in token_ids)


class FakeModelRunner:
    def __init__(
        self,
        sample_tokens: list[int] | None = None,
        prefill_batches: list[list[PrefillBatchOutput]] | None = None,
        decode_batches: list[list[DecodeBatchOutput]] | None = None,
        token_map: dict[int, str] | None = None,
    ) -> None:
        self.tokenizer = FakeTokenizer(token_map)
        self._sample_tokens = list(sample_tokens or [])
        self._prefill_batches = list(prefill_batches or [])
        self._decode_batches = list(decode_batches or [])
        self.decode_batch_calls: list[tuple[list[int], list[DynamicCache]]] = []

    @property
    def eos_token_id(self) -> int:
        return EOS

    def sample_token(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> int:
        return self._sample_tokens.pop(0)

    def prefill_batch(self, prompts: list[str]) -> list[PrefillBatchOutput]:
        return self._prefill_batches.pop(0)

    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[DynamicCache],
    ) -> list[DecodeBatchOutput]:
        self.decode_batch_calls.append((token_ids, past_key_values))
        return self._decode_batches.pop(0)


_counter = itertools.count()


def make_req(
    request_id: str | None = None, max_new_tokens: int = 10
) -> GenerationRequestState:
    rid = request_id if request_id is not None else f"req-{next(_counter)}"
    return GenerationRequestState(
        request_id=rid,
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens, temperature=1.0, top_p=1.0
        ),
        prompt=f"prompt-{rid}",
    )


def make_decode_req(
    request_id: str = "r0",
    max_new_tokens: int = 10,
    output_tokens: list[str] | None = None,
) -> GenerationRequestState:
    req = make_req(request_id, max_new_tokens=max_new_tokens)
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()
    req.output_tokens.extend(output_tokens or [])
    return req


def make_prefill_output(num_prompt_tokens: int = 5) -> PrefillBatchOutput:
    return PrefillBatchOutput(
        logits=torch.randn(1, num_prompt_tokens, VOCAB),
        past_key_values=DynamicCache(),
        num_prompt_tokens=num_prompt_tokens,
    )


def make_decode_output() -> DecodeBatchOutput:
    return DecodeBatchOutput(
        logits=torch.randn(1, 1, VOCAB),
        past_key_values=DynamicCache(),
    )


def assert_failures(
    results: list[PrefillResult | DecodeResult | RequestFailure], message: str
) -> None:
    assert all(isinstance(result, RequestFailure) for result in results)
    assert all(
        message in result.error
        for result in results
        if isinstance(result, RequestFailure)
    )


def test_batched_prefill_returns_one_result_per_request() -> None:
    runner = FakeModelRunner(
        prefill_batches=[[make_prefill_output(3), make_prefill_output(4)]],
    )
    reqs = [make_req("r0"), make_req("r1")]

    results = BatchExecutor(runner).batched_prefill(reqs)

    assert len(results) == len(reqs)
    assert all(isinstance(result, PrefillResult) for result in results)


def test_batched_prefill_preserves_request_output_order() -> None:
    out0 = make_prefill_output(3)
    out1 = make_prefill_output(7)
    runner = FakeModelRunner(prefill_batches=[[out0, out1]])

    results = BatchExecutor(runner).batched_prefill([make_req("r0"), make_req("r1")])

    assert isinstance(results[0], PrefillResult)
    assert isinstance(results[1], PrefillResult)
    assert results[0].all_logits is out0.logits
    assert results[0].past_key_values is out0.past_key_values
    assert results[0].num_prompt_tokens == 3
    assert results[1].all_logits is out1.logits
    assert results[1].past_key_values is out1.past_key_values
    assert results[1].num_prompt_tokens == 7


def test_batched_prefill_results_share_start_ns() -> None:
    runner = FakeModelRunner(
        prefill_batches=[[make_prefill_output(), make_prefill_output()]],
    )

    results = BatchExecutor(runner).batched_prefill([make_req("r0"), make_req("r1")])

    assert isinstance(results[0], PrefillResult)
    assert isinstance(results[1], PrefillResult)
    assert results[0].start_ns == results[1].start_ns


def test_batched_prefill_runner_exception_returns_failure_for_all_requests() -> None:
    runner = FakeModelRunner()

    def raise_prefill(prompts: list[str]) -> list[PrefillBatchOutput]:
        raise RuntimeError("model crash")

    runner.prefill_batch = raise_prefill

    results = BatchExecutor(runner).batched_prefill([make_req("r0"), make_req("r1")])

    assert_failures(results, "model crash")


def test_batched_prefill_output_count_mismatch_returns_failure_for_all_requests() -> (
    None
):
    runner = FakeModelRunner(prefill_batches=[[make_prefill_output()]])

    results = BatchExecutor(runner).batched_prefill([make_req("r0"), make_req("r1")])

    assert_failures(results, "Expected 2 prefill outputs, but got 1")


def test_batched_decode_returns_token_text_and_updated_model_state() -> None:
    req = make_decode_req("r0")
    decode_output = make_decode_output()
    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[decode_output]],
        token_map={42: "hello"},
    )

    results = BatchExecutor(runner).batched_decode([req])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DecodeResult)
    assert result.token_id == 42
    assert result.token == "hello"
    assert result.finish_reason is None
    assert result.all_logits is decode_output.logits
    assert result.past_key_values is decode_output.past_key_values
    assert runner.decode_batch_calls == [([42], [req.past_key_values])]


def test_batched_decode_eos_returns_finished_result_without_decode_batch() -> None:
    req = make_decode_req("r0")
    runner = FakeModelRunner(sample_tokens=[EOS])

    results = BatchExecutor(runner).batched_decode([req])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DecodeResult)
    assert result.token_id == EOS
    assert result.token == ""
    assert result.finish_reason == FinishReason.EOS
    assert result.all_logits is None
    assert result.past_key_values is None
    assert runner.decode_batch_calls == []


def test_batched_decode_max_length_returns_finished_result_without_decode_batch() -> (
    None
):
    req = make_decode_req("r0", max_new_tokens=1)
    runner = FakeModelRunner(sample_tokens=[42], token_map={42: "a"})

    results = BatchExecutor(runner).batched_decode([req])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, DecodeResult)
    assert result.token_id == 42
    assert result.token == "a"
    assert result.finish_reason == FinishReason.MAX_LENGTH
    assert result.all_logits is None
    assert result.past_key_values is None
    assert runner.decode_batch_calls == []


def test_batched_decode_missing_logits_returns_request_failure() -> None:
    req = make_decode_req("r0")
    req.all_logits = None

    results = BatchExecutor(FakeModelRunner()).batched_decode([req])

    assert len(results) == 1
    assert isinstance(results[0], RequestFailure)
    assert "No logits available" in results[0].error


def test_batched_decode_missing_past_key_values_returns_request_failure() -> None:
    req = make_decode_req("r0")
    req.past_key_values = None

    results = BatchExecutor(FakeModelRunner()).batched_decode([req])

    assert len(results) == 1
    assert isinstance(results[0], RequestFailure)
    assert "No past_key_values available" in results[0].error


def test_batched_decode_one_bad_request_does_not_block_valid_requests() -> None:
    bad = make_decode_req("bad")
    bad.all_logits = None
    good = make_decode_req("good")
    decode_output = make_decode_output()
    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[decode_output]],
        token_map={42: "a"},
    )

    results = BatchExecutor(runner).batched_decode([bad, good])

    assert isinstance(results[0], RequestFailure)
    assert isinstance(results[1], DecodeResult)
    assert results[1].token == "a"
    assert results[1].all_logits is decode_output.logits
    assert runner.decode_batch_calls == [([42], [good.past_key_values])]


def test_batched_decode_decode_batch_exception_fails_unfinished_requests_only() -> None:
    eos = make_decode_req("eos")
    unfinished = make_decode_req("unfinished")
    runner = FakeModelRunner(sample_tokens=[EOS, 42], token_map={42: "a"})

    def raise_decode(
        token_ids: list[int], past_key_values: list[DynamicCache]
    ) -> list[DecodeBatchOutput]:
        runner.decode_batch_calls.append((token_ids, past_key_values))
        raise RuntimeError("batch decode crash")

    runner.decode_batch = raise_decode

    results = BatchExecutor(runner).batched_decode([eos, unfinished])

    assert isinstance(results[0], DecodeResult)
    assert results[0].finish_reason == FinishReason.EOS
    assert isinstance(results[1], RequestFailure)
    assert "batch decode crash" in results[1].error
    assert runner.decode_batch_calls == [([42], [unfinished.past_key_values])]


def test_batched_decode_output_count_mismatch_fails_unfinished_requests_only() -> None:
    eos = make_decode_req("eos")
    unfinished = make_decode_req("unfinished")
    runner = FakeModelRunner(
        sample_tokens=[EOS, 42],
        decode_batches=[[]],
        token_map={42: "a"},
    )

    results = BatchExecutor(runner).batched_decode([eos, unfinished])

    assert isinstance(results[0], DecodeResult)
    assert results[0].finish_reason == FinishReason.EOS
    assert isinstance(results[1], RequestFailure)
    assert "Expected 1 decode outputs, but got 0" in results[1].error
