import torch
from transformers import DynamicCache

from server.executor.executor import Executor
from server.executor.types import (
    DecodeResult,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
    RequestStatus,
)
from server.model.sampling import SamplingParams

VOCAB = 100
EOS = 2


class FakeTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return {42: "a"}.get(token_ids[0], "x")


class FakeModelOutput:
    def __init__(self, logits: torch.Tensor, past_key_values: DynamicCache) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class FakeModel:
    def __init__(self, output: FakeModelOutput) -> None:
        self._output = output
        self.calls = 0
        self.exception: Exception | None = None

    def __call__(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None,
        use_cache: bool,
    ) -> FakeModelOutput:
        self.calls += 1
        if self.exception is not None:
            raise self.exception
        return self._output


class FakeRunner:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self._eos_token_id = EOS
        self._sample_tokens: list[int] = []
        self._prefill_output = (
            torch.randn(1, 3, VOCAB),
            DynamicCache(),
            3,
        )
        self.prefill_exception: Exception | None = None
        self._decode_output = FakeModelOutput(torch.randn(1, 1, VOCAB), DynamicCache())
        self.model = FakeModel(self._decode_output)

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def prefill(self, prompt: str) -> tuple[torch.Tensor, DynamicCache, int]:
        if self.prefill_exception is not None:
            raise self.prefill_exception
        return self._prefill_output

    def sample_token(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> int:
        return self._sample_tokens.pop(0)


def make_req() -> GenerationRequestState:
    return GenerationRequestState(
        request_id="r0",
        sampling_params=SamplingParams(max_new_tokens=10, temperature=1.0, top_p=1.0),
        prompt="hello",
    )


def test_prefill_returns_prefill_result_without_mutating_request_status() -> None:
    runner = FakeRunner()
    executor = Executor(runner)
    req = make_req()

    result = executor.prefill(req)

    assert isinstance(result, PrefillResult)
    assert result.all_logits is runner._prefill_output[0]
    assert result.past_key_values is runner._prefill_output[1]
    assert result.num_prompt_tokens == 3
    assert req.all_logits is None
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()


def test_prefill_returns_request_failure_when_runner_raises() -> None:
    runner = FakeRunner()
    runner.prefill_exception = RuntimeError("prefill boom")
    executor = Executor(runner)
    req = make_req()

    result = executor.prefill(req)

    assert isinstance(result, RequestFailure)
    assert "prefill boom" in result.error
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()


def test_decode_returns_decode_result_with_updated_model_state() -> None:
    runner = FakeRunner()
    runner._sample_tokens = [42]
    executor = Executor(runner)
    req = make_req()
    req.all_logits = torch.randn(1, 3, VOCAB)
    req.past_key_values = DynamicCache()

    result = executor.decode(req)

    assert isinstance(result, DecodeResult)
    assert result.token_id == 42
    assert result.token == "a"
    assert result.is_finished is False
    assert result.all_logits is runner._decode_output.logits
    assert result.past_key_values is runner._decode_output.past_key_values
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()


def test_decode_returns_request_failure_when_logits_missing() -> None:
    runner = FakeRunner()
    executor = Executor(runner)
    req = make_req()

    result = executor.decode(req)

    assert isinstance(result, RequestFailure)
    assert "No logits available" in result.error
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()


def test_decode_returns_eos_without_calling_model() -> None:
    runner = FakeRunner()
    runner._sample_tokens = [EOS]
    executor = Executor(runner)
    req = make_req()
    req.all_logits = torch.randn(1, 3, VOCAB)
    req.past_key_values = DynamicCache()

    result = executor.decode(req)

    assert isinstance(result, DecodeResult)
    assert result.token_id == EOS
    assert result.token == ""
    assert result.finish_reason == FinishReason.EOS
    assert runner.model.calls == 0
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()


def test_decode_returns_max_length_without_calling_model() -> None:
    runner = FakeRunner()
    runner._sample_tokens = [42]
    executor = Executor(runner)
    req = make_req()
    req.sampling_params = SamplingParams(max_new_tokens=1, temperature=1.0, top_p=1.0)
    req.all_logits = torch.randn(1, 3, VOCAB)
    req.past_key_values = DynamicCache()

    result = executor.decode(req)

    assert isinstance(result, DecodeResult)
    assert result.token_id == 42
    assert result.token == "a"
    assert result.finish_reason == FinishReason.MAX_LENGTH
    assert runner.model.calls == 0
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()


def test_decode_returns_request_failure_when_model_decode_raises() -> None:
    runner = FakeRunner()
    runner._sample_tokens = [42]
    runner.model.exception = RuntimeError("decode boom")
    executor = Executor(runner)
    req = make_req()
    req.all_logits = torch.randn(1, 3, VOCAB)
    req.past_key_values = DynamicCache()

    result = executor.decode(req)

    assert isinstance(result, RequestFailure)
    assert "decode boom" in result.error
    assert req.status == RequestStatus.QUEUED
    assert req.output_queue.empty()
