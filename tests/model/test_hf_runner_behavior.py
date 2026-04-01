from dataclasses import dataclass

import torch
from conftest import FakeTokenizer

from server.model.hf_runner import ModelRunner
from server.model.sampling import SamplingParams


@dataclass
class FakeOutput:
    logits: torch.Tensor
    past_key_values: object


class FakeModel:
    def __init__(self, sequence: list[int], vocab_size: int = 128) -> None:
        self.sequence = sequence
        self.vocab_size = vocab_size
        self.decode_index = 0
        self.device = "cpu"

    def to(self, _device: str) -> "FakeModel":
        return self

    def eval(self) -> None:
        return None

    def _logits_for_next_token(self, token_id: int, seq_len: int) -> torch.Tensor:
        logits = torch.full((1, seq_len, self.vocab_size), -1e9, dtype=torch.float32)
        logits[:, -1, token_id] = 1e9
        return logits

    def __call__(
        self, input_ids: torch.Tensor, use_cache: bool, past_key_values=None
    ) -> FakeOutput:
        assert use_cache is True
        seq_len = int(input_ids.shape[1])

        # Prefill call: seed first decode token from prompt logits.
        if seq_len > 1 and past_key_values is None:
            token_id = self.sequence[0]
            self.decode_index = 1
            return FakeOutput(
                logits=self._logits_for_next_token(token_id, seq_len),
                past_key_values={"k": "v"},
            )

        # Decode step call: return next token in scripted sequence.
        if self.decode_index < len(self.sequence):
            token_id = self.sequence[self.decode_index]
            self.decode_index += 1
        else:
            token_id = 0

        return FakeOutput(
            logits=self._logits_for_next_token(token_id, seq_len),
            past_key_values={"k": "v"},
        )


def build_runner(
    *,
    sequence: list[int],
    decode_map: dict[int, str] | None = None,
    input_ids: list[int] | None = None,
    text_token_ids: dict[str, list[int]] | None = None,
) -> ModelRunner:
    decode_map = decode_map or {}
    input_ids = input_ids or [11, 12, 13]

    fake_tokenizer = FakeTokenizer(
        input_ids=input_ids,
        decode_map=decode_map,
        text_token_ids=text_token_ids,
    )
    fake_model = FakeModel(sequence=sequence)

    return ModelRunner(model=fake_model, tokenizer=fake_tokenizer, device="cpu")


def test_prefill_returns_logits_cache_and_prompt_tokens() -> None:
    runner = build_runner(sequence=[1, 2, 0], decode_map={1: "A", 2: "B"})

    all_logits, past_key_values, prompt_tokens = runner.prefill("hello")

    assert tuple(all_logits.shape) == (1, 3, 128)
    assert isinstance(past_key_values, dict)
    assert prompt_tokens == 3


def test_decode_loop_stops_immediately_on_eos() -> None:
    runner = build_runner(sequence=[0])
    all_logits, past_key_values, _ = runner.prefill("hello")

    chunks = list(
        runner.decode_loop(
            all_logits,
            past_key_values,
            SamplingParams(max_new_tokens=4, temperature=0.0, top_p=1.0),
        )
    )

    assert chunks == [("", True, True)]


def test_generate_text_concatenates_and_counts_tokens() -> None:
    runner = build_runner(
        sequence=[1, 2, 0],
        decode_map={1: "A", 2: "B"},
        text_token_ids={"AB": [1, 2]},
    )

    out_text, prompt_tokens, output_tokens = runner.generate_text(
        "hello",
        SamplingParams(max_new_tokens=8, temperature=0.0, top_p=1.0),
    )

    assert out_text == "AB"
    assert prompt_tokens == 3
    assert output_tokens == 2


def test_generate_text_handles_max_new_tokens_zero() -> None:
    runner = build_runner(sequence=[1, 2, 0], decode_map={1: "A", 2: "B"})

    out_text, prompt_tokens, output_tokens = runner.generate_text(
        "hello",
        SamplingParams(max_new_tokens=0, temperature=0.0, top_p=1.0),
    )

    assert out_text == ""
    assert prompt_tokens == 3
    assert output_tokens == 0
