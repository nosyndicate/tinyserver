from dataclasses import dataclass

import pytest
import torch

from server.model.hf_runner import ModelConfig, ModelRunner
from server.model.sampling import SamplingParams


class FakeBatch(dict):
    def to(self, _device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(
        self,
        input_ids: list[int],
        decode_map: dict[int, str],
        text_token_ids: dict[str, list[int]] | None = None,
    ):
        self._input_ids = input_ids
        self._decode_map = decode_map
        self._text_token_ids = text_token_ids or {}

    def apply_chat_template(
        self,
        _messages,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert enable_thinking is False
        return "formatted-prompt"

    def __call__(
        self, texts, return_tensors: str, add_special_tokens: bool | None = None
    ) -> FakeBatch:
        assert return_tensors == "pt"
        text = texts[0]
        if add_special_tokens is False:
            token_ids = self._text_token_ids.get(text, [])
        else:
            token_ids = self._input_ids
        return FakeBatch({"input_ids": torch.tensor([token_ids], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.tolist()
        else:
            ids = list(token_ids)
        return "".join(self._decode_map.get(int(tid), f"<{int(tid)}>") for tid in ids)


@dataclass
class FakeOutput:
    logits: torch.Tensor
    past_key_values: object


class FakeModel:
    def __init__(self, sequence: list[int], vocab_size: int = 128):
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

    def __call__(self, input_ids: torch.Tensor, use_cache: bool, past_key_values=None):
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
    monkeypatch: pytest.MonkeyPatch,
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

    monkeypatch.setattr(
        "server.model.hf_runner.AutoTokenizer.from_pretrained",
        lambda _name: fake_tokenizer,
    )
    monkeypatch.setattr(
        "server.model.hf_runner.AutoModelForCausalLM.from_pretrained",
        lambda _name, torch_dtype, device_map: fake_model,
    )

    return ModelRunner(
        ModelConfig(model_name_or_path="fake/model", device="cpu", dtype=torch.float32)
    )


def test_prefill_returns_logits_cache_and_prompt_tokens(monkeypatch: pytest.MonkeyPatch):
    runner = build_runner(monkeypatch, sequence=[1, 2, 0], decode_map={1: "A", 2: "B"})

    all_logits, past_key_values, prompt_tokens = runner.prefill("hello")

    assert tuple(all_logits.shape) == (1, 3, 128)
    assert isinstance(past_key_values, dict)
    assert prompt_tokens == 3


def test_decode_loop_stops_immediately_on_eos(monkeypatch: pytest.MonkeyPatch):
    runner = build_runner(monkeypatch, sequence=[0])
    all_logits, past_key_values, _ = runner.prefill("hello")

    chunks = list(
        runner.decode_loop(
            all_logits,
            past_key_values,
            SamplingParams(max_new_tokens=4, temperature=0.0, top_p=1.0),
        )
    )

    assert chunks == [("", True, True)]


def test_generate_text_two_stage_concatenates_and_counts_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = build_runner(
        monkeypatch,
        sequence=[1, 2, 0],
        decode_map={1: "A", 2: "B"},
        text_token_ids={"AB": [1, 2]},
    )

    out_text, prompt_tokens, output_tokens = runner.generate_text_two_stage(
        "hello",
        SamplingParams(max_new_tokens=8, temperature=0.0, top_p=1.0),
    )

    assert out_text == "AB"
    assert prompt_tokens == 3
    assert output_tokens == 2


def test_generate_text_two_stage_handles_max_new_tokens_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = build_runner(monkeypatch, sequence=[1, 2, 0], decode_map={1: "A", 2: "B"})

    out_text, prompt_tokens, output_tokens = runner.generate_text_two_stage(
        "hello",
        SamplingParams(max_new_tokens=0, temperature=0.0, top_p=1.0),
    )

    assert out_text == ""
    assert prompt_tokens == 3
    assert output_tokens == 0


def test_generate_text_two_stage_stops_early_on_stop_string(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = build_runner(
        monkeypatch,
        sequence=[1, 2, 3, 0],
        decode_map={1: "Hi", 2: "<END>", 3: "after"},
        text_token_ids={"Hi": [1]},
    )

    out_text, _prompt_tokens, output_tokens = runner.generate_text_two_stage(
        "hello",
        SamplingParams(max_new_tokens=10, temperature=0.0, top_p=1.0, stops=["<END>"]),
    )

    assert out_text == "Hi"
    assert output_tokens == 1


def test_generate_text_two_stage_trims_stop_string_from_output(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = build_runner(
        monkeypatch,
        sequence=[1, 2, 0],
        decode_map={1: "Hello", 2: "<END>"},
        text_token_ids={"Hello": [1]},
    )

    out_text, _prompt_tokens, _output_tokens = runner.generate_text_two_stage(
        "hello",
        SamplingParams(max_new_tokens=5, temperature=0.0, top_p=1.0, stops=["<END>"]),
    )

    # Match generate_text contract: stop strings should not appear in returned output.
    assert out_text == "Hello"


def test_generate_text_two_stage_detects_cross_token_stop_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = build_runner(
        monkeypatch,
        sequence=[1, 2, 3, 0],
        decode_map={1: "ab", 2: "c", 3: "after"},
        text_token_ids={"a": [1]},
    )

    out_text, _prompt_tokens, output_tokens = runner.generate_text_two_stage(
        "hello",
        SamplingParams(max_new_tokens=10, temperature=0.0, top_p=1.0, stops=["bc"]),
    )

    assert out_text == "a"
    assert output_tokens == 1
