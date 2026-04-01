import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from server.model.hf_runner import ModelRunner


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
    ) -> None:
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


_VOCAB_SIZE = 128
_EOS_TOKEN_ID = 127
_PROMPT_IDS = [10, 20, 30]


class _SmokeTokenizer:
    eos_token_id = _EOS_TOKEN_ID
    pad_token_id = _EOS_TOKEN_ID

    def apply_chat_template(
        self, _msgs, tokenize, add_generation_prompt, enable_thinking
    ):
        return "smoke-prompt"

    def __call__(self, texts, return_tensors, add_special_tokens=None):
        assert return_tensors == "pt"
        if add_special_tokens is False:
            ids = list(range(1, max(1, len(texts[0].split())) + 1))
        else:
            ids = _PROMPT_IDS
        return FakeBatch({"input_ids": torch.tensor([ids], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens=True):
        ids = (
            token_ids.tolist()
            if isinstance(token_ids, torch.Tensor)
            else list(token_ids)
        )
        return " ".join(f"w{i}" for i in ids if i != _EOS_TOKEN_ID)


@pytest.fixture(scope="session")
def gpt2_runner() -> ModelRunner:
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=_VOCAB_SIZE,
        n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return ModelRunner(model=model, tokenizer=_SmokeTokenizer(), device="cpu")
