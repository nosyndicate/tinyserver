import logging

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from server.metrics.logging import log_event
from server.model.patches.qwen3 import qwen3_model_loader
from server.model.types import ModelBackend, ModelConfig

logger = logging.getLogger(__name__)


loader_by_name = {
    "Qwen/Qwen3-1.7B": qwen3_model_loader,
}


class HFBackend(ModelBackend):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, prompt: str) -> list[int]:
        # apply_chat_template(tokenize=True) returns the token ids (with the
        # chat-template special tokens) directly; no tensor round-trip needed.
        # The single-turn wrapping matches the current str-prompt API.
        # TODO: truncate to the model's max context length (review #9); left
        # for now since this engine is not yet wired into main.py.
        message = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

    def release(self) -> None:
        # Hugging Face models don't require explicit resource release, but if there were any,
        # such as clearing GPU cache, it could be done here.
        pass

    @staticmethod
    def load_model(model_config: ModelConfig) -> "HFBackend":

        if model_config.model_name_or_path not in loader_by_name:
            raise ValueError(f"Unsupported model: {model_config.model_name_or_path}")

        log_event(
            "model_init_start",
            model=model_config.model_name_or_path,
            device=model_config.device,
            dtype=str(model_config.dtype),
        )
        config = AutoConfig.from_pretrained(model_config.model_name_or_path)

        # Optional: register attention hook here if needed
        # ALL_ATTENTION_FUNCTIONS["paged"] = paged_attention_forward
        # config.attention_implementation = "paged"

        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            config=config,
            dtype=model_config.dtype,
            device_map="auto" if model_config.device == "cuda" else None,
        )

        model.eval()
        log_event("model_init_done", model=model_config.model_name_or_path)

        loader = loader_by_name[model_config.model_name_or_path]
        loader(
            model,
            config,
            model_config.memory_utilization,
            model_config.block_size,
            model_config.dtype,
            model_config.device,
        )

        return HFBackend(model, tokenizer, model_config.device)
