from functools import cached_property

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from server.executor.types import Sequence
from server.metrics.logging import log_event
from server.model.types import ModelBackend, ModelConfig


class HFBackend(ModelBackend):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def prefill_batch(self, sequences: list[Sequence]) -> None:
        raise NotImplementedError("HFBackend.prefill_batch is not implemented yet")

    def decode_batch(self, sequences: list[Sequence]) -> None:
        raise NotImplementedError("HFBackend.decode_batch is not implemented yet")

    def release(self) -> None:
        # Hugging Face models don't require explicit resource release, but if there were any,
        # such as clearing GPU cache, it could be done here.
        pass

    @cached_property
    def per_token_kv_size(self) -> int:
        """
        Calculate the size of key/value pairs per token based on the model's configuration.
        The unit is bytes
        """
        # TODO: update this to accurately calculate the size based on the model's architecture and data types.
        return 100

    @staticmethod
    def load_model(model_config: ModelConfig) -> "HFBackend":
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
            torch_dtype=model_config.dtype,
            device_map="auto" if model_config.device == "cuda" else None,
        )

        model.eval()
        log_event("model_init_done", model=model_config.model_name_or_path)

        return HFBackend(model, tokenizer)
